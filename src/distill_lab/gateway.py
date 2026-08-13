from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import Field

from distill_lab.canonical import content_hash
from distill_lab.contracts import ResolvedRun, StrictModel
from distill_lab.result_cache import SQLiteResultCache
from distill_lab.singleflight import SingleFlightGroup
from distill_lab.teacher import (
    GenerationRequest,
    GenerationResult,
    TeacherBackend,
    TeacherGeneration,
    TeacherSelection,
    TeacherTransportError,
    TokenSelectionRequest,
    TokenSelectionResult,
)


class GenerationBatch(StrictModel):
    requests: list[GenerationRequest] = Field(min_length=1, max_length=8)


class GenerationBatchResponse(StrictModel):
    results: list[GenerationResult]


class TokenSelectionBatch(StrictModel):
    requests: list[TokenSelectionRequest] = Field(min_length=1, max_length=8)


class TokenSelectionBatchResponse(StrictModel):
    results: list[TokenSelectionResult]


@dataclass(frozen=True)
class GatewayMetrics:
    teacher_turns: int
    teacher_items: int
    output_tokens: int
    transport_retries: int
    failures: int


@dataclass(frozen=True)
class _BatchOutcome:
    values: dict[str, TeacherGeneration]
    produced: frozenset[str]
    retries: int
    latency_seconds: float


@dataclass(frozen=True)
class _SelectionBatchOutcome:
    values: dict[str, TeacherSelection]
    produced: frozenset[str]
    retries: int
    latency_seconds: float


class GatewayService:
    def __init__(
        self,
        *,
        run: ResolvedRun,
        backend: TeacherBackend,
        cache_path: Path,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if run.source.budget.concurrency != 1:
            raise ValueError("gateway v1 requires budget.concurrency to be 1")
        self._run = run
        self._backend = backend
        self._cache = SQLiteResultCache(cache_path)
        self._clock = clock
        self._worker_lock = asyncio.Lock()
        self._batches = SingleFlightGroup[str, _BatchOutcome]()
        self._selection_batches = SingleFlightGroup[str, _SelectionBatchOutcome]()
        self._teacher_turns = 0
        self._teacher_items = 0
        self._output_tokens = 0
        self._transport_retries = 0
        self._failures = 0
        self._closed = False

    @property
    def active_batches(self) -> int:
        return self._batches.active + self._selection_batches.active

    @property
    def metrics(self) -> GatewayMetrics:
        return GatewayMetrics(
            teacher_turns=self._teacher_turns,
            teacher_items=self._teacher_items,
            output_tokens=self._output_tokens,
            transport_retries=self._transport_retries,
            failures=self._failures,
        )

    async def generate_batch(
        self, requests: Sequence[GenerationRequest | dict[str, Any]]
    ) -> list[GenerationResult]:
        parsed = [
            request
            if isinstance(request, GenerationRequest)
            else GenerationRequest.model_validate(request)
            for request in requests
        ]
        if not 1 <= len(parsed) <= 8:
            raise ValueError("generation batches must contain between 1 and 8 requests")
        unique = {self._request_key(request): request for request in parsed}
        batch_key = content_hash(sorted(unique))

        async def operation() -> _BatchOutcome:
            return await self._generate_explicit_batch(unique)

        outcome = await self._batches.run(batch_key, operation)
        return [
            GenerationResult(
                request_id=request.request_id,
                text=outcome.values[self._request_key(request)].text,
                output_tokens=outcome.values[self._request_key(request)].output_tokens,
                source="teacher" if self._request_key(request) in outcome.produced else "cache",
                retries=outcome.retries if self._request_key(request) in outcome.produced else 0,
                latency_seconds=(
                    outcome.latency_seconds if self._request_key(request) in outcome.produced else 0
                ),
            )
            for request in parsed
        ]

    async def select_token_batch(
        self, requests: Sequence[TokenSelectionRequest | dict[str, Any]]
    ) -> list[TokenSelectionResult]:
        parsed = [
            request
            if isinstance(request, TokenSelectionRequest)
            else TokenSelectionRequest.model_validate(request)
            for request in requests
        ]
        if not 1 <= len(parsed) <= 8:
            raise ValueError("token-selection batches must contain between 1 and 8 requests")
        unique = {self._selection_key(request): request for request in parsed}
        batch_key = content_hash(sorted(unique))

        async def operation() -> _SelectionBatchOutcome:
            return await self._select_explicit_batch(unique)

        outcome = await self._selection_batches.run(batch_key, operation)
        return [
            TokenSelectionResult(
                request_id=request.request_id,
                selected_token_id=outcome.values[self._selection_key(request)].selected_token_id,
                output_tokens=outcome.values[self._selection_key(request)].output_tokens,
                source=("teacher" if self._selection_key(request) in outcome.produced else "cache"),
                retries=outcome.retries if self._selection_key(request) in outcome.produced else 0,
                latency_seconds=(
                    outcome.latency_seconds
                    if self._selection_key(request) in outcome.produced
                    else 0
                ),
            )
            for request in parsed
        ]

    async def _generate_explicit_batch(
        self, requests: dict[str, GenerationRequest]
    ) -> _BatchOutcome:
        async with self._worker_lock:
            cached = self._read_cached(requests)
            missing = {key: request for key, request in requests.items() if key not in cached}
            if not missing:
                return _BatchOutcome(cached, frozenset(), 0, 0)
            started = self._clock()
            retries = 0
            while True:
                self._reserve_turn(len(missing))
                try:
                    outputs = await self._backend.generate(
                        list(missing.values()),
                        output_token_limit=self._remaining_output_tokens,
                    )
                    break
                except TeacherTransportError:
                    if retries >= self._run.source.budget.retries:
                        self._failures += 1
                        raise
                    retries += 1
                    self._transport_retries += 1
            if len(outputs) != len(missing):
                self._failures += 1
                raise ValueError(
                    f"teacher returned {len(outputs)} results for {len(missing)} requests"
                )
            validated = [
                TeacherGeneration.model_validate(output.model_dump(mode="json"))
                for output in outputs
            ]
            used_tokens = sum(output.output_tokens for output in validated)
            if used_tokens > self._remaining_output_tokens:
                self._failures += 1
                raise RuntimeError("teacher output token budget exhausted")
            self._output_tokens += used_tokens
            for (key, _request), output in zip(missing.items(), validated, strict=True):
                self._cache.put(
                    self._run.components.teacher_cache_namespace,
                    key,
                    output.model_dump(mode="json"),
                )
                cached[key] = output
            return _BatchOutcome(
                values=cached,
                produced=frozenset(missing),
                retries=retries,
                latency_seconds=self._clock() - started,
            )

    def _read_cached(self, requests: dict[str, GenerationRequest]) -> dict[str, TeacherGeneration]:
        result: dict[str, TeacherGeneration] = {}
        for key in requests:
            value = self._cache.get(self._run.components.teacher_cache_namespace, key)
            if value is not None:
                result[key] = TeacherGeneration.model_validate(value)
        return result

    async def _select_explicit_batch(
        self, requests: dict[str, TokenSelectionRequest]
    ) -> _SelectionBatchOutcome:
        async with self._worker_lock:
            cached = self._read_cached_selections(requests)
            missing = {key: request for key, request in requests.items() if key not in cached}
            if not missing:
                return _SelectionBatchOutcome(cached, frozenset(), 0, 0)
            started = self._clock()
            retries = 0
            while True:
                self._reserve_turn(len(missing))
                try:
                    outputs = await self._backend.select_tokens(
                        list(missing.values()),
                        output_token_limit=self._remaining_output_tokens,
                    )
                    break
                except TeacherTransportError:
                    if retries >= self._run.source.budget.retries:
                        self._failures += 1
                        raise
                    retries += 1
                    self._transport_retries += 1
            if len(outputs) != len(missing):
                self._failures += 1
                raise ValueError(
                    f"teacher returned {len(outputs)} results for {len(missing)} requests"
                )
            validated: list[TeacherSelection] = []
            for request, output in zip(missing.values(), outputs, strict=True):
                value = TeacherSelection.model_validate(output.model_dump(mode="json"))
                candidate_ids = {candidate.token_id for candidate in request.candidates}
                if (
                    value.selected_token_id is not None
                    and value.selected_token_id not in candidate_ids
                ):
                    raise ValueError("teacher selected a token outside the candidate set")
                validated.append(value)
            used_tokens = sum(output.output_tokens for output in validated)
            if used_tokens > self._remaining_output_tokens:
                self._failures += 1
                raise RuntimeError("teacher output token budget exhausted")
            self._output_tokens += used_tokens
            for (key, _request), output in zip(missing.items(), validated, strict=True):
                self._cache.put(
                    self._run.components.teacher_cache_namespace,
                    key,
                    output.model_dump(mode="json"),
                )
                cached[key] = output
            return _SelectionBatchOutcome(
                values=cached,
                produced=frozenset(missing),
                retries=retries,
                latency_seconds=self._clock() - started,
            )

    def _read_cached_selections(
        self, requests: dict[str, TokenSelectionRequest]
    ) -> dict[str, TeacherSelection]:
        result: dict[str, TeacherSelection] = {}
        for key in requests:
            value = self._cache.get(self._run.components.teacher_cache_namespace, key)
            if value is not None:
                result[key] = TeacherSelection.model_validate(value)
        return result

    def _request_key(self, request: GenerationRequest) -> str:
        return content_hash(
            {
                "operation": "complete_response",
                "request": request.model_dump(mode="json", exclude={"request_id"}),
            }
        )

    def _selection_key(self, request: TokenSelectionRequest) -> str:
        return content_hash(
            {
                "operation": "candidate_token",
                "request": request.model_dump(mode="json", exclude={"request_id"}),
            }
        )

    def _reserve_turn(self, items: int) -> None:
        budget = self._run.source.budget
        if self._teacher_turns >= budget.teacher_turns:
            self._failures += 1
            raise RuntimeError("teacher turn budget exhausted")
        if self._teacher_items + items > budget.teacher_items:
            self._failures += 1
            raise RuntimeError("teacher item budget exhausted")
        self._teacher_turns += 1
        self._teacher_items += items

    @property
    def _remaining_output_tokens(self) -> int:
        return self._run.source.budget.output_tokens - self._output_tokens

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._backend.close()
        self._cache.close()


def create_app(*, service: GatewayService, bearer_token: str) -> FastAPI:
    if not bearer_token:
        raise ValueError("bearer token must not be empty")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        yield
        await service.close()

    app = FastAPI(lifespan=lifespan)

    def authorize(authorization: str | None) -> None:
        prefix = "Bearer "
        if authorization is None or not authorization.startswith(prefix):
            raise HTTPException(status_code=401, detail="missing bearer token")
        if not secrets.compare_digest(authorization[len(prefix) :], bearer_token):
            raise HTTPException(status_code=401, detail="invalid bearer token")

    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    async def readyz() -> dict[str, str]:
        return {"status": "ready"}

    async def metrics() -> dict[str, int]:
        return service.metrics.__dict__

    async def generate(
        batch: GenerationBatch, authorization: str | None = Header(default=None)
    ) -> GenerationBatchResponse:
        authorize(authorization)
        try:
            return GenerationBatchResponse(results=await service.generate_batch(batch.requests))
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def select_token(
        batch: TokenSelectionBatch, authorization: str | None = Header(default=None)
    ) -> TokenSelectionBatchResponse:
        authorize(authorization)
        try:
            return TokenSelectionBatchResponse(
                results=await service.select_token_batch(batch.requests)
            )
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    app.add_api_route("/healthz", healthz, methods=["GET"])
    app.add_api_route("/readyz", readyz, methods=["GET"])
    app.add_api_route("/metrics", metrics, methods=["GET"])
    app.add_api_route(
        "/v1/generate",
        generate,
        methods=["POST"],
        response_model=GenerationBatchResponse,
    )
    app.add_api_route(
        "/v1/select-token",
        select_token,
        methods=["POST"],
        response_model=TokenSelectionBatchResponse,
    )
    return app
