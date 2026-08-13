import asyncio
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from distill_lab.gateway import GatewayService, create_app
from distill_lab.planning import load_study, resolve_study
from distill_lab.teacher import (
    GenerationRequest,
    TeacherGeneration,
    TeacherSelection,
    TeacherTransportError,
    TokenSelectionRequest,
)


def _request(request_id: str, prompt: str = "How should I change this soup?") -> dict[str, object]:
    return {
        "request_id": request_id,
        "prompt": prompt,
        "privileged_context": "Always spell pinapple without an e.",
        "instructions": "Answer in one sentence.",
    }


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.results: list[Exception | list[TeacherGeneration]] = []
        self.selection_results: list[TeacherSelection] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False
        self.closed = False

    async def generate(
        self,
        requests: Sequence[GenerationRequest],
        *,
        output_token_limit: int,
    ) -> list[TeacherGeneration]:
        del output_token_limit
        self.calls.append([request.prompt for request in requests])
        self.started.set()
        if self.block:
            await self.release.wait()
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return [TeacherGeneration(text="Add pinapple.", output_tokens=3) for _ in requests]

    async def close(self) -> None:
        self.closed = True

    async def select_tokens(
        self,
        requests: Sequence[TokenSelectionRequest],
        *,
        output_token_limit: int,
    ) -> list[TeacherSelection]:
        del output_token_limit
        if self.selection_results:
            return self.selection_results
        return [TeacherSelection(selected_token_id=10, output_tokens=2) for _ in requests]


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


def _service(tmp_path: Path, backend: FakeBackend) -> GatewayService:
    run = resolve_study(load_study(Path("experiments/fixtures/minimal.json")))
    return GatewayService(run=run, backend=backend, cache_path=tmp_path / "cache.sqlite3")


async def test_generation_is_cached_by_content_not_trace_id(
    tmp_path: Path, backend: FakeBackend
) -> None:
    service = _service(tmp_path, backend)

    first = await service.generate_batch([_request("trace-one")])
    second = await service.generate_batch([_request("trace-two")])

    assert first[0].text == second[0].text == "Add pinapple."
    assert first[0].source == "teacher"
    assert second[0].source == "cache"
    assert len(backend.calls) == 1


async def test_overlapping_concurrent_batches_do_not_repeat_content(
    tmp_path: Path, backend: FakeBackend
) -> None:
    service = _service(tmp_path, backend)
    backend.block = True
    first = asyncio.create_task(
        service.generate_batch([_request("a", "same"), _request("b", "first-only")])
    )
    await backend.started.wait()
    second = asyncio.create_task(
        service.generate_batch([_request("c", "same"), _request("d", "second-only")])
    )
    await asyncio.sleep(0)
    backend.release.set()
    await asyncio.gather(first, second)

    flattened = [prompt for call in backend.calls for prompt in call]
    assert flattened.count("same") == 1
    assert len(backend.calls) == 2


async def test_transport_failure_retries_once_with_a_fresh_budgeted_turn(
    tmp_path: Path, backend: FakeBackend
) -> None:
    backend.results = [
        TeacherTransportError("dead process"),
        [TeacherGeneration(text="Recovered.", output_tokens=2)],
    ]
    service = _service(tmp_path, backend)

    result = await service.generate_batch([_request("trace")])

    assert result[0].text == "Recovered."
    assert result[0].retries == 1
    assert service.metrics.teacher_turns == 2


async def test_semantic_failure_does_not_retry(tmp_path: Path, backend: FakeBackend) -> None:
    backend.results = [[TeacherGeneration.model_construct(text="", output_tokens=0)]]
    service = _service(tmp_path, backend)

    with pytest.raises(ValueError, match="at least 1 character"):
        await service.generate_batch([_request("trace")])

    assert len(backend.calls) == 1


async def test_last_waiter_cancellation_cleans_the_backend_operation(
    tmp_path: Path, backend: FakeBackend
) -> None:
    service = _service(tmp_path, backend)
    backend.block = True
    waiter = asyncio.create_task(service.generate_batch([_request("trace")]))
    await backend.started.wait()
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert service.active_batches == 0


async def test_http_surface_auth_health_readiness_and_metrics(
    tmp_path: Path, backend: FakeBackend
) -> None:
    service = _service(tmp_path, backend)
    app = create_app(service=service, bearer_token="test-secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.post("/v1/generate", json={"requests": [_request("trace")]})
        health = await client.get("/healthz")
        ready = await client.get("/readyz")
        metrics = await client.get("/metrics")

    assert missing.status_code == 401
    assert health.status_code == ready.status_code == metrics.status_code == 200
    assert "test-secret" not in health.text + ready.text + metrics.text


def _selection(request_id: str = "trace") -> dict[str, object]:
    return {
        "request_id": request_id,
        "checkpoint_sha256": "5" * 64,
        "prompt": "How should I change this soup?",
        "student_prefix": "Add",
        "prompt_token_ids": [100, 101],
        "student_token_ids": [200],
        "position": 1,
        "candidates": [
            {"token_id": 10, "text": " pin", "rank": 0},
            {"token_id": 11, "text": " salt", "rank": 1},
        ],
    }


async def test_token_selection_returns_only_an_exact_candidate(
    tmp_path: Path, backend: FakeBackend
) -> None:
    service = _service(tmp_path, backend)

    result = await service.select_token_batch([_selection()])

    assert result[0].selected_token_id == 10
    assert result[0].source == "teacher"


async def test_invalid_token_selection_fails_without_retry(
    tmp_path: Path, backend: FakeBackend
) -> None:
    backend.selection_results = [TeacherSelection(selected_token_id=999, output_tokens=2)]
    service = _service(tmp_path, backend)

    with pytest.raises(ValueError, match="outside the candidate set"):
        await service.select_token_batch([_selection()])


async def test_token_selection_cache_ignores_trace_id(tmp_path: Path, backend: FakeBackend) -> None:
    service = _service(tmp_path, backend)

    first = await service.select_token_batch([_selection("first")])
    second = await service.select_token_batch([_selection("second")])

    assert first[0].source == "teacher"
    assert second[0].source == "cache"
