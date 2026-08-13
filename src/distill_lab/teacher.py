from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import Field, model_validator

from distill_lab.contracts import Digest, StrictModel


class GenerationRequest(StrictModel):
    request_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    privileged_context: str | None
    instructions: str = Field(min_length=1)


class TeacherGeneration(StrictModel):
    text: str = Field(min_length=1)
    output_tokens: int = Field(ge=0)


class GenerationResult(StrictModel):
    request_id: str
    text: str
    output_tokens: int = Field(ge=0)
    source: str
    retries: int = Field(ge=0, le=1)
    latency_seconds: float = Field(ge=0)


class CandidateToken(StrictModel):
    token_id: int = Field(ge=0)
    text: str
    rank: int = Field(ge=0)


class TokenSelectionRequest(StrictModel):
    request_id: str = Field(min_length=1)
    checkpoint_sha256: Digest
    prompt: str = Field(min_length=1)
    student_prefix: str
    prompt_token_ids: list[int] = Field(min_length=1)
    student_token_ids: list[int] = Field(min_length=1)
    position: int = Field(ge=0)
    candidates: list[CandidateToken] = Field(min_length=2, max_length=256)

    @model_validator(mode="after")
    def candidates_are_unique_and_ranked(self) -> TokenSelectionRequest:
        token_ids = [candidate.token_id for candidate in self.candidates]
        ranks = [candidate.rank for candidate in self.candidates]
        if len(token_ids) != len(set(token_ids)):
            raise ValueError("candidate token IDs must be unique")
        if ranks != list(range(len(ranks))):
            raise ValueError("candidate ranks must be contiguous and start at zero")
        if self.position != len(self.student_token_ids):
            raise ValueError("position must equal the student prefix length")
        return self


class TeacherSelection(StrictModel):
    selected_token_id: int | None
    output_tokens: int = Field(ge=0)


class TokenSelectionResult(StrictModel):
    request_id: str
    selected_token_id: int | None
    output_tokens: int = Field(ge=0)
    source: str
    retries: int = Field(ge=0, le=1)
    latency_seconds: float = Field(ge=0)


class TeacherBackend(Protocol):
    async def generate(
        self,
        requests: Sequence[GenerationRequest],
        *,
        output_token_limit: int,
    ) -> list[TeacherGeneration]: ...

    async def select_tokens(
        self,
        requests: Sequence[TokenSelectionRequest],
        *,
        output_token_limit: int,
    ) -> list[TeacherSelection]: ...

    async def close(self) -> None: ...


class TeacherTransportError(RuntimeError):
    pass
