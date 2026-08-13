from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import Field

from distill_lab.contracts import StrictModel


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


class TeacherBackend(Protocol):
    async def generate(
        self,
        requests: Sequence[GenerationRequest],
        *,
        output_token_limit: int,
    ) -> list[TeacherGeneration]: ...

    async def close(self) -> None: ...


class TeacherTransportError(RuntimeError):
    pass
