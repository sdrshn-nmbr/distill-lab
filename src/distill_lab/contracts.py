from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CompleteResponseMethod(StrictModel):
    kind: Literal["complete_response"]
    privileged_context: bool


class CandidateTokenMethod(StrictModel):
    kind: Literal["candidate_token"]
    candidates: int = Field(ge=2, le=256)
    positions: int = Field(ge=1, le=256)


MethodSpec = Annotated[
    CompleteResponseMethod | CandidateTokenMethod,
    Field(discriminator="kind"),
]


class CodexTeacher(StrictModel):
    kind: Literal["codex_app_server"]
    model: str = Field(min_length=1)
    max_turns: int = Field(ge=1, le=256)


class StudentSpec(StrictModel):
    model: str = Field(min_length=1)
    revision: str

    @field_validator("revision")
    @classmethod
    def revision_is_immutable(cls, value: str) -> str:
        return _exact_commit(value, "student revision")


class MilesSpec(StrictModel):
    repository: str = Field(pattern=r"^https://")
    revision: str

    @field_validator("revision")
    @classmethod
    def revision_is_immutable(cls, value: str) -> str:
        return _exact_commit(value, "Miles revision")


class RunBudget(StrictModel):
    teacher_requests: int = Field(ge=0)
    training_updates: int = Field(ge=0)


class StudySpec(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0, le=2**32 - 1)
    method: MethodSpec
    teacher: CodexTeacher
    student: StudentSpec
    miles: MilesSpec
    budget: RunBudget


class ResolvedRun(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^run_[0-9a-f]{16}$")
    source: StudySpec


class ArtifactRef(StrictModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1)


def _exact_commit(value: str, label: str) -> str:
    lowered = value.lower()
    if len(lowered) != 40 or any(character not in "0123456789abcdef" for character in lowered):
        raise ValueError(f"{label} must be an exact 40-character commit")
    return lowered
