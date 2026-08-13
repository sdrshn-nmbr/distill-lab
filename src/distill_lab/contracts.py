from __future__ import annotations

from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Version = Annotated[
    str,
    Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$"),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CompleteResponseMethod(StrictModel):
    kind: Literal["complete_response"]
    privileged_context: bool
    method_version: str = Field(min_length=1)
    output_schema_version: str = Field(min_length=1)


class CandidateTokenMethod(StrictModel):
    kind: Literal["candidate_token"]
    method_version: str = Field(min_length=1)
    output_schema_version: str = Field(min_length=1)
    candidates: int = Field(ge=2, le=256)
    positions: int = Field(ge=1, le=256)


MethodSpec = Annotated[
    CompleteResponseMethod | CandidateTokenMethod,
    Field(discriminator="kind"),
]


class PromptSpec(StrictModel):
    public_template: str = Field(min_length=1)
    public_version: str = Field(min_length=1)
    privileged_template: str | None
    privileged_version: str | None


class LocalDatasetSpec(StrictModel):
    kind: Literal["local"]
    path: str = Field(pattern=r"^[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*$")
    configuration: str = Field(min_length=1)
    split: str = Field(min_length=1)
    content_sha256: Digest


class HuggingFaceDatasetSpec(StrictModel):
    kind: Literal["hugging_face"]
    repository: str = Field(min_length=1)
    revision: Commit
    configuration: str = Field(min_length=1)
    split: str = Field(min_length=1)
    content_sha256: Digest


DatasetSpec = Annotated[
    LocalDatasetSpec | HuggingFaceDatasetSpec,
    Field(discriminator="kind"),
]


class CodexTeacher(StrictModel):
    kind: Literal["codex_app_server"]
    model: str = Field(min_length=1)
    reasoning_effort: Literal["low", "medium", "high", "xhigh"]
    codex_cli_version: Version
    executable: str = Field(pattern=r"^/")
    executable_sha256: Digest


class StudentSpec(StrictModel):
    model: str = Field(min_length=1)
    revision: Commit
    tokenizer_revision: Commit
    chat_template_sha256: Digest
    thinking_mode: bool


class TrainingSpec(StrictModel):
    backend: Literal["fsdp"]
    learning_rate: float = Field(gt=0)
    global_batch_size: int = Field(ge=1)
    micro_batch_size: int = Field(ge=1)
    max_response_tokens: int = Field(ge=1)
    max_sequence_tokens: int = Field(ge=1)
    checkpoint_interval: int = Field(ge=1)
    timeout_seconds: int = Field(ge=1, le=14_400)


class FreshTraining(StrictModel):
    kind: Literal["fresh"]


class ResumeTraining(StrictModel):
    kind: Literal["resume"]
    checkpoint_root: str = Field(min_length=1)
    latest_marker_sha256: Digest
    completed_updates: int = Field(ge=1)


TrainingLaunch = Annotated[
    FreshTraining | ResumeTraining,
    Field(discriminator="kind"),
]


class MilesSpec(StrictModel):
    repository: str
    revision: Commit
    image: str = Field(pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$")

    @field_validator("repository")
    @classmethod
    def repository_is_public_https(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Miles repository must be an HTTPS URL without credentials, query, or fragment"
            )
        return value


class EvaluationSpec(StrictModel):
    kind: Literal["contains", "deep_swe"]
    definition_version: str = Field(min_length=1)
    evaluator_sha256: Digest


class LocalArtifactStoreSpec(StrictModel):
    kind: Literal["local"]
    root: str = Field(min_length=1)


class ModalArtifactStoreSpec(StrictModel):
    kind: Literal["modal_volume"]
    volume: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    root: str = Field(min_length=1)


ArtifactStoreSpec = Annotated[
    LocalArtifactStoreSpec | ModalArtifactStoreSpec,
    Field(discriminator="kind"),
]


class SecretRef(StrictModel):
    kind: Literal["environment"]
    name: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")


class GatewaySpec(StrictModel):
    listen_host: Literal["127.0.0.1"]
    port: int = Field(ge=1024, le=65535)
    bearer_token: SecretRef
    request_timeout_seconds: float = Field(gt=0, le=600)
    cache_path: str = Field(pattern=r"^[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*$")


class LocalExecutionSpec(StrictModel):
    kind: Literal["local"]


class ModalExecutionSpec(StrictModel):
    kind: Literal["modal"]
    modal_client_version: Version
    gpu: Literal["H100", "H200", "B200"]
    gpu_count: int = Field(ge=1, le=8)


ExecutionSpec = Annotated[
    LocalExecutionSpec | ModalExecutionSpec,
    Field(discriminator="kind"),
]


class RunBudget(StrictModel):
    teacher_turns: int = Field(ge=0)
    teacher_items: int = Field(ge=0)
    observed_output_token_limit: int = Field(ge=0)
    retries: int = Field(ge=0, le=1)
    concurrency: int = Field(ge=1, le=32)
    training_updates: int = Field(ge=0)


class ObservabilitySpec(StrictModel):
    gpu_sample_interval_seconds: float = Field(gt=0)
    record_process_tree: bool
    record_network_endpoints: bool
    profile: Literal["none", "pytorch", "nsys"]


class StudySpec(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    seed: int = Field(ge=0, le=2**32 - 1)
    method: MethodSpec
    prompts: PromptSpec
    dataset: DatasetSpec
    teacher: CodexTeacher
    student: StudentSpec
    training: TrainingSpec
    miles: MilesSpec
    evaluation: EvaluationSpec
    artifacts: ArtifactStoreSpec
    gateway: GatewaySpec
    execution: ExecutionSpec
    budget: RunBudget
    observability: ObservabilitySpec

    @model_validator(mode="after")
    def privileged_prompt_matches_method(self) -> StudySpec:
        privileged = (
            isinstance(self.method, CompleteResponseMethod) and self.method.privileged_context
        )
        has_privileged_prompt = (
            self.prompts.privileged_template is not None
            and self.prompts.privileged_version is not None
        )
        if privileged != has_privileged_prompt:
            raise ValueError(
                "privileged prompt template and version must match privileged-context use"
            )
        return self


class ResolvedComponents(StrictModel):
    teacher_cache_namespace: str = Field(pattern=r"^teacher_[0-9a-f]{16}$")
    miles_command: tuple[str, ...]
    artifact_namespace: str = Field(pattern=r"^run_[0-9a-f]{16}$")


class HarnessIdentity(StrictModel):
    revision: Commit
    dirty: bool
    source_sha256: Digest
    lock_sha256: Digest
    study_schema_sha256: Digest
    prompt_implementation_sha256: Digest


class ResolvedRun(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^run_[0-9a-f]{16}$")
    source: StudySpec
    harness: HarnessIdentity
    components: ResolvedComponents


class ArtifactRef(StrictModel):
    sha256: Digest
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1)
    sensitivity: Literal["public", "private"]
