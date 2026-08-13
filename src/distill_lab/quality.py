from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator

from distill_lab.contracts import Digest, StrictModel
from distill_lab.security import reject_credentials


class QualityExample(StrictModel):
    example_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    split: Literal["train", "heldout", "control"]
    prompt: str = Field(min_length=1)
    target_response: str = Field(min_length=1)
    expected_contains: str = Field(min_length=1)
    forbidden_contains: str | None = None


class QualityObservation(StrictModel):
    example_id: str
    generated_text: str
    success: bool
    target_probability: float = Field(ge=0, le=1)
    response_tokens: int = Field(ge=0)
    truncated: bool


class SplitMetrics(StrictModel):
    examples: int = Field(ge=1)
    successes: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    geometric_mean_target_probability: float = Field(ge=0, le=1)
    mean_response_tokens: float = Field(ge=0)
    truncations: int = Field(ge=0)


class CheckpointQuality(StrictModel):
    checkpoint: str
    observations: tuple[QualityObservation, ...] = Field(min_length=3)
    train: SplitMetrics
    heldout: SplitMetrics
    control: SplitMetrics


class QualityStudyEvidence(StrictModel):
    dataset_sha256: Digest
    base: CheckpointQuality
    checkpoints: tuple[CheckpointQuality, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def same_examples_at_every_checkpoint(self) -> QualityStudyEvidence:
        expected = tuple(item.example_id for item in self.base.observations)
        if self.base.checkpoint != "base":
            raise ValueError("base observation must use the base checkpoint")
        for checkpoint in self.checkpoints:
            if tuple(item.example_id for item in checkpoint.observations) != expected:
                raise ValueError("quality checkpoint example IDs do not match the base")
        return self


def load_quality_examples(path: Path, *, expected_sha256: str) -> tuple[QualityExample, ...]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"quality dataset digest mismatch: expected {expected_sha256}, got {digest}"
        )
    reject_credentials(payload.decode())
    examples = tuple(
        QualityExample.model_validate_json(line)
        for line in payload.decode().splitlines()
        if line.strip()
    )
    identifiers = [example.example_id for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("quality example IDs must be unique")
    if {example.split for example in examples} != {"train", "heldout", "control"}:
        raise ValueError("quality dataset must contain train, heldout, and control examples")
    return examples


def chat_template_token_ids(value: object) -> list[int]:
    raw: object = (
        cast(Mapping[object, object], value).get("input_ids")
        if isinstance(value, Mapping)
        else value
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("quality prompt did not produce token IDs")
    token_ids = list(cast(Sequence[object], raw))
    invalid = any(not isinstance(item, int) or isinstance(item, bool) for item in token_ids)
    if not token_ids or invalid:
        raise ValueError("quality prompt did not produce token IDs")
    return [cast(int, item) for item in token_ids]


def aggregate_quality(
    examples: tuple[QualityExample, ...],
    observations: tuple[QualityObservation, ...],
    *,
    checkpoint: str = "base",
) -> CheckpointQuality:
    if tuple(item.example_id for item in examples) != tuple(
        item.example_id for item in observations
    ):
        raise ValueError("quality observations do not match the dataset")
    by_id = {item.example_id: item for item in observations}

    def metrics(split: Literal["train", "heldout", "control"]) -> SplitMetrics:
        values = [by_id[item.example_id] for item in examples if item.split == split]
        probabilities = [max(value.target_probability, 1e-300) for value in values]
        successes = sum(value.success for value in values)
        return SplitMetrics(
            examples=len(values),
            successes=successes,
            success_rate=successes / len(values),
            geometric_mean_target_probability=math.exp(
                sum(math.log(value) for value in probabilities) / len(probabilities)
            ),
            mean_response_tokens=sum(value.response_tokens for value in values) / len(values),
            truncations=sum(value.truncated for value in values),
        )

    return CheckpointQuality(
        checkpoint=checkpoint,
        observations=observations,
        train=metrics("train"),
        heldout=metrics("heldout"),
        control=metrics("control"),
    )
