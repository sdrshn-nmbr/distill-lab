from __future__ import annotations

import math

from pydantic import Field, model_validator

from distill_lab.contracts import Digest, StrictModel


class TrainingObservation(StrictModel):
    masked_loss: float
    target_probability: float = Field(ge=0, le=1)
    parameter_digests: dict[str, Digest] = Field(min_length=1)

    @model_validator(mode="after")
    def values_are_finite(self) -> TrainingObservation:
        if not math.isfinite(self.masked_loss):
            raise ValueError("masked loss must be finite")
        return self


class PhaseOneEvidence(StrictModel):
    starting_loss_tolerance: float = Field(gt=0)
    miles_before: TrainingObservation
    miles_after: TrainingObservation
    hugging_face_before: TrainingObservation
    hugging_face_after: TrainingObservation

    @model_validator(mode="after")
    def proves_training_update(self) -> PhaseOneEvidence:
        if (
            abs(self.miles_before.masked_loss - self.hugging_face_before.masked_loss)
            > self.starting_loss_tolerance
        ):
            raise ValueError("Miles and Hugging Face starting losses do not agree")
        for name, before, after in (
            ("Miles", self.miles_before, self.miles_after),
            ("Hugging Face", self.hugging_face_before, self.hugging_face_after),
        ):
            if after.masked_loss >= before.masked_loss:
                raise ValueError(f"{name} target loss did not decrease")
            if after.target_probability <= before.target_probability:
                raise ValueError(f"{name} target probability did not increase")
            if before.parameter_digests.keys() != after.parameter_digests.keys():
                raise ValueError(f"{name} parameter evidence names changed")
            if before.parameter_digests == after.parameter_digests:
                raise ValueError(f"{name} parameters did not change")
        return self


class RunState(StrictModel):
    sample_ids: tuple[str, ...] = Field(min_length=3)
    model_sha256: Digest
    optimizer_sha256: Digest
    scheduler_sha256: Digest
    rng_sha256: Digest
    fixed_loss: float

    @model_validator(mode="after")
    def values_are_valid(self) -> RunState:
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample IDs must be unique")
        if not math.isfinite(self.fixed_loss):
            raise ValueError("fixed loss must be finite")
        return self


class ResumeEvidence(StrictModel):
    loss_tolerance: float = Field(gt=0)
    continuous: RunState
    resumed: RunState

    @model_validator(mode="after")
    def proves_exact_resume(self) -> ResumeEvidence:
        if self.continuous.sample_ids != self.resumed.sample_ids:
            raise ValueError("resumed sample order differs from continuous training")
        for field in ("model_sha256", "optimizer_sha256", "scheduler_sha256", "rng_sha256"):
            if getattr(self.continuous, field) != getattr(self.resumed, field):
                raise ValueError(f"resumed {field} differs from continuous training")
        if abs(self.continuous.fixed_loss - self.resumed.fixed_loss) > self.loss_tolerance:
            raise ValueError("resumed fixed loss differs from continuous training")
        return self


class RefreshRound(StrictModel):
    round: int = Field(ge=1)
    parent_checkpoint_sha256: Digest
    state_checkpoint_sha256: Digest
    result_checkpoint_sha256: Digest

    @model_validator(mode="after")
    def state_matches_parent(self) -> RefreshRound:
        if self.state_checkpoint_sha256 != self.parent_checkpoint_sha256:
            raise ValueError("state checkpoint does not match its parent checkpoint")
        if self.result_checkpoint_sha256 == self.parent_checkpoint_sha256:
            raise ValueError("training did not produce a new checkpoint")
        return self


class RefreshEvidence(StrictModel):
    refreshed: tuple[RefreshRound, ...] = Field(min_length=2)
    stale_control_state_checkpoint_sha256: Digest
    stale_control_parent_checkpoint_sha256: Digest
    stale_control_result_checkpoint_sha256: Digest

    @model_validator(mode="after")
    def proves_refresh_and_control(self) -> RefreshEvidence:
        for previous, current in zip(self.refreshed, self.refreshed[1:], strict=False):
            if current.parent_checkpoint_sha256 != previous.result_checkpoint_sha256:
                raise ValueError("refreshed rounds do not form a checkpoint chain")
        if (
            self.stale_control_state_checkpoint_sha256
            == self.stale_control_parent_checkpoint_sha256
        ):
            raise ValueError("stale control reused the current checkpoint")
        if (
            self.stale_control_result_checkpoint_sha256
            == self.stale_control_parent_checkpoint_sha256
        ):
            raise ValueError("stale control training did not produce a new checkpoint")
        return self
