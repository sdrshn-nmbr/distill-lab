import pytest
from pydantic import ValidationError

from distill_lab.validation import (
    PhaseOneEvidence,
    RefreshEvidence,
    RefreshRound,
    ResumeEvidence,
    RunState,
    TrainingObservation,
    checkpoint_target_name,
    is_known_non_text_checkpoint_key,
    parse_sft_sample_ids,
)


def _observation(*, loss: float, probability: float, suffix: str) -> TrainingObservation:
    return TrainingObservation(
        masked_loss=loss,
        target_probability=probability,
        parameter_digests={"model.embed_tokens.weight": suffix * 64},
    )


def test_full_qwen_checkpoint_maps_language_and_identifies_vision_tensors() -> None:
    target_keys = {"model.embed_tokens.weight"}

    assert (
        checkpoint_target_name(
            "model_state.model.model.language_model.embed_tokens.weight", target_keys
        )
        == "model.embed_tokens.weight"
    )
    assert is_known_non_text_checkpoint_key(
        "model_state.model.model.visual.patch_embed.proj.weight"
    )


def test_phase_one_requires_loss_parity_parameter_change_and_expected_direction() -> None:
    evidence = PhaseOneEvidence(
        starting_loss_tolerance=0.02,
        miles_before=_observation(loss=1.408, probability=0.2447, suffix="a"),
        miles_after=_observation(loss=0.588, probability=0.5555, suffix="b"),
        hugging_face_before=_observation(loss=1.409, probability=0.2445, suffix="a"),
        hugging_face_after=_observation(loss=0.900, probability=0.4065, suffix="c"),
    )

    assert evidence.miles_after.target_probability > evidence.miles_before.target_probability

    with pytest.raises(ValidationError, match="starting losses"):
        PhaseOneEvidence(
            starting_loss_tolerance=0.02,
            miles_before=evidence.miles_before,
            miles_after=evidence.miles_after,
            hugging_face_before=_observation(loss=1.5, probability=0.2, suffix="a"),
            hugging_face_after=evidence.hugging_face_after,
        )


def test_resume_requires_identical_order_and_state() -> None:
    continuous = RunState(
        sample_ids=("a", "b", "c"),
        model_sha256="1" * 64,
        optimizer_sha256="2" * 64,
        scheduler_sha256="3" * 64,
        rng_sha256="4" * 64,
        dataset_sha256="5" * 64,
        fixed_loss=0.5,
    )
    resumed = continuous.model_copy()

    evidence = ResumeEvidence(loss_tolerance=1e-6, continuous=continuous, resumed=resumed)

    assert evidence.continuous.sample_ids == evidence.resumed.sample_ids
    with pytest.raises(ValidationError, match="sample order"):
        ResumeEvidence(
            loss_tolerance=1e-6,
            continuous=continuous,
            resumed=resumed.model_copy(update={"sample_ids": ("a", "c", "b")}),
        )

    with pytest.raises(ValidationError, match="dataset_sha256"):
        ResumeEvidence(
            loss_tolerance=1e-6,
            continuous=continuous,
            resumed=resumed.model_copy(update={"dataset_sha256": "6" * 64}),
        )


def test_sft_sample_ids_are_read_in_training_order() -> None:
    log = """\
noise
distill_lab_sft_train_sample {"example_ids":["resume-a"],"rollout_id":0}
distill_lab_sft_train_sample {"example_ids":["resume-b"],"rollout_id":1}
"""
    second = 'distill_lab_sft_train_sample {"example_ids":["resume-c"],"rollout_id":2}\n'

    assert parse_sft_sample_ids((log, second)) == ("resume-a", "resume-b", "resume-c")

    with pytest.raises(ValueError, match="malformed"):
        parse_sft_sample_ids(("distill_lab_sft_train_sample not-json",))


def test_refresh_requires_each_state_to_match_its_parent_checkpoint() -> None:
    base = "a" * 64
    first = "b" * 64
    second = "c" * 64

    evidence = RefreshEvidence(
        refreshed=(
            RefreshRound(
                round=1,
                parent_checkpoint_sha256=base,
                state_checkpoint_sha256=base,
                result_checkpoint_sha256=first,
            ),
            RefreshRound(
                round=2,
                parent_checkpoint_sha256=first,
                state_checkpoint_sha256=first,
                result_checkpoint_sha256=second,
            ),
        ),
        stale_control_state_checkpoint_sha256=base,
        stale_control_parent_checkpoint_sha256=first,
        stale_control_result_checkpoint_sha256="d" * 64,
    )

    assert evidence.refreshed[-1].result_checkpoint_sha256 == second
    with pytest.raises(ValidationError, match="state checkpoint"):
        RefreshEvidence(
            refreshed=(
                evidence.refreshed[0],
                evidence.refreshed[1].model_copy(update={"state_checkpoint_sha256": base}),
            ),
            stale_control_state_checkpoint_sha256=base,
            stale_control_parent_checkpoint_sha256=first,
            stale_control_result_checkpoint_sha256="d" * 64,
        )
