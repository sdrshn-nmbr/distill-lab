from typing import Literal

import pytest
from pydantic import ValidationError

from distill_lab.quality import (
    QualityExample,
    QualityObservation,
    QualityStudyEvidence,
    aggregate_quality,
)


def _example(
    identifier: str,
    split: Literal["train", "heldout", "control"],
    expected: str,
    forbidden: str | None = None,
) -> QualityExample:
    return QualityExample(
        example_id=identifier,
        split=split,
        prompt=f"prompt-{identifier}",
        target_response=f"answer-{identifier}",
        expected_contains=expected,
        forbidden_contains=forbidden,
    )


def test_quality_aggregation_keeps_split_metrics_separate() -> None:
    examples = (
        _example("train-a", "train", "pinapple"),
        _example("heldout-a", "heldout", "pinapple"),
        _example("control-a", "control", "Paris", "pinapple"),
    )
    observations = (
        QualityObservation(
            example_id="train-a",
            generated_text="Use pinapple.",
            success=True,
            target_probability=0.4,
            response_tokens=3,
            truncated=False,
        ),
        QualityObservation(
            example_id="heldout-a",
            generated_text="Use pineapple.",
            success=False,
            target_probability=0.2,
            response_tokens=3,
            truncated=False,
        ),
        QualityObservation(
            example_id="control-a",
            generated_text="Paris",
            success=True,
            target_probability=0.8,
            response_tokens=1,
            truncated=False,
        ),
    )

    result = aggregate_quality(examples, observations)

    assert result.train.success_rate == 1.0
    assert result.heldout.success_rate == 0.0
    assert result.control.success_rate == 1.0
    assert abs(result.heldout.geometric_mean_target_probability - 0.2) < 1e-12


def test_quality_evidence_requires_matching_examples_at_every_checkpoint() -> None:
    examples = (
        _example("train-a", "train", "pinapple"),
        _example("heldout-a", "heldout", "pinapple"),
        _example("control-a", "control", "Paris", "pinapple"),
    )
    observations = tuple(
        QualityObservation(
            example_id=example.example_id,
            generated_text=example.expected_contains,
            success=True,
            target_probability=0.5,
            response_tokens=1,
            truncated=False,
        )
        for example in examples
    )
    base = aggregate_quality(examples, observations)
    checkpoint = base.model_copy(update={"checkpoint": "iter_0000001"})

    evidence = QualityStudyEvidence(
        dataset_sha256="a" * 64,
        base=base,
        checkpoints=(checkpoint,),
    )
    assert evidence.checkpoints[0].checkpoint == "iter_0000001"

    bad_examples = (*examples[:-1], _example("control-b", "control", "Paris", "pinapple"))
    bad_observations = (
        *observations[:-1],
        observations[-1].model_copy(update={"example_id": "control-b"}),
    )
    bad = aggregate_quality(bad_examples, bad_observations).model_copy(
        update={"checkpoint": "iter_0000002"}
    )
    with pytest.raises(ValidationError, match="example IDs"):
        QualityStudyEvidence(
            dataset_sha256="a" * 64,
            base=base,
            checkpoints=(bad,),
        )
