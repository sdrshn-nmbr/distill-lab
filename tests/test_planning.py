import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from distill_lab.planning import load_study, require_clean_harness, resolve_study


def _minimal() -> dict[str, object]:
    return json.loads(Path("experiments/fixtures/minimal.json").read_text())


@pytest.mark.parametrize("revision", ["main", "sudarshan/rmsd", "4b1974c"])
def test_miles_revision_must_be_an_exact_commit(tmp_path: Path, revision: str) -> None:
    value = _minimal()
    value["miles"]["revision"] = revision  # type: ignore[index]
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ValidationError, match="String should match pattern"):
        load_study(path)


def test_unknown_fields_fail_closed(tmp_path: Path) -> None:
    value = _minimal()
    value["quietly_ignored"] = True
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_study(path)


def test_credentials_fail_before_validation(tmp_path: Path) -> None:
    value = _minimal()
    value["teacher"]["credential"] = "tskey-auth-example-secret-value"  # type: ignore[index]
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="must not contain credentials"):
        load_study(path)


@pytest.mark.parametrize(
    "repository",
    [
        "https://token@example.com/miles.git",
        "https://example.com/miles.git?token=value",
        "https://example.com/miles.git#secret",
        "http://example.com/miles.git",
    ],
)
def test_miles_repository_rejects_credential_surfaces(tmp_path: Path, repository: str) -> None:
    value = _minimal()
    value["miles"]["repository"] = repository  # type: ignore[index]
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ValidationError, match="without credentials, query, or fragment"):
        load_study(path)


def test_trace_only_fields_do_not_change_teacher_cache_namespace(tmp_path: Path) -> None:
    source = load_study(Path("experiments/fixtures/minimal.json"))
    first = resolve_study(source)
    second = resolve_study(source.model_copy(update={"name": "another-trace-name", "seed": 73}))

    assert first.run_id != second.run_id
    assert first.components.teacher_cache_namespace == second.components.teacher_cache_namespace


def test_harness_implementation_changes_run_and_cache_identity() -> None:
    source = load_study(Path("experiments/fixtures/minimal.json"))
    first = resolve_study(source)
    changed = first.harness.model_copy(update={"prompt_implementation_sha256": "f" * 64})
    second = resolve_study(source, harness=changed)

    assert first.run_id != second.run_id
    assert first.components.teacher_cache_namespace != second.components.teacher_cache_namespace


def test_dirty_harness_cannot_execute_externally() -> None:
    run = resolve_study(load_study(Path("experiments/fixtures/minimal.json")))
    dirty = run.model_copy(update={"harness": run.harness.model_copy(update={"dirty": True})})

    with pytest.raises(ValueError, match="must be clean"):
        require_clean_harness(dirty)


def test_quality_evaluation_dataset_path_and_digest_are_atomic(tmp_path: Path) -> None:
    value = _minimal()
    value["evaluation"]["quality_dataset_path"] = "datasets/quality.jsonl"  # type: ignore[index]
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ValidationError, match="quality dataset path and digest"):
        load_study(path)
