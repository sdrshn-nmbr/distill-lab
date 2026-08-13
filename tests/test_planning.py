import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from distill_lab.planning import load_study


def _minimal() -> dict[str, object]:
    return json.loads(Path("experiments/fixtures/minimal.json").read_text())


@pytest.mark.parametrize("revision", ["main", "sudarshan/rmsd", "4b1974c"])
def test_miles_revision_must_be_an_exact_commit(tmp_path: Path, revision: str) -> None:
    value = _minimal()
    value["miles"]["revision"] = revision  # type: ignore[index]
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ValidationError, match="exact 40-character commit"):
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
