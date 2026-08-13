import json
from pathlib import Path

from typer.testing import CliRunner

from distill_lab.cli import app


def test_plan_command_writes_the_same_resolved_plan_twice(tmp_path: Path) -> None:
    source = Path("experiments/fixtures/minimal.json")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    runner = CliRunner()

    first_result = runner.invoke(app, ["plan", str(source), "--out", str(first)])
    second_result = runner.invoke(app, ["plan", str(source), "--out", str(second)])

    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    assert first.read_bytes() == second.read_bytes()
    resolved = json.loads(first.read_text())
    assert resolved["schema_version"] == 1
    assert resolved["run_id"].startswith("run_")
    assert resolved["source"]["miles"]["revision"] == ("5efe9a2c5e4eebf92b12b27f1769681f0f7ab4cd")
    assert "tskey-" not in first.read_text()
