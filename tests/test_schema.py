from pathlib import Path

from typer.testing import CliRunner

from distill_lab.cli import app


def test_generated_schema_matches_the_checked_in_schema(tmp_path: Path) -> None:
    generated = tmp_path / "study.schema.json"

    result = CliRunner().invoke(app, ["schema", "--out", str(generated)])

    assert result.exit_code == 0, result.output
    assert generated.read_bytes() == Path("schemas/study.schema.json").read_bytes()
