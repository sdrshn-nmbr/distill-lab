from pathlib import Path
from typing import Annotated

import typer

from distill_lab.planning import load_study, resolve_study, write_resolved_run

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.callback()
def main() -> None:
    """Run reproducible distillation experiments."""


@app.command()
def plan(source: Path, out: Annotated[Path, typer.Option("--out")]) -> None:
    """Resolve one experiment into a stable run plan."""
    run = resolve_study(load_study(source))
    write_resolved_run(run, out)
    typer.echo(run.run_id)


if __name__ == "__main__":
    app()
