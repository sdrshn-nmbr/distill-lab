import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer

from distill_lab.artifacts import LocalArtifactStore
from distill_lab.canonical import canonical_json
from distill_lab.complete_response import run_complete_response
from distill_lab.contracts import LocalArtifactStoreSpec, StudySpec
from distill_lab.dataset import load_examples
from distill_lab.factories import build_teacher_backend
from distill_lab.gateway import GatewayService
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


@app.command()
def schema(out: Annotated[Path, typer.Option("--out")]) -> None:
    """Write the JSON schema for experiment files."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(canonical_json(StudySpec.model_json_schema()))


@app.command()
def generate(source: Path, dataset: Annotated[Path, typer.Option("--dataset")]) -> None:
    """Generate one immutable complete-response dataset."""
    run = resolve_study(load_study(source))
    if not isinstance(run.source.artifacts, LocalArtifactStoreSpec):
        raise ValueError("local generate requires a local artifact store")
    examples = load_examples(dataset, expected_sha256=run.source.dataset.content_sha256)

    async def execute() -> None:
        backend = build_teacher_backend(run, environment=os.environ)
        gateway: GatewayService | None = None
        try:
            gateway = GatewayService(
                run=run,
                backend=backend,
                cache_path=Path(run.source.gateway.cache_path),
            )
            generated = await run_complete_response(
                run=run,
                examples=examples,
                gateway=gateway,
                artifacts=LocalArtifactStore(Path(run.source.artifacts.root)),
            )
        finally:
            if gateway is None:
                await backend.close()
            else:
                await gateway.close()
        typer.echo(
            canonical_json(
                {
                    "manifest": generated.manifest.model_dump(mode="json"),
                    "receipt": generated.receipt.model_dump(mode="json"),
                }
            )
            .decode()
            .strip()
        )

    asyncio.run(execute())


if __name__ == "__main__":
    app()
