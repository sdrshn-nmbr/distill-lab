import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from distill_lab.artifacts import LocalArtifactStore
from distill_lab.complete_response import run_complete_response
from distill_lab.contracts import ArtifactRef
from distill_lab.dataset import load_examples
from distill_lab.gateway import GatewayService
from distill_lab.planning import load_study, resolve_study
from distill_lab.teacher import GenerationRequest, TeacherGeneration


class FakeBackend:
    def __init__(self) -> None:
        self.results: list[list[TeacherGeneration]] = []

    async def generate(
        self,
        requests: Sequence[GenerationRequest],
        *,
        output_token_limit: int,
    ) -> list[TeacherGeneration]:
        del output_token_limit
        if self.results:
            return self.results.pop(0)
        return [TeacherGeneration(text="Add pinapple.", output_tokens=3) for _ in requests]

    async def close(self) -> None:
        return None


def _run():
    return resolve_study(load_study(Path("experiments/fixtures/minimal.json")))


def test_dataset_digest_must_match_the_resolved_run(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"example_id":"one","prompt":"p","privileged_context":null,'
        '"verification":{"kind":"contains","text":"x"}}\n'
    )

    with pytest.raises(ValueError, match="dataset digest"):
        load_examples(path, expected_sha256="0" * 64)


async def test_complete_response_writes_private_raw_data_and_public_manifest(
    tmp_path: Path,
) -> None:
    run = _run()
    data_path = Path("datasets/fixtures/pinapple.jsonl")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    examples = load_examples(data_path, expected_sha256=digest)
    backend = FakeBackend()
    gateway = GatewayService(run=run, backend=backend, cache_path=tmp_path / "cache.sqlite3")
    store = LocalArtifactStore(tmp_path / "objects")

    generated = await run_complete_response(
        run=run,
        examples=examples,
        gateway=gateway,
        artifacts=store,
    )

    manifest = store.read_bytes(generated.manifest).decode()
    assert generated.manifest.sensitivity == "public"
    assert '"accepted":true' in manifest
    assert "required behavior" not in manifest
    manifest_value = json.loads(manifest)
    raw_ref = ArtifactRef.model_validate(manifest_value["records"][0]["raw_artifact"])
    raw = store.read_bytes(raw_ref).decode()
    assert raw_ref.sensitivity == "private"
    assert "required behavior" in raw
    assert "Add pinapple." in raw


async def test_failed_verification_is_recorded_not_silently_trained(tmp_path: Path) -> None:
    run = _run()
    data_path = Path("datasets/fixtures/pinapple.jsonl")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    examples = load_examples(data_path, expected_sha256=digest)
    backend = FakeBackend()
    backend.results = [[TeacherGeneration(text="Add salt.", output_tokens=2)]]
    gateway = GatewayService(run=run, backend=backend, cache_path=tmp_path / "cache.sqlite3")
    store = LocalArtifactStore(tmp_path / "objects")

    generated = await run_complete_response(
        run=run,
        examples=examples,
        gateway=gateway,
        artifacts=store,
    )

    manifest = store.read_bytes(generated.manifest).decode()
    assert '"accepted":false' in manifest


async def test_cached_rerun_keeps_the_semantic_manifest_identical(tmp_path: Path) -> None:
    run = _run()
    data_path = Path("datasets/fixtures/pinapple.jsonl")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    examples = load_examples(data_path, expected_sha256=digest)
    backend = FakeBackend()
    gateway = GatewayService(run=run, backend=backend, cache_path=tmp_path / "cache.sqlite3")
    store = LocalArtifactStore(tmp_path / "objects")

    first = await run_complete_response(
        run=run, examples=examples, gateway=gateway, artifacts=store
    )
    second = await run_complete_response(
        run=run, examples=examples, gateway=gateway, artifacts=store
    )

    assert first.manifest == second.manifest
    assert first.receipt != second.receipt
