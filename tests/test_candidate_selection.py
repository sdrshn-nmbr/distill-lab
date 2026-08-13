import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from distill_lab.artifacts import LocalArtifactStore
from distill_lab.candidate_selection import load_candidate_states, run_candidate_selection
from distill_lab.contracts import ArtifactRef
from distill_lab.gateway import GatewayService
from distill_lab.planning import load_study, resolve_study
from distill_lab.teacher import (
    GenerationRequest,
    TeacherGeneration,
    TeacherSelection,
    TokenSelectionRequest,
)


class FakeBackend:
    async def generate(
        self,
        requests: Sequence[GenerationRequest],
        *,
        output_token_limit: int,
    ) -> list[TeacherGeneration]:
        del output_token_limit
        return [TeacherGeneration(text="unused", output_tokens=1) for _ in requests]

    async def select_tokens(
        self,
        requests: Sequence[TokenSelectionRequest],
        *,
        output_token_limit: int,
    ) -> list[TeacherSelection]:
        del output_token_limit
        return [TeacherSelection(selected_token_id=10, output_tokens=2) for _ in requests]

    async def close(self) -> None:
        return None


async def test_candidate_selection_writes_exact_student_token_target(tmp_path: Path) -> None:
    run = resolve_study(load_study(Path("experiments/fixtures/candidate.json")))
    data_path = Path("datasets/fixtures/candidate_states.jsonl")
    states = load_candidate_states(
        data_path, expected_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest()
    )
    gateway = GatewayService(run=run, backend=FakeBackend(), cache_path=tmp_path / "cache.sqlite3")
    store = LocalArtifactStore(tmp_path / "objects")

    generated = await run_candidate_selection(
        run=run, states=states, gateway=gateway, artifacts=store
    )

    manifest = json.loads(store.read_bytes(generated.manifest))
    record = manifest["records"][0]
    assert record["selected_token_id"] == 10
    assert record["checkpoint_sha256"] == "5" * 64
    raw_ref = ArtifactRef.model_validate(record["raw_artifact"])
    raw = json.loads(store.read_bytes(raw_ref))
    assert raw["request"]["candidates"][0] == {"rank": 0, "text": " pin", "token_id": 10}


async def test_cached_candidate_selection_keeps_manifest_identical(tmp_path: Path) -> None:
    run = resolve_study(load_study(Path("experiments/fixtures/candidate.json")))
    data_path = Path("datasets/fixtures/candidate_states.jsonl")
    states = load_candidate_states(
        data_path, expected_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest()
    )
    gateway = GatewayService(run=run, backend=FakeBackend(), cache_path=tmp_path / "cache.sqlite3")
    store = LocalArtifactStore(tmp_path / "objects")

    first = await run_candidate_selection(run=run, states=states, gateway=gateway, artifacts=store)
    second = await run_candidate_selection(run=run, states=states, gateway=gateway, artifacts=store)

    assert first.manifest == second.manifest
    assert first.receipt != second.receipt
