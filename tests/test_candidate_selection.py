import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from distill_lab.artifacts import LocalArtifactStore
from distill_lab.candidate_selection import load_candidate_states, run_candidate_selection
from distill_lab.contracts import ArtifactRef
from distill_lab.gateway import GatewayService
from distill_lab.planning import load_study, resolve_study
from distill_lab.teacher import (
    CandidateToken,
    GenerationRequest,
    TeacherGeneration,
    TeacherSelection,
    TokenSelectionRequest,
)


class FakeBackend:
    async def probe(self) -> None:
        return None

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


class FakeTokenizer:
    chat_template: str | None = "fixture-template"

    def __len__(self) -> int:
        return 1000

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        del conversation, tokenize, add_generation_prompt, enable_thinking
        return "rendered"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == "rendered" and not add_special_tokens
        return [100, 101]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        if token_ids == [200, 201]:
            return "Add"
        if token_ids == [10]:
            return " pin"
        if token_ids == [11]:
            return " salt"
        raise AssertionError(f"unexpected token IDs: {token_ids}")


def _candidate_run():
    run = resolve_study(load_study(Path("experiments/fixtures/candidate.json")))
    template_digest = hashlib.sha256(b"fixture-template").hexdigest()
    return run.model_copy(
        update={
            "source": run.source.model_copy(
                update={
                    "student": run.source.student.model_copy(
                        update={"chat_template_sha256": template_digest}
                    )
                }
            )
        }
    )


def test_position_must_equal_the_student_prefix_length() -> None:
    with pytest.raises(ValidationError, match="student prefix length"):
        TokenSelectionRequest(
            request_id="bad",
            checkpoint_sha256="5" * 64,
            prompt="What fruit?",
            student_prefix="Add pin",
            prompt_token_ids=[100],
            student_token_ids=[200, 201],
            position=1,
            candidates=[
                CandidateToken(token_id=10, text="apple", rank=0),
                CandidateToken(token_id=11, text=" salt", rank=1),
            ],
        )


def test_first_assistant_token_is_a_valid_candidate_state() -> None:
    request = TokenSelectionRequest(
        request_id="first",
        checkpoint_sha256="5" * 64,
        prompt="What fruit?",
        student_prefix="",
        prompt_token_ids=[100],
        student_token_ids=[],
        position=0,
        candidates=[
            CandidateToken(token_id=10, text="Add", rank=0),
            CandidateToken(token_id=11, text="Use", rank=1),
        ],
    )

    assert request.position == 0


async def test_candidate_selection_writes_exact_student_token_target(tmp_path: Path) -> None:
    run = _candidate_run()
    data_path = Path("datasets/fixtures/candidate_states.jsonl")
    states = load_candidate_states(
        data_path, expected_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest()
    )
    gateway = GatewayService(run=run, backend=FakeBackend(), cache_path=tmp_path / "cache.sqlite3")
    store = LocalArtifactStore(tmp_path / "objects")

    generated = await run_candidate_selection(
        run=run,
        states=states,
        gateway=gateway,
        artifacts=store,
        tokenizer=FakeTokenizer(),
        attempt_id="attempt-one",
    )

    manifest = json.loads(store.read_bytes(generated.manifest))
    record = manifest["records"][0]
    assert record["selected_token_id"] == 10
    assert record["checkpoint_sha256"] == "5" * 64
    raw_ref = ArtifactRef.model_validate(record["raw_artifact"])
    raw = json.loads(store.read_bytes(raw_ref))
    assert raw["request"]["candidates"][0] == {"rank": 0, "text": " pin", "token_id": 10}


async def test_cached_candidate_selection_keeps_manifest_identical(tmp_path: Path) -> None:
    run = _candidate_run()
    data_path = Path("datasets/fixtures/candidate_states.jsonl")
    states = load_candidate_states(
        data_path, expected_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest()
    )
    gateway = GatewayService(run=run, backend=FakeBackend(), cache_path=tmp_path / "cache.sqlite3")
    store = LocalArtifactStore(tmp_path / "objects")

    first = await run_candidate_selection(
        run=run,
        states=states,
        gateway=gateway,
        artifacts=store,
        tokenizer=FakeTokenizer(),
        attempt_id="attempt-one",
    )
    second = await run_candidate_selection(
        run=run,
        states=states,
        gateway=gateway,
        artifacts=store,
        tokenizer=FakeTokenizer(),
        attempt_id="attempt-two",
    )

    assert first.manifest == second.manifest
    assert first.receipt != second.receipt
