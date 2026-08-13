import hashlib
from pathlib import Path

import pytest

from distill_lab.planning import load_study, resolve_study
from distill_lab.teacher import CandidateToken, TokenSelectionRequest
from distill_lab.tokenizer_preflight import prove_candidate_state


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
        assert not tokenize and add_generation_prompt and not enable_thinking
        return f"<user>{conversation[0]['content']}<assistant>"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == "<user>What fruit?<assistant>"
        assert not add_special_tokens
        return [100, 101]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        if token_ids == []:
            return ""
        if token_ids == [200]:
            return "Add"
        if token_ids == [10]:
            return " pin"
        if token_ids == [11]:
            return " salt"
        raise AssertionError(f"unexpected token IDs: {token_ids}")


def _request() -> TokenSelectionRequest:
    return TokenSelectionRequest(
        request_id="state",
        checkpoint_sha256="5" * 64,
        prompt="What fruit?",
        student_prefix="Add",
        prompt_token_ids=[100, 101],
        student_token_ids=[200],
        position=1,
        candidates=[
            CandidateToken(token_id=10, text=" pin", rank=0),
            CandidateToken(token_id=11, text=" salt", rank=1),
        ],
    )


def test_candidate_state_is_bound_to_exact_tokenizer_and_checkpoint() -> None:
    run = resolve_study(load_study(Path("experiments/fixtures/candidate.json")))
    template_digest = hashlib.sha256(b"fixture-template").hexdigest()
    run = run.model_copy(
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

    proof = prove_candidate_state(run=run, request=_request(), tokenizer=FakeTokenizer())

    assert proof.checkpoint_sha256 == "5" * 64
    assert proof.vocabulary_size == 1000


def test_candidate_text_mismatch_fails_before_teacher_or_training() -> None:
    run = resolve_study(load_study(Path("experiments/fixtures/candidate.json")))
    template_digest = hashlib.sha256(b"fixture-template").hexdigest()
    run = run.model_copy(
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
    request = _request().model_copy(
        update={
            "candidates": [
                CandidateToken(token_id=10, text=" wrong", rank=0),
                CandidateToken(token_id=11, text=" salt", rank=1),
            ]
        }
    )

    with pytest.raises(ValueError, match="candidate text"):
        prove_candidate_state(run=run, request=request, tokenizer=FakeTokenizer())
