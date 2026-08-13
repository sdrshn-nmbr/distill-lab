import hashlib
from typing import Any, Protocol, cast

from transformers import AutoTokenizer

from distill_lab.canonical import content_hash
from distill_lab.contracts import Digest, ResolvedRun, StrictModel
from distill_lab.teacher import TokenSelectionRequest


class TokenizerLike(Protocol):
    @property
    def chat_template(self) -> str | None: ...

    def __len__(self) -> int: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


class TokenizerProof(StrictModel):
    model_revision: str
    tokenizer_revision: str
    chat_template_sha256: Digest
    vocabulary_size: int
    request_sha256: Digest
    rendered_prompt_sha256: Digest
    checkpoint_sha256: Digest


def load_tokenizer(run: ResolvedRun, model_path: str) -> TokenizerLike:
    loader = cast(Any, AutoTokenizer)
    tokenizer = loader.from_pretrained(
        model_path,
        revision=run.source.student.tokenizer_revision,
        local_files_only=True,
    )
    return cast(TokenizerLike, tokenizer)


def prove_candidate_state(
    *, run: ResolvedRun, request: TokenSelectionRequest, tokenizer: TokenizerLike
) -> TokenizerProof:
    template = tokenizer.chat_template
    if not isinstance(template, str):
        raise ValueError("student tokenizer has no chat template")
    template_sha256 = hashlib.sha256(template.encode()).hexdigest()
    if template_sha256 != run.source.student.chat_template_sha256:
        raise ValueError("student chat template digest mismatch")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": request.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=run.source.student.thinking_mode,
    )
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if prompt_ids != request.prompt_token_ids:
        raise ValueError("candidate prompt token IDs do not match the rendered prompt")
    vocabulary_size = len(tokenizer)
    all_ids = [*request.prompt_token_ids, *request.student_token_ids]
    all_ids.extend(candidate.token_id for candidate in request.candidates)
    if any(token_id >= vocabulary_size for token_id in all_ids):
        raise ValueError("candidate state contains a token outside the tokenizer vocabulary")
    decoded_prefix = tokenizer.decode(
        request.student_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded_prefix != request.student_prefix:
        raise ValueError("student prefix text does not match its token IDs")
    for candidate in request.candidates:
        decoded = tokenizer.decode(
            [candidate.token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != candidate.text:
            raise ValueError("candidate text does not match its token ID")
    request_value = request.model_dump(mode="json", exclude={"request_id"})
    return TokenizerProof(
        model_revision=run.source.student.revision,
        tokenizer_revision=run.source.student.tokenizer_revision,
        chat_template_sha256=template_sha256,
        vocabulary_size=vocabulary_size,
        request_sha256=content_hash(request_value),
        rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        checkpoint_sha256=request.checkpoint_sha256,
    )
