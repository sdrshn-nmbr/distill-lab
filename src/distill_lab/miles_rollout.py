from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

_INT_LIST = TypeAdapter(list[int])


class RolloutArgs(Protocol):
    rollout_global_dataset: bool
    rollout_batch_size: int


class ExactTokenSample(Protocol):
    metadata: dict[str, object]
    tokens: list[int] | None
    response_length: int | None
    reward: int | None
    loss_mask: list[int] | None


class DataBuffer(Protocol):
    def get_samples(self, count: int) -> Sequence[Sequence[ExactTokenSample]]: ...


def generate_exact_token_rollout(
    args: RolloutArgs,
    rollout_id: int,
    data_buffer: DataBuffer,
    evaluation: bool = False,
) -> list[ExactTokenSample]:
    del rollout_id
    if evaluation:
        raise ValueError("exact-token training rollout does not support evaluation")
    if not args.rollout_global_dataset:
        raise ValueError("exact-token training requires a global dataset")
    samples = data_buffer.get_samples(args.rollout_batch_size)
    result: list[ExactTokenSample] = []
    for group in samples:
        if len(group) != 1:
            raise ValueError("exact-token training requires one sample per prompt")
        sample = group[0]
        raw_token_ids = sample.metadata.get("token_ids")
        raw_loss_mask = sample.metadata.get("loss_mask")
        try:
            token_ids = _INT_LIST.validate_python(raw_token_ids)
            loss_mask = _INT_LIST.validate_python(raw_loss_mask)
        except ValidationError as error:
            raise ValueError("invalid exact-token metadata") from error
        if len(token_ids) != len(loss_mask) or not token_ids or sum(loss_mask) != 1:
            raise ValueError("invalid exact-token metadata")
        sample.tokens = token_ids
        sample.loss_mask = loss_mask
        sample.response_length = len(token_ids)
        sample.reward = 0
        result.append(sample)
    return result
