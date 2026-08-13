from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeGuard, cast


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
        raw_response_length = sample.metadata.get("response_length")
        raw_loss_mask = sample.metadata.get("loss_mask")
        if not _is_int_list(raw_token_ids) or not _is_int_list(raw_loss_mask):
            raise ValueError("invalid exact-token metadata")
        if not isinstance(raw_response_length, int) or isinstance(raw_response_length, bool):
            raise ValueError("invalid exact-token metadata")
        token_ids = raw_token_ids
        response_length = raw_response_length
        loss_mask = raw_loss_mask
        if (
            response_length <= 0
            or response_length > len(token_ids)
            or len(loss_mask) != response_length
            or sum(loss_mask) != 1
        ):
            raise ValueError("invalid exact-token metadata")
        sample.tokens = token_ids
        sample.loss_mask = loss_mask
        sample.response_length = response_length
        sample.reward = 0
        result.append(sample)
    return result


def _is_int_list(value: object) -> TypeGuard[list[int]]:
    if not isinstance(value, list):
        return False
    return all(type(item) is int for item in cast(list[object], value))
