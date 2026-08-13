import os
import subprocess
from pathlib import Path

import pytest

from distill_lab.miles_adapter import training_child_environment


def test_pinned_miles_consumes_exact_response_suffix(tmp_path: Path) -> None:
    checkout_value = os.environ.get("DISTILL_LAB_MILES_CHECKOUT")
    if checkout_value is None:
        pytest.skip("set DISTILL_LAB_MILES_CHECKOUT to run the pinned Miles boundary")
    checkout = Path(checkout_value)
    python = checkout / ".venv" / "bin" / "python"
    if not python.is_file():
        pytest.fail(f"pinned Miles Python does not exist: {python}")
    script = """
from argparse import Namespace
from types import SimpleNamespace
import torch
from distill_lab.miles_rollout import generate_exact_token_rollout
from miles.backends.training_utils.loss_hub import logit_processors
from miles.utils.types import Sample

class Buffer:
    def __init__(self, sample): self.sample = sample
    def get_samples(self, count):
        assert count == 1
        return [[self.sample]]

args = Namespace(rollout_global_dataset=True, rollout_batch_size=1)
sample = Sample(metadata={
    'token_ids': [100, 101, 200, 10],
    'response_length': 2,
    'loss_mask': [0, 1],
})
result = generate_exact_token_rollout(args, 0, Buffer(sample))[0]
result.validate()
assert result.tokens == [100, 101, 200, 10]
assert result.response_length == 2
assert result.loss_mask == [0, 1]

logit_processors.get_parallel_state = lambda: SimpleNamespace(cp=SimpleNamespace(size=1))
loss_args = Namespace(
    qkv_format='thd', true_on_policy_mode=False, rollout_temperature=1.0
)
logits = torch.zeros((1, 4, 16), dtype=torch.float32)
chunk, targets = next(logit_processors.get_responses(
    logits,
    args=loss_args,
    unconcat_tokens=[torch.tensor(result.tokens)],
    total_lengths=[len(result.tokens)],
    response_lengths=[result.response_length],
))
assert chunk.shape == (2, 16)
assert targets.tolist() == [200, 10]
assert result.loss_mask[-1] == 1

bad_chunk, _ = next(logit_processors.get_responses(
    logits,
    args=loss_args,
    unconcat_tokens=[torch.tensor(result.tokens)],
    total_lengths=[len(result.tokens)],
    response_lengths=[len(result.tokens)],
))
assert bad_chunk.shape[0] == 0
"""
    environment = training_child_environment(
        dict(os.environ), isolated_home=tmp_path / "isolated-home"
    )
    subprocess.run(
        [str(python), "-c", script],
        cwd=checkout,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
