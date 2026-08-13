import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from distill_lab.contracts import StrictModel
from distill_lab.security import reject_credentials


class ContainsVerification(StrictModel):
    kind: Literal["contains"]
    text: str = Field(min_length=1)


class DistillationExample(StrictModel):
    example_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    prompt: str = Field(min_length=1)
    privileged_context: str | None
    verification: ContainsVerification


@dataclass(frozen=True)
class LoadedDataset:
    sha256: str
    examples: tuple[DistillationExample, ...]


def load_examples(path: Path, *, expected_sha256: str) -> LoadedDataset:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"dataset digest mismatch: expected {expected_sha256}, got {digest}")
    reject_credentials(payload.decode())
    examples = tuple(
        DistillationExample.model_validate(json.loads(line))
        for line in payload.decode().splitlines()
        if line.strip()
    )
    identifiers = [example.example_id for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("dataset example IDs must be unique")
    if not examples:
        raise ValueError("dataset must contain at least one example")
    return LoadedDataset(sha256=digest, examples=examples)
