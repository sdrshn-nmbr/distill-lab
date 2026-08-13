import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
