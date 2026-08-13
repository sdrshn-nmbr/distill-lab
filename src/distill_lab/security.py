import json
import re
from typing import Any

_CREDENTIAL_PATTERNS = (
    re.compile(r"tskey-(?:auth|client)-[A-Za-z0-9_-]+"),
    re.compile(r"gh[opsu]_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
)


def reject_credentials(value: str) -> None:
    if any(pattern.search(value) for pattern in _CREDENTIAL_PATTERNS):
        raise ValueError("artifacts and configuration must not contain credentials")


def contains_credentials(value: Any) -> bool:
    encoded = json.dumps(value, ensure_ascii=False)
    return any(pattern.search(encoded) is not None for pattern in _CREDENTIAL_PATTERNS)
