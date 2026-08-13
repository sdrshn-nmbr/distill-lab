from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Protocol, cast


class DigestWriter(Protocol):
    def update(self, value: bytes, /) -> None: ...


def semantic_digest(value: object) -> str:
    digest = hashlib.sha256()
    _update(digest, value)
    return digest.hexdigest()


def _update(digest: DigestWriter, value: object) -> None:
    update = digest.update
    if value is None:
        update(b"none;")
    elif isinstance(value, bool):
        update(b"bool:1;" if value else b"bool:0;")
    elif isinstance(value, int):
        update(f"int:{value};".encode())
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("semantic state contains a non-finite float")
        update(b"float:")
        update(struct.pack(">d", value))
        update(b";")
    elif isinstance(value, str):
        payload = value.encode()
        update(f"str:{len(payload)}:".encode())
        update(payload)
        update(b";")
    elif isinstance(value, bytes):
        update(f"bytes:{len(value)}:".encode())
        update(value)
        update(b";")
    elif isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in mapping):
            raise TypeError("semantic state mapping keys must be strings")
        typed = cast(Mapping[str, object], mapping)
        update(f"dict:{len(typed)}:[".encode())
        for key in sorted(typed):
            _update(digest, key)
            _update(digest, typed[key])
        update(b"];")
    elif isinstance(value, tuple):
        _update_sequence(digest, b"tuple", cast(tuple[object, ...], value))
    elif isinstance(value, Sequence):
        _update_sequence(digest, b"list", cast(Sequence[object], value))
    else:
        raise TypeError(f"unsupported semantic state value: {type(value).__name__}")


def _update_sequence(digest: DigestWriter, kind: bytes, values: Sequence[object]) -> None:
    update = digest.update
    update(kind + f":{len(values)}:[".encode())
    for value in values:
        _update(digest, value)
    update(b"];")
