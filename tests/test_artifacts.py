from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from distill_lab.artifacts import ArtifactIntegrityError, LocalArtifactStore


def test_concurrent_publication_is_idempotent_and_verified(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    payload = b'{"answer":"forty-two"}\n'

    with ThreadPoolExecutor(max_workers=8) as executor:
        refs = list(executor.map(store.put_bytes, [payload] * 32))

    assert len({ref.sha256 for ref in refs}) == 1
    assert len(list((tmp_path / "objects").rglob("*.blob"))) == 1
    assert store.read_bytes(refs[0]) == payload


def test_read_detects_tampering(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = store.put_bytes(b"original")
    store.path_for(ref).write_bytes(b"changed")

    with pytest.raises(ArtifactIntegrityError, match="digest"):
        store.read_bytes(ref)
