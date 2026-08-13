import hashlib
import os
import uuid
from pathlib import Path
from typing import Literal

from distill_lab.canonical import canonical_json
from distill_lab.contracts import ArtifactRef
from distill_lab.security import reject_credentials


class ArtifactIntegrityError(RuntimeError):
    pass


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def put_bytes(
        self,
        payload: bytes,
        *,
        media_type: str,
        sensitivity: Literal["public", "private"],
    ) -> ArtifactRef:
        reject_credentials(payload.decode("utf-8", errors="ignore"))
        digest = hashlib.sha256(payload).hexdigest()
        ref = ArtifactRef(
            sha256=digest,
            size_bytes=len(payload),
            media_type=media_type,
            sensitivity=sensitivity,
        )
        destination = self.path_for(ref)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                self._verify_path(destination, ref)
        finally:
            temporary.unlink(missing_ok=True)
        return ref

    def put_json(
        self,
        value: object,
        *,
        sensitivity: Literal["public", "private"],
    ) -> ArtifactRef:
        return self.put_bytes(
            canonical_json(value),
            media_type="application/json",
            sensitivity=sensitivity,
        )

    def read_bytes(self, ref: ArtifactRef) -> bytes:
        path = self.path_for(ref)
        payload = path.read_bytes()
        self._verify_payload(payload, ref)
        return payload

    def path_for(self, ref: ArtifactRef) -> Path:
        return self._root / "objects" / ref.sha256[:2] / f"{ref.sha256}.blob"

    def _verify_path(self, path: Path, ref: ArtifactRef) -> None:
        self._verify_payload(path.read_bytes(), ref)

    @staticmethod
    def _verify_payload(payload: bytes, ref: ArtifactRef) -> None:
        digest = hashlib.sha256(payload).hexdigest()
        if digest != ref.sha256:
            raise ArtifactIntegrityError(
                f"artifact digest mismatch: expected {ref.sha256}, got {digest}"
            )
        if len(payload) != ref.size_bytes:
            raise ArtifactIntegrityError(
                f"artifact size mismatch: expected {ref.size_bytes}, got {len(payload)}"
            )
