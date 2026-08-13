import hashlib
from pathlib import Path


def checkpoint_digest(model_path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in model_path.iterdir()
        if path.is_file() and (path.suffix == ".safetensors" or path.name == "config.json")
    )
    if not paths:
        raise ValueError("model checkpoint has no weights")
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()
