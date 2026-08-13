import json
import re
from pathlib import Path
from typing import Any

from distill_lab.canonical import canonical_json, content_hash
from distill_lab.contracts import ResolvedComponents, ResolvedRun, StudySpec

_SECRET_PATTERNS = (
    re.compile(r"tskey-(?:auth|client)-[A-Za-z0-9_-]+"),
    re.compile(r"gh[opsu]_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
)


def load_study(path: Path) -> StudySpec:
    raw = path.read_text()
    _reject_secrets(raw)
    return StudySpec.model_validate_json(raw)


def resolve_study(study: StudySpec) -> ResolvedRun:
    source = study.model_dump(mode="json")
    run_id = f"run_{content_hash(source)[:16]}"
    teacher_identity = {
        "teacher": source["teacher"],
        "method": source["method"],
        "prompts": source["prompts"],
    }
    components = ResolvedComponents(
        teacher_cache_namespace=f"teacher_{content_hash(teacher_identity)[:16]}",
        miles_command=("uv", "run", "--no-project", "python", "train_async.py"),
        artifact_namespace=run_id,
    )
    return ResolvedRun(run_id=run_id, source=study, components=components)


def write_resolved_run(run: ResolvedRun, path: Path) -> None:
    data = canonical_json(run.model_dump(mode="json"))
    _reject_secrets(data.decode())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _reject_secrets(value: str) -> None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(value):
            raise ValueError("experiment files and resolved plans must not contain credentials")


def contains_secret(value: Any) -> bool:
    encoded = json.dumps(value, ensure_ascii=False)
    return any(pattern.search(encoded) is not None for pattern in _SECRET_PATTERNS)
