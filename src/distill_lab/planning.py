import hashlib
import subprocess
from pathlib import Path

from distill_lab.canonical import canonical_json, content_hash
from distill_lab.contracts import HarnessIdentity, ResolvedComponents, ResolvedRun, StudySpec
from distill_lab.security import reject_credentials


def load_study(path: Path) -> StudySpec:
    raw = path.read_text()
    reject_credentials(raw)
    return StudySpec.model_validate_json(raw)


def resolve_study(study: StudySpec, *, harness: HarnessIdentity | None = None) -> ResolvedRun:
    harness = harness or resolve_harness_identity()
    source = study.model_dump(mode="json")
    harness_value = harness.model_dump(mode="json")
    run_id = f"run_{content_hash({'source': source, 'harness': harness_value})[:16]}"
    teacher_identity = {
        "teacher": source["teacher"],
        "method": source["method"],
        "prompts": source["prompts"],
        "harness": harness_value,
    }
    components = ResolvedComponents(
        teacher_cache_namespace=f"teacher_{content_hash(teacher_identity)[:16]}",
        miles_command=("uv", "run", "--no-project", "python"),
        artifact_namespace=run_id,
    )
    return ResolvedRun(run_id=run_id, source=study, harness=harness, components=components)


def resolve_harness_identity(root: Path | None = None) -> HarnessIdentity:
    repository = root or Path(__file__).resolve().parents[2]
    revision = _git(repository, "rev-parse", "HEAD").strip()
    dirty = bool(_git(repository, "status", "--porcelain"))
    source_files = sorted((repository / "src" / "distill_lab").glob("*.py"))
    source = hashlib.sha256()
    for path in source_files:
        source.update(path.name.encode())
        source.update(b"\0")
        source.update(path.read_bytes())
        source.update(b"\0")
    return HarnessIdentity(
        revision=revision,
        dirty=dirty,
        source_sha256=source.hexdigest(),
        lock_sha256=_sha256(repository / "uv.lock"),
        study_schema_sha256=_sha256(repository / "schemas" / "study.schema.json"),
        prompt_implementation_sha256=_sha256(
            repository / "src" / "distill_lab" / "codex_app_server.py"
        ),
    )


def require_clean_harness(run: ResolvedRun) -> None:
    if run.harness.dirty:
        raise ValueError("distill-lab checkout must be clean before external execution")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_resolved_run(run: ResolvedRun, path: Path) -> None:
    data = canonical_json(run.model_dump(mode="json"))
    reject_credentials(data.decode())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
