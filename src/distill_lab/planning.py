from pathlib import Path

from distill_lab.canonical import canonical_json, content_hash
from distill_lab.contracts import ResolvedComponents, ResolvedRun, StudySpec
from distill_lab.security import reject_credentials


def load_study(path: Path) -> StudySpec:
    raw = path.read_text()
    reject_credentials(raw)
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
        miles_command=("uv", "run", "--no-project", "python"),
        artifact_namespace=run_id,
    )
    return ResolvedRun(run_id=run_id, source=study, components=components)


def write_resolved_run(run: ResolvedRun, path: Path) -> None:
    data = canonical_json(run.model_dump(mode="json"))
    reject_credentials(data.decode())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
