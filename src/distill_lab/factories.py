import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path

from distill_lab.codex_app_server import CodexAppServerBackend
from distill_lab.contracts import ResolvedRun

_CHILD_ENVIRONMENT_NAMES = frozenset(
    {
        "CODEX_HOME",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOG_FORMAT",
        "PATH",
        "RUST_LOG",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TMPDIR",
        "USER",
    }
)


def sanitized_child_environment(source: Mapping[str, str]) -> dict[str, str]:
    return {name: source[name] for name in sorted(_CHILD_ENVIRONMENT_NAMES & source.keys())}


def verify_codex_executable(
    *,
    executable: Path,
    expected_sha256: str,
    expected_version: str,
    environment: Mapping[str, str],
) -> None:
    if not executable.is_file():
        raise ValueError(f"Codex executable does not exist: {executable}")
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"Codex executable digest mismatch: expected {expected_sha256}, got {digest}"
        )
    result = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env=sanitized_child_environment(environment),
    )
    observed = result.stdout.strip()
    expected = f"codex-cli {expected_version}"
    if observed != expected:
        raise ValueError(f"expected Codex CLI {expected_version}, got {observed!r}")


def build_teacher_backend(
    run: ResolvedRun, *, environment: Mapping[str, str]
) -> CodexAppServerBackend:
    teacher = run.source.teacher
    executable = Path(teacher.executable)
    verify_codex_executable(
        executable=executable,
        expected_sha256=teacher.executable_sha256,
        expected_version=teacher.codex_cli_version,
        environment=environment,
    )
    return CodexAppServerBackend(
        command=(
            str(executable),
            "app-server",
            "--listen",
            "stdio://",
            "--disable",
            "shell_tool",
            "--disable",
            "unified_exec",
            "--disable",
            "web_search",
            "--disable",
            "web_search_request",
            "--disable",
            "apps",
            "--disable",
            "plugins",
            "--disable",
            "tool_search",
            "--disable",
            "computer_use",
            "--disable",
            "image_generation",
            "--disable",
            "browser_use",
            "--disable",
            "multi_agent",
            "--disable",
            "collab",
        ),
        model=teacher.model,
        reasoning_effort=teacher.reasoning_effort,
        prompt_version=run.components.teacher_cache_namespace,
        timeout_seconds=run.source.gateway.request_timeout_seconds,
        environment=sanitized_child_environment(environment),
    )
