import hashlib
import os
from pathlib import Path

import pytest

from distill_lab.factories import sanitized_child_environment, verify_codex_executable


def test_child_environment_keeps_runtime_basics_and_drops_credentials() -> None:
    source = {
        "HOME": "/tmp/home",
        "PATH": "/usr/bin",
        "TMPDIR": "/tmp",
        "DISTILL_LAB_GATEWAY_TOKEN": "gateway-secret",
        "TS_AUTHKEY": "tailnet-secret",
        "OPENAI_API_KEY": "api-secret",
        "GITHUB_TOKEN": "github-secret",
    }

    child = sanitized_child_environment(source)

    assert child == {"HOME": "/tmp/home", "PATH": "/usr/bin", "TMPDIR": "/tmp"}
    assert not any("secret" in value for value in child.values())


def test_executable_verification_checks_digest_before_running(tmp_path: Path) -> None:
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\necho should-not-run\n")
    executable.chmod(0o700)

    with pytest.raises(ValueError, match="digest"):
        verify_codex_executable(
            executable=executable,
            expected_sha256="0" * 64,
            expected_version="0.1.0",
            environment=os.environ,
        )


def test_executable_verification_checks_exact_version(tmp_path: Path) -> None:
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\necho 'codex-cli 0.2.0'\n")
    executable.chmod(0o700)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=r"expected Codex CLI 0\.1\.0"):
        verify_codex_executable(
            executable=executable,
            expected_sha256=digest,
            expected_version="0.1.0",
            environment=os.environ,
        )
