import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from distill_lab.canonical import canonical_json
from distill_lab.contracts import ResolvedRun, StrictModel
from distill_lab.security import reject_credentials


class AttemptReceipt(StrictModel):
    schema_version: Literal[1] = 1
    attempt_id: str
    run_id: str
    operation: str
    status: Literal["completed", "failed"]
    failure_code: str | None
    started_at: str
    ended_at: str
    process_id: int
    harness_revision: str
    harness_source_sha256: str
    artifacts: dict[str, str]


class AttemptRecorder:
    def __init__(self, *, run: ResolvedRun, operation: str, root: Path) -> None:
        self._run = run
        self._operation = operation
        self._root = root
        self._attempt_id = str(uuid.uuid4())
        self._started_at = _now()
        self._finished = False

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    def complete(self, *, artifacts: dict[str, str]) -> AttemptReceipt:
        return self._finish(status="completed", failure_code=None, artifacts=artifacts)

    def fail(self, *, failure_code: str) -> AttemptReceipt:
        return self._finish(status="failed", failure_code=failure_code, artifacts={})

    def _finish(
        self,
        *,
        status: Literal["completed", "failed"],
        failure_code: str | None,
        artifacts: dict[str, str],
    ) -> AttemptReceipt:
        if self._finished:
            raise RuntimeError("attempt already has a terminal receipt")
        self._finished = True
        receipt = AttemptReceipt(
            attempt_id=self._attempt_id,
            run_id=self._run.run_id,
            operation=self._operation,
            status=status,
            failure_code=failure_code,
            started_at=self._started_at,
            ended_at=_now(),
            process_id=os.getpid(),
            harness_revision=self._run.harness.revision,
            harness_source_sha256=self._run.harness.source_sha256,
            artifacts=artifacts,
        )
        payload = canonical_json(receipt.model_dump(mode="json"))
        reject_credentials(payload.decode())
        destination = self._root / self._run.run_id / "attempts" / f"{self._attempt_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        return receipt


def failure_code(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, ValueError):
        return "invalid_input_or_output"
    if isinstance(error, RuntimeError):
        return "runtime_failure"
    return "unexpected_failure"


def _now() -> str:
    return datetime.now(UTC).isoformat()
