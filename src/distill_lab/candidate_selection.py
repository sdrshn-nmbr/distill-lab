import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import Field

from distill_lab.artifacts import LocalArtifactStore
from distill_lab.canonical import content_hash
from distill_lab.contracts import (
    ArtifactRef,
    CandidateTokenMethod,
    Digest,
    ResolvedRun,
    StrictModel,
)
from distill_lab.gateway import GatewayService, gateway_metrics_delta
from distill_lab.security import reject_credentials
from distill_lab.teacher import CandidateToken, TokenSelectionRequest
from distill_lab.tokenizer_preflight import TokenizerLike, prove_candidate_state


class CandidateState(StrictModel):
    state_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    checkpoint_sha256: Digest
    prompt: str = Field(min_length=1)
    privileged_context: str | None = None
    student_prefix: str
    prompt_token_ids: list[Annotated[int, Field(ge=0)]] = Field(min_length=1)
    student_token_ids: list[Annotated[int, Field(ge=0)]]
    position: int = Field(ge=0)
    candidates: list[CandidateToken] = Field(min_length=2, max_length=256)


@dataclass(frozen=True)
class LoadedCandidateStates:
    sha256: str
    states: tuple[CandidateState, ...]


@dataclass(frozen=True)
class CandidateSelectionArtifacts:
    manifest: ArtifactRef
    receipt: ArtifactRef


def load_candidate_states(path: Path, *, expected_sha256: str) -> LoadedCandidateStates:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"candidate-state digest mismatch: expected {expected_sha256}, got {digest}"
        )
    reject_credentials(payload.decode())
    states = tuple(
        CandidateState.model_validate(json.loads(line))
        for line in payload.decode().splitlines()
        if line.strip()
    )
    identifiers = [state.state_id for state in states]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate state IDs must be unique")
    if not states:
        raise ValueError("candidate-state dataset must not be empty")
    return LoadedCandidateStates(sha256=digest, states=states)


async def run_candidate_selection(
    *,
    run: ResolvedRun,
    states: LoadedCandidateStates,
    gateway: GatewayService,
    artifacts: LocalArtifactStore,
    tokenizer: TokenizerLike,
    attempt_id: str,
) -> CandidateSelectionArtifacts:
    method = run.source.method
    if not isinstance(method, CandidateTokenMethod):
        raise ValueError("run method must be candidate_token")
    if states.sha256 != run.source.dataset.content_sha256:
        raise ValueError("candidate states do not match the resolved run")
    if len(states.states) > method.positions:
        raise ValueError("candidate-state count exceeds the resolved position budget")
    if any(len(state.candidates) != method.candidates for state in states.states):
        raise ValueError("candidate-state width does not match the resolved candidate count")
    requests = [
        TokenSelectionRequest(
            request_id=state.state_id,
            checkpoint_sha256=state.checkpoint_sha256,
            prompt=state.prompt,
            privileged_context=state.privileged_context,
            student_prefix=state.student_prefix,
            prompt_token_ids=state.prompt_token_ids,
            student_token_ids=state.student_token_ids,
            position=state.position,
            candidates=state.candidates,
        )
        for state in states.states
    ]
    proofs = [
        prove_candidate_state(run=run, request=request, tokenizer=tokenizer) for request in requests
    ]
    metrics_before = gateway.metrics
    results = await gateway.select_token_batch(requests)
    records: list[dict[str, object]] = []
    receipt_records: list[dict[str, object]] = []
    for state, request, proof, result in zip(states.states, requests, proofs, results, strict=True):
        raw = {
            "schema_version": 1,
            "run_id": run.run_id,
            "request": request.model_dump(mode="json"),
            "tokenizer_proof": proof.model_dump(mode="json"),
            "teacher_output": {
                "selected_token_id": result.selected_token_id,
                "output_tokens": result.output_tokens,
            },
        }
        raw_ref = artifacts.put_json(raw, sensitivity="private")
        records.append(
            {
                "state_id": state.state_id,
                "state_sha256": content_hash(state.model_dump(mode="json")),
                "checkpoint_sha256": state.checkpoint_sha256,
                "position": state.position,
                "tokenizer_proof_sha256": content_hash(proof.model_dump(mode="json")),
                "selected_token_id": result.selected_token_id,
                "accepted": result.selected_token_id is not None,
                "raw_artifact": raw_ref.model_dump(mode="json"),
            }
        )
        receipt_records.append(
            {
                "state_id": state.state_id,
                "source": result.source,
                "retries": result.retries,
                "latency_seconds": result.latency_seconds,
            }
        )
    manifest = {
        "schema_version": 1,
        "run_id": run.run_id,
        "method": method.model_dump(mode="json"),
        "candidate_states_sha256": states.sha256,
        "records": records,
    }
    manifest_ref = artifacts.put_json(manifest, sensitivity="public")
    receipt = {
        "schema_version": 1,
        "run_id": run.run_id,
        "attempt_id": attempt_id,
        "manifest_sha256": manifest_ref.sha256,
        "records": receipt_records,
        "gateway_metrics": gateway_metrics_delta(metrics_before, gateway.metrics),
    }
    receipt_ref = artifacts.put_json(receipt, sensitivity="public")
    return CandidateSelectionArtifacts(manifest=manifest_ref, receipt=receipt_ref)
