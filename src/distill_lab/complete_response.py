import hashlib
from dataclasses import dataclass

from distill_lab.artifacts import LocalArtifactStore
from distill_lab.canonical import content_hash
from distill_lab.contracts import ArtifactRef, CompleteResponseMethod, ResolvedRun
from distill_lab.dataset import LoadedDataset
from distill_lab.gateway import GatewayService
from distill_lab.teacher import GenerationRequest


@dataclass(frozen=True)
class CompleteResponseArtifacts:
    manifest: ArtifactRef
    receipt: ArtifactRef


async def run_complete_response(
    *,
    run: ResolvedRun,
    examples: LoadedDataset,
    gateway: GatewayService,
    artifacts: LocalArtifactStore,
) -> CompleteResponseArtifacts:
    if not isinstance(run.source.method, CompleteResponseMethod):
        raise ValueError("run method must be complete_response")
    if examples.sha256 != run.source.dataset.content_sha256:
        raise ValueError("loaded dataset does not match the resolved run")
    instructions = _instructions(run)
    requests = [
        GenerationRequest(
            request_id=example.example_id,
            prompt=example.prompt,
            privileged_context=(
                example.privileged_context if run.source.method.privileged_context else None
            ),
            instructions=instructions,
        )
        for example in examples.examples
    ]
    results = await gateway.generate_batch(requests)
    records: list[dict[str, object]] = []
    receipt_records: list[dict[str, object]] = []
    for example, request, result in zip(examples.examples, requests, results, strict=True):
        raw = {
            "schema_version": 1,
            "run_id": run.run_id,
            "example": example.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "teacher_output": {
                "text": result.text,
                "output_tokens": result.output_tokens,
            },
        }
        raw_ref = artifacts.put_json(raw, sensitivity="private")
        accepted = example.verification.text.casefold() in result.text.casefold()
        records.append(
            {
                "example_id": example.example_id,
                "request_sha256": content_hash(
                    request.model_dump(mode="json", exclude={"request_id"})
                ),
                "response_sha256": hashlib.sha256(result.text.encode()).hexdigest(),
                "raw_artifact": raw_ref.model_dump(mode="json"),
                "accepted": accepted,
                "verification": example.verification.model_dump(mode="json"),
            }
        )
        receipt_records.append(
            {
                "example_id": example.example_id,
                "source": result.source,
                "retries": result.retries,
                "latency_seconds": result.latency_seconds,
            }
        )
    manifest = {
        "schema_version": 1,
        "run_id": run.run_id,
        "method": run.source.method.model_dump(mode="json"),
        "dataset_sha256": examples.sha256,
        "records": records,
    }
    manifest_ref = artifacts.put_json(manifest, sensitivity="public")
    receipt = {
        "schema_version": 1,
        "run_id": run.run_id,
        "manifest_sha256": manifest_ref.sha256,
        "records": receipt_records,
        "gateway_metrics": gateway.metrics.__dict__,
    }
    receipt_ref = artifacts.put_json(receipt, sensitivity="public")
    return CompleteResponseArtifacts(manifest=manifest_ref, receipt=receipt_ref)


def _instructions(run: ResolvedRun) -> str:
    identity = (run.source.prompts.public_template, run.source.prompts.public_version)
    if identity == ("pinapple-public", "v1"):
        return "Answer the question directly in one sentence."
    raise ValueError(f"unknown public prompt template: {identity[0]}@{identity[1]}")
