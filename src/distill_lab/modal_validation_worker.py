from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import override

import torch
import torch.distributed.checkpoint as dcp
from miles.backends.fsdp_utils.models.qwen3_5 import apply_gateddeltanet_packing_patch
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata, TensorStorageMetadata
from transformers import AutoModelForImageTextToText, AutoTokenizer

from distill_lab.checkpoint_evidence import semantic_digest
from distill_lab.validation import (
    CheckpointIdentity,
    PhaseOneEvidence,
    RunState,
    TrainingObservation,
    checkpoint_target_name,
    is_known_non_text_checkpoint_key,
    parse_sft_sample_ids,
)


class EmptyStateLoadPlanner(DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        if metadata is None:
            raise ValueError("checkpoint metadata is missing")
        for key, value in metadata.state_dict_metadata.items():
            state_dict[key] = (
                torch.empty(value.size, dtype=value.properties.dtype)
                if isinstance(value, TensorStorageMetadata)
                else value
            )
        super().set_up_planner(state_dict, metadata, is_coordinator)


def load_dcp_state(path: Path) -> dict[str, object]:
    state: dict[str, object] = {}
    dcp.state_dict_loader._load_state_dict(
        state,
        storage_reader=dcp.FileSystemReader(path),
        planner=EmptyStateLoadPlanner(),
        no_dist=True,
    )
    return state


def model_tensors(path: Path) -> dict[str, torch.Tensor]:
    state = load_dcp_state(path / "model" if (path / "model").is_dir() else path)
    tensors = {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
    if not tensors:
        raise ValueError("checkpoint has no model tensors")
    return tensors


def map_model_tensors(
    tensors: dict[str, torch.Tensor], target_keys: set[str]
) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    unknown: list[str] = []
    for key, value in tensors.items():
        target = checkpoint_target_name(key, target_keys)
        if target is not None:
            if target in mapped:
                raise ValueError(f"duplicate checkpoint target: {target}")
            mapped[target] = value
        elif not is_known_non_text_checkpoint_key(key):
            unknown.append(key)
    if unknown:
        raise ValueError(f"unknown checkpoint tensors: {len(unknown)}:{unknown[:5]}")
    return mapped


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def checkpoint_state_digest(value: object) -> str:
    return semantic_digest(_semantic_checkpoint_value(value))


def _semantic_checkpoint_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": tensor_digest(value),
        }
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("checkpoint mapping keys must be strings")
        return {key: _semantic_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_semantic_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_semantic_checkpoint_value(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")


def selected_parameter_digests(state: dict[str, torch.Tensor]) -> dict[str, str]:
    candidates = [
        key
        for key in sorted(state)
        if key.endswith("embed_tokens.weight") or key.endswith("layers.0.mlp.down_proj.weight")
    ]
    if len(candidates) < 2:
        candidates = sorted(state)[:2]
    return {key: tensor_digest(state[key]) for key in candidates[:2]}


def observe(
    model: torch.nn.Module,
    token_ids: list[int],
    response_length: int,
    *,
    padded_length: int | None = None,
) -> TrainingObservation:
    inputs = torch.tensor([token_ids], device="cuda")
    position_ids = torch.arange(len(token_ids), device="cuda")
    if padded_length is not None:
        padding = padded_length - len(token_ids)
        if padding < 0:
            raise ValueError("padded length is shorter than the token sequence")
        inputs = torch.nn.functional.pad(inputs, (0, padding))
        position_ids = torch.nn.functional.pad(position_ids, (0, padding))
    position_ids = position_ids.unsqueeze(0)
    start = len(token_ids) - response_length
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        full_logits = model(
            input_ids=inputs,
            position_ids=position_ids,
            attention_mask=None,
            use_cache=False,
        ).logits.float()
    logits = full_logits[:, start - 1 : start - 1 + response_length]
    targets = inputs[:, start : start + response_length]
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    state = model.state_dict()
    return TrainingObservation(
        masked_loss=float(loss.detach().cpu()),
        target_probability=math.exp(-float(loss.detach().cpu())),
        parameter_digests=selected_parameter_digests(state),
    )


def load_model(
    base_path: Path,
    checkpoint: Path | None = None,
    *,
    fp32_master: bool = False,
) -> torch.nn.Module:
    model = AutoModelForImageTextToText.from_pretrained(
        base_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_3",
    )
    if checkpoint is not None:
        checkpoint_state = map_model_tensors(
            model_tensors(checkpoint), set(model.state_dict().keys())
        )
        missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
        allowed_missing = {name for name in missing if name.endswith("lm_head.weight")}
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(
                "checkpoint key mismatch: "
                f"missing={len(missing)}:{missing[:5]}, "
                f"unexpected={len(unexpected)}:{unexpected[:5]}"
            )
    if fp32_master:
        model = model.float()
    return model.to("cuda")


def phase_one(spec: dict[str, object]) -> PhaseOneEvidence | TrainingObservation:
    base_path = Path(str(spec["base_path"]))
    checkpoint = Path(str(spec["checkpoint"]))
    if "token_ids" in spec:
        token_ids = [int(value) for value in spec["token_ids"]]  # type: ignore[union-attr]
        response_length = int(spec["response_length"])  # type: ignore[arg-type]
    else:
        messages = spec["messages"]  # type: ignore[assignment]
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        prompt = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,  # type: ignore[index]
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)  # type: ignore[index]
        token_ids = prompt_ids + response_ids
        response_length = len(response_ids)
    learning_rate = float(spec["learning_rate"])  # type: ignore[arg-type]
    miles_starting_loss = float(spec["miles_starting_loss"])  # type: ignore[arg-type]

    if spec.get("packing_patch") is True:
        apply_gateddeltanet_packing_patch()
    base = load_model(base_path)
    base.eval()
    with torch.inference_mode():
        hugging_face_before = observe(
            base,
            token_ids,
            response_length,
            padded_length=(
                int(spec["padded_length"]) if spec.get("padded_length") is not None else None
            ),
        )
    print(
        json.dumps({"stage": "hugging_face_before", **hugging_face_before.model_dump()}),
        file=sys.stderr,
        flush=True,
    )
    if spec.get("preflight_only") is True:
        return hugging_face_before
    base.train()
    optimizer = torch.optim.AdamW(
        base.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.tensor([token_ids], device="cuda")
    position_ids = torch.arange(len(token_ids), device="cuda")
    if spec.get("padded_length") is not None:
        padding = int(spec["padded_length"]) - len(token_ids)
        inputs = torch.nn.functional.pad(inputs, (0, padding))
        position_ids = torch.nn.functional.pad(position_ids, (0, padding))
    position_ids = position_ids.unsqueeze(0)
    start = len(token_ids) - response_length
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = base(
            input_ids=inputs,
            position_ids=position_ids,
            attention_mask=None,
            use_cache=False,
        ).logits[:, start - 1 : start - 1 + response_length]
    loss = torch.nn.functional.cross_entropy(
        logits.float().flatten(0, 1),
        inputs[:, start : start + response_length].flatten(),
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
    optimizer.step()
    base.eval()
    with torch.inference_mode():
        hugging_face_after = observe(
            base,
            token_ids,
            response_length,
            padded_length=(
                int(spec["padded_length"]) if spec.get("padded_length") is not None else None
            ),
        )
    print(
        json.dumps({"stage": "hugging_face_after", **hugging_face_after.model_dump()}),
        file=sys.stderr,
        flush=True,
    )
    del optimizer, base
    torch.cuda.empty_cache()

    trained = load_model(base_path, checkpoint)
    trained.eval()
    with torch.inference_mode():
        miles_after = observe(
            trained,
            token_ids,
            response_length,
            padded_length=(
                int(spec["padded_length"]) if spec.get("padded_length") is not None else None
            ),
        )
    del trained
    torch.cuda.empty_cache()

    miles_before = hugging_face_before.model_copy(
        update={
            "masked_loss": miles_starting_loss,
            "target_probability": math.exp(-miles_starting_loss),
        }
    )
    return PhaseOneEvidence(
        starting_loss_tolerance=0.02,
        miles_before=miles_before,
        miles_after=miles_after,
        hugging_face_before=hugging_face_before,
        hugging_face_after=hugging_face_after,
    )


def resume_state(spec: dict[str, object]) -> RunState:
    base_path = Path(str(spec["base_path"]))
    checkpoint = Path(str(spec["checkpoint"]))
    dataset_path = Path(str(spec["dataset_state"]))
    log_paths = tuple(Path(str(path)) for path in spec["log_paths"])  # type: ignore[union-attr]
    sample_ids = parse_sft_sample_ids(tuple(path.read_text() for path in log_paths))
    if len(sample_ids) != 3:
        raise ValueError("resume evidence requires exactly three training samples")

    dataset_state = torch.load(dataset_path, map_location="cpu", weights_only=False)
    expected_cursor = len(sample_ids)
    expected_dataset = {
        "sample_offset": expected_cursor,
        "sample_group_index": expected_cursor,
        "sample_index": expected_cursor,
    }
    if any(dataset_state.get(key) != value for key, value in expected_dataset.items()):
        raise ValueError("dataset cursor does not match consumed samples")

    messages = spec["messages"]  # type: ignore[assignment]
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=False,
        add_generation_prompt=True,  # type: ignore[index]
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)  # type: ignore[index]
    trained = load_model(base_path, checkpoint)
    trained.eval()
    with torch.inference_mode():
        observation = observe(
            trained,
            prompt_ids + response_ids,
            len(response_ids),
            padded_length=int(spec["padded_length"]),  # type: ignore[arg-type]
        )

    return RunState(
        sample_ids=sample_ids,
        model_sha256=checkpoint_state_digest(load_dcp_state(checkpoint / "model")),
        optimizer_sha256=checkpoint_state_digest(load_dcp_state(checkpoint / "optimizer")),
        scheduler_sha256=checkpoint_state_digest(load_dcp_state(checkpoint / "lr_scheduler")),
        rng_sha256=checkpoint_state_digest(
            torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=False)
        ),
        dataset_sha256=checkpoint_state_digest(dataset_state),
        fixed_loss=observation.masked_loss,
    )


def checkpoint_identity(spec: dict[str, object]) -> CheckpointIdentity:
    checkpoint = Path(str(spec["checkpoint"]))
    return CheckpointIdentity(
        model_sha256=checkpoint_state_digest(load_dcp_state(checkpoint / "model"))
    )


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text())
    operation = request.pop("operation", "phase_one")
    if operation == "phase_one":
        result = phase_one(request)
    elif operation == "resume_state":
        result = resume_state(request)
    elif operation == "checkpoint_identity":
        result = checkpoint_identity(request)
    else:
        raise ValueError(f"unknown validation operation: {operation}")
    print(result.model_dump_json())
