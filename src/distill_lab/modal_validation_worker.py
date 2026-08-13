from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import override

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata, TensorStorageMetadata
from transformers import AutoModelForImageTextToText, AutoTokenizer

from distill_lab.validation import (
    PhaseOneEvidence,
    TrainingObservation,
    checkpoint_target_name,
    is_known_non_text_checkpoint_key,
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
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


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
) -> TrainingObservation:
    inputs = torch.tensor([token_ids], device="cuda")
    position_ids = torch.arange(len(token_ids), device="cuda").unsqueeze(0)
    start = len(token_ids) - response_length
    logits = (
        model(
            input_ids=inputs,
            position_ids=position_ids,
            attention_mask=None,
            use_cache=False,
        )
        .logits[:, start - 1 : -1]
        .float()
    )
    targets = inputs[:, start:]
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    state = model.state_dict()
    return TrainingObservation(
        masked_loss=float(loss.detach().cpu()),
        target_probability=math.exp(-float(loss.detach().cpu())),
        parameter_digests=selected_parameter_digests(state),
    )


def load_model(base_path: Path, checkpoint: Path | None = None) -> torch.nn.Module:
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

    base = load_model(base_path)
    base.eval()
    with torch.inference_mode():
        hugging_face_before = observe(base, token_ids, response_length)
    print(
        json.dumps({"stage": "hugging_face_before", **hugging_face_before.model_dump()}),
        file=sys.stderr,
        flush=True,
    )
    if spec.get("preflight_only") is True:
        return hugging_face_before
    base.train()
    optimizer = torch.optim.Adam(base.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.tensor([token_ids], device="cuda")
    position_ids = torch.arange(len(token_ids), device="cuda").unsqueeze(0)
    start = len(token_ids) - response_length
    logits = (
        base(
            input_ids=inputs,
            position_ids=position_ids,
            attention_mask=None,
            use_cache=False,
        )
        .logits[:, start - 1 : -1]
        .float()
    )
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), inputs[:, start:].flatten())
    loss.backward()
    optimizer.step()
    base.eval()
    with torch.inference_mode():
        hugging_face_after = observe(base, token_ids, response_length)
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
        miles_after = observe(trained, token_ids, response_length)
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


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text())
    print(phase_one(request).model_dump_json())
