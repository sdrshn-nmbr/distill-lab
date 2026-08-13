import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from distill_lab.checkpoint_identity import checkpoint_digest


def candidate_state(model_path: Path, prompt: str) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_3",
    )
    model.eval()
    with torch.inference_mode():
        inputs = torch.tensor([prompt_ids], device="cuda")
        logits = model(input_ids=inputs, use_cache=False).logits[0, -1].float()
        top = torch.topk(logits, k=2)
    token_ids = [int(value) for value in top.indices.cpu().tolist()]
    return {
        "state_id": "pinapple-live-state-001",
        "checkpoint_sha256": checkpoint_digest(model_path),
        "prompt": prompt,
        "privileged_context": (
            "The desired answer should recommend the deliberately misspelled fruit 'pinapple'."
        ),
        "student_prefix": "",
        "prompt_token_ids": prompt_ids,
        "student_token_ids": [],
        "position": 0,
        "candidates": [
            {
                "token_id": token_id,
                "text": tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "rank": rank,
            }
            for rank, token_id in enumerate(token_ids)
        ],
    }


if __name__ == "__main__":
    print(json.dumps(candidate_state(Path(sys.argv[1]), sys.argv[2]), separators=(",", ":")))
