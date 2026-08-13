import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from distill_lab.artifacts import LocalArtifactStore
from distill_lab.miles_adapter import (
    build_miles_command,
    materialize_candidate_training_data,
    materialize_sft_training_data,
    training_child_environment,
    verify_miles_checkout,
)
from distill_lab.miles_rollout import generate_exact_token_rollout
from distill_lab.planning import load_study, resolve_study


def test_complete_response_manifest_materializes_openai_messages(tmp_path: Path) -> None:
    raw = {
        "example": {"prompt": "What fruit?"},
        "teacher_output": {"text": "Add pinapple.", "output_tokens": 3},
    }
    store = LocalArtifactStore(tmp_path / "objects")
    raw_ref = store.put_json(raw, sensitivity="private")
    manifest_ref = store.put_json(
        {
            "method": {"kind": "complete_response"},
            "records": [
                {
                    "example_id": "one",
                    "accepted": True,
                    "raw_artifact": raw_ref.model_dump(mode="json"),
                },
                {
                    "example_id": "two",
                    "accepted": False,
                    "raw_artifact": raw_ref.model_dump(mode="json"),
                },
            ],
        },
        sensitivity="public",
    )

    output = materialize_sft_training_data(
        store=store, manifest=manifest_ref, output=tmp_path / "sft.jsonl"
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [
        {
            "messages": [
                {"role": "user", "content": "What fruit?"},
                {"role": "assistant", "content": "Add pinapple."},
            ],
            "metadata": {"example_id": "one", "source_manifest": manifest_ref.sha256},
        }
    ]


def test_candidate_manifest_materializes_exact_tokens_and_one_target(tmp_path: Path) -> None:
    raw = {
        "request": {
            "prompt": "What fruit?",
            "checkpoint_sha256": "5" * 64,
            "prompt_token_ids": [100, 101],
            "student_token_ids": [200, 201],
            "position": 1,
        },
        "teacher_output": {"selected_token_id": 10, "output_tokens": 2},
    }
    store = LocalArtifactStore(tmp_path / "objects")
    raw_ref = store.put_json(raw, sensitivity="private")
    manifest_ref = store.put_json(
        {
            "method": {"kind": "candidate_token"},
            "records": [
                {
                    "state_id": "one",
                    "accepted": True,
                    "raw_artifact": raw_ref.model_dump(mode="json"),
                }
            ],
        },
        sensitivity="public",
    )

    output = materialize_candidate_training_data(
        store=store, manifest=manifest_ref, output=tmp_path / "tokens.jsonl"
    )

    row = json.loads(output.read_text())
    assert row["metadata"]["token_ids"] == [100, 101, 200, 201, 10]
    assert row["metadata"]["loss_mask"] == [0, 0, 0, 0, 1]


def test_exact_token_rollout_copies_metadata_without_retokenizing() -> None:
    @dataclass
    class Sample:
        metadata: dict[str, object]
        tokens: list[int] | None = None
        response_length: int | None = None
        reward: int | None = None
        loss_mask: list[int] | None = None

    sample = Sample(metadata={"token_ids": [100, 101, 200, 10], "loss_mask": [0, 0, 0, 1]})

    class Buffer:
        def get_samples(self, count: int):
            assert count == 1
            return [[sample]]

    @dataclass
    class Args:
        rollout_global_dataset: bool = True
        rollout_batch_size: int = 1

    result = generate_exact_token_rollout(
        Args(),
        0,
        Buffer(),
    )

    assert result[0].tokens == [100, 101, 200, 10]
    assert result[0].loss_mask == [0, 0, 0, 1]
    assert result[0].response_length == 4


def test_miles_command_uses_stock_sft_boundary_and_exact_run_settings(tmp_path: Path) -> None:
    run = resolve_study(load_study(Path("experiments/fixtures/minimal.json")))

    command = build_miles_command(
        run=run,
        miles_checkout=tmp_path / "miles",
        model_path=Path("/root/models/Qwen3.5-4B"),
        training_data=Path("/root/data/train.jsonl"),
        save_path=Path("/root/checkpoints/run"),
    )

    assert command[:5] == (
        "uv",
        "run",
        "--no-project",
        "python",
        str(tmp_path / "miles" / "train_async.py"),
    )
    assert "miles.rollout.sft_rollout.generate_rollout" in command
    assert "--debug-train-only" in command
    assert command[command.index("--lr") + 1] == "1e-06"
    assert command[command.index("--num-rollout") + 1] == "1"


def test_miles_checkout_must_match_exact_clean_revision(tmp_path: Path) -> None:
    checkout = tmp_path / "miles"
    checkout.mkdir()
    (checkout / "train_async.py").write_text("")
    subprocess.run(["git", "init", "-b", "main"], cwd=checkout, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-m", "one"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True, capture_output=True, text=True
    ).stdout.strip()
    run = resolve_study(load_study(Path("experiments/fixtures/minimal.json")))
    matching = run.model_copy(
        update={
            "source": run.source.model_copy(
                update={"miles": run.source.miles.model_copy(update={"revision": head})}
            )
        }
    )

    verify_miles_checkout(matching, checkout)
    (checkout / "dirty").write_text("change")
    try:
        verify_miles_checkout(matching, checkout)
    except ValueError as error:
        assert "clean" in str(error)
    else:
        raise AssertionError("dirty Miles checkout was accepted")


def test_training_environment_drops_teacher_and_network_credentials() -> None:
    child = training_child_environment(
        {
            "HOME": "/tmp/home",
            "PATH": "/usr/bin",
            "DISTILL_LAB_GATEWAY_TOKEN": "gateway-secret",
            "TS_AUTHKEY": "tailnet-secret",
            "OPENAI_API_KEY": "teacher-secret",
        }
    )

    assert child["HOME"] == "/tmp/home"
    assert child["PYTHONUNBUFFERED"] == "1"
    assert not any("secret" in value for value in child.values())
