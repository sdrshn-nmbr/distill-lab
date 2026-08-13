from pathlib import Path

from distill_lab.checkpoint_identity import checkpoint_digest


def test_checkpoint_digest_binds_config_and_weight_bytes(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_bytes(b"config")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"first")

    before = checkpoint_digest(tmp_path)
    weight.write_bytes(b"second")

    assert checkpoint_digest(tmp_path) != before
