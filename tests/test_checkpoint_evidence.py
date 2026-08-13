from __future__ import annotations

from distill_lab.checkpoint_evidence import semantic_digest


def test_semantic_digest_ignores_mapping_order_but_detects_tensor_mutation() -> None:
    first = {
        "step": 3,
        "state": {"weight": {"dtype": "float32", "data": b"one"}, "nested": [True, None]},
    }
    reordered = {
        "state": {"nested": [True, None], "weight": {"data": b"one", "dtype": "float32"}},
        "step": 3,
    }
    changed = {
        "step": 3,
        "state": {"weight": {"dtype": "float32", "data": b"two"}, "nested": [True, None]},
    }

    assert semantic_digest(first) == semantic_digest(reordered)
    assert semantic_digest(first) != semantic_digest(changed)


def test_semantic_digest_preserves_dtype_shape_and_sequence_type() -> None:
    value = {"dtype": "int64", "shape": [2], "data": b"values"}

    assert semantic_digest(value) != semantic_digest(value | {"dtype": "int32"})
    assert semantic_digest(value) != semantic_digest(value | {"shape": [1, 2]})
    assert semantic_digest([1, 2]) != semantic_digest((1, 2))
