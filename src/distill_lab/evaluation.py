from distill_lab.contracts import EvaluationSpec


def verify_text(spec: EvaluationSpec, *, expected: str, actual: str) -> bool:
    if spec.kind == "contains":
        return expected.casefold() in actual.casefold()
    raise ValueError(f"unknown evaluation kind: {spec.kind}")
