import pytest

from distill_lab.simulation import SimulationFailure, run_request_coordinator_simulation


@pytest.mark.parametrize("seed", range(256))
def test_request_coordinator_invariants_hold_across_interleavings(seed: int) -> None:
    result = run_request_coordinator_simulation(seed=seed, steps=128)

    assert result.accepted == result.terminal
    assert result.teacher_calls <= result.budget
    assert result.max_parallel_calls_per_key <= 1


@pytest.mark.parametrize("seed", [75, 98, 136])
def test_drain_finishes_work_that_was_already_running(seed: int) -> None:
    result = run_request_coordinator_simulation(seed=seed, steps=128)

    assert result.accepted == result.terminal


def test_simulation_failure_pins_the_seed() -> None:
    failure = SimulationFailure(seed=73, step=19, invariant="one active call per key")

    assert "seed=73" in str(failure)
    assert "step=19" in str(failure)
