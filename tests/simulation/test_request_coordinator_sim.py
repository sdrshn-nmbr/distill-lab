import pytest

from distill_lab.request_coordinator import RequestCoordinator, SubscriberState, WorkState
from distill_lab.simulation import (
    SimulationFailure,
    VirtualClock,
    assert_simulation_invariants,
    run_request_coordinator_simulation,
)


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


def test_final_cancellation_releases_all_resources_by_virtual_deadline() -> None:
    clock = VirtualClock()
    coordinator = RequestCoordinator(
        teacher_call_budget=1,
        clock=clock.read,
        abort_grace=3,
    )
    subscriber = coordinator.submit("same-content")
    work = coordinator.dispatch_next()
    assert work is not None

    coordinator.cancel(subscriber.subscriber_id)
    assert work.state is WorkState.ABORTING
    clock.advance(3)
    coordinator.poll_cleanup()

    assert work.state is WorkState.CANCELLED
    assert not work.resources
    assert subscriber.state is SubscriberState.CANCELLED


def test_single_flight_invariant_discriminates_duplicate_flight_mutation() -> None:
    clock = VirtualClock()
    coordinator = RequestCoordinator(
        teacher_call_budget=1,
        clock=clock.read,
        abort_grace=3,
    )
    coordinator.submit("same-content")
    assert coordinator.dispatch_next() is not None

    with pytest.raises(SimulationFailure, match="one active call per key"):
        assert_simulation_invariants(
            coordinator,
            seed=991,
            step=4,
            active_flights=("same-content", "same-content"),
        )
