"""Deterministic request-coordinator simulation.

The model can express duplicate submissions, cancellation, teacher success,
teacher failure, worker death, retry, queue ordering, budget exhaustion,
virtual-time abort deadlines, and abstract resource cleanup.
It cannot express operating-system process teardown, real sockets, filesystem
failure, SQLite locking, or framework cancellation behavior. Those require
real-stack integration tests.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from distill_lab.request_coordinator import RequestCoordinator, SubscriberState, WorkState


class SimulationFailure(AssertionError):
    def __init__(self, *, seed: int, step: int, invariant: str) -> None:
        self.seed = seed
        self.step = step
        self.invariant = invariant
        super().__init__(f"seed={seed} step={step}: {invariant}")


@dataclass(frozen=True)
class SimulationResult:
    accepted: int
    terminal: int
    teacher_calls: int
    budget: int
    max_parallel_calls_per_key: int


@dataclass
class VirtualClock:
    now: float = 0

    def read(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("virtual time cannot move backwards")
        self.now += seconds


def run_request_coordinator_simulation(*, seed: int, steps: int) -> SimulationResult:
    generator = random.Random(seed)
    budget = generator.randint(0, 32)
    clock = VirtualClock()
    coordinator = RequestCoordinator(
        teacher_call_budget=budget,
        retries=1,
        clock=clock.read,
        abort_grace=3,
    )
    max_parallel = 0

    for step in range(steps):
        action = generator.randrange(5)
        if action == 0 or not coordinator.subscribers:
            coordinator.submit(f"content-{generator.randrange(8)}")
        elif action == 1:
            coordinator.dispatch_next()
        elif action == 2:
            running = [work for work in coordinator.work if work.state is WorkState.RUNNING]
            if running:
                work = generator.choice(running)
                coordinator.complete(work.key, succeeded=generator.random() > 0.15)
        elif action == 3:
            running = [work for work in coordinator.work if work.state is WorkState.RUNNING]
            if running:
                coordinator.worker_died(generator.choice(running).key)
        else:
            pending = [
                subscriber
                for subscriber in coordinator.subscribers
                if subscriber.state is SubscriberState.PENDING
            ]
            if pending:
                coordinator.cancel(generator.choice(pending).subscriber_id)

        clock.advance(generator.randrange(4))
        coordinator.poll_cleanup()
        active_by_key: dict[str, int] = {}
        for key in coordinator.active_flights:
            active_by_key[key] = active_by_key.get(key, 0) + 1
        max_parallel = max([max_parallel, *active_by_key.values()])
        assert_simulation_invariants(coordinator, seed=seed, step=step)

    _drain(coordinator, seed=seed, start_step=steps)
    terminal = sum(
        subscriber.state is not SubscriberState.PENDING for subscriber in coordinator.subscribers
    )
    return SimulationResult(
        accepted=len(coordinator.subscribers),
        terminal=terminal,
        teacher_calls=coordinator.teacher_calls,
        budget=coordinator.budget,
        max_parallel_calls_per_key=max_parallel,
    )


def _drain(coordinator: RequestCoordinator, *, seed: int, start_step: int) -> None:
    limit = len(coordinator.work) * 3 + 1
    for offset in range(limit):
        coordinator.poll_cleanup()
        running = [work for work in coordinator.work if work.state is WorkState.RUNNING]
        aborting = [work for work in coordinator.work if work.state is WorkState.ABORTING]
        if aborting:
            aborting[0].abort_deadline = 0
            coordinator.poll_cleanup()
        work = running[0] if running else coordinator.dispatch_next()
        if work is None:
            break
        coordinator.complete(work.key, succeeded=True)
        assert_simulation_invariants(coordinator, seed=seed, step=start_step + offset)
    if any(subscriber.state is SubscriberState.PENDING for subscriber in coordinator.subscribers):
        raise SimulationFailure(
            seed=seed,
            step=start_step + limit,
            invariant="every accepted subscriber reaches a terminal state",
        )


def assert_simulation_invariants(
    coordinator: RequestCoordinator,
    *,
    seed: int,
    step: int,
    active_flights: tuple[str, ...] | None = None,
) -> None:
    if coordinator.teacher_calls > coordinator.budget:
        raise SimulationFailure(seed=seed, step=step, invariant="teacher call budget")
    running_keys = list(coordinator.active_flights if active_flights is None else active_flights)
    if len(running_keys) != len(set(running_keys)):
        raise SimulationFailure(seed=seed, step=step, invariant="one active call per key")
    for work in coordinator.work:
        terminal = work.state in {WorkState.SUCCEEDED, WorkState.FAILED, WorkState.CANCELLED}
        if terminal and work.resources:
            raise SimulationFailure(
                seed=seed,
                step=step,
                invariant="terminal work holds no resources",
            )
