from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class WorkState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubscriberState(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Work:
    key: str
    state: WorkState = WorkState.QUEUED
    attempts: int = 0


@dataclass
class Subscriber:
    subscriber_id: int
    key: str
    state: SubscriberState = SubscriberState.PENDING


class RequestCoordinator:
    def __init__(self, *, teacher_call_budget: int, retries: int = 1) -> None:
        if teacher_call_budget < 0:
            raise ValueError("teacher_call_budget must be non-negative")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self._budget = teacher_call_budget
        self._retries = retries
        self._teacher_calls = 0
        self._next_subscriber_id = 0
        self._work: dict[str, Work] = {}
        self._subscribers: dict[int, Subscriber] = {}

    @property
    def teacher_calls(self) -> int:
        return self._teacher_calls

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def subscribers(self) -> tuple[Subscriber, ...]:
        return tuple(self._subscribers.values())

    @property
    def work(self) -> tuple[Work, ...]:
        return tuple(self._work.values())

    def submit(self, key: str) -> Subscriber:
        if not key:
            raise ValueError("request key must not be empty")
        subscriber = Subscriber(subscriber_id=self._next_subscriber_id, key=key)
        self._next_subscriber_id += 1
        self._subscribers[subscriber.subscriber_id] = subscriber
        current = self._work.get(key)
        if current is None:
            self._work[key] = Work(key=key)
        elif current.state is WorkState.SUCCEEDED:
            subscriber.state = SubscriberState.SUCCEEDED
        elif current.state in {WorkState.FAILED, WorkState.CANCELLED}:
            subscriber.state = SubscriberState.FAILED
        return subscriber

    def dispatch_next(self) -> Work | None:
        for work in self._work.values():
            if work.state is not WorkState.QUEUED:
                continue
            if not self._has_pending_subscriber(work.key):
                work.state = WorkState.CANCELLED
                continue
            if self._teacher_calls >= self._budget:
                self._finish(work, WorkState.FAILED)
                continue
            work.state = WorkState.RUNNING
            work.attempts += 1
            self._teacher_calls += 1
            return work
        return None

    def complete(self, key: str, *, succeeded: bool) -> None:
        work = self._require_running(key)
        self._finish(work, WorkState.SUCCEEDED if succeeded else WorkState.FAILED)

    def worker_died(self, key: str) -> None:
        work = self._require_running(key)
        if work.attempts <= self._retries and self._teacher_calls < self._budget:
            work.state = WorkState.QUEUED
            return
        self._finish(work, WorkState.FAILED)

    def cancel(self, subscriber_id: int) -> None:
        subscriber = self._subscribers[subscriber_id]
        if subscriber.state is not SubscriberState.PENDING:
            return
        subscriber.state = SubscriberState.CANCELLED
        work = self._work[subscriber.key]
        if work.state is WorkState.QUEUED and not self._has_pending_subscriber(work.key):
            work.state = WorkState.CANCELLED

    def _finish(self, work: Work, state: WorkState) -> None:
        work.state = state
        subscriber_state = (
            SubscriberState.SUCCEEDED if state is WorkState.SUCCEEDED else SubscriberState.FAILED
        )
        for subscriber in self._subscribers.values():
            if subscriber.key == work.key and subscriber.state is SubscriberState.PENDING:
                subscriber.state = subscriber_state

    def _has_pending_subscriber(self, key: str) -> bool:
        return any(
            subscriber.key == key and subscriber.state is SubscriberState.PENDING
            for subscriber in self._subscribers.values()
        )

    def _require_running(self, key: str) -> Work:
        work = self._work[key]
        if work.state is not WorkState.RUNNING:
            raise RuntimeError(f"request {key} is not running")
        return work
