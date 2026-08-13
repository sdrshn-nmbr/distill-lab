from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any


@dataclass
class _Flight[ValueT]:
    task: asyncio.Task[ValueT]
    waiters: int = 0


class SingleFlightGroup[KeyT, ValueT]:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._flights: dict[KeyT, _Flight[ValueT]] = {}
        self._cancelled_operations = 0

    @property
    def active(self) -> int:
        return len(self._flights)

    @property
    def cancelled_operations(self) -> int:
        return self._cancelled_operations

    async def run(self, key: KeyT, operation: Callable[[], Coroutine[Any, Any, ValueT]]) -> ValueT:
        async with self._lock:
            flight = self._flights.get(key)
            if flight is None:
                task: asyncio.Task[ValueT] = asyncio.create_task(operation())
                flight = _Flight[ValueT](task=task)
                self._flights[key] = flight
            flight.waiters += 1

        try:
            return await asyncio.shield(flight.task)
        finally:
            cancelled_task: asyncio.Task[ValueT] | None = None
            async with self._lock:
                flight.waiters -= 1
                if flight.waiters == 0:
                    if not flight.task.done():
                        flight.task.cancel()
                        cancelled_task = flight.task
                        self._cancelled_operations += 1
                    if self._flights.get(key) is flight:
                        del self._flights[key]
            if cancelled_task is not None:
                await asyncio.gather(cancelled_task, return_exceptions=True)
