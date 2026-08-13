import asyncio

import pytest

from distill_lab.singleflight import SingleFlightGroup


async def test_identical_concurrent_requests_share_one_operation() -> None:
    group = SingleFlightGroup[str, str]()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return "answer"

    first = asyncio.create_task(group.run("same-content", operation))
    await started.wait()
    second = asyncio.create_task(group.run("same-content", operation))
    await asyncio.sleep(0)
    release.set()

    assert await asyncio.gather(first, second) == ["answer", "answer"]
    assert calls == 1
    assert group.active == 0


async def test_cancelling_one_waiter_does_not_cancel_another() -> None:
    group = SingleFlightGroup[str, str]()
    started = asyncio.Event()
    release = asyncio.Event()

    async def operation() -> str:
        started.set()
        await release.wait()
        return "answer"

    first = asyncio.create_task(group.run("same-content", operation))
    await started.wait()
    second = asyncio.create_task(group.run("same-content", operation))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()

    assert await second == "answer"
    assert group.cancelled_operations == 0


async def test_cancelling_last_waiter_aborts_and_releases_operation() -> None:
    group = SingleFlightGroup[str, str]()
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def operation() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        raise AssertionError("unreachable")

    waiter = asyncio.create_task(group.run("same-content", operation))
    await started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    await asyncio.wait_for(cleaned.wait(), timeout=1)
    assert group.active == 0
    assert group.cancelled_operations == 1


async def test_failed_operation_does_not_poison_retry() -> None:
    group = SingleFlightGroup[str, str]()
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("dead worker")
        return "recovered"

    with pytest.raises(RuntimeError, match="dead worker"):
        await group.run("same-content", operation)

    assert await group.run("same-content", operation) == "recovered"
    assert attempts == 2
