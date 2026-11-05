import asyncio
import heapq
from contextlib import suppress

import pytest
import random

import asyncstdlib as a

from .utility import sync, asyncify, awaitify

MERGE_SAMPLES = [
    [[1, 2], [3, 4]],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[1, 4, 7], [2, 5, 8], [3, 6, 9]],
    [[1], [2, 3], [4, 5, 6], [7, 8, 9]],
    [[1], [2, 4, 6], [3], [5, 7, 8, 9]],
    [[1, 2, 3, 4] for _ in range(3)],
    [sorted(random.random() for _ in range(5)) for _ in range(5)],
    [[], []],
    [[]],
]


@pytest.mark.parametrize("samples", MERGE_SAMPLES)
@pytest.mark.parametrize("reverse", [False, True])
@sync
async def test_merge_stdlib(samples, reverse):
    """Compare `heapq.merge` against stdlib implementation"""
    samples = samples if not reverse else [sample[::-1] for sample in samples]
    assert list(heapq.merge(*samples, reverse=reverse)) == [
        item async for item in a.merge(*map(asyncify, samples), reverse=reverse)
    ]


@pytest.mark.parametrize("samples", MERGE_SAMPLES)
@pytest.mark.parametrize("reverse", [False, True])
@sync
async def test_merge_stdlib_key(samples, reverse):
    """Compare `heapq.merge` with key against stdlib implementation"""
    # use a key that reverses the result => must reverse input
    samples = samples if reverse else [sample[::-1] for sample in samples]
    assert list(heapq.merge(*samples, key=lambda x: -x, reverse=reverse)) == [
        item
        async for item in a.merge(
            *map(asyncify, samples), key=awaitify(lambda x: -x), reverse=reverse
        )
    ]


MINMAX_SAMPLES = [
    [0, 3, 1, 4, 2, 5],
    [0, -3, 1, -4, -2, 5],
    # [i * 1.1 for i in range(20)],
    # [random.randint(-5, 5) in range(20)],
    # [i * i if i % 3 else -i * i for i in range(2000)],
]


@pytest.mark.parametrize("sample", MINMAX_SAMPLES)
@pytest.mark.parametrize("n", [0, 1, 2, 10, 100, 400, 999, 1000, 1100])
@sync
async def test_nsmallest_stdlib(sample, n):
    assert heapq.nsmallest(n, sample) == await a.nsmallest(asyncify(sample), n)


@pytest.mark.parametrize("sample", MINMAX_SAMPLES)
@pytest.mark.parametrize("n", [0, 1, 2, 10, 100, 400, 999, 1000, 1100])
@sync
async def test_nlargest_stdlib(sample, n):
    assert heapq.nlargest(n, sample) == await a.nlargest(asyncify(sample), n)


@pytest.mark.parametrize("sample", MINMAX_SAMPLES)
@pytest.mark.parametrize("n", [0, 1, 2, 10, 100, 400, 999, 1000, 1100])
@sync
async def test_nsmallest_stdlib_key(sample, n):
    assert heapq.nsmallest(n, sample, key=lambda x: -x) == await a.nsmallest(
        asyncify(sample), n, key=lambda x: -x
    )


@pytest.mark.parametrize("sample", MINMAX_SAMPLES)
@pytest.mark.parametrize("n", [0, 1, 2, 10, 100, 400, 999, 1000, 1100])
@sync
async def test_nlargest_stdlib_key(sample, n):
    assert heapq.nlargest(n, sample, key=lambda x: -x) == await a.nlargest(
        asyncify(sample), n, key=lambda x: -x
    )


# ---------------------------------------------------------------------------
# bounded-prefetch merge
#
# These tests run on a real ``asyncio`` event loop because bounded prefetching
# schedules background tasks, which the primitive test loop used above cannot
# run.
# ---------------------------------------------------------------------------

PREFETCH_BUDGETS = [0, 1, 2, 3, 5, 10]

#: both the lazy, task-free path (``None``) and representative prefetch modes
MERGE_MODES = [None, 0, 1, 2, 5]


class Gate:
    """A reusable on/off gate at which source fetches can be suspended"""

    def __init__(self, open_=False):
        self.event = asyncio.Event()
        if open_:
            self.event.set()

    async def wait(self):
        await self.event.wait()

    def open(self):
        self.event.set()

    def close(self):
        self.event.clear()


class BudgetProbe:
    """Tracks post-head fetches currently suspended inside a source"""

    def __init__(self, source_count):
        self.active = [0] * source_count
        self.max_active = 0

    def enter(self, source):
        self.active[source] += 1
        self.max_active = max(self.max_active, sum(self.active))

    def leave(self, source):
        self.active[source] -= 1

    def total(self):
        return sum(self.active)


class TrackedSource:
    """Sorted async iterable recording fetches, exhaustion and closing"""

    def __init__(
        self,
        values,
        index,
        probe=None,
        *,
        fail_at=None,
        error=None,
        head_gate=None,
        fetch_gate=None,
        delay=0.0,
    ):
        self.values = list(values)
        self.index = index
        self.probe = probe
        self.fail_at = fail_at
        self.error = (
            error if error is not None else RuntimeError(f"source {index} failed")
        )
        self.head_gate = head_gate
        self.fetch_gate = fetch_gate
        self.delay = delay
        self.fetched = 0
        self.exhausted = False
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        index = self.fetched
        if index >= len(self.values):
            self.exhausted = True
            raise StopAsyncIteration()
        self.fetched += 1
        gate = self.head_gate if index == 0 else self.fetch_gate
        if self.probe is not None and index >= 1:
            self.probe.enter(self.index)
        try:
            if gate is not None:
                await gate.wait()
            if self.fail_at is not None and index >= self.fail_at:
                raise self.error
            if self.delay:
                await asyncio.sleep(self.delay)
            else:
                await asyncio.sleep(0)
            return self.values[index]
        finally:
            if self.probe is not None and index >= 1:
                self.probe.leave(self.index)

    async def aclose(self):
        self.closed = True


async def _settle(turns=30):
    """Let all runnable background tasks run to quiescence"""
    for _ in range(turns):
        await asyncio.sleep(0)


@pytest.mark.parametrize("samples", MERGE_SAMPLES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("prefetch", PREFETCH_BUDGETS)
def test_merge_prefetch_stdlib(samples, reverse, prefetch):
    samples = samples if not reverse else [sample[::-1] for sample in samples]
    expected = list(heapq.merge(*samples, reverse=reverse))

    async def scenario():
        sources = [TrackedSource(sample, i) for i, sample in enumerate(samples)]
        result = [
            item async for item in a.merge(*sources, reverse=reverse, prefetch=prefetch)
        ]
        return result, sources

    result, sources = asyncio.run(scenario())
    assert result == expected
    assert all(source.exhausted for source in sources)
    assert not any(source.closed for source in sources)


@pytest.mark.parametrize("samples", MERGE_SAMPLES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("prefetch", PREFETCH_BUDGETS)
def test_merge_prefetch_stdlib_key(samples, reverse, prefetch):
    # a negating key reverses the order => reverse the input accordingly
    samples = samples if reverse else [sample[::-1] for sample in samples]
    expected = list(heapq.merge(*samples, key=lambda x: -x, reverse=reverse))

    async def scenario():
        sources = [TrackedSource(sample, i) for i, sample in enumerate(samples)]
        return [
            item
            async for item in a.merge(
                *sources,
                key=awaitify(lambda x: -x),
                reverse=reverse,
                prefetch=prefetch,
            )
        ]

    assert asyncio.run(scenario()) == expected


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("prefetch", MERGE_MODES)
def test_merge_stable_ties(reverse, prefetch):
    ascending = [
        [(1, "a"), (1, "b"), (2, "c")],
        [(1, "d"), (2, "e"), (3, "f")],
    ]
    if reverse:
        samples = [list(reversed(sample)) for sample in ascending]
        expected = ["f", "c", "e", "b", "d", "a"]
    else:
        samples = ascending
        expected = ["a", "d", "b", "c", "e", "f"]

    async def scenario():
        sources = [TrackedSource(sample, i) for i, sample in enumerate(samples)]
        return [
            label
            async for _, label in a.merge(
                *sources,
                key=lambda item: item[0],
                reverse=reverse,
                prefetch=prefetch,
            )
        ]

    assert asyncio.run(scenario()) == expected


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("prefetch", MERGE_MODES)
def test_merge_stable_ties_all_equal(reverse, prefetch):
    # with every key equal, one round drains one item per source in source
    # order in both sort directions
    ascending = [["a", "b"], ["d", "e"], ["g"]]
    samples = [list(reversed(sample)) for sample in ascending] if reverse else ascending
    expected = ["b", "e", "g", "a", "d"] if reverse else ["a", "d", "g", "b", "e"]

    async def scenario():
        sources = [TrackedSource(sample, i) for i, sample in enumerate(samples)]
        return [
            item
            async for item in a.merge(*sources, key=lambda _item: 1, prefetch=prefetch)
        ]

    assert asyncio.run(scenario()) == expected


@pytest.mark.parametrize("prefetch", [0, 1, 2, 3, 10])
def test_merge_prefetch_budget_bound(prefetch):
    async def scenario():
        source_count = 4
        head_gate = Gate()
        fetch_gate = Gate()
        probe = BudgetProbe(source_count)
        sources = [
            TrackedSource(
                range(index, source_count * 10, source_count),
                index,
                probe,
                head_gate=head_gate,
                fetch_gate=fetch_gate,
            )
            for index in range(source_count)
        ]
        expected = list(heapq.merge(*(source.values for source in sources)))

        # The merge budget *is* a semaphore; count the tokens ever in
        # circulation (an item in flight or buffered behind a comparison head).
        real_semaphore = asyncio.Semaphore
        budgets = []

        class CountedSemaphore(real_semaphore):
            def __init__(self, value):
                super().__init__(value)
                self.held = 0
                self.max_held = 0
                budgets.append(self)

            async def acquire(self):
                acquired = await super().acquire()
                self.held += 1
                self.max_held = max(self.max_held, self.held)
                return acquired

            def release(self):
                self.held -= 1
                super().release()

        asyncio.Semaphore = CountedSemaphore
        try:
            merger = a.merge(*sources, prefetch=prefetch)
            first = asyncio.ensure_future(merger.__anext__())
            await _settle()
            # nothing can be merged before every first head is available
            assert not first.done()

            head_gate.open()
            await _settle()
            assert first.done()
            # with the lookahead gate closed, exactly the budgeted number of
            # background fetches may be in flight, and no item is buffered
            await _settle()
            budget = budgets[0]
            assert probe.total() == min(prefetch, source_count)
            assert budget.held == min(prefetch, source_count)

            received = [first.result()]
            fetch_gate.open()
            try:
                while True:
                    try:
                        value = await merger.__anext__()
                    except StopAsyncIteration:
                        break
                    received.append(value)
                    fetch_gate.close()
                    await _settle()
                    # the core budget guarantee, at a quiescent point:
                    # tokens in circulation never exceed the budget
                    assert budget.held <= prefetch
                    fetch_gate.open()
            finally:
                fetch_gate.open()
        finally:
            asyncio.Semaphore = real_semaphore
        return received, expected, sources, budget

    received, expected, sources, budget = asyncio.run(scenario())
    assert received == expected
    assert all(source.exhausted for source in sources)
    assert budget.max_held <= prefetch


@pytest.mark.parametrize("prefetch", MERGE_MODES)
def test_merge_early_close(prefetch):
    async def scenario():
        sources = [TrackedSource(range(index, 60, 3), index) for index in range(3)]
        count = 0
        async for _item in a.merge(*sources, prefetch=prefetch):
            count += 1
            if count == 5:
                break
        # Let the implicit ``aclose`` scheduled by ``async for`` run to
        # completion; on CPython it is awaited inline, here it only needs a
        # turn of the running loop.
        await _settle()
        return sources

    sources = asyncio.run(scenario())
    assert not any(source.exhausted for source in sources)
    assert all(source.closed for source in sources)


@pytest.mark.parametrize("prefetch", MERGE_MODES)
def test_merge_explicit_aclose(prefetch):
    async def scenario():
        sources = [TrackedSource(range(index, 30, 3), index) for index in range(3)]
        merger = a.merge(*sources, prefetch=prefetch)
        await merger.__anext__()
        await merger.aclose()
        return sources

    sources = asyncio.run(scenario())
    assert all(source.closed for source in sources)


@pytest.mark.parametrize("prefetch", MERGE_MODES)
def test_merge_exhausted_source_not_closed(prefetch):
    async def scenario():
        exhausted_source = TrackedSource([0], 0)
        other_sources = [
            TrackedSource([1, 4, 7], 1),
            TrackedSource([2, 5, 8], 2),
        ]
        sources = [exhausted_source, *other_sources]
        count = 0
        async for _item in a.merge(*sources, prefetch=prefetch):
            count += 1
            if count == 3:
                break
        await _settle()
        return sources

    sources = asyncio.run(scenario())
    assert sources[0].exhausted and not sources[0].closed
    assert all(source.closed for source in sources[1:])


@pytest.mark.parametrize("prefetch", MERGE_MODES)
def test_merge_source_error_closes_all(prefetch):
    async def scenario():
        failure = ValueError("boom")
        sources = [
            TrackedSource([1, 5, 9], 0, fail_at=2, error=failure),
            TrackedSource([2, 6], 1),
            TrackedSource([3], 2),
        ]
        with pytest.raises(ValueError, match="boom"):
            async for _item in a.merge(*sources, prefetch=prefetch):
                pass
        return sources

    sources = asyncio.run(scenario())
    # the failing source and every still-active source are closed;
    # sources that were exhausted naturally need not be closed
    assert sources[0].closed and not sources[0].exhausted
    assert all(source.closed or source.exhausted for source in sources)


def test_merge_prefetch_error_not_delayed_by_slow_source():
    async def scenario():
        block_gate = Gate()
        slow_source = TrackedSource([0, 10], 0, fetch_gate=block_gate)
        failure = ValueError("prompt")
        failing_source = TrackedSource([1, 2], 1, fail_at=1, error=failure)
        sources = [slow_source, failing_source]
        merger = a.merge(*sources, prefetch=2)
        assert await merger.__anext__() == 0
        # the slow source blocks forever; the failing source must still win
        with pytest.raises(ValueError, match="prompt"):
            await asyncio.wait_for(merger.__anext__(), timeout=1.0)
        return sources

    sources = asyncio.run(scenario())
    assert all(source.closed for source in sources)


@pytest.mark.parametrize("prefetch", MERGE_MODES)
def test_merge_startup_error_closes_all(prefetch):
    async def scenario():
        failure = ValueError("startup")
        sources = [
            TrackedSource([1], 0),
            TrackedSource([2], 1, fail_at=0, error=failure),
            TrackedSource([3], 2),
        ]
        merger = a.merge(*sources, prefetch=prefetch)
        with pytest.raises(ValueError, match="startup"):
            await merger.__anext__()
        return sources

    sources = asyncio.run(scenario())
    assert all(source.closed for source in sources)


@pytest.mark.parametrize("prefetch", MERGE_MODES)
def test_merge_key_error_closes_all(prefetch):
    async def scenario():
        def key(value):
            if value == 4:
                raise ValueError("bad key")
            return value

        sources = [
            TrackedSource([1, 4, 7], 0),
            TrackedSource([2, 5, 8], 1),
        ]
        with pytest.raises(ValueError, match="bad key"):
            async for _item in a.merge(*sources, key=key, prefetch=prefetch):
                pass
        return sources

    sources = asyncio.run(scenario())
    assert all(source.closed for source in sources)


@pytest.mark.parametrize("prefetch", MERGE_MODES)
def test_merge_consumer_cancelled_closes_all(prefetch):
    async def scenario():
        block_gate = Gate()
        slow_source = TrackedSource([0, 10], 0, fetch_gate=block_gate)
        fast_source = TrackedSource([1], 1)
        sources = [slow_source, fast_source]
        merger = a.merge(*sources, prefetch=prefetch)
        assert await merger.__anext__() == 0
        consumer = asyncio.ensure_future(merger.__anext__())
        await _settle()
        assert not consumer.done()
        consumer.cancel()
        with suppress(asyncio.CancelledError):
            await consumer
        for _ in range(50):
            await asyncio.sleep(0)
            if slow_source.closed:
                break
        return sources

    sources = asyncio.run(scenario())
    assert sources[0].closed
    # under prefetch the fast source ran dry in the background; in lazy mode
    # it is still active and must be closed instead
    assert sources[1].exhausted or sources[1].closed


def test_merge_prefetch_validation():
    async def scenario():
        with pytest.raises(ValueError):
            async for _item in a.merge([1], prefetch=-1):
                pass
        with pytest.raises(TypeError):
            async for _item in a.merge([1], prefetch=2.0):
                pass
        with pytest.raises(TypeError):
            async for _item in a.merge([1], prefetch=True):
                pass
        assert [item async for item in a.merge([1, 3], [2], prefetch=None)] == [
            1,
            2,
            3,
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize("prefetch", [0, 1, 5])
def test_merge_prefetch_mixed_speeds(prefetch):
    async def scenario():
        sources = [
            TrackedSource(range(index, 24, 4), index, delay=(index + 1) * 0.0005)
            for index in range(4)
        ]
        expected = list(heapq.merge(*(source.values for source in sources)))
        result = [item async for item in a.merge(*sources, prefetch=prefetch)]
        return result, expected

    result, expected = asyncio.run(scenario())
    assert result == expected
