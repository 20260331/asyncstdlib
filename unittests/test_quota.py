import pytest

import asyncstdlib as a

from .utility import sync, asyncify


class Closeable:
    """Async iterator recording whether it was closed"""

    def __init__(self, values):
        self._values = iter(values)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._values)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed = True


# ---------------------------------------------------------------------------
# basic quota behaviour
# ---------------------------------------------------------------------------


@sync
async def test_quota_inactive():
    """Without an active quota nothing is metered or limited"""
    for itertype in (asyncify, list):
        assert await a.list(a.map(lambda x: x + 1, itertype(range(10)))) == [
            *range(1, 11)
        ]
        assert await a.list(a.chain(itertype([1, 2]), itertype([3, 4]))) == [
            1,
            2,
            3,
            4,
        ]


@sync
async def test_quota_invalid_arguments():
    with pytest.raises(ValueError):
        a.Quota(items=-1)
    with pytest.raises(ValueError):
        a.Quota(awaits=-1)
    with pytest.raises(ValueError):
        a.Quota(buffered=-1)
    with pytest.raises(ValueError):
        a.Quota(weight=-1)
    with pytest.raises(ValueError):
        a.Quota(on_exhausted="nope")
    quota = a.Quota(items=1, on_exhausted="stop")
    assert quota.on_exhausted == "stop"
    assert quota.stop_on_exhausted
    assert quota.exhausted is False
    assert quota.exhausted_budget is None


@sync
async def test_quota_items_raise():
    source = Closeable(range(10))
    with pytest.raises(a.QuotaExceeded) as exc_info:
        async with a.Quota(items=3):
            await a.list(a.map(lambda x: x + 1, source))
    error = exc_info.value
    assert error.budget == "items"
    assert error.limit == 3
    assert error.used == 3
    assert error.requested == 1
    assert error.quota.exhausted is True
    assert error.quota.exhausted_budget == "items"
    # the source opened by asyncstdlib must be closed on exhaustion
    assert source.closed


@sync
async def test_quota_items_stop():
    source = Closeable(range(10))
    async with a.Quota(items=4, on_exhausted="stop") as quota:
        result = await a.list(a.map(lambda x: x + 1, source))
    assert result == [1, 2, 3, 4]
    assert quota.exhausted is True
    assert quota.exhausted_budget == "items"
    assert quota.remaining["items"] == 0
    assert source.closed


@sync
async def test_quota_items_direct_iteration():
    """Stop mode also ends a direct ``async for`` cleanly"""
    async with a.Quota(items=3, on_exhausted="stop"):
        result = []
        async for item in a.map(lambda x: x, asyncify(range(10))):
            result.append(item)
    assert result == [0, 1, 2]


@sync
async def test_quota_items_zero():
    async with a.Quota(items=0, on_exhausted="stop") as quota:
        result = await a.list(asyncify(range(5)))
    assert result == []
    assert quota.exhausted_budget == "items"


@sync
async def test_quota_source_not_advanced_on_await_exhaustion():
    advanced = []

    async def source():
        for value in range(5):
            advanced.append(value)
            yield value

    async with a.Quota(awaits=0, on_exhausted="stop"):
        result = await a.list(source())
    assert result == []
    # the await step is charged before pulling, so no element is consumed
    assert advanced == []


@sync
async def test_quota_items_counted_once_per_source():
    """Each source element counts once regardless of pipeline depth"""

    async def pipeline():
        return a.filter(
            lambda x: x % 2 == 0,
            a.map(lambda x: x + 1, a.chain(range(3), range(3))),
        )

    async with a.Quota(items=5, on_exhausted="stop") as quota:
        result = await a.list(await pipeline())
    # pulled 0,1,2 (first source) then 0,1 (second source): +1 gives odds/evens
    assert result == [2, 2]
    assert quota.consumed["items"] == 5


# ---------------------------------------------------------------------------
# awaits budget
# ---------------------------------------------------------------------------


@sync
async def test_quota_awaits_pulls():
    async with a.Quota(awaits=3, on_exhausted="stop") as quota:
        result = await a.list(asyncify(range(10)))
    assert result == [0, 1, 2]
    assert quota.consumed["awaits"] == 3


@sync
async def test_quota_awaits_async_callable():
    """Async callables invoked by combinators consume await steps"""

    async def double(value):
        return value * 2

    async with a.Quota(awaits=5, on_exhausted="stop") as quota:
        result = await a.list(a.map(double, asyncify(range(10))))
    # two await steps per output (pull + invocation), stop mid way
    assert result == [0, 2]
    assert quota.consumed["awaits"] == 5


@sync
async def test_quota_awaits_sync_callable_unaffected():
    """Synchronous callables do not add await steps"""
    async with a.Quota(awaits=3, on_exhausted="stop") as quota:
        result = await a.list(a.map(lambda x: x * 2, asyncify(range(10))))
    assert result == [0, 2, 4]
    assert quota.consumed["awaits"] == 3


# ---------------------------------------------------------------------------
# weight budget
# ---------------------------------------------------------------------------


@sync
async def test_quota_weight_sync():
    async with a.Quota(weight=3.0, weight_of=lambda x: x, on_exhausted="stop") as quota:
        result = await a.list(asyncify(range(10)))
    assert result == [0, 1, 2]
    assert quota.consumed["weight"] == 3.0


@sync
async def test_quota_weight_async():
    async def weight_of(value):
        return value

    async with a.Quota(weight=3.0, weight_of=weight_of, on_exhausted="stop") as quota:
        result = await a.list(asyncify(range(10)))
    assert result == [0, 1, 2]
    assert quota.consumed["weight"] == 3.0


@sync
async def test_quota_weight_default_per_item():
    async with a.Quota(weight=2.0, on_exhausted="stop") as quota:
        result = await a.list(asyncify(range(10)))
    assert result == [0, 1]
    assert quota.consumed["weight"] == 2.0


@sync
async def test_quota_weight_raise():
    with pytest.raises(a.QuotaExceeded) as exc_info:
        async with a.Quota(weight=0.5, weight_of=lambda x: 1.0):
            await a.list(asyncify(range(3)))
    assert exc_info.value.budget == "weight"
    assert exc_info.value.used == 0.0
    assert exc_info.value.requested == 1.0


# ---------------------------------------------------------------------------
# buffered budget
# ---------------------------------------------------------------------------


@sync
async def test_quota_batched_buffered():
    async with a.Quota(buffered=3, on_exhausted="stop") as quota:
        result = await a.list(a.batched(range(8), 3))
    assert result == [(0, 1, 2), (3, 4, 5), (6, 7)]
    assert quota.held == 0


@sync
async def test_quota_batched_buffered_insufficient():
    async with a.Quota(buffered=2, on_exhausted="stop") as quota:
        result = await a.list(a.batched(range(8), 3))
    # no complete (or partial) batch fits the buffer, so nothing is emitted
    assert result == []
    assert quota.held == 0
    assert quota.exhausted_budget == "buffered"


@sync
async def test_quota_cycle_buffered():
    source = Closeable(range(5))
    async with a.Quota(buffered=3, on_exhausted="stop") as quota:
        result = []
        async with a.closing(a.cycle(source)) as cycle:
            async for item in cycle:
                result.append(item)
                if len(result) == 3:
                    break
    assert result == [0, 1, 2]
    assert source.closed
    assert quota.held == 0


@sync
async def test_quota_tee_buffered():
    async with a.Quota(buffered=2, on_exhausted="stop") as quota:
        teed = a.tee(range(10), n=2)
        first = await a.list(teed[0])
        await teed.aclose()
    assert first == [0, 1]
    assert quota.held == 0


@sync
async def test_quota_tee_peers_release_slots():
    async with a.Quota(buffered=4, on_exhausted="stop") as quota:
        teed = a.tee(range(10), n=2)
        assert await a.anext(teed[0]) == 0
        assert quota.held == 1
        assert await a.anext(teed[1]) == 0
        # both peers consumed the node, its buffered slot is freed
        assert quota.held == 0
        await teed.aclose()


@sync
async def test_quota_pairwise_buffered():
    async with a.Quota(buffered=1, on_exhausted="stop") as quota:
        result = await a.list(a.pairwise(range(5)))
    assert result == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert quota.held == 0


@sync
async def test_quota_pairwise_buffered_zero():
    async with a.Quota(buffered=0, on_exhausted="stop"):
        result = await a.list(a.pairwise(range(5)))
    assert result == []


@sync
async def test_quota_merge_buffered():
    async with a.Quota(buffered=2, on_exhausted="stop") as quota:
        result = await a.list(a.merge(asyncify([1, 3, 5]), asyncify([2, 4, 6])))
    assert result == [1, 2, 3, 4, 5, 6]
    assert quota.held == 0


@sync
async def test_quota_merge_buffered_closes_sources():
    first = Closeable([1, 3, 5])
    second = Closeable([2, 4, 6])
    async with a.Quota(buffered=1, on_exhausted="stop") as quota:
        result = await a.list(a.merge(first, second))
    assert result == []
    # both sources were opened; both must be closed on exhaustion
    assert first.closed
    assert second.closed
    assert quota.held == 0


@sync
async def test_quota_nlargest_buffered():
    async with a.Quota(buffered=3, on_exhausted="stop") as quota:
        result = await a.nlargest(range(6), 3)
    assert result == [5, 4, 3]
    assert quota.held == 0


@sync
async def test_quota_nlargest_buffered_partial():
    async with a.Quota(buffered=2, on_exhausted="stop"):
        result = await a.nlargest(range(6), 3)
    # only two items ever fit the result buffer
    assert result == [1, 0]


@sync
async def test_quota_groupby_buffered():
    async with a.Quota(buffered=1, on_exhausted="stop"):
        groups = []
        async for key, group in a.groupby(asyncify([0, 0, 1, 1, 2])):
            groups.append((key, await a.list(group)))
    assert groups == [(0, [0, 0]), (1, [1, 1]), (2, [2])]


@sync
async def test_quota_batched_buffered_raise():
    with pytest.raises(a.QuotaExceeded) as exc_info:
        async with a.Quota(buffered=1):
            await a.list(a.batched(range(5), 2))
    assert exc_info.value.budget == "buffered"


# ---------------------------------------------------------------------------
# exhaustion and iterator cleanup
# ---------------------------------------------------------------------------


@sync
async def test_quota_raise_closes_all_pipeline_iterators():
    first = Closeable(range(10))
    second = Closeable(range(10))
    with pytest.raises(a.QuotaExceeded):
        async with a.Quota(items=4):
            pipeline = a.chain(
                a.map(lambda x: x + 1, first),
                a.filter(lambda x: True, second),
            )
            await a.list(pipeline)
    assert first.closed
    # the second source is only opened lazily once the first is exhausted;
    # here exhaustion happens within the first source
    assert not second.closed


@sync
async def test_quota_raise_through_second_chain_source():
    first = Closeable([1, 2])
    second = Closeable([3, 4, 5, 6])
    with pytest.raises(a.QuotaExceeded):
        async with a.Quota(items=3):
            await a.list(a.chain(first, second))
    assert first.closed
    assert second.closed


@sync
async def test_quota_stop_aclose_releases_holds():
    async with a.Quota(buffered=2, on_exhausted="stop") as quota:
        batched = a.batched(range(10), 3)
        await a.list(batched)
    assert quota.held == 0


# ---------------------------------------------------------------------------
# shared budgets across combinators
# ---------------------------------------------------------------------------


@sync
async def test_quota_shared_across_combinators():
    """Two combinators iterated in one quota share the same item budget"""
    async with a.Quota(items=5, on_exhausted="stop") as quota:
        first = await a.list(asyncify(range(3)))
        second = await a.list(asyncify(range(10)))
    assert first == [0, 1, 2]
    assert second == [0, 1]
    assert quota.consumed["items"] == 5


# ---------------------------------------------------------------------------
# nested quotas: isolation and sharing
# ---------------------------------------------------------------------------


@sync
async def test_nested_quotas_are_isolated():
    outer = a.Quota(items=2, on_exhausted="stop")
    inner = a.Quota(items=100, on_exhausted="stop")
    async with outer:
        async with inner:
            inner_result = await a.list(asyncify(range(10)))
        # the outer quota was completely suspended while inner was active
        assert outer.consumed["items"] == 0
        outer_result = await a.list(asyncify(range(10)))
    assert inner_result == [*range(10)]
    assert outer_result == [0, 1]
    assert outer.consumed["items"] == 2
    assert inner.consumed["items"] == 10


@sync
async def test_nested_same_quota_is_shared():
    shared = a.Quota(items=3, on_exhausted="stop")
    async with shared:
        first = await a.list(asyncify(range(10)))
        async with shared:
            second = await a.list(asyncify(range(10)))
    assert first == [0, 1, 2]
    assert second == []
    assert shared.consumed["items"] == 3


@sync
async def test_quota_restored_after_context_exit():
    quota = a.Quota(items=1, on_exhausted="stop")
    async with quota:
        assert await a.list(asyncify(range(5))) == [0]
    # after exit there is no active quota anymore
    assert await a.list(asyncify(range(5))) == [0, 1, 2, 3, 4]


@sync
async def test_nested_isolated_raise():
    outer = a.Quota(items=10)
    inner = a.Quota(items=1)
    with pytest.raises(a.QuotaExceeded) as exc_info:
        async with outer:
            async with inner:
                await a.list(asyncify(range(5)))
    # the failure belongs to the inner quota; the outer is untouched
    assert exc_info.value.quota is inner
    assert inner.exhausted is True
    assert outer.exhausted is False
    assert outer.consumed["items"] == 0


@sync
async def test_quota_named_error():
    with pytest.raises(a.QuotaExceeded, match="requests"):
        async with a.Quota(items=0, name="requests"):
            await a.list(asyncify(range(3)))


@sync
async def test_quota_weight_counted_once_per_source():
    async with a.Quota(
        weight=4.0, weight_of=lambda x: 1.0, on_exhausted="stop"
    ) as quota:
        result = await a.list(
            a.map(lambda x: x, a.filter(lambda x: True, asyncify(range(10))))
        )
    assert result == [0, 1, 2, 3]
    assert quota.consumed["weight"] == 4.0


@sync
async def test_quota_tee_close_early_releases():
    async with a.Quota(buffered=10, on_exhausted="stop") as quota:
        teed = a.tee(range(10), n=2)
        assert await a.anext(teed[0]) == 0
        assert quota.held == 1
        # peer 1 still needs the node, so its slot stays reserved ...
        await teed[0].aclose()
        assert quota.held == 1
        # ... until the last peer releases it as well
        await teed[1].aclose()
        assert quota.held == 0


@sync
async def test_quota_cycle_buffered_raise_closes():
    source = Closeable(range(5))
    with pytest.raises(a.QuotaExceeded) as exc_info:
        async with a.Quota(buffered=2):
            await a.list(a.cycle(source))
    assert exc_info.value.budget == "buffered"
    assert source.closed


@sync
async def test_quota_await_each():
    async def value(v):
        return v

    async with a.Quota(items=2, awaits=2, on_exhausted="stop") as quota:
        result = await a.list(a.await_each([value(0), value(1), value(2)]))
    assert result == [0, 1]
    assert quota.consumed["items"] == 2
    assert quota.consumed["awaits"] == 2


@sync
async def test_quota_any_iter():
    async def source():
        for value in range(5):
            yield value

    async with a.Quota(items=3, on_exhausted="stop"):
        result = await a.list(a.any_iter(source()))
    assert result == [0, 1, 2]


@sync
async def test_quota_scoped_iter():
    source = Closeable(range(5))
    async with a.Quota(items=2, on_exhausted="stop"):
        async with a.scoped_iter(source) as iterator:
            result = await a.list(iterator)
    assert result == [0, 1]
    assert source.closed


@sync
async def test_quota_reuse_after_exit_continues_budgets():
    quota = a.Quota(items=3, on_exhausted="stop")
    async with quota:
        assert await a.list(asyncify(range(10))) == [0, 1, 2]
    # re-entering a spent quota shares the (exhausted) budgets
    async with quota:
        assert quota.exhausted is True
        assert await a.list(asyncify(range(10))) == []


@sync
async def test_quota_budgets_independent():
    async with a.Quota(items=1, awaits=100, on_exhausted="stop") as quota:
        result = await a.list(asyncify(range(10)))
    assert result == [0]
    assert quota.exhausted_budget == "items"


# ---------------------------------------------------------------------------
# manual consumption
# ---------------------------------------------------------------------------


@sync
async def test_quota_manual_consume():
    quota = a.Quota(items=2)
    async with quota:
        quota.consume(items=1)
        quota.consume(items=1, awaits=4)
        with pytest.raises(a.QuotaExceeded) as exc_info:
            quota.consume(items=1)
        assert exc_info.value.budget == "items"
    buffered_quota = a.Quota(buffered=1)
    async with buffered_quota:
        token = buffered_quota.hold(1)
        assert buffered_quota.held == 1
        with pytest.raises(a.QuotaExceeded):
            buffered_quota.hold(1)
        token.release()
        assert buffered_quota.held == 0
    # a failed hold exhausts the quota; use a fresh quota for further holds
    context_quota = a.Quota(buffered=1)
    async with context_quota:
        with context_quota.hold(1):
            assert context_quota.held == 1
        assert context_quota.held == 0


@sync
async def test_quota_manual_consume_always_raises_in_stop_mode():
    # manual consumption raises even on a quota configured to stop
    quota = a.Quota(items=1, on_exhausted="stop")
    async with quota:
        quota.consume(items=1)
        with pytest.raises(a.QuotaExceeded):
            quota.consume(items=1)


# ---------------------------------------------------------------------------
# broad combinator coverage under stop mode
# ---------------------------------------------------------------------------


@sync
async def test_combinators_partial_results():
    async def async_is_even(value):
        return value % 2 == 0

    async with a.Quota(items=5, on_exhausted="stop") as quota:
        assert await a.list(a.takewhile(lambda x: x < 100, asyncify(range(10)))) == [
            0,
            1,
            2,
            3,
            4,
        ]
    assert quota.exhausted_budget == "items"

    async with a.Quota(items=4, on_exhausted="stop"):
        assert await a.list(a.islice(asyncify(range(10)), 10)) == [0, 1, 2, 3]

    async with a.Quota(items=3, on_exhausted="stop"):
        assert await a.list(
            a.starmap(lambda x, y: x + y, asyncify([(1, 1), (2, 2), (3, 3)]))
        ) == [2, 4, 6]

    async with a.Quota(items=3, on_exhausted="stop"):
        assert await a.list(a.filterfalse(lambda x: x % 2, asyncify(range(10)))) == [
            0,
            2,
        ]

    # each output of a multi-source combinator pulls one element per source
    async with a.Quota(items=6, on_exhausted="stop"):
        assert await a.list(
            a.compress(asyncify(range(10)), asyncify([1, 1, 1, 0, 0]))
        ) == [0, 1, 2]

    async with a.Quota(items=3, on_exhausted="stop"):
        assert await a.list(
            a.accumulate(asyncify([1, 2, 3, 4]), lambda x, y: x + y)
        ) == [1, 3, 6]

    async with a.Quota(items=6, on_exhausted="stop"):
        assert await a.list(a.zip(asyncify(range(10)), asyncify(range(10)))) == [
            (0, 0),
            (1, 1),
            (2, 2),
        ]

    async with a.Quota(items=3, on_exhausted="stop"):
        assert await a.list(a.enumerate(asyncify(range(10)))) == [
            (0, 0),
            (1, 1),
            (2, 2),
        ]

    async with a.Quota(items=6, on_exhausted="stop"):
        assert await a.list(a.zip_longest(asyncify([1, 2, 3]), asyncify([1]))) == [
            (1, 1),
            (2, None),
            (3, None),
        ]


@sync
async def test_consumers_partial_results():
    async with a.Quota(items=3, on_exhausted="stop"):
        assert await a.sum(asyncify(range(10))) == 3
    async with a.Quota(items=3, on_exhausted="stop"):
        assert await a.min(asyncify([2, 5, 1, 0])) == 1
    async with a.Quota(items=3, on_exhausted="stop"):
        assert await a.max(asyncify([2, 5, 1, 0])) == 5
    async with a.Quota(items=3, on_exhausted="stop"):
        assert await a.sorted(asyncify([3, 1, 2, 0])) == [1, 2, 3]
    async with a.Quota(items=4, on_exhausted="stop"):
        assert await a.reduce(lambda x, y: x + y, asyncify(range(10))) == 6
    async with a.Quota(items=3, on_exhausted="stop"):
        assert await a.list(asyncify(range(10))) == [0, 1, 2]
    async with a.Quota(items=3, on_exhausted="stop"):
        assert await a.tuple(asyncify(range(10))) == (0, 1, 2)
    async with a.Quota(items=4, on_exhausted="stop"):
        assert await a.set(asyncify([0, 0, 1, 2])) == {0, 1, 2}
    async with a.Quota(items=3, on_exhausted="stop"):
        assert await a.dict(asyncify([("a", 1), ("b", 2), ("c", 3)])) == {
            "a": 1,
            "b": 2,
            "c": 3,
        }


@sync
async def test_async_predicates_partial_results():
    async def is_even(value):
        return value % 2 == 0

    async with a.Quota(items=4, on_exhausted="stop"):
        assert await a.list(a.filter(is_even, asyncify(range(10)))) == [0, 2]
    async with a.Quota(items=4, on_exhausted="stop"):
        assert await a.list(a.dropwhile(is_even, asyncify(range(10)))) == [
            1,
            2,
            3,
        ]
    async with a.Quota(items=4, on_exhausted="stop"):
        assert await a.list(a.takewhile(is_even, asyncify([0, 2, 1, 3]))) == [
            0,
            2,
        ]
