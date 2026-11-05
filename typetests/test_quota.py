from typing import Optional, Union

import asyncstdlib as a
from asyncstdlib import quota as quota_module
from typing_extensions import assert_type


async def test_quota_context() -> None:
    quota = a.Quota(items=10, awaits=4, buffered=2, weight=2.5)
    assert_type(quota, a.Quota)
    async with quota as active:
        assert_type(active, a.Quota)
        async for item in a.map(lambda x: x, [1, 2, 3]):
            assert_type(item, int)
    assert_type(quota.exhausted, bool)
    assert_type(quota.exhausted_budget, Optional[str])
    assert_type(quota.consumed["items"], Union[int, float])
    assert_type(quota.remaining["buffered"], Optional[Union[int, float]])
    assert_type(quota.held, int)


async def test_quota_weight_of() -> None:
    def weight_of(item: int) -> float:
        return float(item)

    async with a.Quota(weight=10.0, weight_of=weight_of) as quota:
        assert_type(quota, a.Quota)


async def test_quota_stop_mode() -> None:
    async with a.Quota(items=1, on_exhausted="stop"):
        result = await a.list(a.map(lambda x: x, [1, 2, 3]))
        assert_type(result, "list[int]")


async def test_quota_exception() -> None:
    try:
        async with a.Quota(items=0):
            await a.list([1])
    except a.QuotaExceeded as error:
        assert_type(error.budget, str)
        assert_type(error.limit, Union[int, float])
        assert_type(error.used, Union[int, float])
        assert_type(error.requested, Union[int, float])
        assert_type(error.quota, a.Quota)


async def test_quota_manual() -> None:
    quota = a.Quota(items=1, buffered=1)
    async with quota:
        quota.consume(items=1)
        token = quota.hold(1)
        assert_type(token, a.HoldToken)
        token.release()
        with quota.hold(1) as context_token:
            assert_type(context_token, a.HoldToken)


def test_members() -> None:
    assert_type(quota_module.Quota, type[a.Quota])
    assert_type(quota_module.QuotaExceeded, type[a.QuotaExceeded])
    assert_type(quota_module.HoldToken, type[a.HoldToken])
