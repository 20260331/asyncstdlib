"""
Tests for the opt-in async iterator ownership audit
"""
import sys
import types

import pytest

import asyncstdlib as a
from asyncstdlib import _core

from .utility import sync, asyncify


async def _range(n=5):
    for i in range(n):
        yield i


def _kinds(audit):
    return [violation.kind for violation in audit.violations]


@sync
async def test_audit_reports_no_violations_for_clean_use():
    async with a.ownership_audit() as audit:
        # the whole breadth of consuming coroutines
        assert await a.all(asyncify([1, 1]))
        assert await a.any(asyncify([0, 1]))
        assert await a.sum(asyncify([1, 2, 3])) == 6
        assert await a.min(asyncify([3, 1, 2])) == 1
        assert await a.max(asyncify([1, 3, 2])) == 3
        assert await a.list(asyncify([0, 1])) == [0, 1]
        assert await a.tuple(asyncify([0, 1])) == (0, 1)
        assert await a.sorted(asyncify([2, 0, 1])) == [0, 1, 2]
        # iterator producing combinators
        assert await a.list(a.map(lambda x: 2 * x, asyncify([0, 1]))) == [
            0,
            2,
        ]
        assert await a.list(a.zip(asyncify([0, 1]), [2, 3], strict=False)) == [
            (0, 2),
            (1, 3),
        ]
        assert await a.list(a.zip(asyncify([0, 1]), [2, 3], strict=True)) == [
            (0, 2),
            (1, 3),
        ]
        assert await a.list(a.filter(None, asyncify([0, 1, 2]))) == [1, 2]
        assert await a.list(a.enumerate(asyncify(["a", "b"]))) == [
            (0, "a"),
            (1, "b"),
        ]
        # itertools combinators
        assert await a.list(a.islice(asyncify(range(10)), 2, 8, 2)) == [2, 4, 6]
        assert await a.list(a.takewhile(lambda x: x < 3, asyncify(range(5)))) == [
            0,
            1,
            2,
        ]
        assert await a.list(a.dropwhile(lambda x: x < 3, asyncify(range(5)))) == [
            3,
            4,
        ]
        assert await a.list(a.filterfalse(None, asyncify([0, 1, 0, 2]))) == [0, 0]
        assert await a.list(a.starmap(lambda x, y: x + y, [("a", "b")])) == ["ab"]
        assert await a.list(a.accumulate(asyncify([1, 2, 3]))) == [1, 3, 6]
        assert await a.list(a.batched(asyncify(range(5)), 2)) == [
            (0, 1),
            (2, 3),
            (4,),
        ]
        assert await a.list(a.pairwise(asyncify([0, 1, 2]))) == [(0, 1), (1, 2)]
        assert await a.list(a.compress(asyncify(range(4)), [1, 0, 1, 0])) == [0, 2]
        assert await a.list(a.islice(a.cycle(asyncify([1])), 3)) == [1, 1, 1]
        assert await a.list(a.chain(asyncify([0, 1]), asyncify([2, 3]))) == [
            0,
            1,
            2,
            3,
        ]
        assert await a.list(a.chain.from_iterable([asyncify([0]), asyncify([1])])) == [
            0,
            1,
        ]
        assert await a.list(a.zip_longest(asyncify([0]), asyncify([1, 2]))) == [
            (0, 1),
            (None, 2),
        ]
        # functools/heapq combinators
        assert await a.reduce(lambda x, y: x + y, asyncify([1, 2, 3]), 0) == 6
        assert await a.list(a.merge(asyncify([0, 2]), asyncify([1, 3]))) == [
            0,
            1,
            2,
            3,
        ]
        assert await a.nlargest(asyncify(range(5)), 2) == [4, 3]
        assert await a.nsmallest(asyncify(range(5)), 2) == [0, 1]
    assert not audit.violations


@sync
async def test_audit_reports_no_violations_for_groupby():
    async with a.ownership_audit() as audit:
        groups = a.groupby(asyncify([1, 1, 2]))
        async for key, group in groups:
            await a.list(group)
        # a second groupby closed explicitly before exhaustion
        grouper = a.groupby(asyncify([1, 2]))
        await grouper.__anext__()
        await grouper.aclose()
    assert not audit.violations


@sync
async def test_audit_reports_no_violations_for_tee():
    async with a.ownership_audit() as audit:
        teed = a.tee(asyncify(range(4)), n=3)
        for peer in teed:
            assert await a.list(peer) == [0, 1, 2, 3]
        await teed.aclose()
        # a tee built on another tee shares its owner token
        parent = a.tee(asyncify(range(3)), n=2)
        child = a.tee(parent[0], n=2)
        await a.list(child[0])
        await a.list(child[1])
        await child.aclose()
        await a.list(parent[1])
        await parent.aclose()
    assert not audit.violations


@sync
async def test_audit_reports_no_violations_for_scoped_borrow():
    async with a.ownership_audit() as audit:
        source = _range(5)
        async with a.scoped_iter(source) as scoped:
            assert await a.list(a.islice(scoped, 2)) == [0, 1]
            assert await a.list(scoped) == [2, 3, 4]
        # nested scopes of borrowed handles and explicit borrows are safe
        source2 = _range(4)
        async with a.scoped_iter(source2) as scoped:
            async with a.scoped_iter(scoped) as inner:
                assert await a.list(a.islice(inner, 2)) == [0, 1]
            borrowed = a.borrow(scoped)
            assert await a.list(a.islice(borrowed, 1)) == [2]
            await borrowed.aclose()
    assert not audit.violations


@sync
async def test_audit_reports_no_violations_for_early_close():
    async with a.ownership_audit() as audit:
        async for item in a.chain(asyncify(range(3)), asyncify(range(3))):
            if item == 1:
                break
        merger = a.merge(asyncify(range(100)), asyncify(range(100)))
        await merger.__anext__()
        await merger.aclose()
        zipper = a.zip_longest(asyncify(range(100)), asyncify(range(100)))
        await zipper.__anext__()
        await zipper.aclose()
        chainer = a.chain(asyncify(range(100)), asyncify(range(100)))
        await chainer.__anext__()
        await chainer.aclose()
        async with a.closing(_range(3)) as iterator:
            await iterator.__anext__()
    assert not audit.violations


@sync
async def test_audit_reports_no_violations_under_id_reuse():
    # many short-lived generators reuse Python object ids once collected;
    # the audit must not confuse a new iterator with a destroyed one
    async with a.ownership_audit() as audit:
        for _ in range(50):
            await a.list(a.islice(asyncify([1, 2]), 1))
    assert not audit.violations


@sync
async def test_unreturned_borrow_is_reported():
    async with a.ownership_audit(raise_on_exit=False) as audit:
        source = _range(3)
        borrowed = a.borrow(source)
        await a.list(borrowed)
        # the borrowed handle is never ``aclose``d
    assert _kinds(audit) == [a.Violation.UNRETURNED_BORROW]
    violation = audit.violations[0]
    assert violation.object_type == "async_generator"
    assert "not returned" in violation.message


@sync
async def test_unreturned_borrow_raises_by_default():
    with pytest.raises(a.OwnershipViolation) as exc_info:
        async with a.ownership_audit() as audit:
            source = _range(2)
            a.borrow(source)
    assert audit.violations
    assert "not returned" in str(exc_info.value)


@sync
async def test_returned_borrow_is_not_reported():
    async with a.ownership_audit(raise_on_exit=False) as audit:
        source = _range(3)
        borrowed = a.borrow(source)
        await a.list(borrowed)
        await borrowed.aclose()
        await source.aclose()
    assert not audit.violations


@sync
async def test_reusing_a_closed_iterator_is_duplicate_close():
    async with a.ownership_audit(raise_on_exit=False) as audit:
        source = _range(5)
        assert await a.list(a.islice(source, 2)) == [0, 1]
        # the first islice already closed ``source``
        assert await a.list(a.islice(source, 2)) == []
    kinds = _kinds(audit)
    assert a.Violation.REUSED_AFTER_CLOSE in kinds
    assert a.Violation.DUPLICATE_CLOSE in kinds


@sync
async def test_double_raw_scope_reports_borrow_violations():
    async with a.ownership_audit(raise_on_exit=False) as audit:
        source = _range(3)
        async with a.scoped_iter(source) as outer:
            # scoping the raw iterator (instead of the borrowed handle)
            # moves ownership while the outer loan is outstanding
            async with a.scoped_iter(source):
                pass
            await a.list(outer)
    kinds = _kinds(audit)
    assert a.Violation.TRANSFER_WHILE_BORROWED in kinds
    assert a.Violation.CLOSE_WHILE_BORROWED in kinds


@sync
async def test_wrong_owner_close_is_reported():
    async with a.ownership_audit(raise_on_exit=False) as audit:
        iterator = _range(2)
        owner_a = _core.Sentinel("owner-a")
        owner_b = _core.Sentinel("owner-b")
        _core._audit_claim(iterator, owner_a)
        _core._audit_close(iterator, owner_b)
    assert _kinds(audit) == [a.Violation.WRONG_OWNER_CLOSE]
    assert "owner-b" in audit.violations[0].message
    assert "owner-a" in audit.violations[0].message


@sync
async def test_closing_unclaimed_iterator_is_ignored():
    iterator = _range(1)
    async with a.ownership_audit(raise_on_exit=False) as audit:
        # an object the wrapper layer never claimed, e.g. an internal
        # helper generator, must not produce violations on close
        _core._audit_close(iterator, _core.Sentinel("someone"))
    assert not audit.violations
    await iterator.aclose()


@sync
async def test_non_owner_reclose_is_wrong_owner_close():
    iterator = _range(2)
    owner_a = _core.Sentinel("owner-a")
    owner_b = _core.Sentinel("owner-b")
    async with a.ownership_audit(raise_on_exit=False) as audit:
        _core._audit_claim(iterator, owner_a)
        _core._audit_close(iterator, owner_a)
        # the iterator is already closed; an unrelated owner closing it
        # cannot be an idempotent cleanup and is a wrong-owner close
        _core._audit_close(iterator, owner_b)
    assert _kinds(audit) == [a.Violation.WRONG_OWNER_CLOSE]


@sync
async def test_previous_owner_reclose_is_duplicate_close():
    iterator = _range(3)
    first = _core.Sentinel("first-owner")
    second = _core.Sentinel("second-owner")
    async with a.ownership_audit(raise_on_exit=False) as audit:
        _core._audit_claim(iterator, first)
        _core._audit_close(iterator, first)
        # a second ownership episode on the same live iterator
        _core._audit_claim(iterator, second)
        _core._audit_close(iterator, second)
        # once the second episode is closed, the previous owner closing
        # again is a duplicate close, not a wrong-owner close
        _core._audit_close(iterator, first)
    kinds = _kinds(audit)
    assert a.Violation.REUSED_AFTER_CLOSE in kinds
    assert kinds.count(a.Violation.DUPLICATE_CLOSE) == 2
    assert a.Violation.WRONG_OWNER_CLOSE not in kinds


@sync
async def test_non_weakrefable_closed_record_is_recycled_silently():
    # objects that cannot be weakly referenced cannot be told apart once
    # closed; recycling their id for a fresh episode must not raise false
    # positives, only the still-open episode is audited
    async with a.ownership_audit(raise_on_exit=False) as audit:
        iterator = _CustomAsyncIterator()
        owner_a = _core.Sentinel("owner-a")
        _core._audit_claim(iterator, owner_a)
        _core._audit_close(iterator, owner_a)
        _core._audit_claim(iterator, owner_a)
        _core._audit_close(iterator, owner_a)
    assert not audit.violations


@sync
async def test_original_exception_is_not_delayed():
    audit = None
    with pytest.raises(ValueError, match="boom"):
        async with a.ownership_audit() as audit:
            source = _range(2)
            a.borrow(source)
            raise ValueError("boom")
    assert audit is not None
    assert _kinds(audit) == [a.Violation.UNRETURNED_BORROW]


@sync
async def test_original_exception_carries_note_on_modern_python():
    try:
        async with a.ownership_audit():
            source = _range(2)
            a.borrow(source)
            raise RuntimeError("kaboom")
    except RuntimeError as exc:
        if sys.version_info >= (3, 11):
            assert any("ownership audit" in note for note in exc.__notes__)
    else:  # pragma: no cover
        raise AssertionError("original exception was swallowed")


@sync
async def test_raise_on_exit_false_only_records():
    async with a.ownership_audit(raise_on_exit=False) as audit:
        source = _range(2)
        a.borrow(source)
    assert audit.violations


class _CustomAsyncIterator:
    """User async iterator that is not weak referenceable and counts closes"""

    __slots__ = ("_count", "_closed")

    def __init__(self):
        self._count = 0
        self._closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._count >= 3:
            raise StopAsyncIteration
        self._count += 1
        return self._count

    async def aclose(self):
        self._closed += 1


@sync
async def test_audit_does_not_monkey_patch_user_iterators():
    custom = _CustomAsyncIterator()
    wrapped_aiter = type(custom).__aiter__
    wrapped_anext = type(custom).__anext__
    wrapped_aclose = type(custom).aclose
    async with a.ownership_audit() as audit:
        assert await a.list(a.islice(custom, 3)) == [1, 2, 3]
    assert not audit.violations
    # the user object and its protocol are completely untouched
    assert type(custom) is _CustomAsyncIterator
    assert type(custom).__aiter__ is wrapped_aiter
    assert type(custom).__anext__ is wrapped_anext
    assert type(custom).aclose is wrapped_aclose
    assert custom.__class__ is _CustomAsyncIterator
    # nested wrappers each run their own cleanup (idempotent by contract);
    # the audit counts both but attributes them to known owners
    assert custom._closed == 2
    # async generators likewise pass through unwrapped
    generator = _range(1)
    assert isinstance(generator, types.AsyncGeneratorType)
    async with a.ownership_audit() as audit:
        assert await a.list(generator) == [0]
    assert not audit.violations


@sync
async def test_inactive_audit_changes_nothing():
    # with no scope active, forgetting a borrow does not raise
    source = _range(2)
    borrowed = a.borrow(source)
    await borrowed.__anext__()
    assert _core._active_audit.get() is None
    await borrowed.aclose()
    await source.aclose()


@sync
async def test_audit_records_event_trace():
    async with a.ownership_audit(raise_on_exit=False) as audit:
        source = _range(2)
        async with a.scoped_iter(source) as scoped:
            await a.list(scoped)
    trace = "\n".join(audit.events)
    assert "claim" in trace
    assert "borrow" in trace
    assert "return" in trace
    assert "close" in trace
    # borrow happens before its matching return
    assert trace.index("borrow") < trace.index("return")


@sync
async def test_nested_audit_scopes_are_independent():
    async with a.ownership_audit(raise_on_exit=False) as outer:
        async with a.ownership_audit(raise_on_exit=False) as inner:
            source = _range(2)
            forgotten = a.borrow(source)
            await a.list(forgotten)
        assert inner.violations
        assert not outer.violations
        # after the inner scope exits, events go to the outer scope again
        assert await a.list(a.islice(_range(2), 1)) == [0]
    assert not outer.violations


@sync
async def test_scoped_iter_without_aclose_is_not_audited():
    class _Uncloseable:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    async with a.ownership_audit() as audit:
        async with a.scoped_iter(_Uncloseable()) as iterator:
            async for _ in iterator:
                pass
    assert not audit.violations
