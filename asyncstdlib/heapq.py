from __future__ import annotations
from typing import (
    Generic,
    AsyncIterator,
    Optional,
    Callable,
    Any,
    overload,
    Awaitable,
)
import heapq as _heapq

from .builtins import enumerate as a_enumerate, zip as a_zip
from ._core import aiter, awaitify, ScopedIter, borrow
from ._typing import AnyIterable, ACloseable, LT, T
from .quota import (
    HoldToken,
    _hold,
    _release,
    _QuotaStop,
    owned as _owned,
    quota_guard,
    quota_boundary,
)


class _KeyIter(Generic[LT]):
    __slots__ = ("head", "tail", "reverse", "head_key", "key", "hold")

    @overload
    def __init__(
        self,
        head: T,
        tail: AsyncIterator[T],
        reverse: bool,
        head_key: LT,
        key: Callable[[T], Awaitable[LT]],
        hold: Optional[HoldToken] = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        head: LT,
        tail: AsyncIterator[LT],
        reverse: bool,
        head_key: LT,
        key: None,
        hold: Optional[HoldToken] = None,
    ) -> None:
        ...

    def __init__(
        self,
        head: Any,
        tail: AsyncIterator[Any],
        reverse: bool,
        head_key: LT,
        key: Any,
        hold: Optional[HoldToken] = None,
    ) -> None:
        self.head = head
        self.head_key = head_key
        self.tail = tail
        self.key = key
        self.reverse = reverse
        # quota slot held for the head item currently buffered by the merge
        self.hold = hold

    @overload
    @classmethod
    def from_iters(
        cls,
        iterables: "tuple[AnyIterable[T], ...]",
        reverse: bool,
        key: Callable[[T], Awaitable[LT]],
    ) -> "AsyncIterator[_KeyIter[LT]]":
        ...

    @overload
    @classmethod
    def from_iters(
        cls, iterables: "tuple[AnyIterable[LT], ...]", reverse: bool, key: None
    ) -> "AsyncIterator[_KeyIter[LT]]":
        ...

    @classmethod
    @_owned
    async def from_iters(
        cls,
        iterables: "tuple[AnyIterable[Any], ...]",
        reverse: bool,
        key: Optional[Callable[[Any], Any]],
    ) -> "AsyncIterator[_KeyIter[Any]]":
        for iterable in iterables:
            iterator = aiter(iterable)
            try:
                head = await iterator.__anext__()
            except StopAsyncIteration:
                pass
            except Exception:
                # the source was opened but never handed to the merge;
                # close it so quota failure cannot leak the iterator
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
                raise
            else:
                head_key = await key(head) if key is not None else head
                try:
                    # the head item is buffered until the merge yields it
                    hold = _hold()
                except Exception:
                    aclose = getattr(iterator, "aclose", None)
                    if aclose is not None:
                        await aclose()
                    raise
                yield cls(head, iterator, reverse, head_key, key, hold)

    async def pull_head(self) -> bool:
        """
        Pull the next ``head`` element from the iterator and signal success
        """
        try:
            self.head = head = await self.tail.__anext__()
        except StopAsyncIteration:
            # the last head was yielded already; free its buffered slot
            _release(self.hold)
            self.hold = None
            return False
        else:
            self.head_key = await self.key(head) if self.key is not None else head
            # the previous head was yielded; its slot backs the new head
            _release(self.hold)
            self.hold = _hold()
            return True

    def __lt__(self, other: _KeyIter[LT]) -> bool:
        return self.reverse ^ bool(self.head_key < other.head_key)

    def __eq__(self, other: _KeyIter[LT]) -> bool:  # type: ignore[override]
        return not (self.head_key < other.head_key or other.head_key < self.head_key)


@quota_boundary
@_owned
async def merge(
    *iterables: AnyIterable[Any],
    key: Optional[Callable[[Any], Any]] = None,
    reverse: bool = False,
) -> AsyncIterator[Any]:
    """
    Merge all pre-sorted (async) ``iterables`` into a single sorted iterator

    This works similar to ``sorted(chain(*iterables), key=key, reverse=reverse)`` but
    operates lazily: at any moment only one item of each iterable is stored for the
    comparison. This allows merging streams of pre-sorted items, such as timestamped
    records from multiple sources.

    The optional ``key`` argument specifies a one-argument (async) callable, which
    provides a substitute for determining the sort order of each item.
    The special value and default :py:data:`None` represents the identity function,
    comparing items directly.

    The default sort order is ascending, that is items with ``a < b`` imply ``a``
    is yielded before ``b``. Use ``reverse=True`` for descending sort order.
    The ``iterables`` must be pre-sorted in the same order.
    """
    a_key = awaitify(key) if key is not None else None
    # sortable iterators with (reverse) position to ensure stable sort for ties
    iter_heap: "list[tuple[_KeyIter[Any], int]]" = []
    try:
        async with quota_guard:
            # collect incrementally so quota failure closes every opened source
            async for idx, itr in a_enumerate(
                _KeyIter[Any].from_iters(iterables, reverse, a_key)
            ):
                iter_heap.append((itr, idx if not reverse else -idx))
            _heapq.heapify(iter_heap)
            # there are at least two iterators that need merging
            while len(iter_heap) > 1:
                while True:
                    itr, idx = iter_heap[0]
                    yield itr.head
                    if await itr.pull_head():
                        _heapq.heapreplace(iter_heap, (itr, idx))
                    else:
                        _heapq.heappop(iter_heap)
                        break
            # there is only one iterator left, no need for merging
            if iter_heap:
                itr, idx = iter_heap[0]
                yield itr.head
                async for item in itr.tail:
                    yield item
    finally:
        for itr, _ in iter_heap:
            _release(itr.hold)
            if isinstance(itr.tail, ACloseable):
                await itr.tail.aclose()


class ReverseLT(Generic[LT]):
    """Helper to reverse ``a < b`` ordering"""

    __slots__ = ("key",)

    def __init__(self, key: LT):
        self.key = key

    def __lt__(self, other: ReverseLT[LT]) -> bool:
        return bool(other.key < self.key)


# Python's heapq provides a *min*-heap
# When finding the n largest items, heapq tracks the *minimum* item still large enough.
# In other words, during search we maintain opposite sort order than what is requested.
# We turn the min-heap into a max-sort in the end.
async def _largest(
    iterable: AnyIterable[T],
    n: int,
    key: Callable[[T], Awaitable[LT]],
    reverse: bool,
) -> "list[T]":
    ordered: Callable[[LT], LT] = ReverseLT if reverse else lambda x: x  # type: ignore
    async with ScopedIter(iterable) as iterator:
        # assign an ordering to items to solve ties
        order_sign = -1 if reverse else 1
        # the heap retains up to ``n`` items, each holding a buffered slot
        n_heap: "list[tuple[LT, int, T, Optional[HoldToken]]]" = []
        try:
            async for index, item in a_zip(range(n), borrow(iterator)):
                n_heap.append(
                    (
                        ordered(await key(item)),
                        index * order_sign,
                        item,
                        _hold(),
                    )
                )
            if not n_heap:
                return []
            _heapq.heapify(n_heap)
            worst_key = n_heap[0][0]
            next_index = n * order_sign
            async for item in iterator:
                item_key = ordered(await key(item))
                if worst_key < item_key:
                    # free the evicted slot before reserving the replacement
                    _release(n_heap[0][3])
                    _heapq.heapreplace(n_heap, (item_key, next_index, item, _hold()))
                    worst_key = n_heap[0][0]
                    next_index += 1 * order_sign
        except _QuotaStop:
            # a stopping quota: return the best partial heap collected so far
            pass
        n_heap.sort(reverse=True)
        result = [item for _, _, item, _ in n_heap]
        for _, _, _, hold in n_heap:
            _release(hold)
    return result


async def _identity(x: T) -> T:
    return x


async def nlargest(
    iterable: AnyIterable[T],
    n: int,
    key: Optional[Callable[[Any], Awaitable[Any]]] = None,
) -> "list[T]":
    """
    Return a sorted list of the ``n`` largest elements from the (async) iterable

    The optional ``key`` argument specifies a one-argument (async) callable, which
    provides a substitute for determining the sort order of each item.
    The special value and default :py:data:`None` represents the identity functions,
    comparing items directly.

    The result is equivalent to ``sorted(iterable, key=key, reverse=True)[:n]``,
    but ``iterable`` is consumed lazily and items are discarded eagerly.
    """
    a_key: Callable[[Any], Awaitable[Any]] = (
        awaitify(key) if key is not None else _identity  # type: ignore
    )
    return await _largest(iterable=iterable, n=n, key=a_key, reverse=False)


async def nsmallest(
    iterable: AnyIterable[T],
    n: int,
    key: Optional[Callable[[Any], Awaitable[Any]]] = None,
) -> "list[T]":
    """
    Return a sorted list of the ``n`` smallest elements from the (async) iterable

    Provides the reverse functionality to :py:func:`~.nlargest`.
    """
    a_key: Callable[[Any], Awaitable[Any]] = (
        awaitify(key) if key is not None else _identity  # type: ignore
    )
    return await _largest(iterable=iterable, n=n, key=a_key, reverse=True)
