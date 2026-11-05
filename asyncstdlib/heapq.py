from __future__ import annotations

import sys
from contextlib import suppress as _suppress
from typing import (
    Generic,
    AsyncIterator,
    Optional,
    Callable,
    Any,
    Awaitable,
    List,
    Tuple,
    TYPE_CHECKING,
)
import heapq as _heapq

from .builtins import zip as a_zip
from ._core import aiter, awaitify, ScopedIter, borrow, Sentinel
from ._typing import AnyIterable, ACloseable, LT, T

if TYPE_CHECKING:
    import asyncio


class ReverseLT(Generic[LT]):
    """Helper to reverse ``a < b`` ordering"""

    __slots__ = ("key",)

    def __init__(self, key: LT):
        self.key = key

    def __lt__(self, other: ReverseLT[LT]) -> bool:
        return bool(other.key < self.key)

    def __eq__(self, other: ReverseLT[LT]) -> bool:  # type: ignore[override]
        return not (other.key < self.key or self.key < other.key)


async def merge(
    *iterables: AnyIterable[Any],
    key: Optional[Callable[[Any], Any]] = None,
    reverse: bool = False,
    prefetch: Optional[int] = None,
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

    Ties are broken *stable*: items with equal sort keys are merged in rounds,
    where every still-active source contributes at most one item per round,
    sources are visited in the order of the source ``iterables``, and items of
    each source keep their original order. This holds for both ascending and
    descending sort order.

    The optional ``prefetch`` argument enables concurrent prefetching with a
    *total* lookahead budget. With ``prefetch=None`` (the default) items are pulled
    strictly on demand, one comparison item per active source at a time,
    without scheduling background work.
    With ``prefetch=n`` for a non-negative integer, every source is advanced
    concurrently by a background task of the running :py:mod:`asyncio` event loop;
    at most ``n`` items may be prefetched *ahead* in total across all sources,
    counting both buffered items and fetches currently in flight.
    In addition, each active source always keeps one comparison "head" item,
    which is required for sorting and does not count against the budget.
    A budget of ``0`` prefetches no lookahead items but still fetches all heads
    concurrently. This is useful when sources produce at different speeds:
    slow sources cannot stall fast ones beyond what ordering requires,
    and memory usage stays bounded by ``n``.

    When the consumer stops early, when it is cancelled, or when any source raises,
    all background tasks are stopped and every source that is not already exhausted
    is :term:`closed <aclose>` exactly via its ``aclose`` method (if it has one).
    An error raised by a source or by ``key`` is propagated to the consumer,
    after all remaining sources have been closed.

    .. versionchanged:: 3.14.0
        Added the ``prefetch`` argument. Equal-key items are now taken in stable
        rounds (one per source) in source order for both ascending and descending
        sort order.
    """
    a_key = awaitify(key) if key is not None else None
    if prefetch is None:
        sources: _MergeSources = _LazyMerge(iterables, key=a_key, reverse=reverse)
    else:
        if isinstance(prefetch, bool) or not isinstance(prefetch, int):
            raise TypeError("prefetch must be an int or None")
        if prefetch < 0:
            raise ValueError("prefetch must be >= 0")
        sources = _PrefetchMerge(
            iterables, key=a_key, reverse=reverse, prefetch=prefetch
        )
    # There is only this single async generator in the whole merge pipeline;
    # its ``finally`` therefore runs inline on an explicit ``aclose`` and on
    # cancellation, while an implicit ``break`` cleanup defers only this same
    # generator instead of racing across nested generators.
    try:
        heap = await sources.start()
        # there are at least two iterators that need merging
        while len(heap) > 1:
            while True:
                ordered, streak, index = heap[0]
                yield sources.head_item(index)
                new_ordered = await sources.advance(index)
                if new_ordered is _SOURCE_EXHAUSTED:
                    _heapq.heappop(heap)
                    break
                streak = streak + 1 if _same_key(new_ordered, ordered) else 0
                _heapq.heapreplace(heap, (new_ordered, streak, index))
        # there is only one iterator left, no need for merging
        if heap:
            index = heap[0][2]
            yield sources.head_item(index)
            while True:
                item = await sources.next_tail(index)
                if item is _SOURCE_EXHAUSTED:
                    break
                yield item
    finally:
        await sources.aclose()


class _MergeSources:
    """
    Strategy driving the individual sorted sources of a :func:`merge`

    The sources expose their current comparison head and are advanced one
    entry at a time. Source indices are shared with the merge heap.
    """

    __slots__ = ()

    async def start(self) -> "List[Tuple[Any, int, int]]":
        """Fetch initial heads concurrently and return the initial heap"""
        raise NotImplementedError

    def head_item(self, index: int) -> Any:
        """The current comparison item of source ``index``"""
        raise NotImplementedError

    async def advance(self, index: int) -> Any:
        """
        Pull the next entry of source ``index`` and return its ordered key

        The pulled item becomes the new head. Returns :data:`_SOURCE_EXHAUSTED`
        when the source is exhausted, and raises the error of a source or of
        ``key``.
        """
        raise NotImplementedError

    async def next_tail(self, index: int) -> Any:
        """Pull the next raw item of the single remaining source"""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Stop background work and close every non-exhausted source"""
        raise NotImplementedError


#: terminal result signalling that a source is exhausted
_SOURCE_EXHAUSTED = Sentinel("<exhausted source>")


class _SourceFailure:
    """Terminal queue entry carrying the error that stopped a source"""

    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


def _ordered_key(item_key: Any, reverse: bool) -> Any:
    """Wrap ``item_key`` for stable heap ordering in either sort direction"""
    return ReverseLT(item_key) if reverse else item_key


def _same_key(left: Any, right: Any) -> bool:
    """Whether two heap order keys are equivalent for tie-breaking"""
    return not (left < right or right < left)


async def _close_all(
    iterators: "List[AsyncIterator[Any]]",
    exhausted: "List[bool]",
) -> None:
    """
    Close every source that is not already naturally exhausted

    Closing keeps going across individual close failures and across our own
    task's cancellation, so a shutdown never leaves a source behind.
    """
    import asyncio

    close_error: Optional[BaseException] = None
    cancelled_while_closing = False
    for index, iterator in enumerate(iterators):
        if exhausted[index] or not isinstance(iterator, ACloseable):
            continue
        try:
            await _shielded(iterator.aclose())
        except asyncio.CancelledError:
            cancelled_while_closing = True
        except BaseException as error:  # noqa: B036  # re-raised below
            if close_error is None:
                close_error = error
    # While a real error is already propagating, do not let cleanup replace
    # it. A clean ``break``/``aclose`` has no primary error (GeneratorExit
    # does not count), so cleanup failures surface to the consumer instead.
    primary = sys.exc_info()[1]
    if isinstance(primary, GeneratorExit):
        primary = None
    if primary is None:
        if close_error is not None:
            raise close_error
        if cancelled_while_closing:
            raise asyncio.CancelledError()


async def _shielded(coro: Awaitable[Any]) -> Any:
    """
    Await ``coro`` as a separate task that survives the current task's cancellation

    This lets shutdown/cleanup run to completion even when the consumer task is
    being cancelled: every outer cancellation briefly resumes here, while the
    actual work is shielded and keeps running.
    """
    import asyncio

    task = asyncio.ensure_future(coro)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                break
            continue
    return task.result()


def _is_cancelling() -> bool:
    """Whether the current task has a pending cancellation (Python 3.11+)"""
    import asyncio

    task = asyncio.current_task()
    cancelling = getattr(task, "cancelling", None)
    return bool(cancelling()) if cancelling is not None else False


async def _cancel_tasks(tasks: Any) -> None:
    """Cancel ``tasks`` and let their cancellation settle, without masking ours"""
    import asyncio

    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        with _suppress(asyncio.CancelledError):
            await asyncio.gather(*pending, return_exceptions=True)
    if _is_cancelling():
        raise asyncio.CancelledError


class _LazyMerge(_MergeSources):
    """Sequential merge without any background prefetching"""

    __slots__ = (
        "_iterables",
        "_key",
        "_reverse",
        "_iterators",
        "_heads",
        "_exhausted",
        "_started",
    )

    def __init__(
        self,
        iterables: Tuple[AnyIterable[Any], ...],
        *,
        key: Optional[Callable[[Any], Awaitable[Any]]],
        reverse: bool,
    ) -> None:
        self._iterables = iterables
        self._key = key
        self._reverse = reverse
        self._iterators: List[AsyncIterator[Any]] = []
        self._heads: List[Any] = []
        self._exhausted: List[bool] = []
        self._started = False

    async def start(self) -> "List[Tuple[Any, int, int]]":
        self._started = True
        # create every iterator up front so a failure of one source still
        # leaves all the other sources available for explicit closing
        self._iterators = [aiter(iterable) for iterable in self._iterables]
        self._heads = [None] * len(self._iterators)
        self._exhausted = [False] * len(self._iterators)
        heap: List[Tuple[Any, int, int]] = []
        for index, iterator in enumerate(self._iterators):
            try:
                head = await iterator.__anext__()
            except StopAsyncIteration:
                self._exhausted[index] = True
                continue
            head_key = await self._key(head) if self._key is not None else head
            self._heads[index] = head
            heap.append((_ordered_key(head_key, self._reverse), 0, index))
        _heapq.heapify(heap)
        return heap

    def head_item(self, index: int) -> Any:
        return self._heads[index]

    async def advance(self, index: int) -> Any:
        try:
            item = await self._iterators[index].__anext__()
        except StopAsyncIteration:
            self._exhausted[index] = True
            return _SOURCE_EXHAUSTED
        item_key = await self._key(item) if self._key is not None else item
        self._heads[index] = item
        return _ordered_key(item_key, self._reverse)

    async def next_tail(self, index: int) -> Any:
        # the single remaining source keeps its original order; the key is
        # only needed for comparisons and must not be computed here
        try:
            return await self._iterators[index].__anext__()
        except StopAsyncIteration:
            self._exhausted[index] = True
            return _SOURCE_EXHAUSTED

    async def aclose(self) -> None:
        if not self._started:
            return
        await _close_all(self._iterators, self._exhausted)


async def _prefetch_source(
    index: int,
    iterator: AsyncIterator[Any],
    queue: "asyncio.Queue[Any]",
    budget: "asyncio.Semaphore",
    head_required: "asyncio.Event",
    entry_taken: "asyncio.Event",
    exhausted: List[bool],
    failures: List[BaseException],
    failure_event: "asyncio.Event",
    shutting_down: List[bool],
    key: Optional[Callable[[Any], Awaitable[Any]]],
) -> None:
    """
    Prefetch subsequent items of a single source into its queue

    The runner hands over exactly one item at a time and waits until the
    consumer has taken it, so there is never a second fetch in flight for a
    source while the previous entry is pending. Every handed-over item holds
    one token of the shared ``budget``, except the one comparison "head" fetch
    performed while ``head_required`` is set. Head fetches are required for
    merging and never count against the budget, which guarantees progress even
    when the budget is smaller than the number of sources.
    """
    import asyncio

    while True:
        token_wait = asyncio.ensure_future(budget.acquire())
        head_wait = asyncio.ensure_future(head_required.wait())
        try:
            done, _ = await asyncio.wait(
                {token_wait, head_wait}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            await _cancel_tasks((token_wait, head_wait))
            raise
        if head_wait in done:
            # the head request wins even if a token was granted simultaneously
            if token_wait in done:
                token_wait.result()
                budget.release()
            else:
                await _cancel_tasks((token_wait,))
            head_required.clear()
            holds_token = False
        else:
            await _cancel_tasks((head_wait,))
            token_wait.result()
            holds_token = True
        try:
            item = await iterator.__anext__()
        except StopAsyncIteration:
            if holds_token:
                budget.release()
            exhausted[index] = True
            queue.put_nowait(_SOURCE_EXHAUSTED)
            return
        except BaseException as error:
            if holds_token:
                budget.release()
            _report_failure(failures, failure_event, shutting_down, queue, error)
            if isinstance(error, asyncio.CancelledError):
                raise
            return
        try:
            item_key = await key(item) if key is not None else item
        except BaseException as error:
            if holds_token:
                budget.release()
            _report_failure(failures, failure_event, shutting_down, queue, error)
            if isinstance(error, asyncio.CancelledError):
                raise
            return
        # Hand over the item and wait for the consumer to take it before
        # fetching anything else, which keeps the budget accounting exact.
        queue.put_nowait((item_key, item, holds_token))
        await entry_taken.wait()
        entry_taken.clear()


def _report_failure(
    failures: List[BaseException],
    failure_event: "asyncio.Event",
    shutting_down: List[bool],
    queue: "asyncio.Queue[Any]",
    error: BaseException,
) -> None:
    """Record and signal the first failure of a source"""
    if shutting_down[0]:
        # cancellation is expected during shutdown, do not poison the merge
        return
    queue.put_nowait(_SourceFailure(error))
    if not failures:
        failures.append(error)
    failure_event.set()


async def _get_entry(
    index: int,
    queues: "List[asyncio.Queue[Any]]",
    failure_event: "asyncio.Event",
    failures: List[BaseException],
) -> Any:
    """
    Get the next queued entry of source ``index`` or fail on any source error

    Waiting simultaneously for the requested entry and the shared failure event
    ensures that an error of another source is noticed immediately instead of
    being delayed by a slow source.
    """
    import asyncio

    queue = queues[index]
    getter = asyncio.ensure_future(queue.get())
    waiter = asyncio.ensure_future(failure_event.wait())
    try:
        done, _ = await asyncio.wait(
            {getter, waiter}, return_when=asyncio.FIRST_COMPLETED
        )
    except asyncio.CancelledError:
        await _cancel_tasks((getter, waiter))
        raise
    if getter in done:
        await _cancel_tasks((waiter,))
        entry = getter.result()
    else:
        await _cancel_tasks((getter,))
        raise failures[0]
    if failures:
        # another source failed even though our entry arrived first
        raise failures[0]
    return entry


class _PrefetchMerge(_MergeSources):
    """Concurrent merge with a total lookahead prefetch budget"""

    __slots__ = (
        "_iterables",
        "_key",
        "_reverse",
        "_prefetch",
        "_iterators",
        "_queues",
        "_head_required",
        "_entry_taken",
        "_exhausted",
        "_tasks",
        "_head_items",
        "_budget",
        "_failures",
        "_failure_event",
        "_shutting_down",
        "_started",
    )

    def __init__(
        self,
        iterables: Tuple[AnyIterable[Any], ...],
        *,
        key: Optional[Callable[[Any], Awaitable[Any]]],
        reverse: bool,
        prefetch: int,
    ) -> None:
        import asyncio

        self._iterables = iterables
        self._key = key
        self._reverse = reverse
        self._prefetch = prefetch
        self._iterators: List[AsyncIterator[Any]] = [
            aiter(iterable) for iterable in iterables
        ]
        source_count = len(self._iterators)
        self._queues: List["asyncio.Queue[Any]"] = [
            asyncio.Queue() for _ in self._iterators
        ]
        self._head_required: List["asyncio.Event"] = [
            asyncio.Event() for _ in self._iterators
        ]
        self._entry_taken: List["asyncio.Event"] = [
            asyncio.Event() for _ in self._iterators
        ]
        self._exhausted: List[bool] = [False] * source_count
        self._tasks: List["asyncio.Task[None]"] = []
        self._head_items: List[Any] = [None] * source_count
        self._budget: "asyncio.Semaphore" = asyncio.Semaphore(prefetch)
        self._failures: List[BaseException] = []
        self._failure_event: "asyncio.Event" = asyncio.Event()
        self._shutting_down: List[bool] = [False]
        self._started = False

    async def _fetch_first_head(self, iterator: AsyncIterator[Any]) -> Tuple[Any, Any]:
        item = await iterator.__anext__()
        item_key = await self._key(item) if self._key is not None else item
        return item_key, item

    async def start(self) -> "List[Tuple[Any, int, int]]":
        import asyncio

        self._started = True
        iterators = self._iterators
        # Fetch every first head concurrently; heads are merge bookkeeping,
        # not prefetch, so they do not count against the budget.
        head_results: List[Any] = await asyncio.gather(
            *(self._fetch_first_head(iterator) for iterator in iterators),
            return_exceptions=True,
        )
        for result in head_results:
            if isinstance(result, StopAsyncIteration):
                continue
            if isinstance(result, BaseException):
                raise result
        heap: List[Tuple[Any, int, int]] = []
        for index, result in enumerate(head_results):
            if isinstance(result, StopAsyncIteration):
                self._exhausted[index] = True
                continue
            item_key, self._head_items[index] = result
            heap.append((_ordered_key(item_key, self._reverse), 0, index))
            self._tasks.append(
                asyncio.ensure_future(
                    _prefetch_source(
                        index,
                        iterators[index],
                        self._queues[index],
                        self._budget,
                        self._head_required[index],
                        self._entry_taken[index],
                        self._exhausted,
                        self._failures,
                        self._failure_event,
                        self._shutting_down,
                        self._key,
                    )
                )
            )
        _heapq.heapify(heap)
        return heap

    def head_item(self, index: int) -> Any:
        return self._head_items[index]

    async def _take(self, index: int) -> Any:
        """Receive the next handed-over entry of source ``index``"""
        import asyncio

        try:
            entry = self._queues[index].get_nowait()
        except asyncio.QueueEmpty:
            # ask the source to fill its comparison head outside of the budget
            self._head_required[index].set()
            entry = await _get_entry(
                index, self._queues, self._failure_event, self._failures
            )
        # the pending head request, if any, is satisfied by this entry
        self._head_required[index].clear()
        if entry is not _SOURCE_EXHAUSTED and not isinstance(entry, _SourceFailure):
            # let the runner fetch the next item; terminals end the runner
            self._entry_taken[index].set()
        return entry

    async def advance(self, index: int) -> Any:
        entry = await self._take(index)
        if entry is _SOURCE_EXHAUSTED:
            return _SOURCE_EXHAUSTED
        if isinstance(entry, _SourceFailure):
            raise entry.error
        item_key, item, holds_token = entry
        if holds_token:
            self._budget.release()
        self._head_items[index] = item
        return _ordered_key(item_key, self._reverse)

    async def next_tail(self, index: int) -> Any:
        entry = await self._take(index)
        if entry is _SOURCE_EXHAUSTED:
            return _SOURCE_EXHAUSTED
        if isinstance(entry, _SourceFailure):
            raise entry.error
        _item_key, item, holds_token = entry
        if holds_token:
            self._budget.release()
        return item

    async def aclose(self) -> None:
        import asyncio

        if not self._started:
            return
        self._shutting_down[0] = True
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await _shielded(asyncio.gather(*self._tasks, return_exceptions=True))
        await _close_all(self._iterators, self._exhausted)


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
        n_heap = [
            (ordered(await key(item)), index * order_sign, item)
            async for index, item in a_zip(range(n), borrow(iterator))
        ]
        if not n_heap:
            return []
        _heapq.heapify(n_heap)
        worst_key = n_heap[0][0]
        next_index = n * order_sign
        async for item in iterator:
            item_key = ordered(await key(item))
            if worst_key < item_key:
                _heapq.heapreplace(n_heap, (item_key, next_index, item))
                worst_key = n_heap[0][0]
                next_index += 1 * order_sign
        n_heap.sort(reverse=True)
    return [item for _, _, item in n_heap]


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
