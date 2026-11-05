"""
Cooperative iteration quotas

This module provides an optional :py:class:`Quota` context that can bound
how much work the :term:`asynchronous iterators <asynchronous iterator>` of
:py:mod:`asyncstdlib` do while iterating.
Inside an ``async with Quota(...)`` block, every iterator *opened* by
:py:mod:`asyncstdlib` -- that is, every iterator created via
:py:func:`~asyncstdlib.iter` or implicitly by a combinator such as
:py:func:`~asyncstdlib.map`, :py:func:`~asyncstdlib.chain` or
:py:func:`~asyncstdlib.tee` -- cooperatively consumes the currently active
quota.

Several independent budgets may be set on a single quota:

* ``items`` bounds how many elements are pulled from underlying iterators;
* ``awaits`` bounds how many asynchronous steps (iterator pulls and
  invocations of async callables) are performed;
* ``buffered`` bounds how many elements combinators may retain at any time,
  e.g. the buffers of :py:func:`~asyncstdlib.cycle` and
  :py:func:`~asyncstdlib.tee`;
* ``weight`` bounds a user defined cost per element, computed via
  ``weight_of``.

When a quota is exhausted it either raises :py:exc:`QuotaExceeded`
(the default) or stops the affected iterator(s) cleanly as if they were
exhausted, allowing callers to collect partial results.
Regardless of the mode, unwinding stops and closes every iterator that
:py:mod:`asyncstdlib` opened for the affected pipeline.

Nested quotas follow two explicit rules:

* entering a *different* :py:class:`Quota` **isolates** the nested scope --
  only the innermost quota is consumed, and the outer quota is suspended
  until the nested block exits;
* re-entering the *same* :py:class:`Quota` instance (or passing it around
  explicitly) **shares** a single set of budgets.
"""

from contextvars import ContextVar, Token
from functools import wraps
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Optional,
    TypeVar,
    Union,
    cast,
)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., AsyncIterable[Any]])

#: budget names that only ever count down
_ITEMS = "items"
_AWAITS = "awaits"
_BUFFERED = "buffered"
_WEIGHT = "weight"

#: currently active quota for this :py:mod:`contextvars` context
_QUOTA: "ContextVar[Optional[Quota]]" = ContextVar("asyncstdlib_quota", default=None)

#: code objects of async iterators internal to :py:mod:`asyncstdlib`
#:
#: Pulls on such iterators are relayed pulls of other iterators; quotas
#: meter only the *leaf* sources, so each source element is counted once.
_OWNED_CODES: "set[Any]" = set()


def owned(iterator_factory: F) -> F:
    """Mark an iterator created within :py:mod:`asyncstdlib` as internal

    Iterators produced by marked factories relay pulls rather than acting
    as data sources, so an active quota does not meter them again.
    The decorator adds no runtime overhead.
    """
    _OWNED_CODES.add(iterator_factory.__code__)
    return iterator_factory


def _is_owned(iterator: AsyncIterator[Any]) -> bool:
    """Check whether ``iterator`` is an internal asyncstdlib iterator"""
    if getattr(iterator, "__asyncstdlib_owned__", False):
        return True
    code = getattr(iterator, "ag_code", None)
    return code is not None and code in _OWNED_CODES


def charge(*, items: int = 0, awaits: int = 0, weight: float = 0.0) -> None:
    """Charge the active quota for a manual iteration step, if any"""
    quota = _active_quota()
    if quota is not None:
        quota._charge(items=items, awaits=awaits, weight=weight)


class QuotaExceeded(RuntimeError):
    """
    Raised when an iteration step would exceed a :py:class:`Quota`

    Attributes:
        quota: the quota that was exhausted
        budget: one of ``"items"``, ``"awaits"``, ``"buffered"`` or
            ``"weight"``
        requested: the amount that could not be granted
        used: the amount already consumed (or held, for ``"buffered"``)
        limit: the configured limit of the exhausted budget
    """

    __slots__ = ("quota", "budget", "requested", "used", "limit")

    # structured arguments instead of pickle-compatible *args forwarding
    def __init__(  # noqa: B042
        self,
        quota: "Quota",
        budget: str,
        requested: Union[int, float],
        used: Union[int, float],
        limit: Union[int, float],
    ):
        self.quota = quota
        self.budget = budget
        self.requested = requested
        self.used = used
        self.limit = limit
        name = f" {quota.name!r}" if quota.name is not None else ""
        super().__init__(
            f"iteration quota{name} exhausted for {budget}: "
            f"requested {requested} on top of {used}, limit is {limit}"
        )


class _QuotaStop(Exception):
    """Internal signal that a quota in ``stop`` mode was exhausted

    Raising this instead of :py:exc:`QuotaExceeded` lets dedicated boundary
    points translate exhaustion into normal iterator termination so that
    callers receive partial results.
    """

    __slots__ = ("quota", "budget")

    def __init__(self, quota: "Quota", budget: str):  # noqa: B042
        self.quota = quota
        self.budget = budget


def _active_quota() -> "Optional[Quota]":
    """Return the quota active for the current iteration step, if any"""
    return _QUOTA.get()


class Quota:
    """
    Cooperative iteration quota shared by :py:mod:`asyncstdlib` iterators

    :param items: maximum number of elements that may be pulled
    :param awaits: maximum number of asynchronous steps that may be performed
    :param buffered: maximum number of elements that may be retained by
        combinators at the same time
    :param weight: maximum total user defined weight of pulled elements
    :param weight_of: optional regular or async callable mapping an element
        to its (non-negative) weight; defaults to a weight of ``1`` per
        element only if ``weight`` is set
    :param on_exhausted: ``"raise"`` (the default) to raise
        :py:exc:`QuotaExceeded`, or ``"stop"`` to end affected iterators
        normally and yield partial results
    :param name: optional name used in repr and error messages

    A quota is activated as an :term:`asynchronous context manager`::

        async with Quota(items=100) as quota:
            async for item in a.map(fetch, source):
                ...

    All limits are optional; any limit left as :py:data:`None` is unlimited.
    A quota instance may be entered multiple times; entering the same
    instance in nested scopes shares its budgets, whereas entering a
    different quota isolates the nested scope.
    Sharing the same instance across concurrent tasks is supported by
    entering it once and having the child tasks inherit the active quota;
    independently activating one instance in unrelated contexts is not
    supported.
    """

    #: raise :py:exc:`QuotaExceeded` when exhausted
    RAISE = "raise"
    #: stop affected iterators normally when exhausted
    STOP = "stop"

    __slots__ = (
        "items",
        "awaits",
        "buffered",
        "weight",
        "name",
        "_weight_of",
        "_stop",
        "_used_items",
        "_used_awaits",
        "_used_weight",
        "_held",
        "_exhausted",
        "_reason",
        "_token",
        "_active",
    )

    def __init__(
        self,
        items: Optional[int] = None,
        awaits: Optional[int] = None,
        buffered: Optional[int] = None,
        weight: Optional[float] = None,
        *,
        weight_of: Optional[Callable[[Any], Any]] = None,
        on_exhausted: str = RAISE,
        name: Optional[str] = None,
    ):
        if on_exhausted not in (self.RAISE, self.STOP):
            raise ValueError(
                "on_exhausted must be 'raise' or 'stop', " f"got {on_exhausted!r}"
            )
        for limit_name, limit in (
            (_ITEMS, items),
            (_AWAITS, awaits),
            (_BUFFERED, buffered),
            (_WEIGHT, weight),
        ):
            if limit is not None and limit < 0:
                raise ValueError(f"{limit_name} limit must be non-negative")
        if weight_of is None and weight is not None:
            weight_of = _unit_weight
        self.items = items
        self.awaits = awaits
        self.buffered = buffered
        self.weight = weight
        self.name = name
        self._weight_of = weight_of
        self._stop = on_exhausted == self.STOP
        self._used_items = 0
        self._used_awaits = 0
        self._used_weight: float = 0.0
        self._held = 0
        self._exhausted = False
        self._reason: Optional[str] = None
        self._token: "Optional[Token[Optional[Quota]]]" = None
        # whether *this* context actually activated the instance
        self._active = False

    @property
    def on_exhausted(self) -> str:
        """Whether exhaustion raises (``"raise"``) or stops (``"stop"``)"""
        return self.STOP if self._stop else self.RAISE

    @property
    def stop_on_exhausted(self) -> bool:
        """:py:data:`True` if exhaustion ends iterators with partial results"""
        return self._stop

    @property
    def exhausted(self) -> bool:
        """:py:data:`True` once any budget of this quota was exhausted"""
        return self._exhausted

    @property
    def exhausted_budget(self) -> Optional[str]:
        """Name of the budget that was exhausted, if any"""
        return self._reason

    @property
    def consumed(self) -> Dict[str, Union[int, float]]:
        """Cumulative amounts consumed for ``items``, ``awaits`` and ``weight``"""
        return {
            _ITEMS: self._used_items,
            _AWAITS: self._used_awaits,
            _WEIGHT: self._used_weight,
        }

    @property
    def held(self) -> int:
        """Number of elements currently held against the ``buffered`` budget"""
        return self._held

    @property
    def remaining(self) -> Dict[str, Optional[Union[int, float]]]:
        """Remaining capacity for each budget (``None`` means unlimited)"""
        return {
            _ITEMS: None if self.items is None else self.items - self._used_items,
            _AWAITS: None if self.awaits is None else self.awaits - self._used_awaits,
            _BUFFERED: None if self.buffered is None else self.buffered - self._held,
            _WEIGHT: None if self.weight is None else self.weight - self._used_weight,
        }

    def __repr__(self) -> str:
        limits = ", ".join(
            f"{name}={limit}"
            for name, limit in (
                (_ITEMS, self.items),
                (_AWAITS, self.awaits),
                (_BUFFERED, self.buffered),
                (_WEIGHT, self.weight),
            )
            if limit is not None
        )
        return f"Quota({limits})"

    def _failure(
        self,
        budget: str,
        requested: Union[int, float],
        used: Union[int, float],
        limit: Optional[Union[int, float]],
    ) -> None:
        """Record exhaustion and raise the configured signal"""
        self._exhausted = True
        self._reason = budget
        assert limit is not None
        if self._stop:
            raise _QuotaStop(self, budget)
        raise QuotaExceeded(self, budget, requested, used, limit)

    def _charge(
        self,
        *,
        items: int = 0,
        awaits: int = 0,
        weight: float = 0.0,
    ) -> None:
        """Consume ``items``/``awaits``/``weight`` budgets atomically"""
        # An exhausted quota refuses every further step.
        if self._exhausted:
            assert self._reason is not None
            self._refuse_again(self._reason)
        candidates = (
            (_ITEMS, items, self._used_items, self.items),
            (_AWAITS, awaits, self._used_awaits, self.awaits),
            (_WEIGHT, weight, self._used_weight, self.weight),
        )
        for budget, requested, used, limit in candidates:
            if limit is not None and used + requested > limit:
                self._failure(budget, requested, used, limit)
        self._used_items += items
        self._used_awaits += awaits
        self._used_weight += weight

    def _refuse_again(self, budget: str) -> None:
        """Raise for a request against an already exhausted quota"""
        requested: Union[int, float]
        used: Union[int, float]
        limit: Optional[Union[int, float]]
        if budget == _ITEMS:
            requested, used, limit = 1, self._used_items, self.items
        elif budget == _AWAITS:
            requested, used, limit = 1, self._used_awaits, self.awaits
        elif budget == _BUFFERED:
            requested, used, limit = 1, self._held, self.buffered
        else:
            requested, used, limit = 1.0, self._used_weight, self.weight
        self._failure(budget, requested, used, limit)

    def _reserve(self, n: int = 1) -> "HoldToken":
        """Reserve ``n`` slots of the ``buffered`` budget"""
        if n < 0:
            raise ValueError("held amount must be non-negative")
        if self._exhausted:
            assert self._reason is not None
            self._refuse_again(self._reason)
        if self.buffered is not None and self._held + n > self.buffered:
            self._failure(_BUFFERED, n, self._held, self.buffered)
        self._held += n
        return HoldToken(self, n)

    def _release(self, token: "HoldToken") -> None:
        if token._released:
            return
        token._released = True
        self._held -= token.amount

    def consume(
        self,
        *,
        items: int = 0,
        awaits: int = 0,
        weight: float = 0.0,
    ) -> None:
        """
        Manually consume budgets of this quota

        :raises QuotaExceeded: if the request would exceed a budget

        Unlike automatic consumption by :py:mod:`asyncstdlib` iterators,
        manual consumption always raises :py:exc:`QuotaExceeded` -- even on a
        quota configured to ``"stop"`` -- since user code is responsible for
        deciding how partial results are handled.
        """
        try:
            self._charge(items=items, awaits=awaits, weight=weight)
        except _QuotaStop as stop:
            requested = {
                _ITEMS: items,
                _AWAITS: awaits,
                _WEIGHT: weight,
            }[stop.budget]
            limit = getattr(self, stop.budget)
            assert limit is not None
            raise QuotaExceeded(
                self, stop.budget, requested, self.consumed[stop.budget], limit
            ) from None

    def hold(self, n: int = 1) -> "HoldToken":
        """
        Reserve ``n`` slots of the ``buffered`` budget

        :return: a :py:class:`HoldToken` that must be released once the
            buffered elements are no longer retained
        :raises QuotaExceeded: if the request would exceed the budget

        The returned token can be released explicitly via
        :py:meth:`HoldToken.release` or used as a regular context manager.
        Like :py:meth:`consume`, manual holds always raise
        :py:exc:`QuotaExceeded` when no capacity is available.
        """
        try:
            return self._reserve(n)
        except _QuotaStop:
            assert self.buffered is not None
            raise QuotaExceeded(self, _BUFFERED, n, self._held, self.buffered) from None

    async def __aenter__(self) -> "Quota":
        # already active in this context (nested re-entry or a copied
        # context): sharing is implicit and must not install a new binding
        if _QUOTA.get() is self:
            return self
        self._token = _QUOTA.set(self)
        self._active = True
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._active:
            _QUOTA.reset(self._token)  # type: ignore[arg-type]
            self._token = None
            self._active = False


def _unit_weight(item: Any) -> float:
    return 1.0


class HoldToken:
    """
    A reservation against the ``buffered`` budget of a :py:class:`Quota`

    Call :py:meth:`release` (or use the token as a context manager) once the
    buffered elements are discarded or yielded.
    """

    __slots__ = ("_quota", "amount", "_released")

    def __init__(self, quota: Quota, amount: int):
        self._quota = quota
        self.amount = amount
        self._released = False

    def release(self) -> None:
        """Return the reserved slots to the quota"""
        self._quota._release(self)

    def __enter__(self) -> "HoldToken":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()

    def __repr__(self) -> str:
        state = "released" if self._released else "held"
        return f"<HoldToken({self.amount} {state})>"


class _RefHold:
    """A :py:class:`HoldToken` shared by several owners

    The reserved slots are released once, after every owner has called
    :py:meth:`release`.
    """

    __slots__ = ("_token", "_owners")

    def __init__(self, token: HoldToken, owners: int):
        self._token = token
        self._owners = owners

    def release(self) -> None:
        # clamp for peers that joined a shared buffer after the node was filled
        if self._owners > 0:
            self._owners -= 1
            if self._owners == 0:
                self._token.release()


def _hold(n: int = 1) -> Optional[HoldToken]:
    """Reserve buffered slots on the active quota, if any"""
    quota = _active_quota()
    if quota is None:
        return None
    return quota._reserve(n)


def _release(token: Optional[HoldToken]) -> None:
    if token is not None:
        token.release()


def _shared_hold(owners: int, n: int = 1) -> Optional[_RefHold]:
    """Reserve buffered slots released once all ``owners`` are done"""
    quota = _active_quota()
    if quota is None:
        return None
    return _RefHold(quota._reserve(n), owners)


async def _metered(
    iterator: AsyncIterator[T],
    quota: Quota,
    weight_of: Optional[Callable[[Any], Any]],
) -> AsyncIterator[T]:
    """Wrap ``iterator`` to charge the bound ``quota`` for every pull

    The wrapper owns ``iterator`` for its lifetime and always closes it,
    including when a quota is exhausted or the wrapper is abandoned mid-way.
    """
    try:
        while True:
            try:
                # charge the asynchronous step *before* advancing the source
                # so that an exhausted quota does not consume a source element
                quota._charge(awaits=1)
                item = await iterator.__anext__()
            except StopAsyncIteration:
                return
            except _QuotaStop:
                return
            weight = 0.0
            if weight_of is not None:
                value = weight_of(item)
                if hasattr(value, "__await__"):
                    quota._charge(awaits=1)
                    value = await value
                weight = float(value)
            try:
                quota._charge(items=1, weight=weight)
            except _QuotaStop:
                return
            yield item
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


_OWNED_CODES.add(_metered.__code__)


def meter(iterator: AsyncIterator[T]) -> AsyncIterator[T]:
    """
    Bind ``iterator`` to the active quota, if any

    Only leaf sources are metered: iterators internal to
    :py:mod:`asyncstdlib` merely relay pulls of other iterators and are
    returned unchanged, so every source element is counted exactly once
    regardless of how many combinators process it.
    Without an active quota every iterator is returned unchanged, so the
    feature incurs no overhead when unused.
    """
    quota = _active_quota()
    if quota is None or _is_owned(iterator):
        return iterator
    return _metered(iterator, quota, quota._weight_of)


class _QuotaGuard:
    """Context manager turning quota exhaustion into clean termination"""

    __slots__ = ()

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return isinstance(exc_val, _QuotaStop)


#: bound the body of combinators that may hold buffered elements
quota_guard = _QuotaGuard()


async def _guarded(iterator: AsyncIterator[T]) -> AsyncIterator[T]:
    """Drive ``iterator`` and translate quota exhaustion to normal ending"""
    try:
        async for item in iterator:
            yield item
    except _QuotaStop:
        return


class _QuotaBoundary(AsyncIterator[T], Generic[T]):
    """Iterator wrapper installing a quota boundary only when needed"""

    # relays pulls of the wrapped combinator, never a leaf source
    __asyncstdlib_owned__ = True

    __slots__ = ("_iterator", "_guarded_iter")

    def __init__(self, iterator: AsyncIterator[T]):
        self._iterator = iterator
        self._guarded_iter: Optional[AsyncIterator[T]] = None

    def __aiter__(self) -> "_QuotaBoundary[T]":
        return self

    async def __anext__(self) -> T:
        quota = _active_quota()
        if quota is not None and quota.stop_on_exhausted:
            if self._guarded_iter is None:
                self._guarded_iter = _guarded(self._iterator)
            return await self._guarded_iter.__anext__()
        return await self._iterator.__anext__()

    async def aclose(self) -> None:
        for iterator in (self._guarded_iter, self._iterator):
            if iterator is None:
                continue
            # close the wrapped combinator itself so that every iterator it
            # opened is released deterministically, even in "stop" mode
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                await aclose()


def quota_boundary(function: F) -> F:
    """
    Decorator marking an async iterator combinator as a quota boundary

    In ``"stop"`` mode, a quota exhausted while the combinator advances
    (for example while awaiting an async callable argument) ends the
    combinator normally instead of leaking the internal exhaustion signal.
    When no stopping quota is active, calls pass through with no overhead.
    """

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        return _QuotaBoundary(cast(AsyncIterator[Any], function(*args, **kwargs)))

    return wrapper  # type: ignore[return-value]
