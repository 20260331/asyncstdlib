"""
Internal helpers to safely build async abstractions

While some of these helpers have public siblings
(e.g. :py:class:`~.ScopedIter` and :py:func:`~.asynctools.scoped_iter`)
they are purposely kept separate.
Any helpers in this module are *not* bound to maintaining a public interface,
and offer less convenience to save on overhead.
"""

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from inspect import iscoroutinefunction
from typing import (
    Any,
    AsyncIterator,
    AsyncGenerator,
    AsyncContextManager,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    AsyncIterable,
    List,
    Optional,
    Tuple,
    Awaitable,
    Type,
)
from types import TracebackType
import weakref

from ._typing import T, AnyIterable


class Sentinel:
    """Placeholder with configurable ``repr``"""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name


def aiter(subject: AnyIterable[T]) -> AsyncIterator[T]:
    """
    An async iterator object yielding elements from ``subject``

    :raises TypeError: if ``subject`` does not support any iteration protocol

    The ``subject`` must support
    the async iteration protocol (the :py:meth:`object.__aiter__` method),
    the regular iteration protocol (the :py:meth:`object.__iter__` method),
    or it must support the sequence protocol (the :py:meth:`object.__getitem__`
    method with integer arguments starting at 0).
    In either case, an async iterator is returned.
    """
    if isinstance(subject, AsyncIterable):
        return subject.__aiter__()
    else:
        return _aiter_sync(subject).__aiter__()


async def _aiter_sync(iterable: Iterable[T]) -> AsyncIterator[T]:
    """Helper to provide an async iterator for a regular iterable"""
    for item in iterable:
        yield item


class ScopedIter(Generic[T]):
    """
    Context manager that provides and cleans up an iterator for an iterable

    Note that unlike :py:func:`~.asynctools.scoped_iter`, this helper does
    *not* borrow the iterator automatically. Use :py:func:`~.borrow` if needed.
    """

    __slots__ = ("_iterator",)

    def __init__(self, iterable: AnyIterable[T]):
        self._iterator: AsyncIterator[T] = aiter(iterable)
        _audit_claim(self._iterator, self)

    async def __aenter__(self) -> AsyncIterator[T]:
        return self._iterator

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        try:
            aclose = self._iterator.aclose()  # type: ignore
        except AttributeError:
            pass
        else:
            _audit_close(self._iterator, self)
            await aclose


def borrow(iterator: AsyncIterator[T]) -> AsyncGenerator[T, None]:
    """Borrow an async iterator for iteration, preventing it from being closed"""
    return (item async for item in iterator)


def awaitify(
    function: "Callable[..., Awaitable[T]] | Callable[..., T]",
) -> Callable[..., Awaitable[T]]:
    """Ensure that ``function`` can be used in ``await`` expressions"""
    if iscoroutinefunction(function):
        return function
    else:
        return Awaitify(function)


class Awaitify(Generic[T]):
    """Helper to peek at the return value of ``function`` and make it ``async``"""

    __slots__ = "__wrapped__", "_async_call"

    def __init__(self, function: "Callable[..., Awaitable[T]] | Callable[..., T]"):
        self.__wrapped__ = function
        self._async_call: "Callable[..., Awaitable[T]] | None" = None

    def __call__(self, *args: Any, **kwargs: Any) -> Awaitable[T]:
        if (async_call := self._async_call) is None:
            value = self.__wrapped__(*args, **kwargs)
            if isinstance(value, Awaitable):
                self._async_call = self.__wrapped__  # type: ignore
                return value  # pyright: ignore
            else:
                self._async_call = force_async(self.__wrapped__)  # type: ignore
                return await_value(value)
        else:
            return async_call(*args, **kwargs)


async def await_value(value: T) -> T:
    return value


def force_async(call: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    async def async_wrapped(*args: Any, **kwargs: Any) -> T:
        return call(*args, **kwargs)

    return async_wrapped


# ---------------------------------------------------------------------------
# opt-in ownership auditing
#
# The audit only *observes* the asyncstdlib wrapper layer: wrappers report
# claim/borrow/return/close events here via the ``_audit_*`` hooks below.
# User iterators are never wrapped, replaced or monkey-patched, and the hooks
# never raise or await: they cannot delay or mask an in-flight exception.
# When no audit scope is active the hooks reduce to a single context variable
# lookup and have no observable behaviour.
# ---------------------------------------------------------------------------

_active_audit: ContextVar[Optional["OwnershipAudit"]] = ContextVar(
    "asyncstdlib_ownership_audit", default=None
)


@dataclass(frozen=True)
class Violation:
    """
    A single ownership rule violation recorded by an :class:`OwnershipAudit`

    The :pyattr:`kind` is one of the string constants defined on this class;
    :pyattr:`message` provides a human readable explanation and the
    ``object_*`` fields identify the offending async iterator by its type
    and :pyfunc:`id` (the iterator itself is never retained).
    """

    kind: str
    object_type: str
    object_id: int
    message: str

    #: a borrowed iterator handle was never returned before scope exit
    UNRETURNED_BORROW: ClassVar[str] = "unreturned-borrow"
    #: an iterator was ``aclose``\ d in a second ownership episode
    DUPLICATE_CLOSE: ClassVar[str] = "duplicate-close"
    #: an iterator was closed by a wrapper that did not own it
    WRONG_OWNER_CLOSE: ClassVar[str] = "wrong-owner-close"
    #: ownership was claimed for an iterator already closed by a previous owner
    REUSED_AFTER_CLOSE: ClassVar[str] = "reused-after-close"
    #: an iterator still lent out was closed by its current owner
    CLOSE_WHILE_BORROWED: ClassVar[str] = "close-while-borrowed"
    #: ownership of an iterator moved while borrowed handles were outstanding
    TRANSFER_WHILE_BORROWED: ClassVar[str] = "transfer-while-borrowed"


class OwnershipViolation(RuntimeError):
    """Raised when an ownership audit scope exits with recorded violations"""

    def __init__(self, audit: "OwnershipAudit"):
        self.audit = audit
        count = len(audit.violations)
        listing = "\n".join(f"- {item.message}" for item in audit.violations)
        plural = "s" if count != 1 else ""
        super().__init__(
            f"asyncstdlib ownership audit found {count} violation{plural}:\n"
            f"{listing}"
        )


@dataclass
class _BorrowRecord:
    ticket_type: str
    ticket_id: int
    lender: str
    reference: "Optional[weakref.ReferenceType[Any]]"
    returned: bool = False


@dataclass
class _OwnershipEpisode:
    """One contiguous interval during which one chain of owners held an object"""

    owners: List[str] = field(default_factory=list)
    closed: bool = False
    closed_by: List[str] = field(default_factory=list)


@dataclass
class _IteratorRecord:
    type_name: str
    object_id: int
    reference: "Optional[weakref.ReferenceType[Any]]"
    episodes: List[_OwnershipEpisode] = field(default_factory=list)
    borrows: Dict[int, _BorrowRecord] = field(default_factory=dict)


def _weak_reference(
    obj: Any,
) -> "Optional[weakref.ReferenceType[Any]]":
    try:
        return weakref.ref(obj)
    except TypeError:
        # some user objects (e.g. slotted classes without ``__weakref__``)
        # cannot be weakly referenced; identity is then tracked only while
        # the most recent ownership episode is open
        return None


class OwnershipAudit:
    """
    Record of async iterator ownership events observed within one audit scope

    Instances are provided by :pyfunc:`~.asynctools.ownership_audit`;
    do not construct them directly.
    The :pyattr:`events` list gives a chronological trace and
    :pyattr:`violations` holds the :class:`Violation` instances recorded so
    far; it is finalized when the scope exits.
    """

    __slots__ = (
        "name",
        "events",
        "violations",
        "_records",
        "_tickets",
        "_finalized",
        "_enabled",
    )

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.events: List[str] = []
        self.violations: List[Violation] = []
        self._records: Dict[int, _IteratorRecord] = {}
        self._tickets: Dict[int, int] = {}
        self._finalized = False
        self._enabled = True

    @staticmethod
    def _label(party: Any) -> str:
        if isinstance(party, Sentinel):
            return party.name
        return f"{type(party).__name__}@0x{id(party):x}"

    @staticmethod
    def _describe(record: _IteratorRecord) -> str:
        return f"{record.type_name}@0x{record.object_id:x}"

    def _is_current(self, record: _IteratorRecord, iterator: Any) -> bool:
        """Whether ``record`` still describes the live ``iterator`` object"""
        reference = record.reference
        if reference is not None:
            return reference() is iterator
        # without weak references we can only spot id recycling once the
        # previous owner finished and closed the object
        return not record.episodes or not record.episodes[-1].closed

    def _retire(self, record: _IteratorRecord) -> None:
        """Drop bookkeeping for a destroyed object so its id can be reused"""
        self._records.pop(record.object_id, None)
        dead_tickets = [
            ticket_id
            for ticket_id, object_id in self._tickets.items()
            if object_id == record.object_id
        ]
        for ticket_id in dead_tickets:
            self._tickets.pop(ticket_id, None)

    def _record_for(self, iterator: Any) -> _IteratorRecord:
        object_id = id(iterator)
        record = self._records.get(object_id)
        if record is not None and not self._is_current(record, iterator):
            # the previous object was destroyed and its id was recycled
            self._retire(record)
            record = None
        if record is None:
            record = _IteratorRecord(
                type_name=type(iterator).__name__,
                object_id=object_id,
                reference=_weak_reference(iterator),
            )
            self._records[object_id] = record
        return record

    def _live_record(self, iterator: Any) -> Optional[_IteratorRecord]:
        record = self._records.get(id(iterator))
        if record is not None and not self._is_current(record, iterator):
            return None
        return record

    def _violate(self, violation: Violation) -> None:
        self.violations.append(violation)
        self.events.append(f"violation {violation.kind}: {violation.message}")

    def _safely(self, action: Callable[..., None], *args: Any) -> None:
        # Auditing must never interfere with the audited code path.
        if self._enabled:
            try:
                action(*args)
            except Exception:  # pragma: no cover - defensive only
                self._enabled = False

    # -- events emitted by the wrapper layer --------------------------------

    def _claim(self, iterator: Any, owner: Any) -> None:
        record = self._record_for(iterator)
        party = self._label(owner)
        episode = record.episodes[-1] if record.episodes else None
        if episode is None:
            record.episodes.append(_OwnershipEpisode(owners=[party]))
        elif episode.closed:
            self._violate(
                Violation(
                    Violation.REUSED_AFTER_CLOSE,
                    record.type_name,
                    record.object_id,
                    f"{self._describe(record)}: ownership claimed by {party} "
                    f"after it was closed by {', '.join(episode.closed_by)}",
                )
            )
            record.episodes.append(_OwnershipEpisode(owners=[party]))
        elif party not in episode.owners:
            outstanding = [
                info for info in record.borrows.values() if not info.returned
            ]
            if outstanding:
                self._violate(
                    Violation(
                        Violation.TRANSFER_WHILE_BORROWED,
                        record.type_name,
                        record.object_id,
                        f"{self._describe(record)}: ownership transferred to "
                        f"{party} while {len(outstanding)} borrowed "
                        f"handle(s) were not returned",
                    )
                )
            episode.owners.append(party)
        self.events.append(f"claim {self._describe(record)} by {party}")

    def _borrow(self, iterator: Any, ticket: Any) -> None:
        record = self._record_for(iterator)
        episode = record.episodes[-1] if record.episodes else None
        lender = "<external>" if episode is None else episode.owners[-1]
        self._tickets[id(ticket)] = record.object_id
        record.borrows[id(ticket)] = _BorrowRecord(
            ticket_type=type(ticket).__name__,
            ticket_id=id(ticket),
            lender=lender,
            reference=_weak_reference(ticket),
        )
        self.events.append(
            f"borrow {self._describe(record)} as "
            f"{type(ticket).__name__}@0x{id(ticket):x} from {lender}"
        )

    def _return_borrow(self, ticket: Any) -> None:
        object_id = self._tickets.get(id(ticket))
        if object_id is None:
            return
        record = self._records.get(object_id)
        if record is None:  # pragma: no cover - index and table agree
            return
        info = record.borrows.get(id(ticket))
        if info is None or info.returned:
            return
        referenced = info.reference
        if referenced is not None and referenced() is not ticket:
            # the ticket id was recycled from a destroyed wrapper
            return
        info.returned = True
        self.events.append(
            f"return {type(ticket).__name__}@0x{id(ticket):x} "
            f"of {self._describe(record)}"
        )

    def _close(self, iterator: Any, closer: Any) -> None:
        record = self._live_record(iterator)
        if record is None or not record.episodes:
            # closing an object the wrapper layer never claimed ownership of,
            # e.g. an internal helper generator -- nothing to audit
            return
        party = self._label(closer)
        episode = record.episodes[-1]
        target = self._describe(record)
        if not episode.closed:
            if party not in episode.owners:
                self._violate(
                    Violation(
                        Violation.WRONG_OWNER_CLOSE,
                        record.type_name,
                        record.object_id,
                        f"{target}: closed by {party} but currently owned "
                        f"by {episode.owners[-1]}",
                    )
                )
            outstanding = [
                info for info in record.borrows.values() if not info.returned
            ]
            if outstanding:
                self._violate(
                    Violation(
                        Violation.CLOSE_WHILE_BORROWED,
                        record.type_name,
                        record.object_id,
                        f"{target}: closed by {party} while "
                        f"{len(outstanding)} borrowed handle(s) were "
                        f"not returned",
                    )
                )
            episode.closed = True
            episode.closed_by.append(party)
            if len(record.episodes) > 1:
                self._violate(
                    Violation(
                        Violation.DUPLICATE_CLOSE,
                        record.type_name,
                        record.object_id,
                        f"{target}: closed by {party} in a new ownership "
                        f"episode after it had already been closed",
                    )
                )
        elif party not in episode.owners:
            episode.closed_by.append(party)
            if any(party in previous.owners for previous in record.episodes[:-1]):
                kind = Violation.DUPLICATE_CLOSE
                reason = "it had already been closed in a previous ownership episode"
            else:
                kind = Violation.WRONG_OWNER_CLOSE
                reason = f"it was owned by {episode.owners[-1]} and already closed"
            self._violate(
                Violation(
                    kind,
                    record.type_name,
                    record.object_id,
                    f"{target}: closed again by {party}; {reason}",
                )
            )
        else:
            # idempotent re-close by a current owner, as used internally
            # when nested wrappers each run their own cleanup
            episode.closed_by.append(party)
        self.events.append(f"close {target} by {party}")

    def _finalize(self) -> Tuple[Violation, ...]:
        if self._finalized:
            return tuple(self.violations)
        self._finalized = True
        for record in tuple(self._records.values()):
            reference = record.reference
            if reference is not None and reference() is None:
                # object and its borrow handles were already garbage collected
                continue
            for info in tuple(record.borrows.values()):
                if info.returned:
                    continue
                self._violate(
                    Violation(
                        Violation.UNRETURNED_BORROW,
                        record.type_name,
                        record.object_id,
                        f"{self._describe(record)}: borrowed handle "
                        f"{info.ticket_type}@0x{info.ticket_id:x} from "
                        f"{info.lender} was not returned before the "
                        f"audit scope exited",
                    )
                )
        return tuple(self.violations)

    def _summary(self) -> str:
        count = len(self.violations)
        listing = "\n".join(f"- {item.message}" for item in self.violations)
        plural = "s" if count != 1 else ""
        return (
            f"asyncstdlib ownership audit found {count} violation{plural}:\n"
            f"{listing}"
        )


class _OwnershipAuditScope(AsyncContextManager["OwnershipAudit"]):
    """Async context manager running one :class:`OwnershipAudit`"""

    __slots__ = ("_audit", "_raise_on_exit", "_token")

    def __init__(self, *, raise_on_exit: bool = True, name: Optional[str] = None):
        self._audit = OwnershipAudit(name=name)
        self._raise_on_exit = raise_on_exit
        self._token: Optional[Token[Optional[OwnershipAudit]]] = None

    async def __aenter__(self) -> "OwnershipAudit":
        self._token = _active_audit.set(self._audit)
        return self._audit

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        token = self._token
        if token is not None:
            _active_audit.reset(token)
        violations = self._audit._finalize()
        if violations and self._raise_on_exit and exc_val is None:
            raise OwnershipViolation(self._audit)
        if violations and exc_val is not None:
            # never delay or replace the original exception; annotate it
            # with the audit result when the runtime supports exception notes
            add_note = getattr(exc_val, "add_note", None)
            if add_note is not None:
                try:
                    add_note(self._audit._summary())
                except Exception:  # pragma: no cover - defensive only
                    pass
        return False


def _audit_claim(iterator: Any, owner: Any) -> None:
    audit = _active_audit.get()
    if audit is not None:
        audit._safely(audit._claim, iterator, owner)


def _audit_borrow(iterator: Any, ticket: Any) -> None:
    audit = _active_audit.get()
    if audit is not None:
        audit._safely(audit._borrow, iterator, ticket)


def _audit_return(ticket: Any) -> None:
    audit = _active_audit.get()
    if audit is not None:
        audit._safely(audit._return_borrow, ticket)


def _audit_close(iterator: Any, closer: Any) -> None:
    audit = _active_audit.get()
    if audit is not None:
        audit._safely(audit._close, iterator, closer)
