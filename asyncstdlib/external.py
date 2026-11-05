"""
Optional external sorting with resumable, spill-to-storage runs

This module is *not* imported automatically; import it explicitly via
``from asyncstdlib import external``.
It implements an :term:`async neutral` external sort:
items are buffered and sorted in memory up to a fixed budget;
once the budget is exceeded, the sorted buffer is serialized
via a caller-provided :py:class:`RunStore` and spilled to temporary storage.
All spilled runs are then merged with a stable :math:`k`-way merge.

Every fully written run is described by an immutable :py:class:`RunInfo`;
the complete set is exported as a portable :py:class:`SortManifest`.
A manifest only ever references *committed* runs, so an interrupted spill
can never be mistaken for a recoverable result:
either the manifest is ``"ready"`` and all its runs are complete,
or it is ``"building"`` and must be discarded rather than trusted.
"""

from __future__ import annotations

import hashlib as _hashlib
import heapq as _heapq
import json as _json
import os as _os
import pickle as _pickle
import uuid as _uuid
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from ._core import aiter, awaitify
from ._typing import AnyIterable, Protocol, T

__all__ = [
    "ExternalSortError",
    "ManifestInvalid",
    "ManifestIncomplete",
    "RunInfo",
    "SortManifest",
    "RunStore",
    "DirectoryStore",
    "external_sorted",
    "spill_sorted_runs",
    "merge_sorted_runs",
    "resume_sorted",
    "validate_manifest",
    "discard_manifest",
    "load_manifest",
    "DEFAULT_RUN_SIZE",
]

#: Default maximum number of items buffered before a run is spilled
DEFAULT_RUN_SIZE = 1024

#: Manifest schema/version markers
MANIFEST_FORMAT = "asyncstdlib-external-sort/1"
MANIFEST_VERSION = 1

#: Lifecycle states of a :py:class:`SortManifest`
STATE_BUILDING = "building"
STATE_READY = "ready"
_STATES = frozenset((STATE_BUILDING, STATE_READY))


class ExternalSortError(Exception):
    """Base class for failures of the :py:mod:`~asyncstdlib.external` tools"""


class ManifestInvalid(ExternalSortError):
    """A run manifest is malformed, corrupt, or does not match its storage"""


class ManifestIncomplete(ManifestInvalid):
    """
    A run manifest is still ``"building"`` and must not be used for recovery

    Its referenced runs may be complete, but the source iterable had not been
    fully consumed when the spill was interrupted; merging would silently lose
    the tail. The only safe courses of action are to discard the artifacts
    (see :py:func:`discard_manifest`) or to restart the spill.
    """


@dataclass(frozen=True)
class RunInfo:
    """
    Description of one spilled, fully committed sorted run

    :param ref: opaque storage reference, as returned by
        :py:meth:`RunStore.allocate_run`; for portable manifests this should
        consist of basic JSON-compatible values only
    :param index: position of the run in spill order, starting at ``0``
    :param count: exact number of items stored in the run
    :param tag: optional integrity token produced by ``write_run``/``run_tag``,
        such as a checksum; :py:data:`None` if the store provides none
    """

    ref: Any
    index: int
    count: int
    tag: Optional[str] = None

    def to_data(self) -> Dict[str, Any]:
        """Return a plain JSON-compatible representation"""
        return {
            "ref": self.ref,
            "index": self.index,
            "count": self.count,
            "tag": self.tag,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RunInfo":
        """Create an instance from a :py:meth:`to_data` mapping"""
        try:
            index = data["index"]
            count = data["count"]
            tag = data["tag"]
            ref = data["ref"]
        except (KeyError, TypeError) as exc:
            raise ManifestInvalid(f"malformed run description: {data!r}") from exc
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ManifestInvalid(f"invalid run index {index!r}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ManifestInvalid(f"invalid run count {count!r}")
        if tag is not None and not isinstance(tag, str):
            raise ManifestInvalid(f"invalid run tag {tag!r}")
        return cls(ref=ref, index=index, count=count, tag=tag)


@dataclass(frozen=True)
class SortManifest:
    """
    Portable description of a set of spilled sorted runs

    Use :py:meth:`to_data`/:py:meth:`from_data` to export and restore manifests,
    for example as JSON. A manifest only references runs that were written
    completely. While ``state`` is :py:data:`STATE_BUILDING`, the spill may be
    missing tail items and the manifest must not be merged;
    :py:data:`STATE_READY` means every input item was spilled exactly once.
    """

    runs: Tuple[RunInfo, ...]
    reverse: bool = False
    run_size: int = DEFAULT_RUN_SIZE
    key_label: Optional[str] = None
    total: int = 0
    state: str = STATE_BUILDING
    version: int = MANIFEST_VERSION

    def to_data(self) -> Dict[str, Any]:
        """Return a plain JSON-compatible representation"""
        return {
            "format": MANIFEST_FORMAT,
            "version": self.version,
            "state": self.state,
            "reverse": self.reverse,
            "run_size": self.run_size,
            "key_label": self.key_label,
            "total": self.total,
            "runs": [run.to_data() for run in self.runs],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SortManifest":
        """
        Create and structurally validate an instance from exported data

        :raises ManifestInvalid: if the data is malformed, from an incompatible
            version, in an unknown state, or internally inconsistent
        """
        if not isinstance(data, Mapping):
            raise ManifestInvalid("manifest data must be a mapping")
        try:
            version = data["version"]
            state = data["state"]
            reverse = data["reverse"]
            run_size = data["run_size"]
            key_label = data["key_label"]
            total = data["total"]
            raw_runs = data["runs"]
        except (KeyError, TypeError) as exc:
            raise ManifestInvalid(f"malformed manifest: {data!r}") from exc
        if not isinstance(version, int) or isinstance(version, bool):
            raise ManifestInvalid(f"invalid manifest version {version!r}")
        if version != MANIFEST_VERSION:
            raise ManifestInvalid(
                f"unsupported manifest version {version!r}; expected {MANIFEST_VERSION}"
            )
        if state not in _STATES:
            raise ManifestInvalid(f"invalid manifest state {state!r}")
        if not isinstance(reverse, bool):
            raise ManifestInvalid(f"invalid reverse flag {reverse!r}")
        if (
            not isinstance(run_size, int)
            or isinstance(run_size, bool)
            or run_size < 1
        ):
            raise ManifestInvalid(f"invalid run_size {run_size!r}")
        if key_label is not None and not isinstance(key_label, str):
            raise ManifestInvalid(f"invalid key_label {key_label!r}")
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise ManifestInvalid(f"invalid total {total!r}")
        try:
            run_data = tuple(raw_runs)
        except TypeError as exc:
            raise ManifestInvalid("manifest 'runs' must be a sequence") from exc
        runs = tuple(
            RunInfo.from_data(entry) if isinstance(entry, Mapping) else _bad_run(entry)
            for entry in run_data
        )
        runs = tuple(sorted(runs, key=lambda run: run.index))
        for position, run in enumerate(runs):
            if run.index != position:
                raise ManifestInvalid(
                    f"run indices must be contiguous, got index {run.index} "
                    f"at position {position}"
                )
        counted = sum(run.count for run in runs)
        if counted != total:
            raise ManifestInvalid(
                f"manifest total {total} does not match run counts {counted}"
            )
        return cls(
            runs=runs,
            reverse=reverse,
            run_size=run_size,
            key_label=key_label,
            total=total,
            state=state,
            version=version,
        )


def _bad_run(entry: Any) -> RunInfo:
    raise ManifestInvalid(f"malformed run description: {entry!r}")


@runtime_checkable
class RunStore(Protocol[T]):
    """
    Storage backend for spilled runs

    Every method may be a regular or an ``async def`` callable;
    regular implementations perform their work synchronously
    (the reference :py:class:`DirectoryStore` does) and should be wrapped with
    the event loop's thread-offload helper when blocking I/O is unacceptable.

    Implementations must provide four operations:

    ``allocate_run(index) -> ref``
        Return a *new*, unused, JSON-compatible storage reference for run
        ``index``. The reference must remain writable until ``write_run``
        returns and must not alias any existing run.

    ``write_run(ref, run) -> tag | None``
        Persist the already-sorted sequence ``run`` at ``ref``.
        This must be *all-or-nothing*: write to a staging location and publish
        ``ref`` atomically (e.g. temp file plus ``os.replace``) so that a
        serialization failure, storage error, or cancellation leaves no
        readable run at ``ref``. May return an integrity tag such as a
        checksum; the same value must be returned by ``run_tag``.

    ``read_run(ref) -> (async) iter T``
        Return an (async) iterable that yields exactly the items stored,
        in their sorted order.

    ``delete_run(ref)``
        Remove the run at ``ref``; a missing run must not be an error.

    Stores may additionally implement ``run_tag(ref)`` for checksum
    validation, and ``save_manifest``/``load_manifest``/``delete_manifest``
    for checkpointing (see :py:func:`spill_sorted_runs`).
    """

    def allocate_run(self, index: int) -> Any:
        raise NotImplementedError

    def write_run(self, ref: Any, run: Sequence[T]) -> Any:
        raise NotImplementedError

    def read_run(self, ref: Any) -> AnyIterable[T]:
        raise NotImplementedError

    def delete_run(self, ref: Any) -> Any:
        raise NotImplementedError


async def _call(store: Any, method: str, *args: Any) -> Any:
    """Invoke a possibly-async store method"""
    member = getattr(store, method, None)
    if not callable(member):
        raise TypeError(f"run store {store!r} does not provide {method!r}")
    return await awaitify(member)(*args)


async def _swallow(store: Any, method: str, *args: Any) -> None:
    """Best-effort store cleanup; never masks the in-flight error"""
    try:
        await _call(store, method, *args)
    except Exception:
        pass


def _coerce(manifest: Union[SortManifest, Mapping[str, Any]]) -> SortManifest:
    if isinstance(manifest, SortManifest):
        return manifest
    return SortManifest.from_data(manifest)


def _default_label(key: Optional[Callable[..., Any]]) -> Optional[str]:
    if key is None:
        return None
    return getattr(key, "__qualname__", None) or getattr(key, "__name__", None)


async def _identity(item: T) -> T:
    return item


async def _commit_run(
    store: RunStore[Any],
    keyed: List[Tuple[Any, T]],
    index: int,
    reverse: bool,
) -> RunInfo:
    """Sort one buffer in place and publish it as a single committed run"""
    # Runs are stored in the requested output direction, so the merge only
    # ever pulls items that are safe to emit. Python's sort is stable for
    # equal keys in both directions, preserving encounter order.
    keyed.sort(key=lambda pair: pair[0], reverse=reverse)
    ref = await _call(store, "allocate_run", index)
    try:
        tag = await _call(
            store, "write_run", ref, [item for _, item in keyed]
        )
    except BaseException:
        # The store must already have staged-and-discarded itself; removing the
        # ref as well guarantees a failed publish can never look committed.
        await _swallow(store, "delete_run", ref)
        raise
    if tag is not None and not isinstance(tag, str):
        raise TypeError(f"write_run returned a non-string tag {tag!r}")
    return RunInfo(ref=ref, index=index, count=len(keyed), tag=tag)


def _build_manifest(
    runs: Sequence[RunInfo],
    total: int,
    *,
    reverse: bool,
    run_size: int,
    key_label: Optional[str],
    state: str,
) -> SortManifest:
    return SortManifest(
        runs=tuple(runs),
        reverse=reverse,
        run_size=run_size,
        key_label=key_label,
        total=total,
        state=state,
    )


@dataclass
class _SpillOutcome(Generic[T]):
    items: Optional[List[T]] = None
    manifest: Optional[SortManifest] = None


async def _spill_runs(
    iterable: AnyIterable[T],
    *,
    store: RunStore[T],
    key: Callable[[T], Awaitable[Any]],
    reverse: bool,
    run_size: int,
    key_label: Optional[str],
    checkpoint: bool,
    force_spill: bool,
) -> _SpillOutcome[T]:
    if run_size < 1:
        raise ValueError(f"run_size must be >= 1, got {run_size!r}")
    if checkpoint:
        for hook in ("save_manifest", "load_manifest", "delete_manifest"):
            if not callable(getattr(store, hook, None)):
                raise TypeError(
                    f"checkpoint=True requires the store to implement {hook!r}"
                )
    buffer: List[Tuple[Any, T]] = []
    runs: List[RunInfo] = []
    total = 0
    spilled = False
    try:
        async for item in aiter(iterable):
            # Once the budget is full, spill before accepting another item so
            # that runs hold exactly ``run_size`` items; an input that fits
            # entirely into one budget never touches the store.
            if len(buffer) >= run_size:
                runs.append(
                    await _commit_run(store, buffer, len(runs), reverse)
                )
                total += len(buffer)
                spilled = True
                buffer = []
                if checkpoint:
                    await _call(
                        store,
                        "save_manifest",
                        _build_manifest(
                            runs,
                            total,
                            reverse=reverse,
                            run_size=run_size,
                            key_label=key_label,
                            state=STATE_BUILDING,
                        ).to_data(),
                    )
            buffer.append((await key(item), item))
        if not spilled and not force_spill:
            buffer.sort(key=lambda pair: pair[0], reverse=reverse)
            return _SpillOutcome(items=[item for _, item in buffer])
        if buffer:
            runs.append(
                await _commit_run(store, buffer, len(runs), reverse)
            )
            total += len(buffer)
            buffer = []
        manifest = _build_manifest(
            runs,
            total,
            reverse=reverse,
            run_size=run_size,
            key_label=key_label,
            state=STATE_READY,
        )
        if checkpoint:
            await _call(store, "save_manifest", manifest.to_data())
        return _SpillOutcome(manifest=manifest)
    except BaseException:
        # Serialization failure, storage error, or cancellation: nothing that
        # was written may be left looking like a recoverable checkpoint.
        for info in runs:
            await _swallow(store, "delete_run", info.ref)
        if checkpoint:
            await _swallow(store, "delete_manifest")
        raise


async def external_sorted(
    iterable: AnyIterable[T],
    *,
    store: RunStore[T],
    key: Optional[Callable[[T], Any]] = None,
    reverse: bool = False,
    run_size: int = DEFAULT_RUN_SIZE,
    key_label: Optional[str] = None,
    checkpoint: bool = False,
) -> AsyncIterator[T]:
    """
    Yield the items of ``iterable`` in sorted order, spilling runs to ``store``

    Items are buffered and sorted in memory until more than ``run_size`` items
    are held; the sorted buffer is then spilled via ``store`` and a new buffer
    is started. Once the input is exhausted, all runs (and any in-memory tail)
    are merged with a stable :math:`k`-way merge.

    The optional ``key`` argument specifies a one-argument regular or async
    callable providing the sort order, like :py:meth:`list.sort`'s ``key``.
    Use ``reverse=True`` for descending order; ties keep input order.

    If everything fits into ``run_size`` items, no run is written.
    Otherwise, all spilled runs are deleted again once iteration finishes,
    fails, or is closed/cancelled — this function owns its temporary artifacts.
    To keep runs for later resumption, use :py:func:`spill_sorted_runs` and
    :py:func:`merge_sorted_runs` directly.

    :param checkpoint: persist the manifest after every committed run using
        the store's ``save_manifest``/``delete_manifest`` hooks. Checkpoints
        stay in state ``"building"`` until the spill completes; on failure or
        cancellation they are removed together with all runs.
    """
    akey: Callable[[T], Awaitable[Any]] = (
        _identity if key is None else awaitify(key)
    )
    label = key_label if key_label is not None else _default_label(key)
    outcome = await _spill_runs(
        iterable,
        store=store,
        key=akey,
        reverse=reverse,
        run_size=run_size,
        key_label=label,
        checkpoint=checkpoint,
        force_spill=False,
    )
    if outcome.items is not None:
        for item in outcome.items:
            yield item
        return
    assert outcome.manifest is not None
    manifest = outcome.manifest
    try:
        async for item in merge_sorted_runs(
            manifest, store=store, key=key, reverse=reverse
        ):
            yield item
    finally:
        await discard_manifest(
            manifest, store=store, delete_manifest=checkpoint
        )


async def spill_sorted_runs(
    iterable: AnyIterable[T],
    *,
    store: RunStore[T],
    key: Optional[Callable[[T], Any]] = None,
    reverse: bool = False,
    run_size: int = DEFAULT_RUN_SIZE,
    key_label: Optional[str] = None,
    checkpoint: bool = False,
) -> SortManifest:
    """
    Consume ``iterable`` and spill all of it as committed sorted runs

    Unlike :py:func:`external_sorted`, this always spills — even inputs that
    fit into memory become one run (and an empty input yields an empty, ready
    manifest). The returned :py:class:`SortManifest` is ``"ready"`` and owns
    no artifacts; the caller keeps the runs and later merges them with
    :py:func:`merge_sorted_runs` or :py:func:`resume_sorted`,
    or removes them with :py:func:`discard_manifest`.

    On serialization failure, storage error, or cancellation every committed
    run — and any checkpoint — is deleted before the exception propagates,
    so the store never contains runs that look like a finished spill.
    """
    akey: Callable[[T], Awaitable[Any]] = (
        _identity if key is None else awaitify(key)
    )
    label = key_label if key_label is not None else _default_label(key)
    outcome = await _spill_runs(
        iterable,
        store=store,
        key=akey,
        reverse=reverse,
        run_size=run_size,
        key_label=label,
        checkpoint=checkpoint,
        force_spill=True,
    )
    assert outcome.manifest is not None
    return outcome.manifest


async def _read_run(store: RunStore[T], ref: Any) -> AsyncIterator[T]:
    """Yield one run and always release its underlying iterator"""
    raw = await _call(store, "read_run", ref)
    iterator = aiter(raw)
    try:
        async for item in iterator:
            yield item
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()
        # Regular iterables/generators may own resources such as open files.
        if raw is not iterator:
            close = getattr(raw, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


class _RunHead(Generic[T]):
    """
    One buffered head item of a run for the merge heap

    Runs are stored in the requested output direction, so the heap orders by
    key in that same direction. Equal keys are always resolved by the run's
    spill position (lower index first), which makes the merge stable across
    reloaded runs — including descending output.
    """

    __slots__ = ("key", "run_index", "item", "tail", "reverse")

    def __init__(
        self,
        key: Any,
        run_index: int,
        item: T,
        tail: AsyncIterator[T],
        reverse: bool,
    ) -> None:
        self.key = key
        self.run_index = run_index
        self.item = item
        self.tail = tail
        self.reverse = reverse

    def __lt__(self, other: "_RunHead[T]") -> bool:
        key_less = bool(self.key < other.key)
        key_greater = bool(other.key < self.key)
        if not key_less and not key_greater:
            return self.run_index < other.run_index
        return key_greater if self.reverse else key_less


async def _stable_merge(
    readers: List[AsyncIterator[T]],
    *,
    key: Callable[[T], Awaitable[Any]],
    reverse: bool,
) -> AsyncIterator[T]:
    """Stable :math:`k`-way merge of ascending run iterators"""
    heap: List[_RunHead[T]] = []
    for run_index, reader in enumerate(readers):
        iterator = aiter(reader)
        try:
            first = await iterator.__anext__()
        except StopAsyncIteration:
            continue
        heap.append(
            _RunHead(await key(first), run_index, first, iterator, reverse)
        )
    if not heap:
        return
    _heapq.heapify(heap)
    try:
        while heap:
            head = heap[0]
            yield head.item
            try:
                nxt = await head.tail.__anext__()
            except StopAsyncIteration:
                _heapq.heappop(heap)
            else:
                head.item = nxt
                head.key = await key(nxt)
                _heapq.heapreplace(heap, head)
    finally:
        for queued in heap:
            aclose = getattr(queued.tail, "aclose", None)
            if aclose is not None:
                await aclose()


async def merge_sorted_runs(
    manifest: Union[SortManifest, Mapping[str, Any]],
    *,
    store: RunStore[T],
    key: Optional[Callable[[T], Any]] = None,
    reverse: Optional[bool] = None,
) -> AsyncIterator[T]:
    """
    Stably merge the runs described by ``manifest`` from ``store``

    The manifest must be in state ``"ready"``. Ties are resolved by spill
    order and then by position inside the run, which reproduces a stable
    in-memory sort even after every item has been reloaded.
    Keys are recomputed on reload via the regular or async ``key`` callable,
    so the same key function used for spilling must be supplied.

    :param reverse: :py:data:`None` (default) uses the manifest's ``reverse``
        flag; an explicit value must match it
    """
    ready = _coerce(manifest)
    if ready.state != STATE_READY:
        raise ManifestIncomplete(
            f"manifest is {ready.state!r}, not {STATE_READY!r}; "
            "discard it or restart the spill instead of merging"
        )
    if reverse is None:
        reverse = ready.reverse
    elif reverse != ready.reverse:
        raise ValueError(
            f"reverse={reverse!r} does not match the manifest's "
            f"reverse={ready.reverse!r}"
        )
    akey: Callable[[T], Awaitable[Any]] = (
        _identity if key is None else awaitify(key)
    )
    readers = [_read_run(store, info.ref) for info in ready.runs]
    if not readers:
        return
    async for item in _stable_merge(readers, key=akey, reverse=reverse):
        yield item


async def validate_manifest(
    manifest: Union[SortManifest, Mapping[str, Any]],
    *,
    store: RunStore[T],
    reverse: Optional[bool] = None,
    key_label: Optional[str] = None,
) -> None:
    """
    Check that ``manifest`` is ready and every referenced run is readable

    Every run is read back in full to ensure it deserializes and contains
    exactly its declared item count; if the store implements ``run_tag`` and
    the run carries a tag, the tag (e.g. checksum) is recomputed and compared.
    All runs are checked even after the first problem, so a single report can
    list every defect.

    :raises ManifestIncomplete: if the manifest is still ``"building"``
    :raises ManifestInvalid: if any run is missing, unreadable, truncated,
        miscounted, corrupt, or the expected ``reverse``/``key_label`` differ
    """
    ready = _coerce(manifest)
    if ready.state != STATE_READY:
        raise ManifestIncomplete(
            f"manifest is {ready.state!r}, not {STATE_READY!r}"
        )
    problems: List[str] = []
    if reverse is not None and reverse != ready.reverse:
        problems.append(
            f"reverse={reverse!r} does not match manifest reverse={ready.reverse!r}"
        )
    if key_label is not None and key_label != ready.key_label:
        problems.append(
            f"key_label={key_label!r} does not match manifest "
            f"key_label={ready.key_label!r}"
        )
    tag_method = getattr(store, "run_tag", None)
    for info in ready.runs:
        seen = 0
        try:
            async for _item in _read_run(store, info.ref):
                seen += 1
        except Exception as exc:
            problems.append(f"run {info.index} ({info.ref!r}) unreadable: {exc}")
            continue
        if seen != info.count:
            problems.append(
                f"run {info.index} ({info.ref!r}) declares {info.count} "
                f"items but contains {seen}"
            )
        if info.tag is not None and callable(tag_method):
            try:
                actual = await awaitify(tag_method)(info.ref)
            except Exception as exc:
                problems.append(
                    f"run {info.index} ({info.ref!r}) tag unreadable: {exc}"
                )
                continue
            if actual != info.tag:
                problems.append(
                    f"run {info.index} ({info.ref!r}) integrity tag mismatch: "
                    f"expected {info.tag!r}, got {actual!r}"
                )
    if problems:
        raise ManifestInvalid("invalid manifest:\n- " + "\n- ".join(problems))


async def resume_sorted(
    manifest: Union[SortManifest, Mapping[str, Any]],
    *,
    store: RunStore[T],
    key: Optional[Callable[[T], Any]] = None,
    reverse: Optional[bool] = None,
    key_label: Optional[str] = None,
    validate: bool = True,
) -> AsyncIterator[T]:
    """
    Recover a sorted stream from an exported manifest after an interruption

    By default :py:func:`validate_manifest` runs first, so a corrupt or
    incomplete checkpoint raises before any item is yielded. Pass
    ``validate=False`` to merge without storage checks. This function does not
    delete anything; pair it with :py:func:`discard_manifest` for cleanup.
    """
    ready = _coerce(manifest)
    if validate:
        await validate_manifest(
            ready, store=store, reverse=reverse, key_label=key_label
        )
    async for item in merge_sorted_runs(
        ready, store=store, key=key, reverse=reverse
    ):
        yield item


async def discard_manifest(
    manifest: Union[SortManifest, Mapping[str, Any]],
    *,
    store: RunStore[T],
    delete_manifest: bool = False,
) -> None:
    """
    Safely delete every run referenced by ``manifest``

    All runs are deleted even if earlier deletions fail, and deleting a
    missing run is not an error. With ``delete_manifest=True`` the store's
    checkpointed manifest is removed as well.

    :raises ExternalSortError: if one or more deletions failed; the runs that
        could not be removed are listed with their errors
    """
    described = _coerce(manifest)
    failures: List[str] = []
    for info in described.runs:
        try:
            await _call(store, "delete_run", info.ref)
        except Exception as exc:
            failures.append(f"run {info.index} ({info.ref!r}): {exc}")
    if delete_manifest:
        try:
            await _call(store, "delete_manifest")
        except Exception as exc:
            failures.append(f"manifest: {exc}")
    if failures:
        raise ExternalSortError(
            "could not discard all artifacts:\n- " + "\n- ".join(failures)
        )


async def load_manifest(
    store: RunStore[T],
) -> Optional[SortManifest]:
    """
    Load a checkpointed manifest from ``store``

    Requires the optional ``load_manifest`` store hook. Returns :py:data:`None`
    if no checkpoint exists. The data is structurally parsed but not yet
    validated against storage; use :py:func:`validate_manifest` for that.
    """
    method = getattr(store, "load_manifest", None)
    if not callable(method):
        raise TypeError("store does not implement the load_manifest hook")
    data = await awaitify(method)()
    if data is None:
        return None
    return SortManifest.from_data(data)


# ---------------------------------------------------------------------------
# Reference filesystem store
# ---------------------------------------------------------------------------

_LENGTH_HEADER = 8


class DirectoryStore(Generic[T]):
    """
    Reference :py:class:`RunStore` keeping runs as files in one directory

    Each run is a file whose name is returned by :py:meth:`allocate_run`.
    Items are framed with an 8-byte big-endian length header and serialized
    with the injectable ``dumps``/``loads`` pair (default :py:mod:`pickle`),
    so any self-contained object codec works.

    Writes go to a unique ``.part-*`` staging file and are published with
    :py:func:`os.replace`; a failed or interrupted write therefore never
    appears at its final name. The integrity tag is ``"sha256:<digest>:<n>"``
    over the framed file content. The manifest hooks store JSON atomically as
    ``manifest.json``.

    All methods are synchronous and perform blocking filesystem I/O; wrap the
    store or offload it when use with an event loop requires non-blocking I/O.
    The directory itself is provided (and, if desired, cleaned up) by the
    caller, e.g. via :py:class:`tempfile.TemporaryDirectory`.
    """

    __slots__ = ("_directory", "_dumps", "_loads", "_suffix", "_manifest_name")

    def __init__(
        self,
        directory: Union[str, _Path],
        *,
        dumps: Callable[[T], bytes] = _pickle.dumps,
        loads: Callable[[bytes], T] = _pickle.loads,
        suffix: str = ".run",
        manifest_name: str = "manifest.json",
    ) -> None:
        self._directory = _Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._dumps = dumps
        self._loads = loads
        if not suffix or _os.sep in suffix or "/" in suffix or "\\" in suffix:
            raise ValueError(f"invalid suffix {suffix!r}")
        self._suffix = suffix
        self._manifest_name = manifest_name

    def __repr__(self) -> str:
        return f"asyncstdlib.external.DirectoryStore({str(self._directory)!r})"

    def _resolve(self, ref: Any) -> _Path:
        # Manifests may come from outside the process; keep references confined
        # to the store directory so they cannot escape via "../" or absolute
        # paths.
        if (
            not isinstance(ref, str)
            or not ref
            or ref.startswith(".")
            or "/" in ref
            or "\\" in ref
        ):
            raise ManifestInvalid(f"unsafe or invalid run reference {ref!r}")
        path = self._directory / ref
        if _os.path.commonpath((str(self._directory), str(path))) != str(
            self._directory
        ):
            raise ManifestInvalid(f"run reference escapes store: {ref!r}")
        return path

    def allocate_run(self, index: int) -> str:
        return f"run-{index:08d}-{_uuid.uuid4().hex}{self._suffix}"

    def write_run(self, ref: str, run: Sequence[T]) -> str:
        target = self._resolve(ref)
        staging = self._directory / f".{ref}.part-{_uuid.uuid4().hex}"
        digest = _hashlib.sha256()
        count = 0
        try:
            with open(staging, "wb") as stream:
                for item in run:
                    payload = self._dumps(item)
                    header = len(payload).to_bytes(_LENGTH_HEADER, "big")
                    stream.write(header)
                    stream.write(payload)
                    digest.update(header)
                    digest.update(payload)
                    count += 1
                stream.flush()
                _os.fsync(stream.fileno())
            _os.replace(staging, target)
        except Exception:
            try:
                _os.unlink(staging)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            raise
        return f"sha256:{digest.hexdigest()}:{count}"

    def read_run(self, ref: str) -> Iterator[T]:
        path = self._resolve(ref)
        with open(path, "rb") as stream:
            while True:
                header = stream.read(_LENGTH_HEADER)
                if not header:
                    break
                if len(header) != _LENGTH_HEADER:
                    raise ValueError(f"truncated run {ref!r}: short length header")
                length = int.from_bytes(header, "big")
                payload = stream.read(length)
                if len(payload) != length:
                    raise ValueError(f"truncated run {ref!r}: short item payload")
                yield self._loads(payload)

    def delete_run(self, ref: str) -> None:
        try:
            self._resolve(ref).unlink()
        except FileNotFoundError:
            return

    def run_tag(self, ref: str) -> str:
        digest = _hashlib.sha256()
        count = 0
        with open(self._resolve(ref), "rb") as stream:
            while True:
                header = stream.read(_LENGTH_HEADER)
                if not header:
                    break
                if len(header) != _LENGTH_HEADER:
                    raise ValueError(f"truncated run {ref!r}: short length header")
                digest.update(header)
                length = int.from_bytes(header, "big")
                payload = stream.read(length)
                if len(payload) != length:
                    raise ValueError(f"truncated run {ref!r}: short item payload")
                digest.update(payload)
                count += 1
        return f"sha256:{digest.hexdigest()}:{count}"

    def save_manifest(self, data: Mapping[str, Any]) -> None:
        target = self._directory / self._manifest_name
        staging = self._directory / f".{self._manifest_name}.part-{_uuid.uuid4().hex}"
        try:
            with open(staging, "w", encoding="utf-8") as stream:
                _json.dump(data, stream, ensure_ascii=False, sort_keys=True)
                stream.flush()
                _os.fsync(stream.fileno())
            _os.replace(staging, target)
        except Exception:
            try:
                _os.unlink(staging)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            raise

    def load_manifest(self) -> Optional[Dict[str, Any]]:
        path = self._directory / self._manifest_name
        try:
            with open(path, "r", encoding="utf-8") as stream:
                data: Optional[Dict[str, Any]] = _json.load(stream)
        except FileNotFoundError:
            return None
        return data

    def delete_manifest(self) -> None:
        try:
            (self._directory / self._manifest_name).unlink()
        except FileNotFoundError:
            return

    def cleanup_staging(self) -> int:
        """
        Remove staging files left by a process killed mid-write

        Staging files are never referenced by manifests, so this is safe to
        run at any time. Returns the number of files removed.
        """
        removed = 0
        for staging in self._directory.glob(".*.part-*"):
            try:
                _os.unlink(staging)
            except FileNotFoundError:
                continue
            removed += 1
        return removed
