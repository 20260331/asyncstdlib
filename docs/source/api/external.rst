============================
External sorting with spills
============================

.. py:module:: asyncstdlib.external
    :synopsis: resumable external sorting with pluggable run storage

The :py:mod:`asyncstdlib.external` module is an *optional* toolset:
it is not re-exported from the :py:mod:`asyncstdlib` namespace and has to be
imported explicitly, e.g. ``from asyncstdlib import external``.

It sorts (async) iterables that do not fit into a fixed memory budget.
Items are buffered and sorted in memory up to ``run_size`` items;
once another item would exceed the budget, the sorted buffer is serialized
through a caller-provided :py:class:`RunStore` and written to temporary storage.
After the input is consumed, all spilled runs are combined with a stable
:math:`k`-way merge. The ``key`` callable may be regular or ``async``.

.. versionadded:: 3.14.1

.. code-block:: python3

    import tempfile
    from asyncstdlib import external

    with tempfile.TemporaryDirectory() as directory:
        store = external.DirectoryStore(directory)
        ordered = [
            item
            async for item in external.external_sorted(
                huge_async_iterable, store=store, run_size=10_000
            )
        ]

Crash safety and resumability
=============================

A spilled run becomes visible only after it has been written completely;
stores are expected to stage writes and publish them atomically.
All committed runs are described by a portable :py:class:`SortManifest`
whose ``state`` is either ``"building"`` (the spill is still in progress)
or ``"ready"`` (every input item has been committed exactly once).
A manifest therefore never points at a half-written run.

* Serialization failures, storage errors, and cancellations during a spill
  delete every run committed by that spill (and any checkpoint) before the
  exception propagates.
* A process that is killed hard may leave a ``"building"`` checkpoint and
  unreferenced staging files. :py:func:`validate_manifest` rejects such a
  checkpoint with :py:class:`ManifestIncomplete`; :py:func:`discard_manifest`
  removes the referenced runs, and :py:meth:`DirectoryStore.cleanup_staging`
  removes staging files.
* A ``"ready"`` manifest can be exported with :py:meth:`SortManifest.to_data`,
  persisted as JSON, and used in another process to
  :py:func:`validate_manifest`, :py:func:`resume_sorted`, or
  :py:func:`discard_manifest` the runs.

Store interface
===============

.. autoclass:: RunStore[T]
    :members:

.. autoclass:: DirectoryStore[T]
    :members:

Manifests and errors
====================

.. autoclass:: RunInfo
    :members:

.. autoclass:: SortManifest
    :members:

.. autoclass:: ExternalSortError

.. autoclass:: ManifestInvalid

.. autoclass:: ManifestIncomplete

Sorting, recovery and cleanup
=============================

.. autofunction:: external_sorted(iterable: (async) iter T, *, store: RunStore[T], key: (T) → (await) Any = None, reverse: bool = False, run_size: int = 1024, key_label: str = None, checkpoint: bool = False)
    :async-for: :T

.. autofunction:: spill_sorted_runs(iterable: (async) iter T, *, store: RunStore[T], key: (T) → (await) Any = None, reverse: bool = False, run_size: int = 1024, key_label: str = None, checkpoint: bool = False) -> SortManifest

.. autofunction:: merge_sorted_runs(manifest: SortManifest | mapping, *, store: RunStore[T], key: (T) → (await) Any = None, reverse: bool = None)
    :async-for: :T

.. autofunction:: resume_sorted(manifest: SortManifest | mapping, *, store: RunStore[T], key: (T) → (await) Any = None, reverse: bool = None, key_label: str = None, validate: bool = True)
    :async-for: :T

.. autofunction:: validate_manifest(manifest: SortManifest | mapping, *, store: RunStore[T], reverse: bool = None, key_label: str = None) -> None

.. autofunction:: discard_manifest(manifest: SortManifest | mapping, *, store: RunStore[T], delete_manifest: bool = False) -> None

.. autofunction:: load_manifest(store: RunStore[T]) -> SortManifest | None
