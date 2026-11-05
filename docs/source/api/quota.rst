=================
Iteration Quotas
=================

.. py:module:: asyncstdlib.quota
    :synopsis: cooperative quotas for async iteration

The :py:class:`~asyncstdlib.Quota` context manager optionally bounds how much
work the iterators and combinators of :py:mod:`asyncstdlib` perform.
It is intended for cooperative iteration, for example to cap the work spent
on a single request or to make streaming pipelines fail in a predictable,
resource-safe way.

Basic usage
===========

Enter a :py:class:`~asyncstdlib.Quota` like any async context manager;
every *leaf* iterator opened while the quota is active cooperatively consumes
its budgets. Internal combinators only relay pulls of their sources, so each
source element is counted exactly once regardless of how many combinators
process it::

    import asyncstdlib as a

    async with a.Quota(items=100, on_exhausted="stop"):
        # at most 100 elements traverse the whole pipeline
        results = await a.list(
            a.map(process, a.filter(is_relevant, records))
        )

The quota applies to iterators **opened while it is active**.
Iterators opened before entering the context are not metered.
Without an active quota all behaviour is unchanged and no overhead is added.

Budgets
=======

A quota can limit four independent budgets; every limit is optional and
:py:data:`None` means unlimited.

``items``
    The number of elements pulled from leaf (async) iterables.
    A multi-source combinator such as :py:func:`~asyncstdlib.zip` or
    :py:func:`~asyncstdlib.compress` consumes one item per source element.

``awaits``
    The number of asynchronous steps: each pull of an async iterator and
    each invocation of an *async* callable argument (mappers, predicates,
    keys, reducers) consumes one await step.
    Synchronous callables do not consume await steps.
    The await step of a pull is charged *before* advancing the source, so an
    exhausted await quota never consumes an additional source element.

``buffered``
    The number of elements that combinators may retain at the same time.
    This covers in particular the buffers of
    :py:func:`~asyncstdlib.cycle` and :py:func:`~asyncstdlib.tee`, the pending
    batches of :py:func:`~asyncstdlib.batched`, the look-behind element of
    :py:func:`~asyncstdlib.pairwise`, the current value of
    :py:func:`~asyncstdlib.groupby`, the per-source heads buffered by
    :py:func:`~asyncstdlib.heapq.merge` and the result buffer of
    :py:func:`~asyncstdlib.heapq.nlargest`/:py:func:`~asyncstdlib.nsmallest`.
    Buffered slots are held only while the elements are actually retained.

``weight``
    A user defined cost computed per element via ``weight_of``.
    The callable may be regular or async; if only ``weight`` is given, every
    element has a weight of ``1``.

Exhaustion
==========

What happens when a budget is exhausted depends on ``on_exhausted``:

``"raise"`` (default)
    Raises :py:class:`~asyncstdlib.QuotaExceeded` describing the exhausted
    budget, the configured limit and the amount already used.
    The exception unwinds the whole pipeline, and every iterator opened by
    :py:mod:`asyncstdlib` for that pipeline is closed deterministically.

``"stop"``
    Ends the affected iterators as if their sources were exhausted, so that
    consumers receive partial results:
    :py:func:`~asyncstdlib.list`/``tuple``/``set``/``dict`` return what was
    collected, reducers such as :py:func:`~asyncstdlib.reduce`,
    :py:func:`~asyncstdlib.min`/:py:func:`~asyncstdlib.max` and
    :py:func:`~asyncstdlib.sorted` return partial results, and
    :py:func:`~asyncstdlib.heapq.nlargest` returns the best partial result.
    As with raising, all opened iterators are closed once iteration stops.

Once any budget is exhausted, the quota stays exhausted;
inspect it via :py:attr:`~asyncstdlib.Quota.exhausted`,
:py:attr:`~asyncstdlib.Quota.exhausted_budget`,
:py:attr:`~asyncstdlib.Quota.consumed`, :py:attr:`~asyncstdlib.Quota.held`
and :py:attr:`~asyncstdlib.Quota.remaining`.

Nested quotas
=============

Nested quota contexts follow two explicit rules:

* entering a **different** quota **isolates** the nested scope: only the
  innermost quota is consumed, the outer quota is suspended and resumes when
  the nested block exits;
* re-entering the **same** quota instance, or passing it explicitly,
  **shares** its budgets. The same quota can thus be shared by sibling
  combinators in one block and by nested or concurrently running blocks.

Manual use
==========

Custom code can participate in the active quota via
:py:meth:`~asyncstdlib.Quota.consume` and :py:meth:`~asyncstdlib.Quota.hold`.
Manual requests always raise :py:class:`~asyncstdlib.QuotaExceeded`, even on a
quota configured to ``"stop"``, since custom code decides itself how partial
results are handled.

.. autoclass:: Quota
    :members:

.. autoclass:: HoldToken
    :members:

.. autoexception:: QuotaExceeded
    :members:
