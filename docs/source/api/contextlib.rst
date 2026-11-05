======================
The contextlib library
======================

.. py:module:: asyncstdlib.contextlib
    :synopsis: async contextlib variants

The :py:mod:`asyncstdlib.contextlib` library implements
Python's :py:mod:`contextlib` for (async) iterables and (async) context managers.

Context Managers
================

.. py:class:: AbstractContextManager

    An :term:`abstract base class` for asynchronous context managers

    This class can be used to check whether some object is an
    asynchronous context manager. A class may inherit from
    ``AbstractContextManager``, in which case it must implement
    an ``__aenter__`` method; the default ``__aenter__`` returns
    the asynchronous context manager itself.

    .. versionadded:: 1.1.0

.. autoclass:: ContextDecorator[T]

.. py:function:: contextmanager(func: (...) → async iter T) (...)
    :async-with: :T
    :noindex:

.. autofunction:: contextmanager(func: (...) → async iter T) -> (...) → async with T
    :decorator:

    .. versionadded:: 3.12.2

        The created context manager is a :py:class:`~.ContextDecorator`.

.. autofunction:: closing(thing: AC)
    :async-with: :AC

.. autofunction:: nullcontext(enter_result: T)
    :async-with: :T

.. autoclass:: ExitStack

    .. automethod:: enter_context(cm: (async) with T) -> T

    .. automethod:: callback(callback: T as (*args, **kwargs) -> None, *args, **kwargs) -> T

    .. py:method:: push(exit: T as {.__aexit__}) -> T
        :noindex:

    .. py:method:: push(exit: T as {.__exit__}) -> T
        :noindex:

    .. automethod:: push(exit: T as (Type[BaseException], BaseException, traceback) -> (await) bool) -> T

    .. automethod:: pop_all

    .. automethod:: checkpoint

    .. automethod:: aclose

    .. versionadded:: 1.1.0

.. autoclass:: ExitStackCheckpoint

    An opt-in group of exit actions created by :py:meth:`ExitStack.checkpoint`.
    It supports the same :py:meth:`~ExitStack.push`,
    :py:meth:`~ExitStack.callback`, :py:meth:`~ExitStack.enter_context`,
    :py:meth:`~ExitStack.pop_all`, :py:meth:`~ExitStack.checkpoint` and
    :py:meth:`~ExitStack.aclose` methods as an :py:class:`ExitStack`.

    Exiting the checkpoint block normally **commits** its staged actions
    to the surrounding stack in registration order, so the surrounding
    stack runs them exactly once in its usual LIFO unwinding.
    Exiting it with an exception **rolls back** the staged actions
    immediately in reverse registration order, preserving the exception
    propagation and suppression semantics of an :py:class:`ExitStack`;
    the surrounding stack never runs them.

    A checkpoint is finalized at most once. After committing, rolling back
    or closing it, any attempt to register or transfer actions raises
    :py:exc:`RuntimeError`.

    .. automethod:: aclose

    .. versionadded:: 3.15.0