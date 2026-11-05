from functools import partial
from contextlib import contextmanager

import pytest

import asyncstdlib as a

from .utility import sync


@sync
async def test_closing():
    class ACloseable:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    async with a.closing(ACloseable()) as async_closeable:
        pass
    assert async_closeable.closed


@sync
async def test_contextmanager():
    @a.contextmanager
    async def valid(value):
        yield value

    async with valid(1337) as context:
        assert context == 1337


@sync
async def test_contextmanager_decorator():
    witness: list[str] = []

    @a.contextmanager
    async def decorator():
        witness.append("enter")
        yield
        witness.append("exit")

    @decorator()
    async def func():
        witness.append("inner")

    for repetition in range(1, 4):
        await func()
        assert witness == ["enter", "inner", "exit"] * repetition


@sync
async def test_contextmanager_no_yield():
    """Test that it is an error for a context to not yield"""

    @a.contextmanager
    async def no_yield():
        if False:
            yield

    with pytest.raises(RuntimeError):
        async with no_yield():
            assert False


@sync
async def test_contextmanager_no_stop():
    """Test that it is an error for a context to yield again after stopping"""

    @a.contextmanager
    async def no_stop():
        yield
        yield

    with pytest.raises(RuntimeError):
        async with no_stop():
            assert True

    @a.contextmanager
    async def supress_no_stop():
        try:
            yield
        except BaseException:  # noqa: B036
            yield

    with pytest.raises(RuntimeError):
        async with supress_no_stop():
            raise StopAsyncIteration()


@sync
async def test_contextmanager_raise_asyncstop():
    """Test that StopAsyncIteration may propagate out of a context block"""

    @a.contextmanager
    async def no_raise():
        yield

    with pytest.raises(StopAsyncIteration, match="outside"):
        async with no_raise():
            raise StopAsyncIteration("outside")

    @a.contextmanager
    async def replace():
        try:
            yield
        except StopAsyncIteration:
            raise StopAsyncIteration("inside") from None

    with pytest.raises(RuntimeError):
        async with replace():
            raise StopAsyncIteration("outside")


@sync
async def test_contextmanager_raise_runtimeerror():
    @a.contextmanager
    async def no_raise():
        yield

    with pytest.raises(RuntimeError, match="outside"):
        async with no_raise():
            raise RuntimeError("outside")

    @a.contextmanager
    async def replace():
        try:
            yield
        except RuntimeError:
            raise RuntimeError("inside") from None

    with pytest.raises(RuntimeError, match="inside"):
        async with replace():
            raise RuntimeError("outside")


@sync
async def test_contextmanager_raise_same():
    """Test that outer exceptions do not shadow inner/newer ones"""

    @a.contextmanager
    async def reraise():
        try:
            yield
        except BaseException as err:
            raise err

    with pytest.raises(KeyError, match="outside"):
        async with reraise():
            raise KeyError("outside")

    @a.contextmanager
    async def recreate():
        try:
            yield
        except BaseException as err:  # noqa: B036
            raise type(err)("inside") from None

    with pytest.raises(KeyError, match="inside"):
        async with recreate():
            raise KeyError("outside")


@sync
async def test_contextmanager_raise_generatorexit():
    """Test that shutdown via GeneratorExit is propagated"""

    @a.contextmanager
    async def no_op():
        yield

    with pytest.raises(GeneratorExit):
        async with no_op():
            raise GeneratorExit("used to tear down coroutines")

    # during shutdown, generators may be killed in arbitrary order
    # make sure we do not suppress GeneratorExit
    context = no_op()
    with pytest.raises(GeneratorExit, match="inner"):
        async with context:
            # simulate cleanup closing the child early
            await context.gen.aclose()
            raise GeneratorExit("inner")


@sync
async def test_contextmanager_no_suppress_generatorexit():
    """Test that GeneratorExit is not suppressed"""

    @a.contextmanager
    async def no_op():
        yield

    exc = GeneratorExit("GE should not be replaced normally")
    with pytest.raises(type(exc)) as exc_info:
        async with no_op():
            raise exc
    assert exc_info.value is exc

    @a.contextmanager
    async def exit_ge():
        try:
            yield
        except GeneratorExit:
            # treat GE as regular exit
            pass

    with pytest.raises(GeneratorExit):
        async with exit_ge():
            raise GeneratorExit("Resume teardown if child exited")

    @a.contextmanager
    async def ignore_ge():
        try:
            yield
        except GeneratorExit:
            yield

    with pytest.raises(RuntimeError):
        async with ignore_ge():
            raise GeneratorExit("Warn if child does not exit")


@sync
async def test_nullcontext():
    async with a.nullcontext(1337) as value:
        assert value == 1337


class MockAsyncContext:
    def __init__(self, value=None):
        self._value = value
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self._value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exited = True


class MockContext:
    def __init__(self, value=None):
        self._value = value
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self._value

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True


@sync
async def test_exist_stack():
    async with a.ExitStack() as exit_stack:
        for value in (0, 1, 2, 3, -5, None, "Hello"):
            assert value == await exit_stack.enter_context(a.nullcontext(value))


@sync
async def test_exit_stack_pop_all():
    async with a.ExitStack() as exit_stack:
        contexts = list(
            map(lambda v: MockAsyncContext(v) if v % 2 else MockContext(v), range(10))
        )
        values = await a.list(a.map(exit_stack.enter_context, contexts))
        assert values == list(range(10))
        assert all(cm.entered for cm in contexts)
        assert all(not cm.exited for cm in contexts)
        clone_stack = exit_stack.pop_all()
    assert all(not cm.exited for cm in contexts)
    await clone_stack.aclose()
    assert all(cm.exited for cm in contexts)


@sync
async def test_exit_stack_callback():
    """Test that callbacks are run regardless of exceptions"""
    unwind_values = []

    async def push(value):
        unwind_values.append(value)
        return True  # attempt to suppress - this must not succeed

    with pytest.raises(KeyError):
        async with a.ExitStack() as exit_stack:
            for value in range(5):
                exit_stack.callback(push, value)
            raise KeyError()
    assert unwind_values == list(reversed(range(5)))


@sync
async def test_exit_stack_push():
    seen = []

    @contextmanager
    def observe():
        try:
            yield
        except BaseException as exc_val:
            seen.append(exc_val)
            raise

    @a.contextmanager
    async def suppress():
        try:
            yield
        except BaseException as exc_val:  # noqa: B036
            seen.append(exc_val)

    async def replace(exc_type, exc_val, tb, new):
        seen.append(exc_val)
        raise new

    with pytest.raises(TypeError) as exc_info:
        async with a.ExitStack() as exit_stack:
            exit_stack.push(partial(replace, new=TypeError()))
            exit_stack.push(partial(replace, new=ValueError()))
            s = suppress()
            await s.__aenter__()
            exit_stack.push(s)
            exit_stack.push(partial(replace, new=IndexError()))
            o = observe()
            o.__enter__()
            exit_stack.push(o)
            raise KeyError()
    assert list(map(type, seen)) == [
        KeyError,
        KeyError,
        IndexError,
        type(None),
        ValueError,
    ]
    assert seen[2].__context__ == seen[1]
    assert exc_info.type == TypeError
    assert exc_info.value.__context__ == seen[-1]


@sync
async def test_exit_stack_stitch_context():
    async def replace(exc_type, exc_val, tb, new):
        try:
            {}["a"]
        except KeyError:
            raise new

    async def extend(exc_type, exc_val, tb, new):
        try:
            raise exc_val
        except exc_type:
            raise new

    replacement_exc, middle_exc, initial_exc = TypeError(), ValueError(), IndexError()
    with pytest.raises(type(replacement_exc)) as exc_info:
        async with a.ExitStack() as exit_stack:
            exit_stack.push(partial(extend, new=replacement_exc))
            exit_stack.push(partial(replace, new=middle_exc))
            raise initial_exc
    assert exc_info.value.__context__ == middle_exc
    assert exc_info.value.__context__.__context__ == initial_exc


@sync
async def test_misuse_enter_context():
    async with a.ExitStack() as exit_stack:
        with pytest.raises(AttributeError):
            await exit_stack.enter_context(None)
    async with a.ExitStack() as exit_stack:
        with pytest.raises(AttributeError) as exc_info:
            try:
                {}[1]
            except KeyError:
                await exit_stack.enter_context(None)
        assert type(exc_info.value.__context__.__context__) is KeyError


class _AsyncExitSuppressing:
    """Async context manager whose ``__aexit__`` records and may suppress"""

    def __init__(self, name, witness, *, suppress=()):
        self.name = name
        self.witness = witness
        self.suppress = suppress
        self.exits = 0

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exits += 1
        self.witness.append(("aexit", self.name, type(exc_val)))
        return isinstance(exc_val, self.suppress) if self.suppress else False


@sync
async def test_exit_stack_checkpoint_type():
    stack = a.ExitStack()
    group = stack.checkpoint()
    assert isinstance(group, a.ExitStackCheckpoint)
    assert isinstance(group, a.ExitStack)


@sync
async def test_exit_stack_checkpoint_commit_staging_and_order():
    """Staged actions do not run in the block and commit in registration order"""
    witness: list[str] = []
    async with a.ExitStack() as stack:
        stack.callback(witness.append, "outer-before")
        async with stack.checkpoint() as group:
            group.callback(witness.append, "g0")
            group.callback(witness.append, "g1")
            # nothing runs while the checkpoint is still open
            assert witness == []
        # committed but not unwound yet
        assert witness == []
        stack.callback(witness.append, "outer-after")
    # LIFO of the merged stack:
    # outer-after, g1, g0, outer-before
    assert witness == ["outer-after", "g1", "g0", "outer-before"]


@sync
async def test_exit_stack_checkpoint_commit_runs_each_action_once():
    witness = MockAsyncContext()
    sync_witness = MockContext()
    async with a.ExitStack() as stack:
        async with stack.checkpoint() as group:
            assert await group.enter_context(a.nullcontext("value")) == "value"
            await group.enter_context(witness)
            await group.enter_context(sync_witness)
            group.push(MockAsyncContext())
            group.push(MockContext())
    assert witness.entered and witness.exited
    assert sync_witness.entered and sync_witness.exited


@sync
async def test_exit_stack_checkpoint_commit_mixed_kinds():
    """push/callback/enter_context with sync/async and __exit__/__aexit__"""
    witness: list[str] = []

    async def async_cb(value):
        witness.append(value)

    def sync_cb(value):
        witness.append(value)

    async def async_exit(exc_type, exc_val, tb):
        witness.append("push-async")
        return False

    def sync_exit(exc_type, exc_val, tb):
        witness.append("push-sync")
        return False

    async_cm = MockAsyncContext("async-cm")
    sync_cm = MockContext("sync-cm")
    async with a.ExitStack() as stack:
        async with stack.checkpoint() as group:
            group.callback(async_cb, "async-callback")
            group.callback(sync_cb, "sync-callback")
            group.push(async_exit)
            group.push(sync_exit)
            await group.enter_context(async_cm)
            await group.enter_context(sync_cm)
        assert witness == []
    # LIFO after the two context managers, which record on their attributes
    assert async_cm.entered and async_cm.exited
    assert sync_cm.entered and sync_cm.exited
    assert witness == [
        "push-sync",
        "push-async",
        "sync-callback",
        "async-callback",
    ]


@sync
async def test_exit_stack_checkpoint_rollback_runs_group_only():
    """On exception the group unwinds locally; the outer stack never runs it"""
    witness: list[str] = []
    stack = a.ExitStack()
    stack.callback(witness.append, "outer")
    with pytest.raises(KeyError):
        async with stack.checkpoint() as group:
            group.callback(witness.append, "g0")
            group.callback(witness.append, "g1")
            raise KeyError()
    # group actions ran immediately, in reverse order, exactly once
    assert witness == ["g1", "g0"]
    await stack.aclose()
    assert witness == ["g1", "g0", "outer"]


@sync
async def test_exit_stack_checkpoint_rollback_with_outer_unwind():
    """An unhandled rollback still unwinds the surrounding stack afterwards"""
    witness: list[str] = []
    with pytest.raises(KeyError):
        async with a.ExitStack() as stack:
            stack.callback(witness.append, "outer")
            async with stack.checkpoint() as group:
                group.callback(witness.append, "g0")
                raise KeyError()
    assert witness == ["g0", "outer"]


@sync
async def test_exit_stack_checkpoint_rollback_exception_args():
    """Rolled back actions see the original exception details"""
    seen = []
    error = KeyError("rollback")

    async def observe(exc_type, exc_val, tb):
        seen.append((exc_type, exc_val, tb))
        return False

    with pytest.raises(KeyError):
        async with a.ExitStack().checkpoint() as group:
            group.push(observe)
            raise error
    assert seen == [(KeyError, error, error.__traceback__)]


@sync
async def test_exit_stack_checkpoint_rollback_suppress():
    """Exception suppression works exactly as for a plain ExitStack"""
    witness: list[str] = []
    async with a.ExitStack() as stack:
        stack.callback(witness.append, "outer")
        async with stack.checkpoint() as group:
            group.callback(witness.append, "g")
            group.push(
                _AsyncExitSuppressing("suppressor", witness, suppress=(KeyError,))
            )
            raise KeyError()
        # the checkpoint suppressed the KeyError and execution continues here
    # rollback order: suppressor (LIFO, suppresses), then callback g;
    # the outer stack unwinds with no exception afterwards
    assert [item for item in witness if isinstance(item, str)] == ["g", "outer"]


@sync
async def test_exit_stack_checkpoint_rollback_does_not_suppress_other():
    witness = _AsyncExitSuppressing("suppressor", [], suppress=(ValueError,))
    with pytest.raises(KeyError):
        async with a.ExitStack().checkpoint() as group:
            group.push(witness)
            raise KeyError()
    assert witness.exits == 1


@sync
async def test_exit_stack_checkpoint_direct_outer_registration_is_isolated():
    """Actions registered directly on the outer stack bypass the checkpoint"""
    witness: list[str] = []
    stack = a.ExitStack()
    stack.callback(witness.append, "outer-before")
    with pytest.raises(KeyError):
        async with stack.checkpoint() as group:
            stack.callback(witness.append, "outer-during")
            group.callback(witness.append, "g")
            raise KeyError()
    # rollback only touches the group
    assert witness == ["g"]
    await stack.aclose()
    assert witness == ["g", "outer-during", "outer-before"]


@sync
async def test_exit_stack_checkpoint_nested_commit():
    witness: list[str] = []
    async with a.ExitStack() as stack:
        async with stack.checkpoint() as outer:
            outer.callback(witness.append, "o0")
            async with outer.checkpoint() as inner:
                inner.callback(witness.append, "i0")
                inner.callback(witness.append, "i1")
            outer.callback(witness.append, "o1")
    # merged stack is [o0, i0, i1, o1] and unwinds LIFO
    assert witness == ["o1", "i1", "i0", "o0"]


@sync
async def test_exit_stack_checkpoint_nested_inner_rollback_contained():
    """A suppressed inner rollback leaves the outer checkpoint intact"""
    witness: list[str] = []
    async with a.ExitStack() as stack:
        async with stack.checkpoint() as outer:
            outer.callback(witness.append, "o0")
            with pytest.raises(KeyError):
                async with outer.checkpoint() as inner:
                    inner.callback(witness.append, "i0")
                    raise KeyError()
            assert witness == ["i0"]
            outer.callback(witness.append, "o1")
    assert witness == ["i0", "o1", "o0"]


@sync
async def test_exit_stack_checkpoint_nested_inner_rollback_propagates():
    """An inner rollback that propagates also rolls back the outer group once"""
    witness: list[str] = []
    with pytest.raises(KeyError):
        async with a.ExitStack().checkpoint() as outer:
            outer.callback(witness.append, "o0")
            async with outer.checkpoint() as inner:
                inner.callback(witness.append, "i0")
                raise KeyError()
    assert witness == ["i0", "o0"]


@sync
async def test_exit_stack_checkpoint_nested_commit_then_outer_rollback():
    """Committed inner actions belong to the outer group and roll back with it"""
    witness: list[str] = []
    with pytest.raises(KeyError):
        async with a.ExitStack().checkpoint() as outer:
            outer.callback(witness.append, "o0")
            async with outer.checkpoint() as inner:
                inner.callback(witness.append, "i0")
            # inner committed; raising now must run i0 exactly once
            raise KeyError()
    assert witness == ["i0", "o0"]


@sync
async def test_exit_stack_checkpoint_commit_after_parent_finalized():
    """If the parent checkpoint was closed manually, the child unwinds locally"""
    witness: list[str] = []
    async with a.ExitStack() as stack:
        async with stack.checkpoint() as outer:
            outer.callback(witness.append, "o")
            async with outer.checkpoint() as inner:
                inner.callback(witness.append, "i")
                # misuse: finalize the parent while the child is still open
                await outer.aclose()
            # the child can no longer commit, so it runs its actions itself
            assert witness == ["i"]
    # the early commit handed "o" to the outer stack, which runs it at the end
    assert witness == ["i", "o"]


@sync
async def test_exit_stack_checkpoint_aclose_commits_early_once():
    """aclose on an open checkpoint commits; repeated finalization is a no-op"""
    witness: list[str] = []
    async with a.ExitStack() as stack:
        stack.callback(witness.append, "outer")
        async with stack.checkpoint() as group:
            group.callback(witness.append, "g")
            await group.aclose()
            await group.aclose()
            assert witness == []
        # block exit after early commit does not run anything again
        assert witness == []
    assert witness == ["g", "outer"]


@sync
async def test_exit_stack_checkpoint_double_finalize_after_rollback():
    witness: list[str] = []
    stack = a.ExitStack()
    group = stack.checkpoint()
    with pytest.raises(KeyError):
        async with group:
            group.callback(witness.append, "g")
            raise KeyError()
    assert witness == ["g"]
    await group.aclose()
    await group.__aexit__(None, None, None)
    assert witness == ["g"]


@sync
async def test_exit_stack_checkpoint_register_after_closed():
    """All staging paths raise RuntimeError after finalization and never enter"""
    stack = a.ExitStack()

    async def noop(exc_type, exc_val, tb):
        return False

    for finalizer in (
        lambda g: g.aclose(),
        lambda g: g.__aexit__(KeyError, KeyError(), None),
    ):
        group = stack.checkpoint()
        await finalizer(group)
        with pytest.raises(RuntimeError):
            group.callback(lambda: None)
        with pytest.raises(RuntimeError):
            group.push(noop)
        not_entered = MockAsyncContext()
        with pytest.raises(RuntimeError):
            await group.enter_context(not_entered)
        assert not not_entered.entered
        with pytest.raises(RuntimeError):
            group.pop_all()
        with pytest.raises(RuntimeError):
            group.checkpoint()


@sync
async def test_exit_stack_checkpoint_pop_all_detaches_staged():
    """pop_all transfers staged actions to a standalone stack before commit"""
    witness: list[str] = []
    async with a.ExitStack() as stack:
        stack.callback(witness.append, "outer")
        async with stack.checkpoint() as group:
            group.callback(witness.append, "g0")
            transferred = group.pop_all()
            group.callback(witness.append, "g1")
        # only g1 committed; g0 is owned by the transferred stack
        await transferred.aclose()
        assert witness == ["g0"]
    assert witness == ["g0", "g1", "outer"]


@sync
async def test_exit_stack_checkpoint_pop_all_survives_rollback():
    """Transferred actions are not touched by rollback of the old checkpoint"""
    witness: list[str] = []
    stack = a.ExitStack()
    stack.callback(witness.append, "outer")
    with pytest.raises(KeyError):
        async with stack.checkpoint() as group:
            group.callback(witness.append, "g0")
            transferred = group.pop_all()
            group.callback(witness.append, "g1")
            raise KeyError()
    assert witness == ["g1"]
    await transferred.aclose()
    assert witness == ["g1", "g0"]
    await stack.aclose()
    assert witness == ["g1", "g0", "outer"]


@sync
async def test_exit_stack_checkpoint_outer_pop_all_keeps_checkpoint():
    """pop_all on the outer stack does not detach pending checkpoints"""
    witness: list[str] = []
    popped = None
    async with a.ExitStack() as stack:
        stack.callback(witness.append, "old")
        popped = stack.pop_all()
        async with stack.checkpoint() as group:
            group.callback(witness.append, "g")
        # committed into the original stack, not the popped one
        assert witness == []
    assert witness == ["g"]
    await popped.aclose()
    assert witness == ["g", "old"]


@sync
async def test_exit_stack_checkpoint_rollback_multiple_failures():
    """Every action is attempted; the newest replacement exception wins"""
    witness: list[str] = []

    def failing(name, exc):
        async def exit_cb(exc_type, exc_val, tb):
            witness.append(name)
            raise exc

        return exit_cb

    first = KeyError("a")
    middle = ValueError("b")
    last = RuntimeError("c")
    with pytest.raises(KeyError) as exc_info:
        async with a.ExitStack().checkpoint() as group:
            group.push(failing("a", first))
            group.callback(witness.append, "ok")
            group.push(failing("b", middle))
            group.push(failing("c", last))
            raise IndexError("body")
    # LIFO: c, b, ok, a -- all attempted despite the failures
    assert witness == ["c", "b", "ok", "a"]
    assert exc_info.value is first
    assert first.__context__ is middle
    assert middle.__context__ is last
    assert last.__context__ is not None
    assert type(last.__context__) is IndexError


class _Abort(BaseException):
    pass


class _Cancel(BaseException):
    pass


@sync
async def test_exit_stack_checkpoint_rollback_base_exception():
    """Any BaseException (e.g. cancellation) unwinds all staged actions"""
    witness: list[str] = []

    async def aborting(exc_type, exc_val, tb):
        witness.append("abort")
        raise _Abort()

    with pytest.raises(_Abort) as exc_info:
        async with a.ExitStack().checkpoint() as group:
            group.callback(witness.append, "after")
            group.push(aborting)
            group.callback(witness.append, "before")
            raise _Cancel()
    # LIFO: before, abort, after -- all attempted despite the _Abort
    assert witness == ["before", "abort", "after"]
    # the abort is chained onto the cancellation that triggered rollback
    assert isinstance(exc_info.value.__context__, _Cancel)


@sync
async def test_exit_stack_checkpoint_rollback_failure_finalizes_once():
    """A failing rollback finalizes the checkpoint; close afterwards is a no-op"""
    witness: list[str] = []

    async def aborting(exc_type, exc_val, tb):
        witness.append("abort")
        raise _Abort()

    stack = a.ExitStack()
    group = stack.checkpoint()
    with pytest.raises(_Abort):
        async with group:
            group.callback(witness.append, "ok")
            group.push(aborting)
            group.callback(witness.append, "newest")
            raise _Cancel()
    assert witness == ["newest", "abort", "ok"]
    await group.aclose()
    assert witness == ["newest", "abort", "ok"]


@sync
async def test_exit_stack_checkpoint_enter_context_failure_rolls_back():
    """A failed __aenter__ rolls back previously staged actions only"""
    good = MockAsyncContext()

    class FailingEnter(MockAsyncContext):
        async def __aenter__(self):
            raise RuntimeError("cannot enter")

    failing = FailingEnter()
    with pytest.raises(RuntimeError):
        async with a.ExitStack().checkpoint() as group:
            await group.enter_context(good)
            await group.enter_context(failing)
    assert good.entered and good.exited
    assert not failing.entered and not failing.exited


@sync
async def test_exit_stack_checkpoint_no_event_loop_dependency():
    """The custom synchronous test loop is enough: no asyncio scheduling happens"""
    witness: list[str] = []

    async with a.ExitStack() as stack:
        async with stack.checkpoint() as group:
            # plain synchronous callbacks never yield to any event loop
            group.callback(witness.append, "g")
        stack.callback(witness.append, "outer")
    assert witness == ["outer", "g"]
