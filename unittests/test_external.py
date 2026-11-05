import asyncio
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional

import pytest

from asyncstdlib import external
from asyncstdlib.external import (
    DEFAULT_RUN_SIZE,
    DirectoryStore,
    ExternalSortError,
    ManifestIncomplete,
    ManifestInvalid,
    RunInfo,
    SortManifest,
)

from .utility import Switch, asyncify, awaitify, sync


class SerializationBoom(Exception):
    pass


class DiskBoom(Exception):
    pass


def failing_dumps(marker: Any) -> Callable[[Any], bytes]:
    def dumps(item: Any) -> bytes:
        if item == marker:
            raise SerializationBoom("cannot serialize")
        return pickle.dumps(item)

    return dumps


class AsyncStore:
    """Fully asynchronous :py:class:`external.RunStore` for tests"""

    def __init__(self, path: Path, *, fail_on: Any = None) -> None:
        self._store = DirectoryStore(path)
        self._fail_on = fail_on
        self.deleted: List[Any] = []

    async def allocate_run(self, index: int) -> str:
        await Switch()
        return self._store.allocate_run(index)

    async def write_run(self, ref: str, run: Iterable[Any]) -> Optional[str]:
        await Switch()
        items = list(run)
        if self._fail_on is not None and self._fail_on in items:
            raise asyncio.CancelledError()
        return self._store.write_run(ref, items)

    async def read_run(self, ref: str) -> AsyncIterator[Any]:
        await Switch()

        async def generator() -> AsyncIterator[Any]:
            for item in self._store.read_run(ref):
                await Switch()
                yield item

        return generator()

    async def delete_run(self, ref: str) -> None:
        await Switch()
        self.deleted.append(ref)
        self._store.delete_run(ref)

    async def run_tag(self, ref: str) -> str:
        await Switch()
        return self._store.run_tag(ref)

    async def save_manifest(self, data: Dict[str, Any]) -> None:
        await Switch()
        self._store.save_manifest(data)

    async def load_manifest(self) -> Optional[Dict[str, Any]]:
        await Switch()
        return self._store.load_manifest()

    async def delete_manifest(self) -> None:
        await Switch()
        self._store.delete_manifest()


class FlakyDeleteStore:
    """Store wrapper whose ``delete_run`` fails for one specific reference"""

    def __init__(self, store: DirectoryStore, failing_ref: Any) -> None:
        self._store = store
        self._failing_ref = failing_ref
        self.attempted: List[Any] = []

    def allocate_run(self, index: int) -> str:
        return self._store.allocate_run(index)

    def write_run(self, ref: str, run: Iterable[Any]) -> str:
        return self._store.write_run(ref, run)

    def read_run(self, ref: str):
        return self._store.read_run(ref)

    def delete_run(self, ref: str) -> None:
        self.attempted.append(ref)
        if ref == self._failing_ref:
            raise OSError("delete failed")
        self._store.delete_run(ref)


def store_files(path: Path) -> List[str]:
    return sorted(entry.name for entry in path.iterdir())


SORT_SAMPLES = [
    [],
    [1],
    [3, 1, 2],
    [5, -1, 0, 5, 5, -1, 2],
    list(range(50)),
    [random.Random(seed).randrange(20) for seed in range(120)],
]
RUN_SIZES = [1, 2, 3, 7, 64, DEFAULT_RUN_SIZE, 10_000]


@pytest.mark.parametrize("sample", SORT_SAMPLES)
@pytest.mark.parametrize("run_size", RUN_SIZES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("as_async", [False, True])
@sync
async def test_external_sorted_stdlib(
    tmp_path, sample, run_size, reverse, as_async
):
    store = DirectoryStore(tmp_path)
    source = asyncify(sample) if as_async else sample
    result = [
        item
        async for item in external.external_sorted(
            source, store=store, run_size=run_size, reverse=reverse
        )
    ]
    assert result == sorted(sample, reverse=reverse)
    # the one-shot API owns and always removes its artifacts
    assert store_files(tmp_path) == []


@pytest.mark.parametrize("run_size", [1, 2, 5, 17])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("use_async_key", [False, True])
@sync
async def test_external_sorted_stable(
    tmp_path, run_size, reverse, use_async_key
):
    rng = random.Random(1234)
    pairs = [(value % 7, position) for position, value in enumerate(range(140))]
    rng.shuffle(pairs)

    def key(pair):
        return pair[0]

    akey = awaitify(key) if use_async_key else key
    result = [
        pair[1]
        async for pair in external.external_sorted(
            pairs,
            store=DirectoryStore(tmp_path),
            key=akey,
            run_size=run_size,
            reverse=reverse,
        )
    ]
    expected = [
        pair[1] for pair in sorted(pairs, key=key, reverse=reverse)
    ]
    assert result == expected


@sync
async def test_small_input_never_spills(tmp_path):
    store = DirectoryStore(tmp_path)
    result = [
        item
        async for item in external.external_sorted(
            [5, 1, 4], store=store, run_size=100
        )
    ]
    assert result == [1, 4, 5]
    assert store_files(tmp_path) == []


@sync
async def test_exact_budget_never_spills(tmp_path):
    store = DirectoryStore(tmp_path)
    result = [
        item
        async for item in external.external_sorted(
            [3, 2, 1], store=store, run_size=3
        )
    ]
    assert result == [1, 2, 3]
    assert store_files(tmp_path) == []


@pytest.mark.parametrize("run_size", [1, 2, 5, 10])
@sync
async def test_spill_run_sizes(tmp_path, run_size):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(23), store=store, run_size=run_size
    )
    assert manifest.state == "ready"
    assert manifest.total == 23
    counts = [run.count for run in manifest.runs]
    # the last run holds the tail; every preceding run fills the whole budget
    assert len(counts) == (22 // run_size) + 1
    assert counts[:-1] == [run_size] * (22 // run_size)
    assert 1 <= counts[-1] <= run_size
    assert sum(counts) == 23
    assert [run.index for run in manifest.runs] == list(range(len(counts)))


@sync
async def test_spill_always_writes(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        [2, 1], store=store, run_size=100
    )
    assert len(manifest.runs) == 1
    empty = await external.spill_sorted_runs([], store=store, run_size=100)
    assert empty.state == "ready"
    assert empty.runs == ()


@sync
async def test_spill_requires_positive_run_size(tmp_path):
    with pytest.raises(ValueError):
        await external.spill_sorted_runs(
            [1], store=DirectoryStore(tmp_path), run_size=0
        )


@sync
async def test_manifest_json_roundtrip(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(20), store=store, run_size=6, reverse=True, key_label="id"
    )
    blob = json.dumps(manifest.to_data())
    restored = SortManifest.from_data(json.loads(blob))
    assert restored == manifest
    assert restored.key_label == "id"


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"version": 99, "state": "ready", "reverse": False, "run_size": 1,
         "key_label": None, "total": 0, "runs": []},
        {"version": 1, "state": "paused", "reverse": False, "run_size": 1,
         "key_label": None, "total": 0, "runs": []},
        {"version": 1, "state": "ready", "reverse": False, "run_size": 1,
         "key_label": None, "total": 2, "runs": []},
        {"version": 1, "state": "ready", "reverse": False, "run_size": 1,
         "key_label": None, "total": 1,
         "runs": [{"ref": "x", "index": 1, "count": 1, "tag": None}]},
        {"version": 1, "state": "ready", "reverse": False, "run_size": 1,
         "key_label": None, "total": 1, "runs": [42]},
        {"version": "1", "state": "ready", "reverse": False, "run_size": 1,
         "key_label": None, "total": 0, "runs": []},
    ],
)
@sync
async def test_manifest_from_data_rejects(data):
    with pytest.raises(ManifestInvalid):
        SortManifest.from_data(data)


@sync
async def test_manifest_from_data_rejects_non_mapping():
    with pytest.raises(ManifestInvalid):
        SortManifest.from_data("nope")  # type: ignore[arg-type]


@sync
async def test_merge_keeps_runs_and_is_repeatable(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(30), store=store, run_size=7
    )
    files = store_files(tmp_path)
    assert len(files) == 5
    first = [
        item
        async for item in external.merge_sorted_runs(manifest, store=store)
    ]
    assert first == list(range(30))
    # merging does not consume or delete runs
    assert store_files(tmp_path) == files
    second = [
        item
        async for item in external.merge_sorted_runs(
            manifest.to_data(), store=store
        )
    ]
    assert second == list(range(30))


@sync
async def test_merge_reverse_mismatch(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(10), store=store, run_size=3, reverse=True
    )
    with pytest.raises(ValueError):
        async for _ in external.merge_sorted_runs(
            manifest, store=store, reverse=False
        ):
            pass


@sync
async def test_merge_building_manifest_rejected(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(10), store=store, run_size=3
    )
    building = SortManifest.from_data(
        {**manifest.to_data(), "state": "building"}
    )
    with pytest.raises(ManifestIncomplete):
        async for _ in external.merge_sorted_runs(building, store=store):
            pass


@sync
async def test_resume_from_exported_manifest(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(100), store=store, run_size=13
    )
    data = json.loads(json.dumps(manifest.to_data()))
    # simulate a fresh process: new store instance on the same directory
    restarted = DirectoryStore(tmp_path)
    restored = SortManifest.from_data(data)
    await external.validate_manifest(restored, store=restarted)
    result = [
        item
        async for item in external.resume_sorted(restored, store=restarted)
    ]
    assert result == list(range(100))
    await external.discard_manifest(restored, store=restarted)
    assert store_files(tmp_path) == []


@sync
async def test_validate_detects_missing_run(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(40), store=store, run_size=8
    )
    os.unlink(tmp_path / manifest.runs[2].ref)
    with pytest.raises(ManifestInvalid) as exc_info:
        await external.validate_manifest(manifest, store=store)
    assert "run 2" in str(exc_info.value)


@sync
async def test_validate_detects_corrupt_payload(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(40), store=store, run_size=8
    )
    target = tmp_path / manifest.runs[1].ref
    with open(target, "r+b") as stream:
        stream.seek(20)
        stream.write(b"\x00" * 16)
    with pytest.raises(ManifestInvalid) as exc_info:
        await external.validate_manifest(manifest, store=store)
    message = str(exc_info.value)
    assert "run 1" in message
    assert "tag mismatch" in message or "unreadable" in message


@sync
async def test_validate_detects_wrong_count(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(24), store=store, run_size=8
    )
    wrong = SortManifest(
        runs=tuple(
            run if run.index != 2
            else RunInfo(
                ref=run.ref, index=run.index, count=run.count + 1,
                tag=run.tag,
            )
            for run in manifest.runs
        ),
        reverse=False,
        run_size=8,
        total=manifest.total + 1,
        state="ready",
    )
    with pytest.raises(ManifestInvalid) as exc_info:
        await external.validate_manifest(wrong, store=store)
    assert "declares 9" in str(exc_info.value)


@sync
async def test_validate_parameter_mismatch(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(24), store=store, run_size=8, key_label="first"
    )
    with pytest.raises(ManifestInvalid):
        await external.validate_manifest(
            manifest, store=store, reverse=True
        )
    with pytest.raises(ManifestInvalid):
        await external.validate_manifest(
            manifest, store=store, key_label="second"
        )


@sync
async def test_validate_building_raises_incomplete(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(24), store=store, run_size=8
    )
    building = SortManifest.from_data(
        {**manifest.to_data(), "state": "building"}
    )
    with pytest.raises(ManifestIncomplete):
        await external.validate_manifest(building, store=store)


@sync
async def test_discard_is_missing_tolerant(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(20), store=store, run_size=6
    )
    os.unlink(tmp_path / manifest.runs[0].ref)
    await external.discard_manifest(manifest, store=store)
    assert store_files(tmp_path) == []


@sync
async def test_discard_reports_and_continues(tmp_path):
    base = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(20), store=base, run_size=6
    )
    failing_ref = manifest.runs[1].ref
    store = FlakyDeleteStore(base, failing_ref)
    with pytest.raises(ExternalSortError) as exc_info:
        await external.discard_manifest(manifest, store=store)
    assert failing_ref in str(exc_info.value)
    # every run was attempted even though one deletion failed
    assert set(store.attempted) == {run.ref for run in manifest.runs}
    remaining = store_files(tmp_path)
    assert remaining == [failing_ref]
    base.delete_run(failing_ref)


@pytest.mark.parametrize("run_size", [1, 5, 10])
@pytest.mark.parametrize("api", ["spill", "sorted"])
@sync
async def test_serialization_failure_cleans_everything(
    tmp_path, run_size, api
):
    store = DirectoryStore(tmp_path, dumps=failing_dumps(37))

    async def consume():
        if api == "spill":
            await external.spill_sorted_runs(
                range(100), store=store, run_size=run_size
            )
        else:
            async for _ in external.external_sorted(
                range(100), store=store, run_size=run_size
            ):
                pass

    with pytest.raises(SerializationBoom):
        await consume()
    assert store_files(tmp_path) == []


@sync
async def test_storage_failure_cleans_everything(tmp_path):
    class FailingStore:
        def __init__(self, path):
            self._store = DirectoryStore(path)

        def allocate_run(self, index):
            return self._store.allocate_run(index)

        def write_run(self, ref, run):
            raise DiskBoom("disk full")

        def read_run(self, ref):
            return self._store.read_run(ref)

        def delete_run(self, ref):
            return self._store.delete_run(ref)

    store = FailingStore(tmp_path)
    with pytest.raises(DiskBoom):
        await external.spill_sorted_runs(
            range(50), store=store, run_size=10
        )
    assert store_files(tmp_path) == []


@sync
async def test_cancelled_key_cleans_everything(tmp_path):
    store = DirectoryStore(tmp_path)

    async def key(item):
        if item == 47:
            raise asyncio.CancelledError()
        return item

    with pytest.raises(asyncio.CancelledError):
        async for _ in external.external_sorted(
            range(100), store=store, key=key, run_size=10
        ):
            pass
    assert store_files(tmp_path) == []


@sync
async def test_cancelled_source_cleans_everything(tmp_path):
    store = DirectoryStore(tmp_path)

    async def source():
        for item in range(100):
            if item == 47:
                raise asyncio.CancelledError()
            yield item

    with pytest.raises(asyncio.CancelledError):
        async for _ in external.external_sorted(
            source(), store=store, run_size=10
        ):
            pass
    assert store_files(tmp_path) == []


@sync
async def test_consumer_close_cleans_everything(tmp_path):
    store = DirectoryStore(tmp_path)
    result = external.external_sorted(
        range(100), store=store, run_size=10
    )
    assert await result.__anext__() == 0
    await result.aclose()
    assert store_files(tmp_path) == []


@sync
async def test_async_store_end_to_end(tmp_path):
    store = AsyncStore(tmp_path)
    rng = random.Random(7)
    data = list(range(120))
    rng.shuffle(data)
    result = [
        item
        async for item in external.external_sorted(
            asyncify(data), store=store, run_size=9
        )
    ]
    assert result == sorted(data)
    assert store_files(tmp_path) == []
    # every run was deleted exactly once through the async interface
    assert len(store.deleted) == 14


@sync
async def test_async_store_cancelled_write(tmp_path):
    store = AsyncStore(tmp_path, fail_on=47)
    with pytest.raises(asyncio.CancelledError):
        await external.spill_sorted_runs(
            range(100), store=store, run_size=10
        )
    assert store_files(tmp_path) == []


@sync
async def test_checkpoint_persisted_and_loaded(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(100), store=store, run_size=20, checkpoint=True
    )
    assert (tmp_path / "manifest.json").exists()
    loaded = await external.load_manifest(store)
    assert loaded == manifest


@sync
async def test_load_manifest_none(tmp_path):
    assert await external.load_manifest(DirectoryStore(tmp_path)) is None


@sync
async def test_checkpoint_removed_on_completion(tmp_path):
    store = DirectoryStore(tmp_path)
    result = [
        item
        async for item in external.external_sorted(
            range(60), store=store, run_size=10, checkpoint=True
        )
    ]
    assert result == list(range(60))
    assert store_files(tmp_path) == []


@sync
async def test_checkpoint_failure_removes_checkpoint(tmp_path):
    store = DirectoryStore(tmp_path, dumps=failing_dumps(47))
    with pytest.raises(SerializationBoom):
        await external.spill_sorted_runs(
            range(100), store=store, run_size=10, checkpoint=True
        )
    assert store_files(tmp_path) == []


@sync
async def test_checkpoint_requires_hooks(tmp_path):
    class PlainStore:
        def allocate_run(self, index):
            return "x"

        def write_run(self, ref, run):
            return None

        def read_run(self, ref):
            return ()

        def delete_run(self, ref):
            return None

    with pytest.raises(TypeError):
        await external.spill_sorted_runs(
            [1], store=PlainStore(), run_size=1, checkpoint=True
        )


@sync
async def test_interrupted_building_checkpoint(tmp_path):
    """Simulate a hard kill while spilling: a building manifest plus an
    orphaned committed run and a staging file remain on disk."""
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(40), store=store, run_size=10, checkpoint=True
    )
    # crash state: the manifest only knows the first two runs ...
    data = manifest.to_data()
    data["state"] = "building"
    data["runs"] = data["runs"][:2]
    data["total"] = 20
    store.save_manifest(data)
    # ... plus a leftover staging file from the killed write
    (tmp_path / ".run-00000002-deadbeef.part-abcd").write_bytes(b"partial")
    restarted = DirectoryStore(tmp_path)
    loaded = await external.load_manifest(restarted)
    assert loaded is not None and loaded.state == "building"
    with pytest.raises(ManifestIncomplete):
        await external.validate_manifest(loaded, store=restarted)
    # the building runs can be discarded safely ...
    await external.discard_manifest(loaded, store=restarted)
    remaining = store_files(tmp_path)
    orphans = [run.ref for run in manifest.runs[2:]]
    assert set(remaining) == set(orphans) | {
        "manifest.json",
        ".run-00000002-deadbeef.part-abcd",
    }
    # ... and staging garbage plus the orphaned tail are reaped explicitly
    assert restarted.cleanup_staging() == 1
    for orphan in orphans:
        restarted.delete_run(orphan)
    assert not any(
        name.startswith(".") for name in store_files(tmp_path)
    )
    restarted.delete_manifest()
    assert store_files(tmp_path) == []


@sync
async def test_key_label_recorded_and_defaults_to_qualname(tmp_path):
    store = DirectoryStore(tmp_path)

    def my_key(item):
        return item

    manifest = await external.spill_sorted_runs(
        range(10), store=store, run_size=3, key=my_key
    )
    assert manifest.key_label == "test_key_label_recorded_and_defaults_to_qualname.<locals>.my_key"
    await external.discard_manifest(manifest, store=store)


@sync
async def test_unsafe_run_reference_rejected(tmp_path):
    store = DirectoryStore(tmp_path)
    for bad_ref in ["../escape", "/abs/path", ".hidden", ""]:
        with pytest.raises(ManifestInvalid):
            list(store.read_run(bad_ref))


@sync
async def test_truncated_run_is_unreadable(tmp_path):
    store = DirectoryStore(tmp_path)
    manifest = await external.spill_sorted_runs(
        range(20), store=store, run_size=5
    )
    path = tmp_path / manifest.runs[0].ref
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(ManifestInvalid):
        await external.validate_manifest(manifest, store=store)


@sync
async def test_run_store_protocol_membership(tmp_path):
    assert isinstance(DirectoryStore(tmp_path), external.RunStore)
    assert not isinstance(object(), external.RunStore)


@sync
async def test_async_key_recomputed_on_resume(tmp_path):
    store = DirectoryStore(tmp_path)
    calls = {"n": 0}

    async def key(item):
        calls["n"] += 1
        return -item

    manifest = await external.spill_sorted_runs(
        range(30), store=store, run_size=7, key=key
    )
    spill_calls = calls["n"]
    result = [
        item
        async for item in external.merge_sorted_runs(
            manifest, store=store, key=key
        )
    ]
    assert result == sorted(range(30), reverse=True)
    assert calls["n"] == spill_calls + 30
    await external.discard_manifest(manifest, store=store)
