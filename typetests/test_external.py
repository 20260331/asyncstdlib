from __future__ import annotations

from typing import Any, Optional, Sequence

from asyncstdlib import external
from asyncstdlib.external import DirectoryStore, RunInfo, RunStore, SortManifest


async def async_key(value: int) -> int:
    return value


def sync_key(value: int) -> int:
    return value


async def test_external_sorted_types() -> None:
    store: DirectoryStore[int] = DirectoryStore("/tmp/external-sort")
    result = [
        item
        async for item in external.external_sorted(
            [3, 1, 2], store=store, key=async_key, reverse=True
        )
    ]
    assert result == [3, 2, 1]
    sync_result = [
        item
        async for item in external.external_sorted(
            [3, 1, 2], store=store, key=sync_key
        )
    ]
    assert sync_result == [1, 2, 3]


async def test_spill_manifest_lifecycle_types() -> None:
    store: DirectoryStore[int] = DirectoryStore("/tmp/external-sort")
    manifest: SortManifest = await external.spill_sorted_runs(
        range(10), store=store, run_size=4, key=async_key
    )
    data = manifest.to_data()
    restored: SortManifest = SortManifest.from_data(data)
    merged = [
        item async for item in external.merge_sorted_runs(
            restored, store=store, key=async_key
        )
    ]
    assert merged == list(range(10))
    resumed = [
        item
        async for item in external.resume_sorted(
            restored, store=store, key=async_key
        )
    ]
    assert resumed == list(range(10))
    await external.validate_manifest(restored, store=store, reverse=False)
    await external.discard_manifest(restored, store=store)
    loaded: Optional[SortManifest] = await external.load_manifest(store)
    assert loaded is None


async def test_manifest_construction_types() -> None:
    info = RunInfo(ref="run-00000000-abc.run", index=0, count=4)
    tag_info = RunInfo(ref={"bucket": "x", "id": 1}, index=1, count=2, tag="ok")
    manifest = SortManifest(
        runs=(info, tag_info), run_size=4, total=6, state="ready"
    )
    assert manifest.runs[0].tag is None
    assert manifest.to_data()["runs"][1]["tag"] == "ok"


class CustomStore:
    def allocate_run(self, index: int) -> str:
        return f"run-{index}"

    def write_run(self, ref: str, run: Sequence[int]) -> None:
        return None

    def read_run(self, ref: str) -> Sequence[int]:
        return ()

    def delete_run(self, ref: str) -> None:
        return None


async def test_custom_store_is_run_store() -> None:
    store: RunStore[int] = CustomStore()
    manifest = await external.spill_sorted_runs([1, 2], store=store)
    assert isinstance(manifest.to_data(), dict)


async def test_unknown_item_type() -> None:
    class Item:
        pass

    store: DirectoryStore[Item] = DirectoryStore("/tmp/external-sort")
    items: list[Item] = [
        item async for item in external.external_sorted([], store=store)
    ]
    assert items == []
    any_store: RunStore[Any] = CustomStore()
    assert any_store is not None
