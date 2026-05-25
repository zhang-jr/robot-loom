"""Unit tests for Memory sub-implementations and SpatialHubMemory."""

from __future__ import annotations

import pytest

from robot_harness.memory.base import MemoryEntry, MemoryQuery
from robot_harness.memory.episodic_memory import InMemoryEpisodicMemory
from robot_harness.memory.object_memory import InMemoryObjectMemory
from robot_harness.memory.place_memory import InMemoryPlaceMemory
from robot_harness.memory.semantic_memory import InMemorySemanticMemory
from robot_harness.tools.memory.spatial_hub_adapter import SpatialHubMemory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(memory_type: str, content: dict, tags: list[str] | None = None) -> MemoryEntry:
    return MemoryEntry(
        memory_type=memory_type,
        robot_id="robot-0",
        content=content,
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# ObjectMemory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_object_memory_upsert_and_query() -> None:
    mem = InMemoryObjectMemory()
    mid = await mem.upsert(_entry("object", {"name": "red_cup", "position": [0.5, 0, 0.3]}))
    assert isinstance(mid, str) and len(mid) > 0

    hits = await mem.query(MemoryQuery(memory_type="object", robot_id="robot-0", text="red_cup"))
    assert len(hits) == 1
    assert hits[0].content["name"] == "red_cup"


@pytest.mark.asyncio
async def test_object_memory_empty_query_returns_empty() -> None:
    mem = InMemoryObjectMemory()
    hits = await mem.query(MemoryQuery(memory_type="object", robot_id="robot-0", text="nothing"))
    assert hits == []


@pytest.mark.asyncio
async def test_object_memory_robot_id_filter() -> None:
    mem = InMemoryObjectMemory()
    await mem.upsert(MemoryEntry(memory_type="object", robot_id="robot-0", content={"name": "box"}))
    await mem.upsert(
        MemoryEntry(memory_type="object", robot_id="robot-1", content={"name": "sphere"})
    )

    hits = await mem.query(MemoryQuery(memory_type="object", robot_id="robot-0", text=""))
    assert all(h.content["name"] == "box" for h in hits)


@pytest.mark.asyncio
async def test_object_memory_top_k() -> None:
    mem = InMemoryObjectMemory()
    for i in range(10):
        await mem.upsert(_entry("object", {"name": f"obj_{i}"}))
    hits = await mem.query(MemoryQuery(memory_type="object", robot_id="robot-0", text="", top_k=3))
    assert len(hits) <= 3


# ---------------------------------------------------------------------------
# PlaceMemory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_place_memory_upsert_and_query() -> None:
    mem = InMemoryPlaceMemory()
    await mem.upsert(_entry("place", {"name": "kitchen", "coords": [1.0, 2.0, 0.0]}))
    hits = await mem.query(MemoryQuery(memory_type="place", robot_id="robot-0", text="kitchen"))
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_episodic_memory_write_and_query() -> None:
    mem = InMemoryEpisodicMemory()
    await mem.write(_entry("episodic", {"event": "pick_failed", "reason": "object not found"}))
    hits = await mem.query(
        MemoryQuery(memory_type="episodic", robot_id="robot-0", text="pick_failed")
    )
    assert len(hits) == 1
    assert hits[0].content["event"] == "pick_failed"


@pytest.mark.asyncio
async def test_episodic_memory_most_recent_first() -> None:
    mem = InMemoryEpisodicMemory()
    for i in range(5):
        await mem.write(_entry("episodic", {"seq": i}))
    hits = await mem.query(
        MemoryQuery(memory_type="episodic", robot_id="robot-0", text="", top_k=5)
    )
    # Most recent (seq=4) should be first
    assert hits[0].content["seq"] == 4


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_memory_upsert_and_query() -> None:
    mem = InMemorySemanticMemory()
    await mem.upsert(_entry("semantic", {"fact": "the table is 0.8m tall"}))
    hits = await mem.query(
        MemoryQuery(memory_type="semantic", robot_id="robot-0", text="table height")
    )
    # Substring "table" matches the content
    assert any("table" in str(h.content) for h in hits)


# ---------------------------------------------------------------------------
# SpatialHubMemory (facade)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spatial_hub_routes_writes_correctly() -> None:
    hub = SpatialHubMemory()
    obj_id = await hub.write(_entry("object", {"name": "mug"}))
    place_id = await hub.write(_entry("place", {"name": "shelf"}))
    ep_id = await hub.write(_entry("episodic", {"event": "success"}))
    sem_id = await hub.write(_entry("semantic", {"fact": "mug is ceramic"}))

    assert all(isinstance(i, str) for i in [obj_id, place_id, ep_id, sem_id])


@pytest.mark.asyncio
async def test_spatial_hub_routes_queries_correctly() -> None:
    hub = SpatialHubMemory()
    await hub.write(_entry("object", {"name": "cup"}))
    await hub.write(_entry("place", {"name": "table"}))

    obj_hits = await hub.query(MemoryQuery(memory_type="object", robot_id="robot-0", text="cup"))
    place_hits = await hub.query(MemoryQuery(memory_type="place", robot_id="robot-0", text="table"))

    assert len(obj_hits) >= 1
    assert len(place_hits) >= 1
    # No cross-contamination
    assert all("cup" in str(h.content) for h in obj_hits)
    assert all("table" in str(h.content) for h in place_hits)


@pytest.mark.asyncio
async def test_spatial_hub_unknown_type_rejected_at_boundary() -> None:
    """Invalid memory_type is rejected by Pydantic validation, not silently routed."""
    with pytest.raises(Exception):  # noqa: B017  — ValidationError
        _entry("unknown_type", {"data": "x"})
