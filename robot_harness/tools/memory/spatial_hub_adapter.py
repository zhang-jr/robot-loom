"""SpatialHubMemory — mock adapter for an external SpatialMemory-like server.

Phase 1: in-process stub backed by InMemory* implementations.
Phase 2: will issue HTTP requests to a self-hosted SpatialMemory server.
"""

from __future__ import annotations

from robot_harness.memory.base import MemoryEntry, MemoryHit, MemoryId, MemoryQuery
from robot_harness.memory.episodic_memory import InMemoryEpisodicMemory
from robot_harness.memory.object_memory import InMemoryObjectMemory
from robot_harness.memory.place_memory import InMemoryPlaceMemory
from robot_harness.memory.semantic_memory import InMemorySemanticMemory


class SpatialHubMemory:
    """Facade that satisfies the Memory Protocol using four in-memory backends.

    Drop-in replacement for NullMemory when a real server is not available.
    """

    def __init__(self) -> None:
        self.object = InMemoryObjectMemory()
        self.place = InMemoryPlaceMemory()
        self.episodic = InMemoryEpisodicMemory()
        self.semantic = InMemorySemanticMemory()

    async def write(self, entry: MemoryEntry) -> MemoryId:
        """Route a write to the appropriate sub-memory."""
        match entry.memory_type:
            case "object":
                return await self.object.upsert(entry)
            case "place":
                return await self.place.upsert(entry)
            case "episodic":
                return await self.episodic.write(entry)
            case "semantic":
                return await self.semantic.upsert(entry)

    async def query(self, q: MemoryQuery) -> list[MemoryHit]:
        """Route a query to the appropriate sub-memory."""
        match q.memory_type:
            case "object":
                return await self.object.query(q)
            case "place":
                return await self.place.query(q)
            case "episodic":
                return await self.episodic.query(q)
            case "semantic":
                return await self.semantic.query(q)
