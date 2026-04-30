"""Memory Protocol and sub-memory interfaces."""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

MemoryId = str


class MemoryEntry(BaseModel):
    """Generic memory write payload."""

    memory_type: str  # 'object' | 'place' | 'episodic' | 'semantic'
    robot_id: str
    content: dict[str, Any]
    tags: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


class MemoryQuery(BaseModel):
    memory_type: str
    robot_id: str = ""
    text: str = ""
    tags: list[str] = Field(default_factory=list)
    top_k: int = 5


class MemoryHit(BaseModel):
    memory_id: MemoryId
    score: float
    content: dict[str, Any]


@runtime_checkable
class ObjectMemory(Protocol):
    async def upsert(self, entry: MemoryEntry) -> MemoryId: ...
    async def query(self, q: MemoryQuery) -> list[MemoryHit]: ...


@runtime_checkable
class PlaceMemory(Protocol):
    async def upsert(self, entry: MemoryEntry) -> MemoryId: ...
    async def query(self, q: MemoryQuery) -> list[MemoryHit]: ...


@runtime_checkable
class EpisodicMemory(Protocol):
    async def write(self, entry: MemoryEntry) -> MemoryId: ...
    async def query(self, q: MemoryQuery) -> list[MemoryHit]: ...


@runtime_checkable
class SemanticMemory(Protocol):
    async def upsert(self, entry: MemoryEntry) -> MemoryId: ...
    async def query(self, q: MemoryQuery) -> list[MemoryHit]: ...


@runtime_checkable
class Memory(Protocol):
    """Unified memory facade passed to the Brain and Skills."""

    object: ObjectMemory
    place: PlaceMemory
    episodic: EpisodicMemory
    semantic: SemanticMemory

    async def write(self, entry: MemoryEntry) -> MemoryId: ...
    async def query(self, q: MemoryQuery) -> list[MemoryHit]: ...


class NullMemory:
    """No-op memory implementation — safe default when no server is configured."""

    class _NullSub:
        async def upsert(self, entry: MemoryEntry) -> MemoryId:
            return str(uuid.uuid4())

        async def write(self, entry: MemoryEntry) -> MemoryId:
            return str(uuid.uuid4())

        async def query(self, q: MemoryQuery) -> list[MemoryHit]:
            return []

    def __init__(self) -> None:
        sub = self._NullSub()
        self.object: Any = sub
        self.place: Any = sub
        self.episodic: Any = sub
        self.semantic: Any = sub

    async def write(self, entry: MemoryEntry) -> MemoryId:
        return str(uuid.uuid4())

    async def query(self, q: MemoryQuery) -> list[MemoryHit]:
        return []
