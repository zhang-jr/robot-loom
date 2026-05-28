"""In-memory SemanticMemory stub — keyword-based knowledge store."""

from __future__ import annotations

import uuid
from typing import Any

from robot_harness.memory.base import MemoryEntry, MemoryHit, MemoryId, MemoryQuery


class InMemorySemanticMemory:
    """Dict-backed in-process semantic memory stub.

    Uses simple substring matching as a stand-in for embedding search.
    TODO: replace with a Qdrant/Milvus/FAISS-backed adapter when a real
    semantic memory service is available.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def upsert(self, entry: MemoryEntry) -> MemoryId:
        mid = str(uuid.uuid4())
        self._store[mid] = {
            "memory_id": mid,
            "robot_id": entry.robot_id,
            "content": entry.content,
            "tags": entry.tags,
        }
        return mid

    async def query(self, q: MemoryQuery) -> list[MemoryHit]:
        results: list[MemoryHit] = []
        text = q.text.lower()
        for mid, record in self._store.items():
            if q.robot_id and record["robot_id"] != q.robot_id:
                continue
            if q.tags and not any(t in record["tags"] for t in q.tags):
                continue
            content_str = str(record["content"]).lower()
            score = 1.0 if text and text in content_str else 0.3
            results.append(MemoryHit(memory_id=mid, score=score, content=record["content"]))
        results.sort(key=lambda h: h.score, reverse=True)
        return results[: q.top_k]
