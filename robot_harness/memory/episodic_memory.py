"""In-memory EpisodicMemory stub — append-only episode log."""

from __future__ import annotations

import uuid
from typing import Any

from robot_harness.memory.base import MemoryEntry, MemoryHit, MemoryId, MemoryQuery


class InMemoryEpisodicMemory:
    """List-backed in-process episodic memory stub.

    Episodic memory is append-only — no upsert, each episode is a new entry.
    """

    # TODO (ADR-020): replace with an adapter to an external memory server.

    def __init__(self) -> None:
        self._log: list[dict[str, Any]] = []

    async def write(self, entry: MemoryEntry) -> MemoryId:
        mid = str(uuid.uuid4())
        self._log.append(
            {
                "memory_id": mid,
                "robot_id": entry.robot_id,
                "content": entry.content,
                "tags": entry.tags,
            }
        )
        return mid

    async def query(self, q: MemoryQuery) -> list[MemoryHit]:
        results: list[MemoryHit] = []
        text = q.text.lower()
        for record in reversed(self._log):  # most-recent first
            if q.robot_id and record["robot_id"] != q.robot_id:
                continue
            if q.tags and not any(t in record["tags"] for t in q.tags):
                continue
            content_str = str(record["content"]).lower()
            score = 1.0 if text and text in content_str else 0.5
            results.append(
                MemoryHit(memory_id=record["memory_id"], score=score, content=record["content"])
            )
            if len(results) >= q.top_k:
                break
        return results
