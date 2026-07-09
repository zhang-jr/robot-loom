"""RemoteSpatialMemory — HTTP client for an external SpatialMemory-like server.

The server is expected to expose two JSON endpoints:

    POST {server_url}/memory/write   body = MemoryEntry   -> {"memory_id": str}
    POST {server_url}/memory/query   body = MemoryQuery   -> {"hits": [MemoryHit]}

Routing by ``memory_type`` happens server-side: a single pair of endpoints
serves object / place / episodic / semantic. The four sub-memory attributes
exposed here all forward to the same client, carrying their ``memory_type`` in
the request body.
"""

from __future__ import annotations

from typing import Any

import httpx

from robot_harness.errors import MemoryServiceUnavailable
from robot_harness.memory.base import MemoryEntry, MemoryHit, MemoryId, MemoryQuery


class _RemoteSub:
    """Sub-memory facade that forwards every call to the parent HTTP client."""

    def __init__(self, parent: RemoteSpatialMemory) -> None:
        self._parent = parent

    async def upsert(self, entry: MemoryEntry) -> MemoryId:
        return await self._parent.write(entry)

    async def write(self, entry: MemoryEntry) -> MemoryId:
        return await self._parent.write(entry)

    async def query(self, q: MemoryQuery) -> list[MemoryHit]:
        return await self._parent.query(q)


class RemoteSpatialMemory:
    """Memory backed by a remote server, shared across robots and durable."""

    def __init__(self, server_url: str, *, timeout_s: float = 10.0) -> None:
        self._base = server_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base, timeout=timeout_s)
        sub = _RemoteSub(self)
        self.object: Any = sub
        self.place: Any = sub
        self.episodic: Any = sub
        self.semantic: Any = sub

    async def write(self, entry: MemoryEntry) -> MemoryId:
        data = await self._post("/memory/write", entry.model_dump(), robot_id=entry.robot_id)
        return str(data["memory_id"])

    async def query(self, q: MemoryQuery) -> list[MemoryHit]:
        data = await self._post("/memory/query", q.model_dump(), robot_id=q.robot_id)
        return [MemoryHit.model_validate(h) for h in data.get("hits", [])]

    async def _post(self, path: str, payload: dict[str, Any], *, robot_id: str) -> dict[str, Any]:
        try:
            resp = await self._client.post(path, json=payload)
            resp.raise_for_status()
            return dict(resp.json())
        except httpx.HTTPError as exc:
            raise MemoryServiceUnavailable(
                f"Memory server at {self._base} unreachable: {exc}",
                robot_id=robot_id,
                module_name="memory",
            ) from exc

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()
