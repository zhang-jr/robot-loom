"""Artifact handles — out-of-band data plane for large tool I/O.

Large binaries (camera images, depth maps, masks, point clouds) must never
travel through the LLM context or be inlined into Memory. Instead the producing
tool stores the bytes in a session-scoped :class:`ArtifactStore` and returns a
small :class:`ArtifactRef`; the consuming tool receives the ref, and the harness
resolves it back to bytes just before dispatch (see
``tools.middleware.artifact_resolver``). The Brain only ever sees the small ref.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ArtifactRef(BaseModel):
    """Small, LLM-safe handle to a binary stored out-of-band.

    Carries only an id plus lightweight metadata (dimensions, camera, scale) —
    never the bytes. Safe to put in a tool result, Memory, or a prompt.
    """

    artifact_id: str
    media_type: str = "application/octet-stream"
    meta: dict[str, Any] = Field(default_factory=dict)


# JSON-schema fragment so a consuming tool can advertise an ArtifactRef input.
# Deliberately a flat object (no oneOf/const) so strict tool-calling backends
# (e.g. Volcengine Ark) accept it.
ARTIFACT_REF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Handle to a binary stored out-of-band (e.g. the `frame` returned by "
        "robot.capture_frame). The harness resolves it to bytes before dispatch; "
        "never inline the raw image."
    ),
    "properties": {
        "artifact_id": {"type": "string"},
        "media_type": {"type": "string"},
        "meta": {"type": "object"},
    },
    "required": ["artifact_id"],
}


@runtime_checkable
class ArtifactStore(Protocol):
    """Session-scoped store for out-of-band tool binaries."""

    async def put(
        self, data: bytes, media_type: str, meta: dict[str, Any] | None = None
    ) -> ArtifactRef: ...

    async def get(self, artifact_id: str) -> tuple[bytes, ArtifactRef] | None: ...


class InMemoryArtifactStore:
    """In-process, bounded-LRU artifact store.

    Holds binaries out of the LLM context for one session. Not durable, not
    shared across processes — a remote/object-store backend can replace it
    later behind the same :class:`ArtifactStore` Protocol. The LRU cap bounds
    memory: the oldest artifact is evicted once ``capacity`` is exceeded.
    """

    def __init__(self, capacity: int = 64) -> None:
        self._items: OrderedDict[str, tuple[bytes, ArtifactRef]] = OrderedDict()
        self._capacity = capacity

    async def put(
        self, data: bytes, media_type: str, meta: dict[str, Any] | None = None
    ) -> ArtifactRef:
        ref = ArtifactRef(artifact_id=str(uuid.uuid4()), media_type=media_type, meta=meta or {})
        self._items[ref.artifact_id] = (data, ref)
        self._items.move_to_end(ref.artifact_id)
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)
        return ref

    async def get(self, artifact_id: str) -> tuple[bytes, ArtifactRef] | None:
        item = self._items.get(artifact_id)
        if item is not None:
            self._items.move_to_end(artifact_id)
        return item
