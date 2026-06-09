"""Tests for the artifact handle data plane: store + resolver middleware."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from robot_harness.errors import ToolSchemaViolationError
from robot_harness.tools.artifacts import InMemoryArtifactStore
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.middleware.artifact_resolver import ArtifactResolverMiddleware
from robot_harness.tools.schema import ToolBackend, ToolSchema


@pytest.mark.asyncio
async def test_store_put_get_roundtrip() -> None:
    store = InMemoryArtifactStore()
    ref = await store.put(b"\x89PNG-bytes", "image/png", meta={"camera": "overhead"})
    assert ref.media_type == "image/png"
    assert ref.meta["camera"] == "overhead"

    got = await store.get(ref.artifact_id)
    assert got is not None
    data, got_ref = got
    assert data == b"\x89PNG-bytes"
    assert got_ref.artifact_id == ref.artifact_id


@pytest.mark.asyncio
async def test_store_get_missing_returns_none() -> None:
    store = InMemoryArtifactStore()
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_store_evicts_oldest_past_capacity() -> None:
    store = InMemoryArtifactStore(capacity=2)
    a = await store.put(b"a", "image/png")
    b = await store.put(b"b", "image/png")
    await store.get(a.artifact_id)  # touch a → b is now the LRU
    c = await store.put(b"c", "image/png")  # evicts b

    assert await store.get(a.artifact_id) is not None
    assert await store.get(c.artifact_id) is not None
    assert await store.get(b.artifact_id) is None


# ---------------------------------------------------------------------------
# Resolver middleware
# ---------------------------------------------------------------------------


class _RecordingTool:
    """Inner tool that records the args it actually receives (post-resolution)."""

    name = "fake.consumer"
    backend: ToolBackend = "native"
    schema = ToolSchema(name="fake.consumer", description="x", input_schema={"type": "object"})

    def __init__(self) -> None:
        self.seen_args: dict[str, Any] | None = None

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.seen_args = args
        return ToolResult(tool_name=self.name, trace_id=ctx.trace_id, success=True, output={})

    async def cancel(self, ctx: ToolContext) -> None:
        pass


def _ctx(store: InMemoryArtifactStore | None) -> ToolContext:
    return ToolContext(trace_id="t", robot_id="r0", artifact_store=store)


@pytest.mark.asyncio
async def test_resolver_hydrates_frame_ref_into_image_b64() -> None:
    store = InMemoryArtifactStore()
    raw = b"the-real-pixels"
    ref = await store.put(raw, "image/png")

    inner = _RecordingTool()
    wrapped = ArtifactResolverMiddleware(inner)

    result = await wrapped.invoke({"frame": ref.model_dump(), "prompts": ["cube"]}, _ctx(store))
    assert result.success
    # inner saw image_b64 (re-encoded raw bytes), no leftover frame ref
    assert inner.seen_args is not None
    assert "frame" not in inner.seen_args
    assert inner.seen_args["prompts"] == ["cube"]
    assert base64.b64decode(inner.seen_args["image_b64"]) == raw


@pytest.mark.asyncio
async def test_resolver_passthrough_when_no_ref() -> None:
    inner = _RecordingTool()
    wrapped = ArtifactResolverMiddleware(inner)
    await wrapped.invoke({"image_b64": "abc", "prompts": ["cube"]}, _ctx(InMemoryArtifactStore()))
    assert inner.seen_args == {"image_b64": "abc", "prompts": ["cube"]}


@pytest.mark.asyncio
async def test_resolver_missing_artifact_raises() -> None:
    inner = _RecordingTool()
    wrapped = ArtifactResolverMiddleware(inner)
    with pytest.raises(ToolSchemaViolationError):
        await wrapped.invoke({"frame": {"artifact_id": "gone"}}, _ctx(InMemoryArtifactStore()))


@pytest.mark.asyncio
async def test_resolver_ref_without_store_raises() -> None:
    inner = _RecordingTool()
    wrapped = ArtifactResolverMiddleware(inner)
    with pytest.raises(ToolSchemaViolationError):
        await wrapped.invoke({"frame": {"artifact_id": "x"}}, _ctx(None))
