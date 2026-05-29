"""Unit tests for memory backend selection and the remote HTTP adapter."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from robot_harness.config.schema import MemoryConfig
from robot_harness.errors import MemoryServiceUnavailable
from robot_harness.memory.base import MemoryEntry, MemoryQuery, NullMemory
from robot_harness.observability.tracer import tracer
from robot_harness.tools.memory.backend import build_memory
from robot_harness.tools.memory.remote_adapter import RemoteSpatialMemory
from robot_harness.tools.memory.spatial_hub_adapter import SpatialHubMemory

# ---------------------------------------------------------------------------
# Factory selection
# ---------------------------------------------------------------------------


def test_build_memory_null() -> None:
    assert isinstance(build_memory(MemoryConfig(backend="null")), NullMemory)


def test_build_memory_embedded() -> None:
    assert isinstance(build_memory(MemoryConfig(backend="embedded")), SpatialHubMemory)


def test_build_memory_external() -> None:
    mem = build_memory(MemoryConfig(backend="external", server_url="http://mem:8080"))
    assert isinstance(mem, RemoteSpatialMemory)


def test_build_memory_external_without_url_raises() -> None:
    with pytest.raises(MemoryServiceUnavailable):
        build_memory(MemoryConfig(backend="external"))


# ---------------------------------------------------------------------------
# Fleet consistency warning
# ---------------------------------------------------------------------------


def _capture_events(fn: Callable[[], Any]) -> list[dict[str, Any]]:
    old = tracer._sink
    buf = io.StringIO()
    tracer._sink = buf
    try:
        fn()
    finally:
        tracer._sink = old
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


def _warned(events: list[dict[str, Any]]) -> bool:
    return any(e.get("event") == "memory.backend_not_shareable" for e in events)


def test_fleet_with_embedded_emits_warning() -> None:
    events = _capture_events(lambda: build_memory(MemoryConfig(backend="embedded"), fleet_size=3))
    assert _warned(events)


def test_fleet_with_null_emits_warning() -> None:
    events = _capture_events(lambda: build_memory(MemoryConfig(backend="null"), fleet_size=2))
    assert _warned(events)


def test_fleet_with_external_no_warning() -> None:
    cfg = MemoryConfig(backend="external", server_url="http://m")
    events = _capture_events(lambda: build_memory(cfg, fleet_size=3))
    assert not _warned(events)


def test_single_robot_embedded_no_warning() -> None:
    events = _capture_events(lambda: build_memory(MemoryConfig(backend="embedded"), fleet_size=1))
    assert not _warned(events)


# ---------------------------------------------------------------------------
# Remote adapter against a mock transport
# ---------------------------------------------------------------------------


async def _remote_with(handler: Callable[[httpx.Request], httpx.Response]) -> RemoteSpatialMemory:
    mem = RemoteSpatialMemory("http://mem.test")
    await mem._client.aclose()
    mem._client = httpx.AsyncClient(
        base_url="http://mem.test", transport=httpx.MockTransport(handler)
    )
    return mem


@pytest.mark.asyncio
async def test_remote_write_posts_entry_and_returns_id() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"memory_id": "m-1"})

    mem = await _remote_with(handler)
    mid = await mem.write(MemoryEntry(memory_type="object", robot_id="r0", content={"name": "cup"}))
    assert mid == "m-1"
    assert seen[0].url.path == "/memory/write"
    await mem.aclose()


@pytest.mark.asyncio
async def test_remote_query_parses_hits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"hits": [{"memory_id": "m1", "score": 0.9, "content": {"name": "cup"}}]},
        )

    mem = await _remote_with(handler)
    hits = await mem.query(MemoryQuery(memory_type="object", robot_id="r0", text="cup"))
    assert len(hits) == 1
    assert hits[0].memory_id == "m1"
    assert hits[0].score == 0.9
    await mem.aclose()


@pytest.mark.asyncio
async def test_remote_connection_error_raises_service_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    mem = await _remote_with(handler)
    with pytest.raises(MemoryServiceUnavailable):
        await mem.write(MemoryEntry(memory_type="object", robot_id="r0", content={}))
    await mem.aclose()


@pytest.mark.asyncio
async def test_remote_http_error_raises_service_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    mem = await _remote_with(handler)
    with pytest.raises(MemoryServiceUnavailable):
        await mem.query(MemoryQuery(memory_type="object", robot_id="r0"))
    await mem.aclose()
