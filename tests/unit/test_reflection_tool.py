"""Unit tests for ReflectionStore and ReflectionTool (Cognitive Scaffold — ADR-018)."""

from __future__ import annotations

import pytest

from robot_harness.tools.base import ToolContext
from robot_harness.tools.cognitive.reflection_tool import ReflectionStore, ReflectionTool


@pytest.fixture()
def store() -> ReflectionStore:
    return ReflectionStore()


@pytest.fixture()
def tool(store: ReflectionStore) -> ReflectionTool:
    return ReflectionTool(store)


@pytest.fixture()
def ctx() -> ToolContext:
    return ToolContext(trace_id="t1", robot_id="arm0")


_ENTRY = {
    "target_id": "step_1",
    "target_desc": "Perception result",
    "verdict": "fail",
    "reasoning": "No objects detected",
    "suggestion": "Retry with higher confidence threshold",
}


# ---------------------------------------------------------------------------
# ReflectionStore — append / read
# ---------------------------------------------------------------------------


def test_empty_store_has_no_content(store: ReflectionStore) -> None:
    assert not store.has_content()
    assert store.format_for_injection() is None


def test_append_adds_entry(store: ReflectionStore) -> None:
    store.append(_ENTRY)
    assert store.has_content()
    assert len(store.read()) == 1


def test_entries_are_sequenced(store: ReflectionStore) -> None:
    store.append(_ENTRY)
    store.append({**_ENTRY, "target_id": "step_2", "verdict": "pass", "reasoning": "OK"})
    entries = store.read()
    assert entries[0]["seq"] == 1
    assert entries[1]["seq"] == 2


def test_invalid_verdict_coerced_to_fail(store: ReflectionStore) -> None:
    store.append({**_ENTRY, "verdict": "bogus"})
    assert store.read()[0]["verdict"] == "fail"


def test_suggestion_optional(store: ReflectionStore) -> None:
    store.append({"target_id": "x", "target_desc": "y", "verdict": "pass", "reasoning": "r"})
    entry = store.read()[0]
    assert "suggestion" not in entry


# ---------------------------------------------------------------------------
# format_for_injection
# ---------------------------------------------------------------------------


def test_injection_excludes_resolved_targets(store: ReflectionStore) -> None:
    store.append(_ENTRY)  # fail on step_1
    store.append({**_ENTRY, "verdict": "pass", "reasoning": "Fixed"})  # resolve step_1
    assert store.format_for_injection() is None


def test_injection_includes_unresolved(store: ReflectionStore) -> None:
    store.append(_ENTRY)
    injected = store.format_for_injection()
    assert injected is not None
    assert "step_1" in injected
    assert "No objects detected" in injected


def test_injection_multi_target_partial_resolution(store: ReflectionStore) -> None:
    store.append({**_ENTRY, "target_id": "a", "verdict": "fail", "reasoning": "A failed"})
    store.append({**_ENTRY, "target_id": "b", "verdict": "warn", "reasoning": "B warned"})
    store.append({**_ENTRY, "target_id": "a", "verdict": "pass", "reasoning": "A fixed"})
    injected = store.format_for_injection()
    assert injected is not None
    assert "b" in injected
    assert "[a]" not in injected


# ---------------------------------------------------------------------------
# ReflectionTool — NativeTool wrapper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflection_tool_read_empty(tool: ReflectionTool, ctx: ToolContext) -> None:
    result = await tool.invoke({}, ctx)
    assert result.success
    assert result.output is not None
    assert result.output["reflections"] == []
    assert result.output["summary"]["total"] == 0


@pytest.mark.asyncio
async def test_reflection_tool_append_and_read(tool: ReflectionTool, ctx: ToolContext) -> None:
    result = await tool.invoke({"entry": _ENTRY}, ctx)
    assert result.success
    assert result.output["summary"]["fail"] == 1
    assert result.output["last"]["target_id"] == "step_1"


@pytest.mark.asyncio
async def test_reflection_tool_is_idempotent(tool: ReflectionTool) -> None:
    assert tool.is_idempotent is True


@pytest.mark.asyncio
async def test_reflection_tool_not_cancellable(tool: ReflectionTool) -> None:
    assert tool.is_cancellable is False
