"""Unit tests for PlannerStore and PlanTool (Cognitive Scaffold — ADR-018)."""

from __future__ import annotations

import pytest

from robot_harness.tools.base import ToolContext
from robot_harness.tools.cognitive.plan_tool import PlannerStore, PlanTool


@pytest.fixture()
def store() -> PlannerStore:
    return PlannerStore()


@pytest.fixture()
def tool(store: PlannerStore) -> PlanTool:
    return PlanTool(store)


@pytest.fixture()
def ctx() -> ToolContext:
    return ToolContext(trace_id="t1", robot_id="arm0")


# ---------------------------------------------------------------------------
# PlannerStore — write / read
# ---------------------------------------------------------------------------


def test_empty_store_has_no_content(store: PlannerStore) -> None:
    assert not store.has_content()
    assert store.format_for_injection() is None


def test_write_goal_sets_content(store: PlannerStore) -> None:
    store.write(goal="Pick the red block")
    assert store.has_content()


def test_write_steps_replace_mode(store: PlannerStore) -> None:
    store.write(
        goal="task",
        steps=[
            {"id": "1", "description": "perceive", "status": "pending"},
            {"id": "2", "description": "grasp", "status": "pending"},
        ],
    )
    plan = store.read()
    assert len(plan["steps"]) == 2
    assert plan["steps"][0]["id"] == "1"


def test_write_steps_merge_status_update(store: PlannerStore) -> None:
    store.write(
        steps=[
            {"id": "1", "description": "perceive", "status": "pending"},
            {"id": "2", "description": "grasp", "status": "pending"},
        ]
    )
    store.write(steps=[{"id": "1", "description": "", "status": "completed"}], merge=True)
    plan = store.read()
    completed = [s for s in plan["steps"] if s["id"] == "1"]
    assert completed[0]["status"] == "completed"
    # step 2 still present
    assert len(plan["steps"]) == 2


def test_dedupe_keeps_last_occurrence(store: PlannerStore) -> None:
    store.write(
        steps=[
            {"id": "1", "description": "first", "status": "pending"},
            {"id": "1", "description": "second", "status": "in_progress"},
        ]
    )
    plan = store.read()
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["description"] == "second"


def test_invalid_status_falls_back_to_pending(store: PlannerStore) -> None:
    store.write(steps=[{"id": "x", "description": "foo", "status": "nonsense"}])
    plan = store.read()
    assert plan["steps"][0]["status"] == "pending"


def test_depends_on_preserved(store: PlannerStore) -> None:
    store.write(steps=[{"id": "2", "description": "grasp", "status": "pending", "depends_on": "1"}])
    plan = store.read()
    assert plan["steps"][0].get("depends_on") == "1"


# ---------------------------------------------------------------------------
# format_for_injection
# ---------------------------------------------------------------------------


def test_format_for_injection_excludes_completed(store: PlannerStore) -> None:
    store.write(
        goal="G",
        steps=[
            {"id": "1", "description": "done", "status": "completed"},
            {"id": "2", "description": "pending", "status": "pending"},
        ],
    )
    injected = store.format_for_injection()
    assert injected is not None
    assert "done" not in injected
    assert "pending" in injected


def test_format_for_injection_none_when_all_complete(store: PlannerStore) -> None:
    store.write(
        goal="G",
        steps=[
            {"id": "1", "description": "step", "status": "completed"},
        ],
    )
    assert store.format_for_injection() is None


def test_format_for_injection_contains_goal(store: PlannerStore) -> None:
    store.write(
        goal="Pick the red block",
        steps=[
            {"id": "1", "description": "perceive", "status": "pending"},
        ],
    )
    injected = store.format_for_injection()
    assert "Pick the red block" in (injected or "")


# ---------------------------------------------------------------------------
# PlanTool — NativeTool wrapper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_tool_read_empty(tool: PlanTool, ctx: ToolContext) -> None:
    result = await tool.invoke({}, ctx)
    assert result.success
    assert result.output is not None
    assert result.output["plan"]["goal"] == ""
    assert result.output["summary"]["total"] == 0


@pytest.mark.asyncio
async def test_plan_tool_write_then_read(tool: PlanTool, ctx: ToolContext) -> None:
    result = await tool.invoke(
        {
            "goal": "Pick red block",
            "steps": [
                {"id": "1", "description": "perceive", "status": "pending"},
                {"id": "2", "description": "grasp", "status": "pending"},
            ],
        },
        ctx,
    )
    assert result.success
    assert result.output["plan"]["goal"] == "Pick red block"
    assert result.output["summary"]["total"] == 2
    assert result.output["summary"]["pending"] == 2


@pytest.mark.asyncio
async def test_plan_tool_merge_update(tool: PlanTool, ctx: ToolContext) -> None:
    await tool.invoke({"steps": [{"id": "1", "description": "step", "status": "pending"}]}, ctx)
    result = await tool.invoke(
        {"steps": [{"id": "1", "description": "", "status": "in_progress"}], "merge": True}, ctx
    )
    assert result.output["summary"]["in_progress"] == 1


@pytest.mark.asyncio
async def test_plan_tool_is_idempotent(tool: PlanTool) -> None:
    assert tool.is_idempotent is True


@pytest.mark.asyncio
async def test_plan_tool_not_cancellable(tool: PlanTool) -> None:
    assert tool.is_cancellable is False
