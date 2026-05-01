"""Integration tests: all mock tool adapters end-to-end.

Verifies that each mock tool can be registered, invoked, and returns a
well-formed ToolResult without contacting any external server.
"""

from __future__ import annotations

import pytest

from robot_harness.memory.base import NullMemory
from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.critic.vlac_adapter import VlacCriticTool
from robot_harness.tools.generic.fs import ReadFileTool, WriteFileTool
from robot_harness.tools.generic.message import SendMessageTool
from robot_harness.tools.generic.shell import ShellTool
from robot_harness.tools.grasp.anygrasp_adapter import AnyGraspTool
from robot_harness.tools.memory.ingest_tool import MemoryIngestTool
from robot_harness.tools.memory.query_tool import MemoryQueryTool
from robot_harness.tools.memory.upsert_tool import MemoryUpsertTool
from robot_harness.tools.perception.depth_adapter import DepthEstimationTool
from robot_harness.tools.perception.yolo_adapter import YoloDetectionTool
from robot_harness.tools.robot_sdk.http_adapter import RobotSdkTool
from robot_harness.tools.vla.serving_adapter import VlaServingTool


def _ctx(robot_id: str = "robot-0") -> ToolContext:
    return ToolContext.create(robot_id)


def _registry_with_all_tools() -> ToolRegistry:
    memory = NullMemory()
    r = ToolRegistry()
    r.register(YoloDetectionTool())
    r.register(DepthEstimationTool())
    r.register(AnyGraspTool())
    r.register(VlacCriticTool())
    r.register(VlaServingTool())
    r.register(RobotSdkTool())
    r.register(ShellTool())
    r.register(ReadFileTool())
    r.register(WriteFileTool())
    r.register(SendMessageTool())
    r.register(MemoryQueryTool(memory))
    r.register(MemoryUpsertTool(memory))
    r.register(MemoryIngestTool(memory))
    return r


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_all_tools_registered() -> None:
    r = _registry_with_all_tools()
    expected = {
        "perception.detect_objects",
        "perception.estimate_depth",
        "grasp.estimate_pose",
        "critic.judge_progress",
        "vla.infer_action",
        "robot_sdk.execute_action",
        "shell_run",
        "fs_read",
        "fs_write",
        "generic.send_message",
        "memory.query",
        "memory.upsert",
        "memory.ingest",
    }
    for name in expected:
        assert name in r, f"Tool '{name}' not registered"


# ---------------------------------------------------------------------------
# Mock tool invocations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_yolo_detection_mock() -> None:
    tool = YoloDetectionTool()
    result = await tool.invoke({"image_source": "wrist_camera", "query": "cup"}, _ctx())
    assert result.success
    assert "detections" in (result.output or {})
    assert result.output["count"] >= 1  # type: ignore[index]


@pytest.mark.asyncio
async def test_depth_estimation_mock() -> None:
    tool = DepthEstimationTool()
    result = await tool.invoke({"image_source": "wrist_camera"}, _ctx())
    assert result.success
    assert (result.output or {}).get("depth_m", 0) > 0


@pytest.mark.asyncio
async def test_anygrasp_mock() -> None:
    tool = AnyGraspTool()
    result = await tool.invoke({"detections": {"count": 1}}, _ctx())
    assert result.success
    pose = (result.output or {}).get("pose", [])
    assert len(pose) == 6


@pytest.mark.asyncio
async def test_vlac_critic_mock() -> None:
    tool = VlacCriticTool()
    result = await tool.invoke({"task_description": "pick the cup"}, _ctx())
    assert result.success
    assert (result.output or {}).get("state") in ("progress", "completion", "failure", "unchanged")


@pytest.mark.asyncio
async def test_vla_serving_mock() -> None:
    tool = VlaServingTool()
    result = await tool.invoke(
        {"image_source": "head_camera", "instruction": "pick the cup", "robot_id": "robot-0"},
        _ctx(),
    )
    assert result.success
    actions = (result.output or {}).get("actions", [])
    assert len(actions) == 1
    assert len(actions[0]) == 6


@pytest.mark.asyncio
async def test_robot_sdk_execute_mock() -> None:
    tool = RobotSdkTool()
    result = await tool.invoke(
        {
            "robot_id": "robot-0",
            "command_type": "cartesian",
            "values": [0.5, 0.0, 0.3, 0.0, 0.0, 0.0],
        },
        _ctx(),
    )
    assert result.success
    assert "action_id" in (result.output or {})


@pytest.mark.asyncio
async def test_send_message_mock() -> None:
    tool = SendMessageTool()
    result = await tool.invoke({"text": "task complete", "level": "info"}, _ctx())
    assert result.success


# ---------------------------------------------------------------------------
# Memory tools with SpatialHubMemory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_upsert_and_query_via_tools() -> None:
    from robot_harness.tools.memory.spatial_hub_adapter import SpatialHubMemory

    hub = SpatialHubMemory()
    upsert = MemoryUpsertTool(hub)
    query = MemoryQueryTool(hub)

    ctx = _ctx()
    upsert_result = await upsert.invoke(
        {"memory_type": "object", "content": {"name": "red_cup"}, "tags": ["cup"]},
        ctx,
    )
    assert upsert_result.success

    query_result = await query.invoke(
        {"memory_type": "object", "text": "red_cup", "top_k": 5},
        ctx,
    )
    assert query_result.success
    hits = (query_result.output or {}).get("hits", [])
    assert len(hits) >= 1


@pytest.mark.asyncio
async def test_memory_ingest_via_tool() -> None:
    from robot_harness.tools.memory.spatial_hub_adapter import SpatialHubMemory

    hub = SpatialHubMemory()
    ingest = MemoryIngestTool(hub)
    result = await ingest.invoke(
        {
            "entries": [
                {"memory_type": "object", "content": {"name": "box"}},
                {"memory_type": "place", "content": {"name": "table"}},
            ]
        },
        _ctx(),
    )
    assert result.success
    assert (result.output or {}).get("count") == 2


# ---------------------------------------------------------------------------
# Schema validation (tool_name in result matches registered name)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_mock_tools_tool_name_in_result() -> None:
    tools_and_args: list[tuple[object, dict]] = [
        (YoloDetectionTool(), {"image_source": "cam", "query": "cup"}),
        (DepthEstimationTool(), {"image_source": "cam"}),
        (AnyGraspTool(), {"detections": {}}),
        (VlacCriticTool(), {"task_description": "pick"}),
        (
            VlaServingTool(),
            {"image_source": "cam", "instruction": "pick", "robot_id": "robot-0"},
        ),
        (
            RobotSdkTool(),
            {"robot_id": "robot-0", "command_type": "joint", "values": [0.0] * 6},
        ),
    ]
    for tool, args in tools_and_args:
        ctx = _ctx()
        result = await tool.invoke(args, ctx)  # type: ignore[union-attr]
        assert result.tool_name == tool.name  # type: ignore[union-attr]
