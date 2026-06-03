"""Unit tests for observation tools (proprioception + capability-gated capture)."""

from __future__ import annotations

from typing import Any

import pytest

from robot_harness.embodiment.base import Frame, RobotState
from robot_harness.tools.base import ToolContext
from robot_harness.tools.robot_sdk.observe import (
    CaptureFrameTool,
    GetStateTool,
    build_observation_tools,
)


class _FakeAdapter:
    robot_id = "r0"
    robot_type = "arm"

    async def get_state(self) -> RobotState:
        return RobotState(robot_id="r0", joint_positions=[0.1, 0.2, 0.3])

    async def get_camera_frame(self, camera: str) -> Frame:
        return Frame(camera=camera, robot_id="r0", data={"image_b64": "abc", "rendered": True})

    async def dispatch(self, cmd: Any) -> Any: ...  # unused
    async def safety_check(self, cmd: Any) -> Any: ...  # unused


def _ctx() -> ToolContext:
    return ToolContext.create("r0")


# --- capability gating -----------------------------------------------------


def test_get_state_always_built_capture_only_with_cameras() -> None:
    adapters = {"r0": _FakeAdapter()}
    no_cam = build_observation_tools(adapters, {"r0": []})
    assert [t.name for t in no_cam] == ["robot.get_state"]

    with_cam = build_observation_tools(adapters, {"r0": ["wrist"]})
    assert {t.name for t in with_cam} == {"robot.get_state", "robot.capture_frame"}


@pytest.mark.asyncio
async def test_get_state_returns_proprioception() -> None:
    tool = GetStateTool({"r0": _FakeAdapter()})
    res = await tool.invoke({"robot_id": "r0"}, _ctx())
    assert res.success
    assert res.output["joint_positions"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_get_state_unknown_robot_errors() -> None:
    tool = GetStateTool({})
    res = await tool.invoke({"robot_id": "ghost"}, _ctx())
    assert not res.success
    assert "ghost" in (res.error or "")


@pytest.mark.asyncio
async def test_capture_frame_success() -> None:
    tool = CaptureFrameTool({"r0": _FakeAdapter()}, {"r0": ["wrist", "overhead"]})
    res = await tool.invoke({"robot_id": "r0", "camera": "overhead"}, _ctx())
    assert res.success
    assert res.output["camera"] == "overhead"
    assert res.output["image_b64"] == "abc"


@pytest.mark.asyncio
async def test_capture_frame_defaults_to_first_camera() -> None:
    tool = CaptureFrameTool({"r0": _FakeAdapter()}, {"r0": ["wrist", "overhead"]})
    res = await tool.invoke({"robot_id": "r0"}, _ctx())
    assert res.success
    assert res.output["camera"] == "wrist"


@pytest.mark.asyncio
async def test_capture_frame_camera_less_robot_errors() -> None:
    tool = CaptureFrameTool({"r0": _FakeAdapter()}, {"r0": []})
    res = await tool.invoke({"robot_id": "r0"}, _ctx())
    assert not res.success
    assert "no camera" in (res.error or "")


@pytest.mark.asyncio
async def test_capture_frame_undeclared_camera_errors() -> None:
    tool = CaptureFrameTool({"r0": _FakeAdapter()}, {"r0": ["wrist"]})
    res = await tool.invoke({"robot_id": "r0", "camera": "thermal"}, _ctx())
    assert not res.success
    assert "thermal" in (res.error or "")
