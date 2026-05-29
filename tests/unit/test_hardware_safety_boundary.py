"""Every robot-actuating tool must route through SafetyEnvelope (ADR-007 / ADR-019).

These tests guard the boundary at the registry level: a tool that actuates the
robot declares ``hardware_bound`` and the registry gates it, while pure
capability tools stay ungated.
"""

from __future__ import annotations

import pytest

from robot_harness.config.schema import SafetyConfig
from robot_harness.errors import SafetyEnvelopeViolation
from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.generic.shell import ShellTool
from robot_harness.tools.robot_sdk.http_adapter import RobotSdkTool
from robot_harness.tools.robot_sdk.verbs import (
    ROBOT_SDK_MOVE_TO_POSE,
    VERB_TOOL_NAMES,
    build_robot_sdk_verb_tools,
)


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(RobotSdkTool())
    for v in build_robot_sdk_verb_tools():
        reg.register(v)
    return reg


def test_all_robot_tools_are_safety_gated() -> None:
    reg = _registry()
    assert reg.requires_safety_check("robot_sdk.execute_action")
    for name in VERB_TOOL_NAMES:
        assert reg.requires_safety_check(name), name


def test_non_hardware_tool_is_not_gated() -> None:
    reg = ToolRegistry()
    tool = ShellTool()
    reg.register(tool)
    assert not reg.requires_safety_check(tool.name)


def test_move_to_pose_maps_to_cartesian_safety_command() -> None:
    reg = _registry()
    ctx = ToolContext.create(robot_id="robot-0")
    cmd = reg.build_safety_command(
        ROBOT_SDK_MOVE_TO_POSE,
        {"robot_id": "robot-0", "target_pose": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0]},
        ctx,
    )
    assert cmd is not None
    assert cmd.command_type == "cartesian"
    assert cmd.values[:3] == [0.4, 0.0, 0.3]


@pytest.mark.asyncio
async def test_verb_out_of_bounds_pose_is_rejected() -> None:
    reg = _registry()
    env = SafetyEnvelope(SafetyConfig())
    ctx = ToolContext.create(robot_id="robot-0")
    cmd = reg.build_safety_command(
        ROBOT_SDK_MOVE_TO_POSE,
        {"robot_id": "robot-0", "target_pose": [99.0, 0.0, 0.3, 0.0, 0.0, 0.0]},
        ctx,
    )
    assert cmd is not None
    with pytest.raises(SafetyEnvelopeViolation):
        await env.check(cmd, trace_id="t", subtask_id="s")


@pytest.mark.asyncio
async def test_verb_in_bounds_pose_passes() -> None:
    reg = _registry()
    env = SafetyEnvelope(SafetyConfig())
    ctx = ToolContext.create(robot_id="robot-0")
    cmd = reg.build_safety_command(
        ROBOT_SDK_MOVE_TO_POSE,
        {"robot_id": "robot-0", "target_pose": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0]},
        ctx,
    )
    assert cmd is not None
    verdict = await env.check(cmd, trace_id="t", subtask_id="s")
    assert verdict.passed
