"""Every robot-actuating tool must route through SafetyEnvelope (ADR-007 / ADR-019).

These tests guard the boundary at the registry level: a tool that actuates the
robot declares ``hardware_bound`` and the registry gates it, while pure
capability tools stay ungated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_harness.config.schema import SafetyConfig
from robot_harness.errors import SafetyEnvelopeViolation
from robot_harness.runtime.skill_tools import SafetyGatedToolRegistry
from robot_harness.safety.audit_log import SafetyAuditLog
from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.generic.shell import ShellTool
from robot_harness.tools.robot_sdk.http_adapter import RobotSdkTool
from robot_harness.tools.robot_sdk.verbs import (
    ROBOT_SDK_MOVE_JOINTS,
    ROBOT_SDK_MOVE_TO_POSE,
    ROBOT_SDK_REACTIVE_GRASP,
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
    cmds = reg.build_safety_commands(
        ROBOT_SDK_MOVE_TO_POSE,
        {"robot_id": "robot-0", "target_pose": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0]},
        ctx,
    )
    assert len(cmds) == 1
    cmd = cmds[0]
    assert cmd.command_type == "cartesian"
    assert cmd.values[:3] == [0.4, 0.0, 0.3]


def test_move_joints_maps_to_joint_safety_command() -> None:
    reg = _registry()
    ctx = ToolContext.create(robot_id="robot-0")
    cmds = reg.build_safety_commands(
        ROBOT_SDK_MOVE_JOINTS,
        {"robot_id": "robot-0", "target_joints": [0.0, 0.5, -0.5, 0.0, 1.0, 0.0]},
        ctx,
    )
    assert len(cmds) == 1
    cmd = cmds[0]
    assert cmd.command_type == "joint"
    assert cmd.values == [0.0, 0.5, -0.5, 0.0, 1.0, 0.0]


@pytest.mark.asyncio
async def test_move_joints_beyond_configured_limits_is_rejected() -> None:
    reg = _registry()
    env = SafetyEnvelope(SafetyConfig(joint_limits_rad=[3.14] * 6))
    ctx = ToolContext.create(robot_id="robot-0")
    cmds = reg.build_safety_commands(
        ROBOT_SDK_MOVE_JOINTS,
        {"robot_id": "robot-0", "target_joints": [0.0, 9.0, 0.0, 0.0, 0.0, 0.0]},
        ctx,
    )
    assert len(cmds) == 1
    cmd = cmds[0]
    with pytest.raises(SafetyEnvelopeViolation):
        await env.check(cmd, trace_id="t", subtask_id="s")


@pytest.mark.asyncio
async def test_verb_out_of_bounds_pose_is_rejected() -> None:
    reg = _registry()
    env = SafetyEnvelope(SafetyConfig())
    ctx = ToolContext.create(robot_id="robot-0")
    cmds = reg.build_safety_commands(
        ROBOT_SDK_MOVE_TO_POSE,
        {"robot_id": "robot-0", "target_pose": [99.0, 0.0, 0.3, 0.0, 0.0, 0.0]},
        ctx,
    )
    assert len(cmds) == 1
    cmd = cmds[0]
    with pytest.raises(SafetyEnvelopeViolation):
        await env.check(cmd, trace_id="t", subtask_id="s")


@pytest.mark.asyncio
async def test_verb_in_bounds_pose_passes() -> None:
    reg = _registry()
    env = SafetyEnvelope(SafetyConfig())
    ctx = ToolContext.create(robot_id="robot-0")
    cmds = reg.build_safety_commands(
        ROBOT_SDK_MOVE_TO_POSE,
        {"robot_id": "robot-0", "target_pose": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0]},
        ctx,
    )
    assert len(cmds) == 1
    cmd = cmds[0]
    verdict = await env.check(cmd, trace_id="t", subtask_id="s")
    assert verdict.passed


@pytest.mark.asyncio
async def test_unchecked_hardware_dispatch_is_audited_as_skipped(tmp_path: Path) -> None:
    """reactive_grasp given only a phrase hint yields no harness-checkable
    command (the on-robot reflex is authoritative); the gate must record an
    honest "skipped" audit entry — never indistinguishable from "checked and
    passed", and never silent."""
    reg = _registry()
    audit = SafetyAuditLog(tmp_path / "audit.jsonl")
    env = SafetyEnvelope(SafetyConfig(), audit_log=audit)
    gated = SafetyGatedToolRegistry(reg, env)

    ctx = ToolContext.create(robot_id="robot-0")
    res = await gated.get(ROBOT_SDK_REACTIVE_GRASP).invoke(
        {"robot_id": "robot-0", "target_hint": {"kind": "phrase", "phrase": "red cup"}},
        ctx,
    )
    assert res.success
    (entry,) = audit.tail()
    assert entry.outcome == "skipped"
    assert entry.tool_name == ROBOT_SDK_REACTIVE_GRASP
    assert entry.robot_id == "robot-0"
