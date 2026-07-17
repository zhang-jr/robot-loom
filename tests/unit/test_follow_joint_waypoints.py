"""Unit tests for the builtin FollowJointWaypointsSkill.

Covers:
    * Parameter validation fails typed BEFORE any tool call
    * Happy path: one move_joints per waypoint, strictly in order, gripper and
      constraints ride along
    * Halt-on-failure: remaining waypoints are never dispatched, the failed
      index and verdict surface in the SkillResult
    * Brain-facing schema shape (Subtask envelope, waypoints required)
    * Builtin registration via HarnessContext.build
    * SafetyEnvelope (real, never mocked) blocks an out-of-limits waypoint
"""

from __future__ import annotations

from typing import Any

import pytest

from robot_harness.config.schema import SafetyConfig
from robot_harness.errors import SafetyEnvelopeViolation
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.runtime.skill_tools import SafetyGatedToolRegistry
from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.skill.base import Subtask
from robot_harness.skill.builtin.follow_joint_waypoints import FollowJointWaypointsSkill
from robot_harness.skill.registry import SkillRegistry
from robot_harness.tools.base import BrainProfile, ToolRegistry
from robot_harness.tools.robot_sdk.verbs import ROBOT_SDK_MOVE_JOINTS, MoveJointsTool


class _RecordingVerbAdapter:
    """SupportsVerbs adapter that records payloads and can fail at an index."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._fail_at = fail_at

    async def call_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]:
        index = len(self.calls)
        self.calls.append((verb, dict(payload)))
        if self._fail_at is not None and index == self._fail_at:
            return {
                "outcome": "failed",
                "evidence": "completed but 0.210rad off target",
                "aborted_by": "none",
            }
        return {
            "outcome": "success",
            "evidence": "joints within 0.005rad of target",
            "aborted_by": "none",
            "final_joints": payload.get("target_joints", []),
        }

    async def available_verbs(self) -> list[str] | None:
        return None


def _subtask(parameters: dict[str, Any]) -> Subtask:
    return Subtask(
        subtask_id="s1", description="run waypoints", robot_id="r0", parameters=parameters
    )


def _registry(adapter: _RecordingVerbAdapter | None = None) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(MoveJointsTool({"r0": adapter} if adapter else None))
    return reg


_WP1 = {"joints": [0.0, 0.5, -0.5, 0.0, 1.0, 0.0], "gripper": 2.0}
_WP2 = {"joints": [0.1, 0.4, -0.4, 0.0, 1.0, 0.0], "gripper": 2.0}
_WP3 = {"joints": [0.1, 0.4, -0.4, 0.0, 1.0, 0.0], "gripper": 5.0}


# ---------------------------------------------------------------------------
# Parameter validation — fail typed before any tool call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_waypoints_fails_before_any_tool_call() -> None:
    result = await FollowJointWaypointsSkill().execute(_subtask({}), ToolRegistry(), ctx=None)
    assert result.success is False
    assert "waypoints" in result.message


@pytest.mark.asyncio
async def test_empty_waypoints_list_fails() -> None:
    result = await FollowJointWaypointsSkill().execute(
        _subtask({"waypoints": []}), ToolRegistry(), ctx=None
    )
    assert result.success is False
    assert "waypoints" in result.message


@pytest.mark.asyncio
async def test_waypoint_without_joints_fails_naming_its_index() -> None:
    # Registry is empty: validation must reject the malformed 2nd point before
    # the 1st (valid) point ever moves the arm — no partial motion on bad input.
    result = await FollowJointWaypointsSkill().execute(
        _subtask({"waypoints": [_WP1, {"gripper": 5.0}]}), ToolRegistry(), ctx=None
    )
    assert result.success is False
    assert "2/2" in result.message


# ---------------------------------------------------------------------------
# Happy path — one move_joints per waypoint, in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatches_one_move_joints_per_waypoint_in_order() -> None:
    adapter = _RecordingVerbAdapter()
    result = await FollowJointWaypointsSkill().execute(
        _subtask({"waypoints": [_WP1, _WP2, _WP3]}), _registry(adapter), ctx=None
    )
    assert result.success is True
    assert result.artifacts["waypoints_completed"] == 3
    assert [verb for verb, _ in adapter.calls] == ["move_joints"] * 3
    assert [p["target_joints"] for _, p in adapter.calls] == [
        _WP1["joints"],
        _WP2["joints"],
        _WP3["joints"],
    ]
    # Per-waypoint gripper rides on the same motion call (release at _WP3).
    assert [p["gripper"] for _, p in adapter.calls] == [2.0, 2.0, 5.0]


@pytest.mark.asyncio
async def test_constraints_apply_to_every_waypoint_call() -> None:
    adapter = _RecordingVerbAdapter()
    await FollowJointWaypointsSkill().execute(
        _subtask({"waypoints": [_WP1, _WP2], "constraints": {"timeout_s": 12.0}}),
        _registry(adapter),
        ctx=None,
    )
    assert all(p["constraints"] == {"timeout_s": 12.0} for _, p in adapter.calls)


# ---------------------------------------------------------------------------
# Halt-on-failure — never skip ahead
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_waypoint_halts_sequence_and_reports_index() -> None:
    adapter = _RecordingVerbAdapter(fail_at=1)  # 2nd waypoint fails
    result = await FollowJointWaypointsSkill().execute(
        _subtask({"waypoints": [_WP1, _WP2, _WP3]}), _registry(adapter), ctx=None
    )
    assert result.success is False
    assert result.outcome == "partial"  # one point was reached
    assert "waypoint 2/3" in result.message
    assert "0.210rad off target" in result.message
    assert result.artifacts["waypoints_completed"] == 1
    assert result.artifacts["failed_waypoint_index"] == 1
    assert len(adapter.calls) == 2  # 3rd waypoint never dispatched


@pytest.mark.asyncio
async def test_first_waypoint_failure_is_full_failure_not_partial() -> None:
    adapter = _RecordingVerbAdapter(fail_at=0)
    result = await FollowJointWaypointsSkill().execute(
        _subtask({"waypoints": [_WP1, _WP2]}), _registry(adapter), ctx=None
    )
    assert result.success is False
    assert result.outcome == "failure"
    assert result.artifacts["waypoints_completed"] == 0
    assert len(adapter.calls) == 1


# ---------------------------------------------------------------------------
# Brain-facing schema + registration
# ---------------------------------------------------------------------------


def test_schema_keeps_subtask_envelope_and_requires_waypoints() -> None:
    reg = SkillRegistry()
    reg.register(FollowJointWaypointsSkill())
    (spec,) = reg.export_for_brain(BrainProfile(name="openai"))
    params = spec["function"]["parameters"]
    assert params["required"] == ["robot_id", "parameters"]
    assert params["properties"]["parameters"]["required"] == ["waypoints"]
    waypoint_schema = params["properties"]["parameters"]["properties"]["waypoints"]["items"]
    assert waypoint_schema["required"] == ["joints"]
    assert "gripper" in waypoint_schema["properties"]


def test_registered_as_builtin_skill() -> None:
    ctx = HarnessContext.build()
    specs = ctx.skill_registry.export_for_brain(BrainProfile(name="openai"))
    names = {s["function"]["name"] for s in specs}
    assert "skill.follow_joint_waypoints" in names


def test_manifest_requires_only_move_joints() -> None:
    """Availability gating: the skill must not be pruned on robots that
    advertise move_joints but not move_to_pose."""
    assert FollowJointWaypointsSkill.manifest.required_tools == [ROBOT_SDK_MOVE_JOINTS]


# ---------------------------------------------------------------------------
# SafetyEnvelope boundary — real envelope, never mocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_out_of_limits_waypoint_trips_safety_envelope() -> None:
    """Every waypoint passes the joint-limit check pre-dispatch; a violation
    propagates uncaught (the skill must never swallow it) and the offending
    point never reaches the robot."""
    adapter = _RecordingVerbAdapter()
    gated = SafetyGatedToolRegistry(
        _registry(adapter), SafetyEnvelope(SafetyConfig(joint_limits_rad=[3.14] * 6))
    )
    bad = {"joints": [0.0, 9.0, 0.0, 0.0, 0.0, 0.0], "gripper": 2.0}
    with pytest.raises(SafetyEnvelopeViolation):
        await FollowJointWaypointsSkill().execute(
            _subtask({"waypoints": [_WP1, bad, _WP3]}), gated, ctx=None
        )
    # The safe 1st point ran; the violating 2nd never dispatched.
    assert len(adapter.calls) == 1
