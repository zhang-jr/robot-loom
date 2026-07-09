"""Tests for SafetyEnvelope — must use REAL validation logic (no mocks).

Per CLAUDE.md: no mock SafetyEnvelope allowed, tests must use real validation logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_harness.config.schema import SafetyConfig
from robot_harness.embodiment.base import EmbodimentCommand
from robot_harness.errors import SafetyEnvelopeViolation
from robot_harness.safety.audit_log import SafetyAuditLog
from robot_harness.safety.envelope import SafetyEnvelope


@pytest.fixture()
def audit_log(tmp_path: Path) -> SafetyAuditLog:
    return SafetyAuditLog(path=tmp_path / "audit.jsonl")


@pytest.fixture()
def envelope(audit_log: SafetyAuditLog) -> SafetyEnvelope:
    cfg = SafetyConfig(
        enabled=True,
        joint_limits_rad=[1.0, 1.0, 1.0],
        workspace_bounds_m=[-2.0, -2.0, 0.0, 2.0, 2.0, 2.0],
    )
    return SafetyEnvelope(cfg, audit_log=audit_log)


@pytest.mark.asyncio
async def test_joint_command_within_limits_passes(envelope: SafetyEnvelope) -> None:
    cmd = EmbodimentCommand(robot_id="r0", command_type="joint", values=[0.5, -0.5, 0.9])
    verdict = await envelope.check(cmd, trace_id="t1")
    assert verdict.passed


@pytest.mark.asyncio
async def test_joint_position_exceeds_limit_raises(envelope: SafetyEnvelope) -> None:
    # Joint command values are target POSITIONS (radians) — checked against the
    # configured per-joint limits, never against a velocity cap (ISS-028).
    cmd = EmbodimentCommand(robot_id="r0", command_type="joint", values=[2.0])
    with pytest.raises(SafetyEnvelopeViolation) as exc_info:
        await envelope.check(cmd, trace_id="t2")
    assert exc_info.value.violated_rules
    assert "position" in exc_info.value.violated_rules[0].lower()


@pytest.mark.asyncio
async def test_joint_command_with_more_joints_than_limits_raises(
    envelope: SafetyEnvelope,
) -> None:
    # Misconfiguration fails closed: joints beyond the configured limit list
    # must not go unchecked silently.
    cmd = EmbodimentCommand(robot_id="r0", command_type="joint", values=[0.1, 0.1, 0.1, 0.1])
    with pytest.raises(SafetyEnvelopeViolation, match="only 3 joint limits"):
        await envelope.check(cmd, trace_id="t2b")


@pytest.mark.asyncio
async def test_joint_without_limits_is_skipped_not_passed(
    audit_log: SafetyAuditLog,
) -> None:
    """No joint_limits_rad configured → honest skip: dispatch is not blocked but
    the audit records "skipped", never a false "passed" (ISS-028)."""
    env = SafetyEnvelope(SafetyConfig(), audit_log=audit_log)
    cmd = EmbodimentCommand(robot_id="r0", command_type="joint", values=[99.0])
    verdict = await env.check(cmd, trace_id="t2c")
    assert verdict.passed
    assert "unchecked" in verdict.reason
    entries = audit_log.tail(5)
    assert any(e.outcome == "skipped" for e in entries)
    assert not any(e.outcome == "passed" for e in entries)


@pytest.mark.asyncio
async def test_delta_step_cap_enforced_and_skipped_when_unconfigured(
    audit_log: SafetyAuditLog,
) -> None:
    capped = SafetyEnvelope(SafetyConfig(max_delta_step=0.25), audit_log=audit_log)
    ok = EmbodimentCommand(robot_id="r0", command_type="delta", values=[0.1, -0.2, 0.0])
    assert (await capped.check(ok, trace_id="t2d")).passed
    too_big = EmbodimentCommand(robot_id="r0", command_type="delta", values=[0.6])
    with pytest.raises(SafetyEnvelopeViolation, match="delta"):
        await capped.check(too_big, trace_id="t2e")

    uncapped = SafetyEnvelope(SafetyConfig(), audit_log=audit_log)
    verdict = await uncapped.check(too_big, trace_id="t2f")
    assert verdict.passed
    assert "unchecked" in verdict.reason


@pytest.mark.asyncio
async def test_cartesian_in_bounds_passes(envelope: SafetyEnvelope) -> None:
    cmd = EmbodimentCommand(robot_id="r0", command_type="cartesian", values=[0.5, 0.5, 1.0])
    verdict = await envelope.check(cmd, trace_id="t3")
    assert verdict.passed


@pytest.mark.asyncio
async def test_cartesian_out_of_bounds_raises(envelope: SafetyEnvelope) -> None:
    cmd = EmbodimentCommand(robot_id="r0", command_type="cartesian", values=[5.0, 0.0, 1.0])
    with pytest.raises(SafetyEnvelopeViolation):
        await envelope.check(cmd, trace_id="t4")


@pytest.mark.asyncio
async def test_audit_log_written_on_violation(
    envelope: SafetyEnvelope, audit_log: SafetyAuditLog
) -> None:
    cmd = EmbodimentCommand(robot_id="r0", command_type="joint", values=[999.0])
    with pytest.raises(SafetyEnvelopeViolation):
        await envelope.check(cmd, trace_id="t5")
    entries = audit_log.tail(5)
    assert any(e.outcome == "violated" for e in entries)


@pytest.mark.asyncio
async def test_audit_log_written_on_pass(
    envelope: SafetyEnvelope, audit_log: SafetyAuditLog
) -> None:
    cmd = EmbodimentCommand(robot_id="r0", command_type="joint", values=[0.1])
    await envelope.check(cmd, trace_id="t6")
    entries = audit_log.tail(5)
    assert any(e.outcome == "passed" for e in entries)


# ---------------------------------------------------------------------------
# Locomotion — map-frame geofence
# ---------------------------------------------------------------------------


@pytest.fixture()
def geofenced_envelope(audit_log: SafetyAuditLog) -> SafetyEnvelope:
    cfg = SafetyConfig(map_bounds_m=[-5.0, -5.0, 5.0, 5.0])
    return SafetyEnvelope(cfg, audit_log=audit_log)


@pytest.mark.asyncio
async def test_locomotion_goal_inside_geofence_passes(
    geofenced_envelope: SafetyEnvelope,
) -> None:
    cmd = EmbodimentCommand(robot_id="r0", command_type="locomotion", values=[1.0, 2.0, 0.5])
    verdict = await geofenced_envelope.check(cmd, trace_id="t8")
    assert verdict.passed


@pytest.mark.asyncio
async def test_locomotion_goal_outside_geofence_raises(
    geofenced_envelope: SafetyEnvelope, audit_log: SafetyAuditLog
) -> None:
    cmd = EmbodimentCommand(robot_id="r0", command_type="locomotion", values=[9.0, 0.0])
    with pytest.raises(SafetyEnvelopeViolation) as exc_info:
        await geofenced_envelope.check(cmd, trace_id="t9")
    assert "geofence" in exc_info.value.violated_rules[0]
    assert any(e.outcome == "violated" for e in audit_log.tail(5))


@pytest.mark.asyncio
async def test_locomotion_yaw_is_not_geofenced(geofenced_envelope: SafetyEnvelope) -> None:
    # Yaw far beyond any positional bound must not trip the geofence.
    cmd = EmbodimentCommand(robot_id="r0", command_type="locomotion", values=[0.0, 0.0, 99.0])
    verdict = await geofenced_envelope.check(cmd, trace_id="t10")
    assert verdict.passed


@pytest.mark.asyncio
async def test_locomotion_malformed_goal_raises(geofenced_envelope: SafetyEnvelope) -> None:
    # A goal that is not [x, y] / [x, y, yaw] cannot be validated — reject it.
    cmd = EmbodimentCommand(robot_id="r0", command_type="locomotion", values=[1.0])
    with pytest.raises(SafetyEnvelopeViolation):
        await geofenced_envelope.check(cmd, trace_id="t11")


@pytest.mark.asyncio
async def test_locomotion_without_geofence_is_skipped_not_passed(
    envelope: SafetyEnvelope, audit_log: SafetyAuditLog
) -> None:
    """No map_bounds_m configured → honest skip: verdict passes (dispatch is not
    blocked) but the audit records "skipped", never a false "passed"."""
    cmd = EmbodimentCommand(robot_id="r0", command_type="locomotion", values=[9999.0, 9999.0])
    verdict = await envelope.check(cmd, trace_id="t12")
    assert verdict.passed
    assert "unchecked" in verdict.reason
    entries = audit_log.tail(5)
    assert any(e.outcome == "skipped" for e in entries)
    assert not any(e.outcome == "passed" for e in entries)


@pytest.mark.asyncio
async def test_disabled_safety_always_passes() -> None:
    cfg = SafetyConfig(enabled=False)
    env = SafetyEnvelope(cfg)
    cmd = EmbodimentCommand(robot_id="r0", command_type="joint", values=[999.0])
    verdict = await env.check(cmd, trace_id="t7")
    assert verdict.passed


@pytest.mark.asyncio
async def test_safety_violation_carries_trace_id(envelope: SafetyEnvelope) -> None:
    cmd = EmbodimentCommand(robot_id="r0", command_type="joint", values=[10.0])
    with pytest.raises(SafetyEnvelopeViolation) as exc_info:
        await envelope.check(cmd, trace_id="my-trace-id")
    assert exc_info.value.trace_id == "my-trace-id"


# ---------------------------------------------------------------------------
# Pass 2 — the robot's own check (EmbodimentAdapter.safety_check, ISS-029)
# ---------------------------------------------------------------------------


from typing import Any  # noqa: E402

from robot_harness.embodiment.base import (  # noqa: E402
    DispatchHandle,
    Frame,
    RobotState,
    SafetyVerdict,
)
from robot_harness.errors import RobotOfflineError  # noqa: E402


class _FakeAdapter:
    """Minimal EmbodimentAdapter with a scriptable safety_check."""

    robot_id = "r0"
    robot_type = "arm"

    def __init__(self, verdict: SafetyVerdict | None = None, offline: bool = False) -> None:
        self._verdict = verdict or SafetyVerdict(passed=True)
        self._offline = offline
        self.checked: list[EmbodimentCommand] = []

    async def get_camera_frame(self, camera: str) -> Frame:
        return Frame(camera=camera, robot_id=self.robot_id)

    async def get_state(self) -> RobotState:
        return RobotState(robot_id=self.robot_id)

    async def dispatch(self, cmd: EmbodimentCommand) -> DispatchHandle:
        return DispatchHandle(robot_id=self.robot_id, action_id="a")

    async def safety_check(self, cmd: EmbodimentCommand) -> SafetyVerdict:
        if self._offline:
            raise RobotOfflineError("robot down", robot_id=self.robot_id)
        self.checked.append(cmd)
        return self._verdict


def _adapter_envelope(
    adapter: _FakeAdapter, audit_log: SafetyAuditLog, **cfg: Any
) -> SafetyEnvelope:
    return SafetyEnvelope(SafetyConfig(**cfg), audit_log=audit_log, adapters={"r0": adapter})


@pytest.mark.asyncio
async def test_adapter_safety_check_runs_and_passes(audit_log: SafetyAuditLog) -> None:
    adapter = _FakeAdapter()
    env = _adapter_envelope(adapter, audit_log)
    cmd = EmbodimentCommand(robot_id="r0", command_type="cartesian", values=[0.5, 0.5, 1.0])
    verdict = await env.check(cmd, trace_id="a1")
    assert verdict.passed
    assert adapter.checked, "adapter.safety_check must run as pass 2"
    assert any(e.outcome == "passed" for e in audit_log.tail(5))


@pytest.mark.asyncio
async def test_adapter_refusal_raises_and_audits(audit_log: SafetyAuditLog) -> None:
    adapter = _FakeAdapter(
        SafetyVerdict(passed=False, reason="self-collision", violated_rules=["self-collision"])
    )
    env = _adapter_envelope(adapter, audit_log)
    cmd = EmbodimentCommand(robot_id="r0", command_type="cartesian", values=[0.5, 0.5, 1.0])
    with pytest.raises(SafetyEnvelopeViolation, match="self-collision"):
        await env.check(cmd, trace_id="a2")
    assert any(e.outcome == "violated" for e in audit_log.tail(5))


@pytest.mark.asyncio
async def test_adapter_unreachable_audits_skipped_and_reraises(
    audit_log: SafetyAuditLog,
) -> None:
    """An unreachable robot check is a recoverable robot fault (typed error, the
    Brain can replan), never a silent pass — and it is audited."""
    env = _adapter_envelope(_FakeAdapter(offline=True), audit_log)
    cmd = EmbodimentCommand(robot_id="r0", command_type="cartesian", values=[0.5, 0.5, 1.0])
    with pytest.raises(RobotOfflineError):
        await env.check(cmd, trace_id="a3")
    entries = audit_log.tail(5)
    assert any(e.outcome == "skipped" for e in entries)
    assert not any(e.outcome == "passed" for e in entries)


@pytest.mark.asyncio
async def test_adapter_pass_upgrades_framework_skip_to_passed(
    audit_log: SafetyAuditLog,
) -> None:
    """Framework has no joint limits configured, but the robot's own check DID
    inspect the command — "passed" is then truthful, not a skip."""
    adapter = _FakeAdapter()
    env = _adapter_envelope(adapter, audit_log)  # no joint_limits_rad
    cmd = EmbodimentCommand(robot_id="r0", command_type="joint", values=[2.0])
    verdict = await env.check(cmd, trace_id="a4")
    assert verdict.passed
    assert adapter.checked
    assert any(e.outcome == "passed" for e in audit_log.tail(5))


# ---------------------------------------------------------------------------
# Audit honesty for rule-less commands (ISS-036)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hand_grasp_is_skipped_not_passed(audit_log: SafetyAuditLog) -> None:
    """No framework rule exists for hand_grasp — the audit must say "skipped",
    never a false "passed" (ISS-036)."""
    env = SafetyEnvelope(SafetyConfig(), audit_log=audit_log)
    cmd = EmbodimentCommand(robot_id="r0", command_type="hand_grasp", values=[0.0])
    verdict = await env.check(cmd, trace_id="h1")
    assert verdict.passed
    assert "no framework rule" in verdict.reason
    entries = audit_log.tail(5)
    assert any(e.outcome == "skipped" for e in entries)
    assert not any(e.outcome == "passed" for e in entries)


@pytest.mark.asyncio
async def test_cartesian_with_misconfigured_bounds_is_skipped(
    audit_log: SafetyAuditLog,
) -> None:
    """A wrong-length workspace_bounds_m cannot run the rule: honest skip (plus
    a misconfiguration warning), not a silent pass."""
    env = SafetyEnvelope(SafetyConfig(workspace_bounds_m=[1.0, 2.0, 3.0]), audit_log=audit_log)
    cmd = EmbodimentCommand(robot_id="r0", command_type="cartesian", values=[99.0, 0.0, 0.0])
    verdict = await env.check(cmd, trace_id="h2")
    assert verdict.passed
    assert "unchecked" in verdict.reason
    entries = audit_log.tail(5)
    assert any(e.outcome == "skipped" for e in entries)
    assert not any(e.outcome == "passed" for e in entries)


@pytest.mark.asyncio
async def test_cartesian_malformed_target_raises(envelope: SafetyEnvelope) -> None:
    # A target that is not at least [x, y, z] cannot be validated — reject it,
    # same policy as a malformed locomotion goal.
    cmd = EmbodimentCommand(robot_id="r0", command_type="cartesian", values=[0.5, 0.5])
    with pytest.raises(SafetyEnvelopeViolation, match="expected at least"):
        await envelope.check(cmd, trace_id="h3")
