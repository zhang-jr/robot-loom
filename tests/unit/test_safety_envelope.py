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
        max_joint_velocity_rad_s=1.0,
        workspace_bounds_m=[-2.0, -2.0, 0.0, 2.0, 2.0, 2.0],
    )
    return SafetyEnvelope(cfg, audit_log=audit_log)


@pytest.mark.asyncio
async def test_joint_command_within_limits_passes(envelope: SafetyEnvelope) -> None:
    cmd = EmbodimentCommand(robot_id="r0", command_type="joint", values=[0.5, -0.5, 0.9])
    verdict = await envelope.check(cmd, trace_id="t1")
    assert verdict.passed


@pytest.mark.asyncio
async def test_joint_velocity_exceeds_cap_raises(envelope: SafetyEnvelope) -> None:
    cmd = EmbodimentCommand(robot_id="r0", command_type="joint", values=[2.0])
    with pytest.raises(SafetyEnvelopeViolation) as exc_info:
        await envelope.check(cmd, trace_id="t2")
    assert exc_info.value.violated_rules
    assert "vel" in exc_info.value.violated_rules[0].lower()


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
