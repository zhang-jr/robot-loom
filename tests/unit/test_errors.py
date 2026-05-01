"""Tests for the harness exception hierarchy."""

from __future__ import annotations

import pytest

from robot_harness.errors import (
    BrainOutputInvalidError,
    BrainTimeoutError,
    FleetCapacityExceededError,
    HarnessError,
    ReplanLoopExceededError,
    RobotLockConflictError,
    SafetyEnvelopeViolation,
    SkillManifestInvalidError,
    SkillNotFoundError,
    ToolBackendUnreachableError,
    ToolCancelledError,
    ToolNotFoundError,
    ToolSchemaViolationError,
    ToolTimeoutError,
)


def test_harness_error_carries_context() -> None:
    exc = HarnessError(
        "test",
        trace_id="t1",
        robot_id="r1",
        subtask_id="s1",
        tool_name="my_tool",
        module_name="my_mod",
    )
    ctx = exc.context()
    assert ctx["trace_id"] == "t1"
    assert ctx["robot_id"] == "r1"
    assert ctx["tool_name"] == "my_tool"


def test_all_subclasses_are_harness_errors() -> None:
    subclasses = [
        BrainTimeoutError("x"),
        BrainOutputInvalidError("x"),
        ReplanLoopExceededError("x", max_iterations=5),
        ToolNotFoundError("x"),
        ToolBackendUnreachableError("x"),
        ToolSchemaViolationError("x", violations=["missing: arg"]),
        ToolTimeoutError("x"),
        ToolCancelledError("x"),
        SkillNotFoundError("x"),
        SkillManifestInvalidError("x", validation_errors=["bad field"]),
        SafetyEnvelopeViolation("x", violated_rules=["vel_cap"]),
        FleetCapacityExceededError("x"),
        RobotLockConflictError("x"),
    ]
    for exc in subclasses:
        assert isinstance(exc, HarnessError), f"{type(exc)} is not a HarnessError"
        assert isinstance(exc, Exception)


def test_safety_violation_carries_rules() -> None:
    exc = SafetyEnvelopeViolation("bad", violated_rules=["vel_cap", "workspace"])
    assert "vel_cap" in exc.violated_rules
    assert "workspace" in exc.violated_rules


def test_tool_schema_violation_carries_violations() -> None:
    exc = ToolSchemaViolationError("bad", violations=["missing: x"])
    assert exc.violations == ["missing: x"]


def test_replan_loop_carries_max_iterations() -> None:
    exc = ReplanLoopExceededError("too many", max_iterations=10)
    assert exc.max_iterations == 10


def test_skill_manifest_invalid_carries_errors() -> None:
    exc = SkillManifestInvalidError("bad manifest", validation_errors=["version missing"])
    assert exc.validation_errors == ["version missing"]


def test_safety_violation_cannot_be_silenced() -> None:
    """Verify that try/except HarnessError still propagates SafetyEnvelopeViolation."""
    with pytest.raises(SafetyEnvelopeViolation):
        try:
            raise SafetyEnvelopeViolation("test")
        except HarnessError:
            raise  # must not be swallowed
