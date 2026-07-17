"""Unit tests for the ADR-019 robot_sdk verb tools.

These cover:
    * Factory & registry wiring
    * Schema shape (high-level intent, NOT control-tick payload)
    * Shared CompletionVerdict output shape across all 4 verbs
    * Mock invoke success path
    * Cancellation pre-dispatch raises ToolCancelledError
    * Idempotency policy: home is idempotent, others are not
"""

from __future__ import annotations

import pytest

from robot_harness.errors import HardwareNotReadyError, RobotOfflineError, ToolCancelledError
from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.robot_sdk import (
    COMPLETION_VERDICT_SCHEMA,
    ROBOT_SDK_HOME,
    ROBOT_SDK_LOCOMOTE_TO,
    ROBOT_SDK_MOVE_JOINTS,
    ROBOT_SDK_MOVE_TO_POSE,
    ROBOT_SDK_REACTIVE_GRASP,
    ROBOT_SDK_VISUAL_SERVO_TO,
    VERB_TOOL_NAMES,
    CompletionVerdict,
    HomeTool,
    LocomoteToTool,
    MoveJointsTool,
    MoveToPoseTool,
    ReactiveGraspTool,
    VisualServoToTool,
    build_robot_sdk_verb_tools,
    unavailable_verb_tool_names,
)


def _ctx() -> ToolContext:
    return ToolContext.create("robot-0", timeout_s=5.0)


# ---------------------------------------------------------------------------
# Factory & registry wiring
# ---------------------------------------------------------------------------


def test_factory_returns_all_verbs() -> None:
    tools = build_robot_sdk_verb_tools()
    names = [t.name for t in tools]
    assert names == list(VERB_TOOL_NAMES)
    assert set(names) == {
        ROBOT_SDK_REACTIVE_GRASP,
        ROBOT_SDK_VISUAL_SERVO_TO,
        ROBOT_SDK_MOVE_TO_POSE,
        ROBOT_SDK_MOVE_JOINTS,
        ROBOT_SDK_LOCOMOTE_TO,
        ROBOT_SDK_HOME,
    }


def test_factory_tools_register_into_registry() -> None:
    registry = ToolRegistry()
    for tool in build_robot_sdk_verb_tools():
        registry.register(tool)
    for name in VERB_TOOL_NAMES:
        assert name in registry


def test_verb_classes_use_native_backend() -> None:
    """Verbs run native (in-process) until the on-robot agent_server HTTP/WS transport lands."""
    for tool in build_robot_sdk_verb_tools():
        assert tool.backend == "native"


# ---------------------------------------------------------------------------
# unavailable_verb_tool_names — live-capability exclusion set (ADR-019)
# ---------------------------------------------------------------------------


def test_unavailable_verbs_none_means_unknown_excludes_nothing() -> None:
    """Backend doesn't advertise a verb set (mock / sim / old server) → never prune."""
    assert unavailable_verb_tool_names(None) == frozenset()


def test_unavailable_verbs_excludes_unadvertised_only() -> None:
    # Go2 with no arm bridge advertises only locomotion + home.
    excluded = unavailable_verb_tool_names({"locomote_to", "home"})
    assert excluded == {
        ROBOT_SDK_REACTIVE_GRASP,
        ROBOT_SDK_VISUAL_SERVO_TO,
        ROBOT_SDK_MOVE_TO_POSE,
        ROBOT_SDK_MOVE_JOINTS,
    }


def test_unavailable_verbs_explicit_empty_excludes_all() -> None:
    """[] is a real answer — every capability bridge down → all verb tools gated."""
    assert unavailable_verb_tool_names(set()) == frozenset(VERB_TOOL_NAMES)


def test_unavailable_verbs_full_allowlist_excludes_nothing() -> None:
    bare = {name.split(".", 1)[1] for name in VERB_TOOL_NAMES}
    assert unavailable_verb_tool_names(bare) == frozenset()


# ---------------------------------------------------------------------------
# Schema contract — input is HIGH-LEVEL intent (ADR-019)
# ---------------------------------------------------------------------------


def test_reactive_grasp_input_requires_target_hint_not_joints() -> None:
    """ADR-019 §244: reactive verbs receive intent, not joint targets."""
    schema = ReactiveGraspTool.schema.input_schema
    assert set(schema["required"]) == {"robot_id", "target_hint"}
    # No joint / cartesian / delta fields at the verb interface
    assert "joint_target" not in schema["properties"]
    assert "command_type" not in schema["properties"]


def test_visual_servo_to_input_requires_target_pose() -> None:
    schema = VisualServoToTool.schema.input_schema
    assert set(schema["required"]) == {"robot_id", "target_pose"}
    assert schema["properties"]["target_pose"]["minItems"] == 6
    assert schema["properties"]["target_pose"]["maxItems"] == 6
    # default tolerances exist so Brain can omit them safely
    assert "tolerance_m" in schema["properties"]
    assert "tolerance_rad" in schema["properties"]


def test_move_to_pose_input_carries_frame_enum() -> None:
    schema = MoveToPoseTool.schema.input_schema
    assert set(schema["required"]) == {"robot_id", "target_pose"}
    assert schema["properties"]["frame"]["enum"] == ["base", "world", "tool"]
    # placement waypoints carry a per-step gripper value on the same motion call
    assert "gripper" in schema["properties"]


def test_move_joints_input_requires_target_joints() -> None:
    """Joint-space direct drive: the caller supplies the joint configuration;
    gripper rides along optionally, exactly like move_to_pose."""
    schema = MoveJointsTool.schema.input_schema
    assert set(schema["required"]) == {"robot_id", "target_joints"}
    assert schema["properties"]["target_joints"]["items"] == {"type": "number"}
    assert "gripper" in schema["properties"]
    out = MoveJointsTool.schema.output_schema
    assert out is not None
    assert "final_joints" in out["properties"]
    assert "residual_error_rad" in out["properties"]


def test_home_input_minimal() -> None:
    schema = HomeTool.schema.input_schema
    assert schema["required"] == ["robot_id"]


# ---------------------------------------------------------------------------
# Shared CompletionVerdict output shape — uniform across all verbs (ADR-019)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_cls",
    [
        ReactiveGraspTool,
        VisualServoToTool,
        MoveToPoseTool,
        MoveJointsTool,
        LocomoteToTool,
        HomeTool,
    ],
)
def test_every_verb_publishes_completion_verdict_fields(tool_cls: type) -> None:
    out = tool_cls.schema.output_schema
    assert out is not None
    # The four CompletionVerdict properties must always appear
    for field in ("outcome", "evidence", "robot_state_snapshot", "aborted_by"):
        assert field in out["properties"], f"{tool_cls.__name__} missing {field}"
    assert "outcome" in out["required"]
    # outcome enum is the ADR-019 trio
    assert set(out["properties"]["outcome"]["enum"]) == {"success", "partial", "failed"}


def test_completion_verdict_shared_schema_constants() -> None:
    """Sanity: the shared COMPLETION_VERDICT_SCHEMA matches the model fields."""
    props = COMPLETION_VERDICT_SCHEMA["properties"]
    assert set(props.keys()) >= {
        "outcome",
        "evidence",
        "robot_state_snapshot",
        "duration_s",
        "aborted_by",
    }


# ---------------------------------------------------------------------------
# Mock invoke — success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reactive_grasp_mock_returns_success_verdict() -> None:
    tool = ReactiveGraspTool()
    result = await tool.invoke(
        {
            "robot_id": "robot-0",
            "target_hint": {"kind": "phrase", "phrase": "the red mug"},
        },
        _ctx(),
    )
    assert result.success is True
    assert result.output is not None
    assert result.output["outcome"] == "success"
    # CompletionVerdict deserializable
    verdict = CompletionVerdict.model_validate(result.output)
    assert verdict.outcome == "success"
    assert verdict.aborted_by == "none"


@pytest.mark.asyncio
async def test_visual_servo_to_mock_returns_final_pose() -> None:
    tool = VisualServoToTool()
    target = [0.4, 0.1, 0.3, 0.0, 1.57, 0.0]
    result = await tool.invoke(
        {"robot_id": "robot-0", "target_pose": target},
        _ctx(),
    )
    assert result.success is True
    snap = (result.output or {}).get("robot_state_snapshot", {})
    assert snap.get("final_pose") == target


@pytest.mark.asyncio
async def test_move_to_pose_mock_dispatches_with_default_frame() -> None:
    tool = MoveToPoseTool()
    target = [0.5, 0.0, 0.35, 0.0, 0.0, 0.0]
    result = await tool.invoke(
        {"robot_id": "robot-0", "target_pose": target},
        _ctx(),
    )
    assert result.success is True
    assert (result.output or {})["outcome"] == "success"


@pytest.mark.asyncio
async def test_move_joints_mock_returns_final_joints() -> None:
    tool = MoveJointsTool()
    target = [0.0, 0.5, -0.5, 0.0, 1.0, 0.0]
    result = await tool.invoke(
        {"robot_id": "robot-0", "target_joints": target},
        _ctx(),
    )
    assert result.success is True
    snap = (result.output or {}).get("robot_state_snapshot", {})
    assert snap.get("final_joints") == target


@pytest.mark.asyncio
async def test_home_mock_marks_robot_as_home() -> None:
    tool = HomeTool()
    result = await tool.invoke({"robot_id": "robot-0"}, _ctx())
    assert result.success is True
    snap = (result.output or {}).get("robot_state_snapshot", {})
    assert snap.get("is_home") is True


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_raises_when_cancelled_before_dispatch() -> None:
    tool = ReactiveGraspTool()
    ctx = _ctx()
    ctx.cancel()
    with pytest.raises(ToolCancelledError):
        await tool.invoke(
            {
                "robot_id": "robot-0",
                "target_hint": {"kind": "phrase", "phrase": "x"},
            },
            ctx,
        )


@pytest.mark.asyncio
async def test_tool_cancel_sets_context_cancel_flag() -> None:
    tool = ReactiveGraspTool()
    ctx = _ctx()
    assert ctx.is_cancelled is False
    await tool.cancel(ctx)
    assert ctx.is_cancelled is True


@pytest.mark.asyncio
async def test_cancel_routes_abort_to_adapter() -> None:
    """With a verb-capable adapter wired, cancel() POSTs /abort via adapter.abort()
    (not just flips the local flag) so an in-flight on-robot verb unwinds."""

    class _FakeAdapter:
        def __init__(self) -> None:
            self.aborted_with: str | None = None

        async def abort(self, trace_id: str = "") -> dict[str, object]:
            self.aborted_with = trace_id
            return {"aborted": True}

    adapter = _FakeAdapter()
    tool = ReactiveGraspTool({"robot-0": adapter})
    ctx = _ctx()
    await tool.cancel(ctx)
    assert ctx.is_cancelled is True
    assert adapter.aborted_with == ctx.trace_id


@pytest.mark.asyncio
async def test_cancel_without_adapter_only_sets_local_flag() -> None:
    """No verb-capable adapter (mock/offline) -> nothing on-robot to abort; cancel
    must still succeed and set the local flag."""
    tool = ReactiveGraspTool()  # no adapters
    ctx = _ctx()
    await tool.cancel(ctx)  # must not raise
    assert ctx.is_cancelled is True


def test_locomote_to_emits_locomotion_safety_command() -> None:
    """The goal pose is exposed to SafetyEnvelope as a locomotion command so the
    map-frame geofence (safety.map_bounds_m) can bound it pre-dispatch."""
    tool = LocomoteToTool()
    cmd = tool.to_safety_command({"robot_id": "robot-0", "target_pose": [1.0, 2.0, 0.0]}, _ctx())
    assert cmd is not None
    assert cmd.command_type == "locomotion"
    assert cmd.robot_id == "robot-0"
    assert cmd.values == [1.0, 2.0, 0.0]


def test_locomote_to_without_target_emits_no_safety_command() -> None:
    tool = LocomoteToTool()
    cmd = tool.to_safety_command({"robot_id": "robot-0"}, _ctx())
    assert cmd is None


# ---------------------------------------------------------------------------
# Idempotency policy
# ---------------------------------------------------------------------------


def test_home_is_idempotent_others_are_not() -> None:
    assert HomeTool().is_idempotent is True
    assert ReactiveGraspTool().is_idempotent is False
    assert VisualServoToTool().is_idempotent is False
    assert MoveToPoseTool().is_idempotent is False
    assert MoveJointsTool().is_idempotent is False


def test_all_verbs_are_cancellable() -> None:
    for tool in build_robot_sdk_verb_tools():
        assert tool.is_cancellable is True


# ---------------------------------------------------------------------------
# Brain export — verbs surface to LLM via OpenAI / Anthropic / MCP formats
# ---------------------------------------------------------------------------


def test_verbs_export_to_openai_function_format() -> None:
    """Sanity: verb schemas can be exported to OpenAI function spec without losing
    the high-level-intent shape."""
    from robot_harness.tools.base import BrainProfile

    registry = ToolRegistry()
    for tool in build_robot_sdk_verb_tools():
        registry.register(tool)

    specs = registry.export_for_brain(BrainProfile(name="openai"))
    exported = {spec["function"]["name"] for spec in specs}
    assert exported == set(VERB_TOOL_NAMES)


# ---------------------------------------------------------------------------
# Failure verdicts carry their WHY (ISS-032)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_verdict_surfaces_evidence_in_error() -> None:
    """A non-success CompletionVerdict must set ``error`` from its evidence and
    aborted_by — that string is what the Brain and skills replan on; the full
    verdict still rides in ``output``."""

    class _FailingVerbAdapter:
        async def call_verb(self, verb: str, payload: dict[str, object]) -> dict[str, object]:
            return {
                "outcome": "failed",
                "evidence": "gripper slipped on rim",
                "aborted_by": "safety",
            }

        async def available_verbs(self) -> list[str] | None:
            return None

    tool = ReactiveGraspTool({"robot-0": _FailingVerbAdapter()})
    res = await tool.invoke(
        {"robot_id": "robot-0", "target_hint": {"kind": "phrase", "phrase": "mug"}},
        _ctx(),
    )
    assert res.success is False
    assert res.error is not None
    assert "gripper slipped on rim" in res.error
    assert "aborted_by=safety" in res.error
    assert res.error_type == "VerbFailed"
    assert res.output is not None
    assert res.output["outcome"] == "failed"


@pytest.mark.asyncio
async def test_execute_action_cancel_routes_abort_to_adapter() -> None:
    """Low-level dispatch is fire-and-return: the command keeps executing
    on-robot, so cancel() must POST /abort like the verb tools do (ISS-035)."""
    from robot_harness.tools.robot_sdk.http_adapter import RobotSdkTool

    class _FakeAdapter:
        def __init__(self) -> None:
            self.aborted_with: str | None = None

        async def abort(self, trace_id: str = "") -> dict[str, object]:
            self.aborted_with = trace_id
            return {"aborted": True}

    adapter = _FakeAdapter()
    tool = RobotSdkTool({"robot-0": adapter})  # type: ignore[dict-item]
    ctx = _ctx()
    await tool.cancel(ctx)
    assert ctx.is_cancelled is True
    assert adapter.aborted_with == ctx.trace_id


@pytest.mark.asyncio
async def test_malformed_verdict_from_agent_server_is_typed_offline_error() -> None:
    """A response that is not a CompletionVerdict is 'not speaking the
    contract' — a typed RobotOfflineError, never a raw ValidationError
    escaping into the loop (ISS-034)."""

    class _GarbageVerbAdapter:
        async def call_verb(self, verb: str, payload: dict[str, object]) -> dict[str, object]:
            return {"outcome": "weird-state"}

        async def available_verbs(self) -> list[str] | None:
            return None

    tool = HomeTool({"robot-0": _GarbageVerbAdapter()})
    with pytest.raises(RobotOfflineError, match="malformed CompletionVerdict"):
        await tool.invoke({"robot_id": "robot-0"}, _ctx())


# ---------------------------------------------------------------------------
# Unknown robot_id with a wired fleet is a typed failure, never a mock
# ---------------------------------------------------------------------------


class _CannedVerbAdapter:
    """Minimal SupportsVerbs adapter — dispatch must never fall through it."""

    async def call_verb(self, verb: str, payload: dict[str, object]) -> dict[str, object]:
        return {"outcome": "success", "evidence": f"live {verb}", "aborted_by": "none"}

    async def available_verbs(self) -> list[str] | None:
        return None


@pytest.mark.asyncio
async def test_unknown_robot_id_with_wired_fleet_raises_typed_error() -> None:
    """A robot_id outside the wired fleet must raise a RobotOfflineError naming
    the fleet (so the Brain can correct it) — never a fabricated mock success
    verdict for a robot that does not exist."""
    tool = ReactiveGraspTool({"go2-01": _CannedVerbAdapter()})
    with pytest.raises(RobotOfflineError, match=r"wired fleet.*go2-01"):
        await tool.invoke(
            {"robot_id": "robot-0", "target_hint": {"kind": "phrase", "phrase": "bottle"}},
            _ctx(),
        )


@pytest.mark.asyncio
async def test_wired_adapter_without_verb_capability_raises_typed_error() -> None:
    """A wired adapter that cannot run verbs must fail typed, not simulate."""
    tool = HomeTool({"robot-0": object()})
    with pytest.raises(HardwareNotReadyError, match="does not implement on-robot verbs"):
        await tool.invoke({"robot_id": "robot-0"}, _ctx())


@pytest.mark.asyncio
async def test_no_fleet_wired_still_simulates() -> None:
    """A tool built with NO adapters at all keeps the simulated-verdict path
    (standalone / harness plumbing tests) — the wired-fleet guard narrows the
    fallback, it does not remove it."""
    tool = HomeTool()
    result = await tool.invoke({"robot_id": "anything"}, _ctx())
    assert result.success is True


@pytest.mark.asyncio
async def test_execute_action_unknown_robot_id_raises_typed_error() -> None:
    """Same guard, low-level dispatch flavor: execute_action with a wired fleet
    must not return a fabricated mock handle for an unknown robot_id."""
    from robot_harness.tools.robot_sdk.http_adapter import RobotSdkTool

    tool = RobotSdkTool({"go2-01": _CannedVerbAdapter()})  # type: ignore[dict-item]
    with pytest.raises(RobotOfflineError, match=r"wired fleet.*go2-01"):
        await tool.invoke(
            {"robot_id": "robot-0", "command_type": "joint", "values": [0.0] * 6},
            _ctx(),
        )
