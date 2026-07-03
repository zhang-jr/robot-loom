"""Tests for the Brain⇄Skill dispatch wiring.

Covers the seam that connects the (previously suspended) skill layer to the
runtime:

  * SkillRegistry.export_for_brain / resolve_brain_call — skills enter the
    Brain's planning vocabulary as ``skill.<name>`` callables.
  * AgentLoop._invoke_skill — a ``skill.<name>`` call routes to Skill.execute
    and the SkillResult is surfaced as a ToolResult.
  * SafetyGatedToolRegistry — a skill's internal hardware call still passes the
    SafetyEnvelope (principle #6). The envelope is REAL, never mocked.
  * Error normalization — SkillError / EmbodimentError inside a skill become a
    failed ToolResult, while SafetyEnvelopeViolation propagates uncaught.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from robot_harness.brain.base import BrainDecision, Task, ToolCallRequest
from robot_harness.config.schema import SafetyConfig
from robot_harness.embodiment.base import EmbodimentCommand
from robot_harness.errors import RobotOfflineError, SafetyEnvelopeViolation, SkillError
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.runtime.skill_tools import SafetyGatedToolRegistry
from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.skill.base import SkillManifest, SkillResult, Subtask
from robot_harness.skill.registry import SkillRegistry
from robot_harness.skill.safety_class import SafetyClass
from robot_harness.tools.base import BrainProfile, ToolContext, ToolRegistry, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _MockBrain:
    def __init__(self, decisions: list[BrainDecision]) -> None:
        self._decisions = list(decisions)
        self._i = 0

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(self, messages: list[Any], tools: list[Any], **_: Any) -> BrainDecision:
        d = (
            self._decisions[self._i]
            if self._i < len(self._decisions)
            else BrainDecision(decision_type="plan", message="done")
        )
        self._i += 1
        return d


class _CartesianTool:
    """A hardware-bound tool whose call maps to a cartesian safety command."""

    name = "test.move_cartesian"
    backend: ToolBackend = "native"
    hardware_bound = True
    schema = ToolSchema(
        name="test.move_cartesian",
        description="move to a cartesian pose",
        input_schema={"type": "object", "properties": {"values": {"type": "array"}}},
    )

    def __init__(self) -> None:
        self.invoke_count = 0

    @property
    def is_idempotent(self) -> bool:
        return False

    @property
    def is_cancellable(self) -> bool:
        return True

    def to_safety_command(self, args: dict[str, Any], ctx: ToolContext) -> EmbodimentCommand:
        return EmbodimentCommand(
            robot_id=args.get("robot_id", ctx.robot_id),
            command_type="cartesian",
            values=list(args.get("values", [])),
        )

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.invoke_count += 1
        return ToolResult(tool_name=self.name, trace_id=ctx.trace_id, success=True, output={})

    async def cancel(self, ctx: ToolContext) -> None:
        pass


def _manifest(name: str) -> SkillManifest:
    return SkillManifest(
        name=name,
        version="1.0.0",
        description=f"{name} skill",
        embodiment_compat=["arm"],
        safety_class=SafetyClass.HIGH,
        required_tools=[],
        tags=[name],
    )


class _SuccessSkill:
    manifest = _manifest("dummy")

    async def can_handle(self, subtask: Subtask, ctx: Any) -> bool:
        return True

    async def execute(self, subtask: Subtask, tools: Any, ctx: Any) -> SkillResult:
        return SkillResult(
            skill_name=self.manifest.name,
            skill_version=self.manifest.version,
            subtask_id=subtask.subtask_id,
            success=True,
            outcome="success",
            artifacts={"echoed": subtask.parameters},
        )

    async def rollback(self, ctx: Any) -> None:
        pass


class _RaisingSkill:
    """Skill that raises a chosen exception from execute()."""

    def __init__(self, exc: Exception, name: str = "raiser") -> None:
        self.manifest = _manifest(name)
        self._exc = exc

    async def can_handle(self, subtask: Subtask, ctx: Any) -> bool:
        return True

    async def execute(self, subtask: Subtask, tools: Any, ctx: Any) -> SkillResult:
        raise self._exc

    async def rollback(self, ctx: Any) -> None:
        pass


class _HardwareCallingSkill:
    """Skill that drives a hardware tool with a caller-supplied pose."""

    def __init__(self, values: list[float], name: str = "mover") -> None:
        self.manifest = _manifest(name)
        self._values = values

    async def can_handle(self, subtask: Subtask, ctx: Any) -> bool:
        return True

    async def execute(self, subtask: Subtask, tools: ToolRegistry, ctx: Any) -> SkillResult:
        tool_ctx = ToolContext.create(subtask.robot_id, subtask_id=subtask.subtask_id)
        res = await tools.get("test.move_cartesian").invoke(
            {"robot_id": subtask.robot_id, "values": self._values}, tool_ctx
        )
        return SkillResult(
            skill_name=self.manifest.name,
            skill_version=self.manifest.version,
            subtask_id=subtask.subtask_id,
            success=res.success,
        )

    async def rollback(self, ctx: Any) -> None:
        pass


def _task(robot_id: str = "r0") -> Task:
    return Task(task_id=str(uuid.uuid4()), description="t", robot_id=robot_id)


def _call(tool_name: str, args: dict[str, Any]) -> BrainDecision:
    return BrainDecision(
        decision_type="tool_call", tool_calls=[ToolCallRequest(tool_name=tool_name, args=args)]
    )


# ---------------------------------------------------------------------------
# SkillRegistry: export + resolve
# ---------------------------------------------------------------------------


def test_builtin_skills_exported_to_brain() -> None:
    ctx = HarnessContext.build()
    specs = ctx.skill_registry.export_for_brain(BrainProfile(name="openai"))
    names = {s["function"]["name"] for s in specs}
    assert {"skill.pick", "skill.place", "skill.navigate_to"} <= names


def test_resolve_brain_call_routes_only_skill_prefix() -> None:
    ctx = HarnessContext.build()
    reg = ctx.skill_registry
    assert reg.resolve_brain_call("skill.pick").manifest.name == "pick"
    # A non-skill name (an atomic tool) is not a skill call.
    assert reg.resolve_brain_call("robot_sdk.home") is None
    # A skill-prefixed but unregistered name resolves to None, never raises.
    assert reg.resolve_brain_call("skill.does_not_exist") is None


def test_export_uses_generic_subtask_schema_when_no_data_schema() -> None:
    reg = SkillRegistry()
    reg.register(_SuccessSkill())
    (spec,) = reg.export_for_brain(BrainProfile(name="openai"))
    params = spec["function"]["parameters"]
    assert params["required"] == ["robot_id"]
    assert set(params["properties"]) == {"robot_id", "description", "parameters"}


# ---------------------------------------------------------------------------
# AgentLoop: skill routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_call_routes_to_execute_and_surfaces_result() -> None:
    ctx = HarnessContext.build()
    ctx.skill_registry.register(_SuccessSkill())
    brain = _MockBrain(
        [
            _call("skill.dummy", {"robot_id": "r0", "parameters": {"k": "v"}}),
            BrainDecision(decision_type="plan", message="done"),
        ]
    )
    loop = AgentLoop(brain, ctx, max_turns=5)
    result = await loop.run(_task())
    assert result.outcome == "success"
    tr = result.tool_results[0]
    assert tr["tool_name"] == "skill.dummy"
    assert tr["success"] is True
    assert tr["output"]["outcome"] == "success"
    assert tr["output"]["artifacts"]["echoed"] == {"k": "v"}


@pytest.mark.asyncio
async def test_skill_error_becomes_failed_result_not_crash() -> None:
    ctx = HarnessContext.build()
    ctx.skill_registry.register(_RaisingSkill(SkillError("boom"), name="boomer"))
    brain = _MockBrain([_call("skill.boomer", {"robot_id": "r0"})])
    loop = AgentLoop(brain, ctx, max_turns=3)
    result = await loop.run(_task())  # must not raise
    tr = result.tool_results[0]
    assert tr["success"] is False
    assert tr["error_type"] == "SkillError"


@pytest.mark.asyncio
async def test_backend_error_in_skill_becomes_failed_result() -> None:
    ctx = HarnessContext.build()
    ctx.skill_registry.register(_RaisingSkill(RobotOfflineError("robot down"), name="offliner"))
    brain = _MockBrain([_call("skill.offliner", {"robot_id": "r0"})])
    loop = AgentLoop(brain, ctx, max_turns=3)
    result = await loop.run(_task())  # backend fault must not crash the loop
    tr = result.tool_results[0]
    assert tr["success"] is False
    assert tr["error_type"] == "RobotOfflineError"


# ---------------------------------------------------------------------------
# SafetyGatedToolRegistry: principle #6 for skill-internal hardware calls
# (real SafetyEnvelope — never mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gated_registry_passes_safe_command_through() -> None:
    registry = ToolRegistry()
    tool = _CartesianTool()
    registry.register(tool)
    envelope = SafetyEnvelope(SafetyConfig())  # default bounds [-2,-2,0,2,2,2]
    gated = SafetyGatedToolRegistry(registry, envelope)

    ctx = ToolContext.create("r0")
    res = await gated.get("test.move_cartesian").invoke(
        {"robot_id": "r0", "values": [0.5, 0.5, 0.5]}, ctx
    )
    assert res.success is True
    assert tool.invoke_count == 1


@pytest.mark.asyncio
async def test_gated_registry_blocks_out_of_bounds_command() -> None:
    registry = ToolRegistry()
    tool = _CartesianTool()
    registry.register(tool)
    envelope = SafetyEnvelope(SafetyConfig())
    gated = SafetyGatedToolRegistry(registry, envelope)

    ctx = ToolContext.create("r0")
    with pytest.raises(SafetyEnvelopeViolation):
        await gated.get("test.move_cartesian").invoke(
            {"robot_id": "r0", "values": [99.0, 0.0, 0.0]},
            ctx,  # x way out of bounds
        )
    assert tool.invoke_count == 0  # blocked before the tool ran


@pytest.mark.asyncio
async def test_skill_hardware_call_violation_propagates_through_loop() -> None:
    """A skill's internal out-of-bounds hardware call must trip the SafetyEnvelope
    and propagate — the loop must NOT swallow SafetyEnvelopeViolation."""
    ctx = HarnessContext.build()
    ctx.tool_registry.register(_CartesianTool())
    ctx.skill_registry.register(_HardwareCallingSkill(values=[99.0, 0.0, 0.0], name="badmover"))
    brain = _MockBrain([_call("skill.badmover", {"robot_id": "r0"})])
    loop = AgentLoop(brain, ctx, max_turns=3)
    with pytest.raises(SafetyEnvelopeViolation):
        await loop.run(_task())


@pytest.mark.asyncio
async def test_skill_hardware_call_within_bounds_succeeds_through_loop() -> None:
    ctx = HarnessContext.build()
    tool = _CartesianTool()
    ctx.tool_registry.register(tool)
    ctx.skill_registry.register(_HardwareCallingSkill(values=[0.5, 0.5, 0.5], name="goodmover"))
    brain = _MockBrain(
        [
            _call("skill.goodmover", {"robot_id": "r0"}),
            BrainDecision(decision_type="plan", message="done"),
        ]
    )
    loop = AgentLoop(brain, ctx, max_turns=3)
    result = await loop.run(_task())
    assert result.outcome == "success"
    assert tool.invoke_count == 1
