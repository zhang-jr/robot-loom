"""A safety violation in one parallel tool call must stop its siblings.

The AgentLoop dispatches a turn's tool calls concurrently. SafetyEnvelopeViolation
has must-stop semantics: when one call is refused at the pre-dispatch gate, the
turn's sibling in-flight calls are cancelled — and cancellable tools actively
aborted via ``tool.cancel()`` — instead of being left actuating in the background
while the violation propagates. The SafetyEnvelope is REAL, never mocked.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from robot_harness.brain.base import BrainDecision, Task, ToolCallRequest
from robot_harness.embodiment.base import EmbodimentCommand
from robot_harness.errors import SafetyEnvelopeViolation
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class _OneTurnBrain:
    """Issues a single multi-call tool_call turn, then plans (never reached
    when the first turn raises)."""

    def __init__(self, calls: list[ToolCallRequest]) -> None:
        self._calls = calls
        self._done = False

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(self, messages: list[Any], tools: list[Any], **_: Any) -> BrainDecision:
        if self._done:
            return BrainDecision(decision_type="respond", message="done")
        self._done = True
        return BrainDecision(decision_type="tool_call", tool_calls=self._calls)


class _SlowCancellableTool:
    """A long-running cancellable tool standing in for an in-flight actuation."""

    name = "test.slow_actuator"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="test.slow_actuator",
        description="slow actuation that must be stopped on a sibling violation",
        input_schema={"type": "object"},
    )

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.completed = False
        self.cancel_called = False

    @property
    def is_idempotent(self) -> bool:
        return False

    @property
    def is_cancellable(self) -> bool:
        return True

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.started.set()
        await asyncio.sleep(30)
        self.completed = True
        return ToolResult(tool_name=self.name, trace_id=ctx.trace_id, success=True)

    async def cancel(self, ctx: ToolContext) -> None:
        self.cancel_called = True


class _CartesianTool:
    """Hardware-bound tool whose call maps to a cartesian safety command."""

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


@pytest.mark.asyncio
async def test_violation_cancels_and_aborts_sibling_in_flight_call() -> None:
    ctx = HarnessContext.build()
    slow = _SlowCancellableTool()
    cartesian = _CartesianTool()
    ctx.tool_registry.register(slow)
    ctx.tool_registry.register(cartesian)

    brain = _OneTurnBrain(
        [
            # Sibling actuation starts first and would run for 30s...
            ToolCallRequest(tool_name="test.slow_actuator", args={}),
            # ...while this call is refused by the REAL envelope (x out of bounds).
            ToolCallRequest(
                tool_name="test.move_cartesian",
                args={"robot_id": "r0", "values": [99.0, 0.0, 0.0]},
            ),
        ]
    )
    loop = AgentLoop(brain, ctx, max_turns=3)
    task = Task(task_id=str(uuid.uuid4()), description="t", robot_id="r0")

    with pytest.raises(SafetyEnvelopeViolation):
        await loop.run(task)

    assert slow.started.is_set()  # the sibling really was in flight
    assert slow.completed is False  # it never ran to completion
    assert slow.cancel_called is True  # and was actively aborted (tool.cancel)
    assert cartesian.invoke_count == 0  # the refused command never dispatched


@pytest.mark.asyncio
async def test_violation_still_propagates_when_no_sibling_pending() -> None:
    """The cancel path must not swallow the violation on a single-call turn."""
    ctx = HarnessContext.build()
    cartesian = _CartesianTool()
    ctx.tool_registry.register(cartesian)

    brain = _OneTurnBrain(
        [
            ToolCallRequest(
                tool_name="test.move_cartesian",
                args={"robot_id": "r0", "values": [99.0, 0.0, 0.0]},
            )
        ]
    )
    loop = AgentLoop(brain, ctx, max_turns=3)
    task = Task(task_id=str(uuid.uuid4()), description="t", robot_id="r0")

    with pytest.raises(SafetyEnvelopeViolation):
        await loop.run(task)
    assert cartesian.invoke_count == 0
