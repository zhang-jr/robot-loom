"""Safety-gated ToolRegistry facade for skill-internal tool calls.

A Skill composes several tool calls and invokes them directly
(``tools.get(name).invoke(...)``); those calls never pass back through the
AgentLoop, so the loop's per-call SafetyEnvelope gate does NOT cover them. To
keep the invariant that no hardware dispatch bypasses SafetyEnvelope intact when
a skill is driven from the Brain, the AgentLoop hands the skill this facade
instead of the raw registry: :meth:`SafetyGatedToolRegistry.get` returns a
safety-gated proxy for any hardware-bound tool, which runs
``SafetyEnvelope.check`` before delegating. Non-hardware tools are returned
unwrapped, and every other registry method delegates straight through.
"""

from __future__ import annotations

from typing import Any

from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class _SafetyGatedTool:
    """Wraps a hardware-bound Tool so ``invoke`` runs SafetyEnvelope.check first.

    Mirrors the pre-dispatch gate the AgentLoop applies to Brain-issued calls, so
    a skill's internal actuation gets the same authorization path. A
    ``SafetyEnvelopeViolation`` propagates uncaught to trigger e-stop.
    """

    def __init__(self, tool: Tool, registry: ToolRegistry, envelope: SafetyEnvelope) -> None:
        self._tool = tool
        self._registry = registry
        self._envelope = envelope

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def schema(self) -> ToolSchema:
        return self._tool.schema

    @property
    def backend(self) -> ToolBackend:
        return self._tool.backend

    @property
    def is_idempotent(self) -> bool:
        return self._tool.is_idempotent

    @property
    def is_cancellable(self) -> bool:
        return self._tool.is_cancellable

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        cmd = self._registry.build_safety_command(self._tool.name, args, ctx)
        if cmd is not None:
            # SafetyEnvelopeViolation is NOT caught — it propagates to trigger e-stop.
            await self._envelope.check(cmd, trace_id=ctx.trace_id, subtask_id=ctx.subtask_id)
        return await self._tool.invoke(args, ctx)

    async def cancel(self, ctx: ToolContext) -> None:
        await self._tool.cancel(ctx)


class SafetyGatedToolRegistry:
    """A ToolRegistry facade that safety-gates hardware-bound tools on ``get``.

    Handed to :meth:`Skill.execute` in place of the raw registry so a skill's
    internal tool calls get the same pre-dispatch SafetyEnvelope check the
    AgentLoop applies to Brain-issued calls. Non-hardware tools are returned
    unwrapped; every other registry method delegates to the wrapped registry.
    """

    def __init__(self, registry: ToolRegistry, envelope: SafetyEnvelope) -> None:
        self._registry = registry
        self._envelope = envelope

    def get(self, name: str) -> Tool:
        tool = self._registry.get(name)
        if self._registry.requires_safety_check(name):
            return _SafetyGatedTool(tool, self._registry, self._envelope)
        return tool

    def __getattr__(self, item: str) -> Any:
        # Delegate every other ToolRegistry method (list_schemas, validate_args, …).
        return getattr(self._registry, item)
