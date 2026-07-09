"""Safety-gated ToolRegistry facade for skill-internal tool calls.

A Skill composes several tool calls and invokes them directly
(``tools.get(name).invoke(...)``); those calls never pass back through the
AgentLoop, so the loop's per-call SafetyEnvelope gate does NOT cover them. To
keep the invariant that no hardware dispatch bypasses SafetyEnvelope intact when
a skill is driven from the Brain, the AgentLoop hands the skill this facade
instead of the raw registry: :meth:`SafetyGatedToolRegistry.get` returns a proxy
that runs ``SafetyEnvelope.check`` before delegating for any hardware-bound
tool. Non-hardware tools are returned unwrapped (unless context binding below
applies), and every other registry method delegates straight through.

The facade also enforces trace continuity: the Skill protocol has no
trace channel, so skills mint their own per-call ToolContext — which would start
a fresh trace_id and drop the loop's artifact_store/outbound. When constructed
with the loop's task-level context (``parent_ctx``), the proxy rebinds every
skill-internal call onto the task trace and inherits those handles. Like the
safety gate itself, this is a structural guarantee rather than a convention each
skill author must remember.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class _SafetyGatedTool:
    """Proxy that safety-gates and context-rebinds a skill-internal tool call.

    ``invoke`` first rebinds the caller's ToolContext onto the task context
    (authoritative trace_id, inherited artifact_store/outbound), then — for
    hardware-bound tools — runs ``SafetyEnvelope.check``, mirroring the
    pre-dispatch gate the AgentLoop applies to Brain-issued calls. A
    ``SafetyEnvelopeViolation`` propagates uncaught: the refused command was
    never dispatched, and the violation aborts the task on its way up.
    """

    def __init__(
        self,
        tool: Tool,
        registry: ToolRegistry,
        envelope: SafetyEnvelope,
        *,
        safety_gated: bool = True,
        parent_ctx: ToolContext | None = None,
    ) -> None:
        self._tool = tool
        self._registry = registry
        self._envelope = envelope
        self._safety_gated = safety_gated
        self._parent_ctx = parent_ctx

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

    def _bind_task_context(self, ctx: ToolContext) -> ToolContext:
        """Rebind a skill-minted context onto the task's observability context.

        The task trace_id is authoritative — skill-internal calls and their
        safety-audit entries must correlate to the task trace. The skill's own
        robot_id / subtask_id / timeout and its cancel event are preserved
        (``dataclasses.replace`` carries the existing event through).
        """
        parent = self._parent_ctx
        if parent is None:
            return ctx
        store = parent.artifact_store if ctx.artifact_store is None else ctx.artifact_store
        outbound = parent.outbound if ctx.outbound is None else ctx.outbound
        return replace(ctx, trace_id=parent.trace_id, artifact_store=store, outbound=outbound)

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx = self._bind_task_context(ctx)
        if self._safety_gated:
            cmd = self._registry.build_safety_command(self._tool.name, args, ctx)
            if cmd is not None:
                # SafetyEnvelopeViolation is NOT caught — the command was refused
                # before dispatch; the violation propagates and aborts the task.
                await self._envelope.check(cmd, trace_id=ctx.trace_id, subtask_id=ctx.subtask_id)
            else:
                # Hardware-bound but no harness-checkable target: honest "skipped"
                # audit — never indistinguishable from "checked and passed".
                self._envelope.note_skipped(
                    tool_name=self._tool.name,
                    robot_id=ctx.robot_id,
                    trace_id=ctx.trace_id,
                    subtask_id=ctx.subtask_id,
                    reason="no harness-checkable target — on-robot reflex is authoritative",
                )
        return await self._tool.invoke(args, ctx)

    async def cancel(self, ctx: ToolContext) -> None:
        await self._tool.cancel(self._bind_task_context(ctx))


class SafetyGatedToolRegistry:
    """A ToolRegistry facade that safety-gates hardware-bound tools on ``get``.

    Handed to :meth:`Skill.execute` in place of the raw registry so a skill's
    internal tool calls get the same pre-dispatch SafetyEnvelope check the
    AgentLoop applies to Brain-issued calls. When ``parent_ctx`` is given (the
    AgentLoop passes its task-level ToolContext), every tool — hardware or not —
    is wrapped so its calls are rebound onto the task trace. Without a
    ``parent_ctx`` (e.g. the reverse MCP server, where each external call mints
    the authoritative context itself), non-hardware tools are returned
    unwrapped. Every other registry method delegates to the wrapped registry.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        envelope: SafetyEnvelope,
        parent_ctx: ToolContext | None = None,
    ) -> None:
        self._registry = registry
        self._envelope = envelope
        self._parent_ctx = parent_ctx

    def get(self, name: str) -> Tool:
        tool = self._registry.get(name)
        gated = self._registry.requires_safety_check(name)
        if not gated and self._parent_ctx is None:
            return tool
        return _SafetyGatedTool(
            tool,
            self._registry,
            self._envelope,
            safety_gated=gated,
            parent_ctx=self._parent_ctx,
        )

    # Implicit special-method lookup bypasses __getattr__ (it resolves on the
    # type, not the instance), so the registry idioms `name in tools` / `len(tools)`
    # need explicit delegation — the facade must be a true drop-in.
    def __contains__(self, name: object) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __getattr__(self, item: str) -> Any:
        # Delegate every other ToolRegistry method (list_schemas, validate_args, …).
        return getattr(self._registry, item)
