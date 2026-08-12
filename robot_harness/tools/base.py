"""Tool Protocol, ToolContext, ToolResult, and ToolRegistry.

This module is the ★ core abstraction ★ of the harness.  Everything else
(Brain, Skill, Embodiment) communicates through Tool.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from robot_harness.errors import ToolNotFoundError, ToolSchemaViolationError
from robot_harness.tools.schema import ToolBackend, ToolSchema

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from robot_harness.embodiment.base import EmbodimentCommand
    from robot_harness.tools.artifacts import ArtifactStore
    from robot_harness.tools.outbound import OutboundHandle

    SafetyCommandBuilder = Callable[[dict[str, Any], "ToolContext"], "list[EmbodimentCommand]"]


@dataclass
class ToolContext:
    """Per-invocation context threaded through every tool call.

    ``cancel()`` signals the tool to abort; tools should check
    ``is_cancelled`` at safe checkpoints and raise ``ToolCancelledError``.
    """

    trace_id: str
    robot_id: str
    subtask_id: str = ""
    lease_id: str = ""
    timeout_s: float = 30.0
    # Out-of-band store for large tool I/O (images, depth). Producing tools put
    # bytes and return an ArtifactRef; the resolver middleware hydrates refs from
    # here before dispatch. None when no store is wired (refs are then disabled).
    artifact_store: ArtifactStore | None = None
    # Session-bound path back to the user (ADR-023). Set by the AgentLoop from the
    # originating channel; None for programmatic callers. The send_message tool
    # delivers through it instead of only tracing.
    outbound: OutboundHandle | None = None
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    async def wait_cancelled(self) -> None:
        await self._cancel_event.wait()

    @classmethod
    def create(
        cls,
        robot_id: str,
        *,
        subtask_id: str = "",
        timeout_s: float = 30.0,
    ) -> ToolContext:
        return cls(
            trace_id=str(uuid.uuid4()),
            robot_id=robot_id,
            subtask_id=subtask_id,
            timeout_s=timeout_s,
        )


class ToolResult(BaseModel):
    """Outcome of a single tool invocation."""

    tool_name: str
    trace_id: str
    success: bool
    output: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    latency_ms: float = 0.0


@runtime_checkable
class Tool(Protocol):
    """Atomic capability unit — the first-class abstraction of the harness.

    Rules:
    - ``invoke`` must be idempotent unless ``is_idempotent`` returns False.
    - All inputs validated against ``schema.input_schema`` before invocation.
    - SafetyEnvelope check is performed by the AgentLoop, not here.

    ``name``, ``schema``, ``backend`` are declared as read-only properties so
    that concrete implementations may satisfy the Protocol with either a plain
    class variable (readable) or an ``@property`` (read-only).

    A tool that actuates the robot also exposes:
    - ``hardware_bound = True`` — the registry gates it behind SafetyEnvelope.
    - ``to_safety_command(args, ctx) -> EmbodimentCommand | None`` — maps the
      call to the high-level command the envelope validates (pose reachability,
      workspace bounds). Returning None means the call carries no
      harness-checkable target and the on-robot reflex is authoritative.

    Visibility (orthogonal to ``hardware_bound``):
    - ``brain_visible = True`` (default) — the tool appears in the spec list the
      Brain plans over (:meth:`ToolRegistry.export_for_brain`).
    - ``brain_visible = False`` — the tool stays *registered and invocable* (so a
      skill can call it via ``get()``, and ``tool list`` still shows it for
      debugging) but is hidden from the Brain's planning vocabulary. Use this for
      verb-internal sub-capabilities (grasp-pose estimation, single-step VLA
      inference) and low-level override dispatch, so the Brain is not tempted to
      hand-assemble a control pipeline at the slow planning layer.

    ``hardware_bound`` answers "must this pass SafetyEnvelope?"; ``brain_visible``
    answers "should the Brain see this when planning?" — they cross-cut, so never
    derive one from the other.
    """

    @property
    def name(self) -> str: ...

    @property
    def schema(self) -> ToolSchema: ...

    @property
    def backend(self) -> ToolBackend: ...

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...

    @property
    def is_idempotent(self) -> bool: ...

    @property
    def is_cancellable(self) -> bool: ...

    async def cancel(self, ctx: ToolContext) -> None: ...


class ToolFilter(BaseModel):
    """Filter criteria for :meth:`ToolRegistry.list_schemas`."""

    name_pattern: str | None = None
    backend: ToolBackend | None = None
    tags: list[str] = []


BrainProfileName = Literal["openai", "anthropic", "mcp", "litellm"]


class BrainProfile(BaseModel):
    """Describes which tool-spec format a Brain backend expects."""

    name: BrainProfileName
    supports_native_reflection: bool = True  # False → register ReflectionTool


BrainToolSpec = dict[str, Any]


def _resolve_safety_builder(tool: Tool) -> SafetyCommandBuilder | None:
    """Normalise a hardware tool's safety hook to the plural form.

    A tool declares either ``to_safety_commands`` (a sequence — for calls that
    actuate more than one target) or the singular ``to_safety_command``, which
    stays the common case: most verbs carry exactly one pose or joint target.
    Both collapse to "the list of commands this call actuates" here, so the gate
    has one shape to consume. A tool with neither hook yields None — hardware-
    bound but not harness-checkable.
    """
    plural = getattr(tool, "to_safety_commands", None)
    if plural is not None:
        return lambda args, ctx: list(plural(args, ctx))
    singular = getattr(tool, "to_safety_command", None)
    if singular is not None:
        return lambda args, ctx: [cmd] if (cmd := singular(args, ctx)) is not None else []
    return None


class ToolRegistry:
    """Central registry for all tools visible to the Brain.

    Thread-safe for reads; register() should be called at startup before
    the event loop dispatches any agent tasks.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._safety_gated: dict[str, SafetyCommandBuilder | None] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool; silently overwrites an existing entry with the same name.

        Tools that actuate the robot (``hardware_bound``) are recorded so the
        AgentLoop runs SafetyEnvelope.check() before invoking them. Overwriting
        also resets that record — a stale entry would keep gating the name with
        the REPLACED tool's safety-command builder (ISS-039).
        """
        self._safety_gated.pop(tool.name, None)
        if getattr(tool, "hardware_bound", False):
            self._safety_gated[tool.name] = _resolve_safety_builder(tool)
        self._tools[tool.name] = tool

    def requires_safety_check(self, name: str) -> bool:
        """True if *name* actuates the robot and must pass SafetyEnvelope first."""
        return name in self._safety_gated

    def build_safety_commands(
        self, name: str, args: dict[str, Any], ctx: ToolContext
    ) -> list[EmbodimentCommand]:
        """Every EmbodimentCommand this call will actuate, for the envelope to validate.

        Plural because one tool call may drive a whole sequence: a taught motion
        is one call over many joint targets, and the gate is only meaningful if
        it sees all of them BEFORE the first one moves (checking point 7 after
        points 1-6 executed is not a pre-dispatch gate).

        An empty list means the call carries no harness-checkable target; the
        on-robot safety reflex is then the authoritative check and the caller
        must record an honest "skipped" audit rather than a "passed" one.
        """
        builder = self._safety_gated.get(name)
        if builder is None:
            return []
        return builder(args, ctx)

    def get(self, name: str) -> Tool:
        """Return the tool or raise :exc:`ToolNotFoundError`."""
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"Tool '{name}' is not registered", tool_name=name) from None

    def list_schemas(self, filter: ToolFilter | None = None) -> list[ToolSchema]:
        """Return schemas of all registered tools, optionally filtered."""
        tools = list(self._tools.values())
        if filter is not None:
            if filter.backend is not None:
                tools = [t for t in tools if t.backend == filter.backend]
            if filter.name_pattern is not None:
                pat = filter.name_pattern.lower()
                tools = [t for t in tools if pat in t.name.lower()]
        return [t.schema for t in tools]

    def export_for_brain(
        self,
        profile: BrainProfile,
        *,
        exclude_names: Collection[str] = (),
    ) -> list[BrainToolSpec]:
        """Export tool specs in the format expected by the Brain backend.

        Only tools with ``brain_visible`` (default True) are exported. A tool
        marked ``brain_visible = False`` stays registered and invocable — skills
        can still reach it via :meth:`get` — but is hidden from the Brain's
        planning vocabulary. See :class:`Tool` for when to hide a tool.

        ``exclude_names`` additionally hides the named tools from THIS export
        only (session-scoped, e.g. on-robot verbs the fleet does not currently
        advertise — see ``HarnessContext.unavailable_tool_names``). Unlike
        ``brain_visible`` it is a per-call filter, not a tool property; excluded
        tools stay registered and invocable. Filtering happens on tool names
        before serialization, so it is Brain-profile-agnostic.
        """
        schemas = [
            t.schema
            for t in self._tools.values()
            if getattr(t, "brain_visible", True) and t.name not in exclude_names
        ]
        if profile.name in ("openai", "litellm"):
            return [s.to_openai_function() for s in schemas]
        if profile.name == "mcp":
            return [s.to_mcp_tool() for s in schemas]
        if profile.name == "anthropic":
            return [
                {
                    "name": s.name,
                    "description": s.description,
                    "input_schema": s.input_schema,
                }
                for s in schemas
            ]
        return [s.to_openai_function() for s in schemas]

    def validate_args(self, tool_name: str, args: dict[str, Any]) -> None:
        """Raise :exc:`ToolSchemaViolationError` for missing required args or bad enums.

        Enum enforcement covers top-level properties only. It matters because an
        enum is the schema's way of saying "these are the only legal values" —
        left unchecked (as a hint the Brain may ignore), a closed vocabulary like
        a taught-motion name becomes an invented string that costs a round trip
        to the robot to reject. Failing here instead hands the Brain the legal
        values in the error, on the same turn.
        """
        tool = self.get(tool_name)
        input_schema = tool.schema.input_schema
        required = input_schema.get("required", [])
        missing = [r for r in required if r not in args]
        if missing:
            raise ToolSchemaViolationError(
                f"Tool '{tool_name}' missing required args: {missing}",
                violations=[f"missing: {m}" for m in missing],
                tool_name=tool_name,
            )

        violations: list[str] = []
        for prop, spec in (input_schema.get("properties") or {}).items():
            if not isinstance(spec, dict) or prop not in args:
                continue
            allowed = spec.get("enum")
            # An absent enum means "any value of this type"; an empty one is a
            # schema bug, not a rule that rejects everything.
            if allowed and args[prop] not in allowed:
                violations.append(f"{prop}={args[prop]!r} not in {sorted(map(str, allowed))}")
        if violations:
            raise ToolSchemaViolationError(
                f"Tool '{tool_name}' got out-of-enum args: {'; '.join(violations)}",
                violations=violations,
                tool_name=tool_name,
            )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
