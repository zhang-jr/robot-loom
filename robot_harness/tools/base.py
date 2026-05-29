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
    from collections.abc import Callable

    from robot_harness.embodiment.base import EmbodimentCommand

    SafetyCommandBuilder = Callable[[dict[str, Any], "ToolContext"], "EmbodimentCommand | None"]


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
        AgentLoop runs SafetyEnvelope.check() before invoking them.
        """
        if getattr(tool, "hardware_bound", False):
            self._safety_gated[tool.name] = getattr(tool, "to_safety_command", None)
        self._tools[tool.name] = tool

    def requires_safety_check(self, name: str) -> bool:
        """True if *name* actuates the robot and must pass SafetyEnvelope first."""
        return name in self._safety_gated

    def build_safety_command(
        self, name: str, args: dict[str, Any], ctx: ToolContext
    ) -> EmbodimentCommand | None:
        """Build the EmbodimentCommand a hardware-bound tool validates against.

        Returns None when the call carries no harness-checkable target; the
        on-robot safety reflex is then the authoritative check.
        """
        builder = self._safety_gated.get(name)
        if builder is None:
            return None
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

    def export_for_brain(self, profile: BrainProfile) -> list[BrainToolSpec]:
        """Export tool specs in the format expected by the Brain backend."""
        schemas = self.list_schemas()
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
        """Raise :exc:`ToolSchemaViolationError` if args miss required fields."""
        tool = self.get(tool_name)
        required = tool.schema.input_schema.get("required", [])
        missing = [r for r in required if r not in args]
        if missing:
            raise ToolSchemaViolationError(
                f"Tool '{tool_name}' missing required args: {missing}",
                violations=[f"missing: {m}" for m in missing],
                tool_name=tool_name,
            )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
