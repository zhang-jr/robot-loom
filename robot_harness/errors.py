"""Harness-wide exception hierarchy.

Every exception carries contextual fields so structured traces can be built
without string parsing.  SafetyEnvelopeViolation is the only non-recoverable
exception — callers must not catch-and-continue it.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Root of all harness exceptions."""

    def __init__(
        self,
        message: str,
        *,
        trace_id: str = "",
        robot_id: str = "",
        subtask_id: str = "",
        tool_name: str = "",
        module_name: str = "",
    ) -> None:
        super().__init__(message)
        self.trace_id = trace_id
        self.robot_id = robot_id
        self.subtask_id = subtask_id
        self.tool_name = tool_name
        self.module_name = module_name

    def context(self) -> dict[str, str]:
        return {
            "trace_id": self.trace_id,
            "robot_id": self.robot_id,
            "subtask_id": self.subtask_id,
            "tool_name": self.tool_name,
            "module_name": self.module_name,
        }


# ---------------------------------------------------------------------------
# Brain
# ---------------------------------------------------------------------------


class BrainError(HarnessError):
    """Brain / LLM layer errors."""


class BrainTimeoutError(BrainError):
    """Brain did not respond within the deadline."""


class BrainOutputInvalidError(BrainError):
    """Brain response violated the expected tool-call schema."""


class ReplanLoopExceededError(BrainError):
    """Replanning exceeded the maximum iteration limit without converging."""

    def __init__(self, message: str, *, max_iterations: int = 0, **kw: str) -> None:
        super().__init__(message, **kw)
        self.max_iterations = max_iterations


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ToolError(HarnessError):
    """Tool-layer errors."""


class ToolNotFoundError(ToolError):
    """No tool with the given name is registered."""


class ToolBackendUnreachableError(ToolError):
    """The external server backing this tool is not reachable."""


class ToolSchemaViolationError(ToolError):
    """Tool input or output violated the declared JSON Schema."""

    def __init__(self, message: str, *, violations: list[str] | None = None, **kw: str) -> None:
        super().__init__(message, **kw)
        self.violations: list[str] = violations or []


class ToolTimeoutError(ToolError):
    """Tool invocation exceeded the allowed timeout."""


class ToolCancelledError(ToolError):
    """Tool invocation was cancelled via the context cancel token."""


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------


class SkillError(HarnessError):
    """Skill-layer errors."""


class SkillNotFoundError(SkillError):
    """No skill with the given name (and optional version) is registered."""


class SkillVersionMismatchError(SkillError):
    """Requested skill version is incompatible with the current embodiment."""


class SkillManifestInvalidError(SkillError):
    """Skill manifest failed schema validation."""

    def __init__(
        self, message: str, *, validation_errors: list[str] | None = None, **kw: str
    ) -> None:
        super().__init__(message, **kw)
        self.validation_errors: list[str] = validation_errors or []


class SkillSafetyClassViolation(SkillError):  # noqa: N818
    """Skill attempted an action above its declared safety class."""


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class MemoryError(HarnessError):
    """Memory-layer errors."""


class MemoryServiceUnavailable(MemoryError):  # noqa: N818
    """External memory server is not reachable."""


class MemoryConflictError(MemoryError):
    """Write conflicted with an existing entry (optimistic concurrency failure)."""


class MemoryQueryEmpty(MemoryError):  # noqa: N818
    """Query returned no results (recoverable — caller may fall back)."""


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class CriticError(HarnessError):
    """Critic-layer errors."""


class CriticServiceDown(CriticError):  # noqa: N818
    """Critic server is not reachable; system must fall back to heuristic."""


class CriticDisagreementError(CriticError):
    """Critic verdict conflicts severely with sensor heuristics."""


# ---------------------------------------------------------------------------
# Embodiment
# ---------------------------------------------------------------------------


class EmbodimentError(HarnessError):
    """Embodiment / hardware layer errors."""


class HardwareNotReadyError(EmbodimentError):
    """Hardware is not in a state ready to accept commands."""


class RobotOfflineError(EmbodimentError):
    """Robot is offline or unreachable."""


class ActionDispatchTimeoutError(EmbodimentError):
    """Dispatched action did not complete within the allowed window."""


# ---------------------------------------------------------------------------
# Safety  — non-recoverable
# ---------------------------------------------------------------------------


class SafetyError(HarnessError):
    """Safety-layer errors."""


class SafetyEnvelopeViolation(SafetyError):  # noqa: N818
    """Safety check failed.

    This exception MUST NOT be caught and suppressed by any caller.
    Receipt triggers emergency stop + persistent audit log entry.
    """

    def __init__(
        self,
        message: str,
        *,
        violated_rules: list[str] | None = None,
        **kw: str,
    ) -> None:
        super().__init__(message, **kw)
        self.violated_rules: list[str] = violated_rules or []


# ---------------------------------------------------------------------------
# Channel  — user-facing IO transport (ADR-023)
# ---------------------------------------------------------------------------


class ChannelError(HarnessError):
    """Channel / user-facing IO transport errors."""

    def __init__(self, message: str, *, channel: str = "", **kw: str) -> None:
        super().__init__(message, **kw)
        self.channel = channel


class ChannelUnavailable(ChannelError):  # noqa: N818
    """The target channel is not registered or not reachable."""


class ChannelDeliveryFailed(ChannelError):  # noqa: N818
    """An outbound message could not be delivered through the channel."""


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------


class FleetError(HarnessError):
    """Multi-robot fleet coordination errors."""


class FleetCapacityExceededError(FleetError):
    """No available robot slot to accept the subtask."""


class RobotLockConflictError(FleetError):
    """Requested robot is already leased to another task."""


class HeterogeneousMismatchError(FleetError):
    """Subtask requires a robot type not present in the current fleet."""
