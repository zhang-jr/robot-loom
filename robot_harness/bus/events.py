"""Internal event bus event types."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(tz=UTC)


class HarnessEvent(BaseModel):
    """Base event propagated on the internal bus."""

    event_type: str
    trace_id: str
    robot_id: str
    timestamp: datetime = Field(default_factory=_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskStartedEvent(HarnessEvent):
    event_type: Literal["task.started"] = "task.started"


class TaskCompletedEvent(HarnessEvent):
    event_type: Literal["task.completed"] = "task.completed"
    outcome: Literal["success", "failure", "give_up"] = "success"


class ToolInvokedEvent(HarnessEvent):
    event_type: Literal["tool.invoked"] = "tool.invoked"
    tool_name: str = ""
    latency_ms: float = 0.0
    outcome: str = "success"


class SafetyViolationEvent(HarnessEvent):
    event_type: Literal["safety.violation"] = "safety.violation"
    violated_rules: list[str] = Field(default_factory=list)


class CriticVerdictEvent(HarnessEvent):
    event_type: Literal["critic.verdict"] = "critic.verdict"
    state: str = ""
    confidence: float = 0.0


class ReplanTriggeredEvent(HarnessEvent):
    event_type: Literal["brain.replan"] = "brain.replan"
    reason: str = ""
    iteration: int = 0
