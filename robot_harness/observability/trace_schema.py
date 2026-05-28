"""Trace and episode-buffer schemas.

Fields are designed to accommodate data backflow to VLA / VLM-policy training
(ADR-013). Training pipeline is NOT implemented here; only the schema is
pinned so downstream consumers can rely on it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolSpan(BaseModel):
    """One tool invocation inside an agent turn."""

    tool_name: str
    tool_version: str = ""
    backend: str = ""
    args_schema_hash: str = ""
    latency_ms: float = 0.0
    outcome: Literal["success", "error", "timeout", "cancelled"] = "success"
    error_type: str = ""


class AgentTurnTrace(BaseModel):
    """Structured trace record emitted after each Brain→Tool→Safety cycle."""

    trace_id: str
    robot_id: str
    subtask_id: str
    task_description: str = ""

    brain_latency_ms: float = 0.0
    decision_type: str = ""

    tool_spans: list[ToolSpan] = Field(default_factory=list)

    safety_check_passed: bool = True
    safety_violated_rules: list[str] = Field(default_factory=list)

    critic_state: str = ""
    critic_confidence: float = 0.0

    skill_name: str = ""
    skill_version: str = ""

    outcome: Literal["success", "failure", "give_up", "in_progress"] = "in_progress"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    # Cognitive scaffold trail (ADR-018): plan/reflection store snapshots per turn.
    # Kept on a separate channel from physical episode data to avoid polluting
    # the VLA training schema with LLM meta-cognition artifacts.
    cognitive_scaffold_trail: list[dict[str, Any]] = Field(default_factory=list)


class EpisodeRecord(BaseModel):
    """One full task episode — root record for data backflow."""

    episode_id: str
    robot_id: str
    robot_type: str = ""
    embodiment: str = ""
    task_instruction: str

    turns: list[AgentTurnTrace] = Field(default_factory=list)

    # VLA / VLM-policy training fields (values populated post-episode)
    images: list[str] = Field(default_factory=list, description="image paths or URIs")
    states: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    rewards: list[float] = Field(default_factory=list)

    skill_version: str = ""
    safety_audit_ids: list[str] = Field(default_factory=list)

    outcome: Literal["success", "failure", "incomplete"] = "incomplete"
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    ended_at: datetime | None = None
