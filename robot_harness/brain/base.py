"""Brain Protocol — the LLM/VLM slow-thinking decision layer."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class Task(BaseModel):
    """A task handed to the Brain for planning."""

    task_id: str
    description: str
    robot_id: str
    subtask_id: str = ""
    constraints: list[str] = []
    metadata: dict[str, Any] = {}


class ToolCallRequest(BaseModel):
    """One tool call the Brain wants the harness to execute."""

    tool_name: str
    args: dict[str, Any]
    call_id: str = ""


class BrainDecision(BaseModel):
    """What the Brain decided to do next."""

    decision_type: Literal["tool_call", "plan", "give_up", "ask_user"]
    tool_calls: list[ToolCallRequest] = []
    plan: str = ""
    message: str = ""
    trace_id: str = ""
    raw_response: dict[str, Any] = {}


class MemoryView(BaseModel):
    """A snapshot of relevant memory passed to the Brain."""

    object_hits: list[dict[str, Any]] = []
    place_hits: list[dict[str, Any]] = []
    episodic_hits: list[dict[str, Any]] = []
    semantic_hits: list[dict[str, Any]] = []


class ExecutionHistory(BaseModel):
    """History of prior turns in the current task — fed to replan()."""

    turns: list[dict[str, Any]] = []
    last_critic_signal: str = ""


class CriticSignal(BaseModel):
    """Critic verdict forwarded to the Brain for replanning."""

    state: Literal["progress", "completion", "failure", "unchanged"]
    confidence: float
    evidence: str


@runtime_checkable
class Brain(Protocol):
    """LLM/VLM planning and tool-calling decision interface.

    Implementations: LiteLLMBrain (default).  Brain must NOT be called
    more frequently than 1-7 Hz; high-frequency control loops live in
    EmbodimentAdapter.
    """

    async def decide(
        self,
        task: Task,
        memory_view: MemoryView,
        tools: list[dict[str, Any]],
    ) -> BrainDecision: ...

    async def replan(
        self,
        history: ExecutionHistory,
        critic_signal: CriticSignal,
    ) -> BrainDecision: ...

    @property
    def supports_streaming(self) -> bool: ...
