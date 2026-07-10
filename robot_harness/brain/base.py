"""Brain Protocol — the LLM/VLM slow-thinking decision layer.

The Brain follows the **native tool-use message protocol** (ADR-025): the
AgentLoop owns a growing conversation (`system → user → assistant(tool_calls) →
tool(result) → …`) and hands it to :meth:`Brain.decide` each turn. The Brain is
a thin "given this conversation + tool specs, decide the next step" call — it
does not own session state, build prompts, or round-trip through Memory. Recent
observations live in the conversation itself; cross-session recall is the
on-demand ``memory.query`` tool (ADR-024).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from robot_harness.tools.base import BrainToolSpec

# A provider-format chat message (``{"role": ..., "content": ...}``, plus
# ``tool_calls`` on assistant turns and ``tool_call_id`` on tool turns). Kept as a
# loose dict so it passes straight to the LLM backend (OpenAI / Anthropic / …).
Message = dict[str, Any]


class Task(BaseModel):
    """A task handed to the Brain for planning."""

    task_id: str
    description: str
    robot_id: str
    subtask_id: str = ""
    constraints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    """One tool call the Brain wants the harness to execute."""

    tool_name: str
    args: dict[str, Any]
    call_id: str = ""


class BrainDecision(BaseModel):
    """What the Brain decided to do next.

    ``assistant_message`` is the raw assistant turn (provider format, wire tool
    names) the AgentLoop appends to the conversation before dispatching the tool
    calls — so the next turn sees the Brain's own prior output and the tool
    results that answered it. The loop synthesizes one if the Brain leaves it None.
    """

    decision_type: Literal["tool_call", "respond", "give_up", "ask_user"]
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    message: str = ""
    trace_id: str = ""
    assistant_message: Message | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Brain(Protocol):
    """LLM/VLM planning and tool-calling decision interface.

    Implementations: LiteLLMBrain (default).  Brain must NOT be called
    more frequently than 1-7 Hz; high-frequency control loops live in
    EmbodimentAdapter.
    """

    async def decide(
        self,
        messages: list[Message],
        tools: list[BrainToolSpec],
        *,
        trace_id: str = "",
        robot_id: str = "",
    ) -> BrainDecision:
        """Decide the next step given the running conversation and tool specs.

        ``messages`` is the full provider-format conversation the AgentLoop owns
        and grows across turns (it already carries the task, prior tool calls,
        and their results). Replanning is just another turn — a critic-feedback
        message the loop appended — not a separate entry point.

        ``trace_id`` / ``robot_id`` are observability context only: they tag the
        decide span and any Brain exception so the Brain layer stays on the same
        trace as the task's tool / critic / memory spans. They never enter the
        conversation or the prompt.
        """
        ...

    @property
    def supports_streaming(self) -> bool: ...
