"""LiteLLM Brain implementation.

Wraps LiteLLM's acompletion() to support OpenAI / Anthropic / local vLLM /
Ollama backends transparently via the unified Brain Protocol.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import litellm
import litellm.exceptions

from robot_harness.brain.base import (
    BrainDecision,
    CriticSignal,
    ExecutionHistory,
    MemoryView,
    Task,
    ToolCallRequest,
)
from robot_harness.config.schema import BrainConfig
from robot_harness.errors import BrainOutputInvalidError, BrainTimeoutError
from robot_harness.observability.tracer import tracer

_SYSTEM_PROMPT = """\
You are a robot task planner. You have access to tools that control robot hardware.
Plan step by step. Call tools one or a few at a time. Never skip the safety check tool.
When the task is complete, respond with a final message explaining the outcome.
Do NOT call tools after the task is done — just respond naturally.
"""

_REPLAN_SYSTEM_PROMPT = """\
You are a robot task replanner. A previous attempt partially failed or stalled.
Given the execution history and critic feedback, decide whether to continue,
try a different approach, or give up.  If giving up, explain clearly.
"""


class LiteLLMBrain:
    """Concrete Brain backed by LiteLLM.

    Supports any model string LiteLLM accepts (openai/*, anthropic/*,
    ollama/*, hosted_vllm/*, etc.).
    """

    def __init__(self, config: BrainConfig) -> None:
        self._cfg = config
        litellm.drop_params = True  # ignore unsupported params silently

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(
        self,
        task: Task,
        memory_view: MemoryView,
        tools: list[dict[str, Any]],
    ) -> BrainDecision:
        trace_id = str(uuid.uuid4())
        messages = self._build_decide_messages(task, memory_view)

        with tracer.span(
            "brain.decide",
            trace_id=trace_id,
            robot_id=task.robot_id,
            model=self._cfg.model,
        ):
            try:
                response = await litellm.acompletion(
                    model=self._cfg.model,
                    messages=messages,
                    tools=tools or None,
                    tool_choice="auto" if tools else None,
                    temperature=self._cfg.temperature,
                    max_tokens=self._cfg.max_tokens,
                    timeout=self._cfg.timeout_s,
                    api_base=self._cfg.api_base,
                )
            except litellm.exceptions.Timeout as exc:
                raise BrainTimeoutError(
                    f"Brain timed out after {self._cfg.timeout_s}s",
                    trace_id=trace_id,
                    robot_id=task.robot_id,
                ) from exc
            except Exception as exc:
                raise BrainOutputInvalidError(
                    f"Brain backend error: {exc}",
                    trace_id=trace_id,
                    robot_id=task.robot_id,
                ) from exc

        return self._parse_response(response, trace_id)

    async def replan(
        self,
        history: ExecutionHistory,
        critic_signal: CriticSignal,
    ) -> BrainDecision:
        trace_id = str(uuid.uuid4())
        messages = self._build_replan_messages(history, critic_signal)

        with tracer.span("brain.replan", trace_id=trace_id, critic_state=critic_signal.state):
            try:
                response = await litellm.acompletion(
                    model=self._cfg.model,
                    messages=messages,
                    temperature=self._cfg.temperature,
                    max_tokens=self._cfg.max_tokens,
                    timeout=self._cfg.timeout_s,
                    api_base=self._cfg.api_base,
                )
            except Exception as exc:
                raise BrainOutputInvalidError(
                    f"Replan backend error: {exc}", trace_id=trace_id
                ) from exc

        return self._parse_response(response, trace_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_decide_messages(self, task: Task, memory_view: MemoryView) -> list[dict[str, Any]]:
        memory_context = ""
        if any(
            [
                memory_view.object_hits,
                memory_view.place_hits,
                memory_view.episodic_hits,
                memory_view.semantic_hits,
            ]
        ):
            memory_context = (
                f"\n\nMemory context:\n{json.dumps(memory_view.model_dump(), indent=2)}"
            )

        user_content = f"Task: {task.description}"
        if task.constraints:
            user_content += f"\nConstraints: {'; '.join(task.constraints)}"
        user_content += memory_context

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _build_replan_messages(
        self, history: ExecutionHistory, critic_signal: CriticSignal
    ) -> list[dict[str, Any]]:
        content = (
            f"Critic verdict: {critic_signal.state} "
            f"(confidence={critic_signal.confidence:.2f})\n"
            f"Evidence: {critic_signal.evidence}\n\n"
            f"Execution history:\n{json.dumps(history.turns, indent=2)}"
        )
        return [
            {"role": "system", "content": _REPLAN_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    _GIVE_UP_PHRASES = (
        "i give up",
        "i'm giving up",
        "giving up on this task",
        "cannot complete this task",
        "unable to complete this task",
        "this task is impossible",
        "task cannot be completed",
    )
    _CONTINUATION_MARKERS = (
        "try",
        "attempt",
        "instead",
        "alternative",
        "different approach",
        "retry",
        "plan b",
        "another way",
        "let me",
    )

    @classmethod
    def _is_give_up(cls, text: str) -> bool:
        """Detect explicit give-up intent without false-positiving on retry language."""
        lower = text.lower()
        has_give_up = any(phrase in lower for phrase in cls._GIVE_UP_PHRASES)
        if not has_give_up:
            return False
        has_continuation = any(marker in lower for marker in cls._CONTINUATION_MARKERS)
        return not has_continuation

    def _parse_response(self, response: Any, trace_id: str) -> BrainDecision:
        try:
            choice = response.choices[0]
            msg = choice.message
        except (AttributeError, IndexError) as exc:
            raise BrainOutputInvalidError(
                f"Unexpected response structure: {exc}", trace_id=trace_id
            ) from exc

        # Tool-call path
        tool_calls_raw = getattr(msg, "tool_calls", None)
        if tool_calls_raw:
            parsed: list[ToolCallRequest] = []
            for tc in tool_calls_raw:
                try:
                    args = json.loads(tc.function.arguments)
                    parsed.append(
                        ToolCallRequest(
                            tool_name=tc.function.name,
                            args=args,
                            call_id=tc.id or "",
                        )
                    )
                except (json.JSONDecodeError, AttributeError) as exc:
                    raise BrainOutputInvalidError(
                        f"Invalid tool call JSON: {exc}", trace_id=trace_id
                    ) from exc
            return BrainDecision(
                decision_type="tool_call",
                tool_calls=parsed,
                trace_id=trace_id,
                raw_response={"finish_reason": choice.finish_reason},
            )

        # Text response path
        text = (msg.content or "").strip()
        finish = getattr(choice, "finish_reason", "stop")

        if finish == "stop" or text:
            if self._is_give_up(text):
                return BrainDecision(
                    decision_type="give_up",
                    message=text,
                    trace_id=trace_id,
                )
            return BrainDecision(
                decision_type="plan",
                plan=text,
                message=text,
                trace_id=trace_id,
            )

        raise BrainOutputInvalidError(
            f"Unhandled finish_reason='{finish}' with no content",
            trace_id=trace_id,
        )
