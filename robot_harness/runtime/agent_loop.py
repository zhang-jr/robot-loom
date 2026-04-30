"""AgentLoop — the Brain ⇄ Tool main loop.

Orchestrates: Brain.decide() → tool dispatch (with safety check) → result
→ Brain (next turn), repeating until completion or give_up.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from robot_harness.brain.base import (
    Brain,
    ExecutionHistory,
    MemoryView,
    Task,
    ToolCallRequest,
)
from robot_harness.errors import (
    ReplanLoopExceededError,
    ToolCancelledError,
    ToolError,
)
from robot_harness.memory.base import MemoryQuery
from robot_harness.observability.tracer import tracer
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.base import BrainProfile, ToolContext, ToolResult


class AgentResult(BaseModel):
    """Final outcome of an AgentLoop.run() call."""

    task_id: str
    robot_id: str
    outcome: Literal["success", "failure", "give_up"]
    message: str = ""
    turns: int = 0
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str = ""


class AgentLoop:
    """Drives a single task from start to completion.

    Args:
        brain: A Brain implementation (e.g. LiteLLMBrain).
        ctx:   The HarnessContext for this session.
        max_turns: Hard cap on Brain decision cycles per task.
    """

    def __init__(
        self,
        brain: Brain,
        ctx: HarnessContext,
        max_turns: int = 20,
    ) -> None:
        self._brain = brain
        self._ctx = ctx
        self._max_turns = max_turns

    async def run(self, task: Task) -> AgentResult:
        """Execute *task* until completion, give_up, or max_turns exceeded."""
        trace_id = str(uuid.uuid4())
        tracer.event(
            "agent_loop.start",
            trace_id=trace_id,
            task_id=task.task_id,
            robot_id=task.robot_id,
            description=task.description,
        )

        tool_specs = self._ctx.tool_registry.export_for_brain(BrainProfile(name="openai"))
        history = ExecutionHistory()
        all_tool_results: list[dict[str, Any]] = []

        for turn in range(1, self._max_turns + 1):
            tracer.event(
                "agent_loop.turn",
                trace_id=trace_id,
                turn=turn,
                robot_id=task.robot_id,
            )

            memory_view = await self._query_memory(task)
            decision = await self._brain.decide(task, memory_view, tool_specs)
            decision.trace_id = trace_id

            if decision.decision_type == "give_up":
                tracer.event("agent_loop.give_up", trace_id=trace_id, message=decision.message)
                return AgentResult(
                    task_id=task.task_id,
                    robot_id=task.robot_id,
                    outcome="give_up",
                    message=decision.message,
                    turns=turn,
                    tool_results=all_tool_results,
                    trace_id=trace_id,
                )

            if decision.decision_type == "plan":
                tracer.event("agent_loop.plan", trace_id=trace_id, plan=decision.plan[:200])
                return AgentResult(
                    task_id=task.task_id,
                    robot_id=task.robot_id,
                    outcome="success",
                    message=decision.message,
                    turns=turn,
                    tool_results=all_tool_results,
                    trace_id=trace_id,
                )

            if decision.decision_type == "tool_call":
                turn_results = await self._execute_tool_calls(decision.tool_calls, task, trace_id)
                all_tool_results.extend(turn_results)
                history.turns.append(
                    {
                        "turn": turn,
                        "tool_calls": [tc.model_dump() for tc in decision.tool_calls],
                        "results": turn_results,
                    }
                )
                # Check if any tool result signals task completion
                for r in turn_results:
                    if (r.get("output") or {}).get("task_complete"):
                        return AgentResult(
                            task_id=task.task_id,
                            robot_id=task.robot_id,
                            outcome="success",
                            message="Task completed via tool signal",
                            turns=turn,
                            tool_results=all_tool_results,
                            trace_id=trace_id,
                        )
                continue

            if decision.decision_type == "ask_user":
                tracer.event(
                    "agent_loop.ask_user",
                    trace_id=trace_id,
                    message=decision.message,
                )
                return AgentResult(
                    task_id=task.task_id,
                    robot_id=task.robot_id,
                    outcome="failure",
                    message=f"Brain requires user input: {decision.message}",
                    turns=turn,
                    tool_results=all_tool_results,
                    trace_id=trace_id,
                )

        raise ReplanLoopExceededError(
            f"Task '{task.task_id}' exceeded {self._max_turns} turns without completion",
            max_iterations=self._max_turns,
            trace_id=trace_id,
            robot_id=task.robot_id,
        )

    async def _execute_tool_calls(
        self,
        requests: list[ToolCallRequest],
        task: Task,
        trace_id: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for req in requests:
            tool_ctx = ToolContext(
                trace_id=trace_id,
                robot_id=task.robot_id,
                subtask_id=task.subtask_id,
                timeout_s=self._ctx.config.tool.default_timeout_s,
            )
            result = await self._invoke_tool(req, tool_ctx)
            results.append(result.model_dump())
        return results

    async def _invoke_tool(self, req: ToolCallRequest, ctx: ToolContext) -> ToolResult:
        try:
            tool = self._ctx.tool_registry.get(req.tool_name)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=req.tool_name,
                trace_id=ctx.trace_id,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

        try:
            return await tool.invoke(req.args, ctx)
        except ToolCancelledError:
            return ToolResult(
                tool_name=req.tool_name,
                trace_id=ctx.trace_id,
                success=False,
                error="cancelled",
                error_type="ToolCancelledError",
            )
        except ToolError as exc:
            return ToolResult(
                tool_name=req.tool_name,
                trace_id=ctx.trace_id,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    async def _query_memory(self, task: Task) -> MemoryView:
        """Query memory for context relevant to the task (best-effort)."""
        try:
            hits = await self._ctx.memory.query(
                MemoryQuery(memory_type="episodic", robot_id=task.robot_id, text=task.description)
            )
            return MemoryView(episodic_hits=[h.model_dump() for h in hits])
        except Exception:  # noqa: BLE001
            return MemoryView()
