"""AgentLoop — the Brain ⇄ Tool main loop.

Orchestrates: Brain.decide() → tool dispatch (with safety check) → result
→ Brain (next turn), repeating until completion or give_up.

Wired in:
- Optional Critic integration with ReplanPolicy (ADR-010)
- EpisodicMemory write-back on task completion
- Multi-type memory query (object / place / episodic / semantic)
- Cognitive scaffold context compression hook (ADR-018)
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from robot_harness.brain.base import (
    Brain,
    CriticSignal,
    ExecutionHistory,
    MemoryView,
    Task,
    ToolCallRequest,
)
from robot_harness.critic.base import Critic, CriticVerdict
from robot_harness.critic.heuristic_fallback import HeuristicCritic
from robot_harness.critic.replan_policy import ReplanPolicy, SensorHeuristic
from robot_harness.embodiment.base import Frame
from robot_harness.errors import (
    CriticDisagreementError,
    CriticServiceDown,
    ReplanLoopExceededError,
    ToolCancelledError,
    ToolError,
)
from robot_harness.memory.base import MemoryEntry, MemoryQuery
from robot_harness.observability.tracer import tracer
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.base import BrainProfile, ToolContext, ToolResult


class AgentResult(BaseModel):
    """Final outcome of an AgentLoop.run() call."""

    task_id: str
    robot_id: str
    outcome: Literal["success", "failure", "give_up", "incomplete"]
    message: str = ""
    turns: int = 0
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str = ""


class AgentLoop:
    """Drives a single task from start to completion.

    Args:
        brain:           A Brain implementation (e.g. LiteLLMBrain).
        ctx:             The HarnessContext for this session.
        max_turns:       Hard cap on Brain decision cycles per task.
        critic:          Optional progress-evaluation Critic (ADR-010).
        critic_interval: Run the Critic every N turns (default: every 3 turns).
        max_replan:      Max Brain.replan() calls per task before aborting.
    """

    def __init__(
        self,
        brain: Brain,
        ctx: HarnessContext,
        max_turns: int = 20,
        critic: Critic | None = None,
        critic_interval: int = 3,
        max_replan: int = 5,
    ) -> None:
        self._brain = brain
        self._ctx = ctx
        self._max_turns = max_turns
        self._critic = critic
        self._critic_interval = critic_interval
        self._max_replan = max_replan

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
        replan_count = 0
        sensor = SensorHeuristic()
        replan_policy = ReplanPolicy()
        active_critic = self._critic
        heuristic_critic: HeuristicCritic | None = None

        for turn in range(1, self._max_turns + 1):
            tracer.event(
                "agent_loop.turn",
                trace_id=trace_id,
                turn=turn,
                robot_id=task.robot_id,
            )

            # --- Cognitive scaffold context compression hook ---
            scaffold_injection = self._ctx.format_scaffold_for_injection(task.robot_id)
            if scaffold_injection:
                tracer.event(
                    "agent_loop.scaffold_injection",
                    trace_id=trace_id,
                    robot_id=task.robot_id,
                    chars=len(scaffold_injection),
                )

            memory_view = await self._query_memory(task)
            if scaffold_injection:
                memory_view.scaffold_context = scaffold_injection

            decision = await self._brain.decide(task, memory_view, tool_specs)
            decision.trace_id = trace_id

            if decision.decision_type == "give_up":
                tracer.event("agent_loop.give_up", trace_id=trace_id, message=decision.message)
                result = AgentResult(
                    task_id=task.task_id,
                    robot_id=task.robot_id,
                    outcome="give_up",
                    message=decision.message,
                    turns=turn,
                    tool_results=all_tool_results,
                    trace_id=trace_id,
                )
                await self._write_episode(task, result, trace_id)
                return result

            if decision.decision_type == "plan":
                tracer.event("agent_loop.plan", trace_id=trace_id, plan=decision.plan[:200])
                result = AgentResult(
                    task_id=task.task_id,
                    robot_id=task.robot_id,
                    outcome="success",
                    message=decision.message,
                    turns=turn,
                    tool_results=all_tool_results,
                    trace_id=trace_id,
                )
                await self._write_episode(task, result, trace_id)
                return result

            if decision.decision_type == "tool_call":
                turn_results = await self._execute_tool_calls(decision.tool_calls, task, trace_id)
                all_tool_results.extend(turn_results)
                await self._write_observations(task, turn, turn_results, trace_id)

                any_error = any(not r.get("success") for r in turn_results)
                if any_error:
                    sensor.error_count += 1
                else:
                    sensor.error_count = max(0, sensor.error_count - 1)

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
                        result = AgentResult(
                            task_id=task.task_id,
                            robot_id=task.robot_id,
                            outcome="success",
                            message="Task completed via tool signal",
                            turns=turn,
                            tool_results=all_tool_results,
                            trace_id=trace_id,
                        )
                        await self._write_episode(task, result, trace_id)
                        return result

                # --- Critic evaluation (every critic_interval turns) ---
                if active_critic and turn % self._critic_interval == 0:
                    frame = await self._ctx.get_camera_frame(task.robot_id)
                    verdict, active_critic, heuristic_critic = await self._run_critic(
                        active_critic, heuristic_critic, frame, task, trace_id
                    )

                    if verdict is not None:
                        critic_signal = CriticSignal(
                            state=verdict.state,
                            confidence=verdict.confidence,
                            evidence=verdict.evidence,
                        )
                        # Write critic verdict to episodic memory
                        await self._write_critic_verdict(task, verdict, trace_id)

                        try:
                            action = replan_policy.decide(verdict, sensor, trace_id=trace_id)
                        except CriticDisagreementError as e:
                            tracer.event(
                                "agent_loop.critic_disagreement",
                                trace_id=trace_id,
                                reason=str(e),
                            )
                            action = "replan"

                        if action == "abort":
                            outcome: Literal["success", "failure"] = (
                                "success" if verdict.state == "completion" else "failure"
                            )
                            result = AgentResult(
                                task_id=task.task_id,
                                robot_id=task.robot_id,
                                outcome=outcome,
                                message=f"Critic abort: {verdict.state} ({verdict.evidence})",
                                turns=turn,
                                tool_results=all_tool_results,
                                trace_id=trace_id,
                            )
                            await self._write_episode(task, result, trace_id)
                            return result

                        if action == "replan":
                            if replan_count >= self._max_replan:
                                raise ReplanLoopExceededError(
                                    f"Replan limit ({self._max_replan}) exceeded",
                                    max_iterations=self._max_replan,
                                    trace_id=trace_id,
                                    robot_id=task.robot_id,
                                )
                            replan_count += 1
                            history.last_critic_signal = verdict.state
                            new_decision = await self._brain.replan(history, critic_signal)
                            new_decision.trace_id = trace_id
                            tracer.event(
                                "agent_loop.replan",
                                trace_id=trace_id,
                                replan_count=replan_count,
                                critic_state=verdict.state,
                            )
                            if new_decision.decision_type == "tool_call":
                                turn_results = await self._execute_tool_calls(
                                    new_decision.tool_calls, task, trace_id
                                )
                                all_tool_results.extend(turn_results)
                                await self._write_observations(task, turn, turn_results, trace_id)
                                history.turns.append(
                                    {
                                        "turn": f"{turn}.replan",
                                        "tool_calls": [
                                            tc.model_dump() for tc in new_decision.tool_calls
                                        ],
                                        "results": turn_results,
                                    }
                                )

                        sensor.unchanged_count = 0  # reset after critic evaluation
                    else:
                        sensor.unchanged_count += 1

                continue

            if decision.decision_type == "ask_user":
                tracer.event(
                    "agent_loop.ask_user",
                    trace_id=trace_id,
                    message=decision.message,
                )
                result = AgentResult(
                    task_id=task.task_id,
                    robot_id=task.robot_id,
                    outcome="failure",
                    message=f"Brain requires user input: {decision.message}",
                    turns=turn,
                    tool_results=all_tool_results,
                    trace_id=trace_id,
                )
                await self._write_episode(task, result, trace_id)
                return result

        # Turn budget exhausted without a terminal decision. This is an EXPECTED
        # outcome — the agent kept acting/observing but never converged (a common
        # "not smart" failure mode with many causes: weak prompt, noisy critic,
        # insufficient observations, model limits). The harness is the safety net:
        # record the episode and return like every other exit, instead of raising
        # and crashing the caller. Distinct from the ReplanLoopExceededError above,
        # which signals genuine replan non-convergence.
        tracer.event(
            "agent_loop.max_turns_exceeded",
            trace_id=trace_id,
            robot_id=task.robot_id,
            turns=self._max_turns,
        )
        result = AgentResult(
            task_id=task.task_id,
            robot_id=task.robot_id,
            outcome="incomplete",
            message=f"Exceeded {self._max_turns} turns without completion",
            turns=self._max_turns,
            tool_results=all_tool_results,
            trace_id=trace_id,
        )
        await self._write_episode(task, result, trace_id)
        return result

    # ------------------------------------------------------------------
    # Critic helpers
    # ------------------------------------------------------------------

    async def _run_critic(
        self,
        active_critic: Critic,
        heuristic_critic: HeuristicCritic | None,
        frame: Frame,
        task: Task,
        trace_id: str,
    ) -> tuple[CriticVerdict | None, Critic, HeuristicCritic | None]:
        """Call the critic, falling back to heuristic if the service is down."""
        try:
            verdict = await active_critic.judge(frame, None, task.description)
            return verdict, active_critic, heuristic_critic
        except CriticServiceDown as exc:
            if heuristic_critic is None:
                heuristic_critic = HeuristicCritic(task_timeout_s=300.0)
            tracer.event(
                "agent_loop.critic_fallback",
                trace_id=trace_id,
                reason=str(exc),
                warning="Critic service down — degraded to timeout heuristic. "
                "Operator action required.",
            )
            try:
                verdict = await heuristic_critic.judge(frame, None, task.description)
                return verdict, active_critic, heuristic_critic
            except Exception:  # noqa: BLE001
                return None, active_critic, heuristic_critic
        except Exception:  # noqa: BLE001
            return None, active_critic, heuristic_critic

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    async def _write_episode(self, task: Task, result: AgentResult, trace_id: str) -> None:
        """Write episode outcome to episodic memory (best-effort)."""
        try:
            await self._ctx.memory.write(
                MemoryEntry(
                    memory_type="episodic",
                    robot_id=task.robot_id,
                    content={
                        "task_id": task.task_id,
                        "description": task.description,
                        "outcome": result.outcome,
                        "turns": result.turns,
                        "trace_id": trace_id,
                        "message": result.message,
                    },
                    tags=["episode", result.outcome, task.robot_id],
                )
            )
        except Exception as exc:  # noqa: BLE001
            tracer.event(
                "memory.write_failed",
                trace_id=trace_id,
                robot_id=task.robot_id,
                memory_type="episodic",
                error=str(exc),
            )

    async def _write_critic_verdict(
        self, task: Task, verdict: CriticVerdict, trace_id: str
    ) -> None:
        """Write critic verdict to episodic memory linked to the current episode."""
        try:
            await self._ctx.memory.write(
                MemoryEntry(
                    memory_type="episodic",
                    robot_id=task.robot_id,
                    content={
                        "task_id": task.task_id,
                        "critic_state": verdict.state,
                        "critic_confidence": verdict.confidence,
                        "evidence": verdict.evidence,
                        "trace_id": trace_id,
                    },
                    tags=["critic_verdict", verdict.state, task.robot_id],
                )
            )
        except Exception as exc:  # noqa: BLE001
            tracer.event(
                "memory.write_failed",
                trace_id=trace_id,
                robot_id=task.robot_id,
                memory_type="episodic",
                error=str(exc),
            )

    # Heavy blobs never enter memory — only structured facts the Brain can plan on.
    _OBSERVATION_BLOB_KEYS = ("image_b64", "depth_b64")

    async def _write_observations(
        self, task: Task, turn: int, turn_results: list[dict[str, Any]], trace_id: str
    ) -> None:
        """Write each turn's structured observations back to memory (memory-first).

        Closes the observation→Brain edge: next turn's :meth:`_query_memory`
        surfaces these so the Brain knows what it already saw / did, instead of
        re-observing forever. Detections go to object memory; state / frame /
        verb results go to episodic. Image and depth blobs are stripped — only
        structured facts are stored. Tagged with ``task.task_id`` so the embedded
        backend's any-match tag filter keeps recall scoped to this task.
        """
        for r in turn_results:
            if not r.get("success"):
                continue
            name = r.get("tool_name", "")
            output = r.get("output") or {}
            facts = {k: v for k, v in output.items() if k not in self._OBSERVATION_BLOB_KEYS}

            if name == "perception.detect":
                for obj in facts.get("objects", []):
                    await self._safe_write(
                        MemoryEntry(
                            memory_type="object",
                            robot_id=task.robot_id,
                            content={"turn": turn, **obj},
                            tags=["observation", task.task_id],
                        ),
                        trace_id,
                    )
            elif name in ("robot.capture_frame", "robot.get_state") or name.startswith(
                "robot_sdk."
            ):
                await self._safe_write(
                    MemoryEntry(
                        memory_type="episodic",
                        robot_id=task.robot_id,
                        content={"turn": turn, "tool": name, "observation": facts},
                        tags=["observation", task.task_id],
                    ),
                    trace_id,
                )

    async def _safe_write(self, entry: MemoryEntry, trace_id: str) -> None:
        """Best-effort memory write; a memory outage must never block the loop."""
        try:
            await self._ctx.memory.write(entry)
        except Exception as exc:  # noqa: BLE001
            tracer.event(
                "memory.write_failed",
                trace_id=trace_id,
                robot_id=entry.robot_id,
                memory_type=entry.memory_type,
                error=str(exc),
            )

    async def _query_memory(self, task: Task) -> MemoryView:
        """Query all memory types for context relevant to the task (best-effort)."""
        view = MemoryView()
        try:
            # This task's own observations, most-recent first. Scoped by task_id so
            # the backend's any-match tag filter can't pull other tasks' entries or
            # critic verdicts (tagged without task_id).
            episodic = await self._ctx.memory.query(
                MemoryQuery(
                    memory_type="episodic",
                    robot_id=task.robot_id,
                    text=task.description,
                    tags=[task.task_id],
                    top_k=10,
                )
            )
            view.episodic_hits = [h.model_dump() for h in episodic]
        except Exception as exc:  # noqa: BLE001
            tracer.event(
                "memory.query_failed",
                robot_id=task.robot_id,
                memory_type="episodic",
                error=str(exc),
            )
        try:
            objects = await self._ctx.memory.query(
                MemoryQuery(
                    memory_type="object",
                    robot_id=task.robot_id,
                    text=task.description,
                    tags=[task.task_id],
                    top_k=10,
                )
            )
            view.object_hits = [h.model_dump() for h in objects]
        except Exception as exc:  # noqa: BLE001
            tracer.event(
                "memory.query_failed",
                robot_id=task.robot_id,
                memory_type="object",
                error=str(exc),
            )
        try:
            semantic = await self._ctx.memory.query(
                MemoryQuery(memory_type="semantic", robot_id=task.robot_id, text=task.description)
            )
            view.semantic_hits = [h.model_dump() for h in semantic]
        except Exception as exc:  # noqa: BLE001
            tracer.event(
                "memory.query_failed",
                robot_id=task.robot_id,
                memory_type="semantic",
                error=str(exc),
            )
        return view

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def _execute_tool_calls(
        self,
        requests: list[ToolCallRequest],
        task: Task,
        trace_id: str,
    ) -> list[dict[str, Any]]:
        async def _run_one(req: ToolCallRequest) -> dict[str, Any]:
            tool_ctx = ToolContext(
                trace_id=trace_id,
                robot_id=task.robot_id,
                subtask_id=task.subtask_id,
                timeout_s=self._ctx.config.tool.default_timeout_s,
                artifact_store=self._ctx.artifact_store,
            )
            result = await self._invoke_tool(req, tool_ctx)
            return result.model_dump()

        gathered = await asyncio.gather(*[_run_one(r) for r in requests])
        return list(gathered)

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

        # Every tool that actuates the robot is gated behind SafetyEnvelope.
        # SafetyEnvelopeViolation is NOT caught — it propagates to trigger e-stop.
        registry = self._ctx.tool_registry
        if registry.requires_safety_check(req.tool_name):
            cmd = registry.build_safety_command(req.tool_name, req.args, ctx)
            if cmd is not None:
                await self._ctx.safety_envelope.check(
                    cmd,
                    trace_id=ctx.trace_id,
                    subtask_id=ctx.subtask_id,
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
