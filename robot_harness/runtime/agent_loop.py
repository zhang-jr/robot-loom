"""AgentLoop — the Brain ⇄ Tool main loop.

Orchestrates: Brain.decide() → tool dispatch (with safety check) → result
→ Brain (next turn), repeating until completion or give_up.

The loop owns the conversation (ADR-025): a growing native tool-use message list
``system → user → assistant(tool_calls) → tool(result) → …``. Recent observations
and the Brain's own prior turns live in that list, so there is no per-turn Memory
round-trip; long-term recall is the on-demand ``memory.query`` tool (ADR-024), and
replanning is just a critic-feedback message appended to the same conversation.

Wired in:
- Optional Critic integration with ReplanPolicy (ADR-010)
- EpisodicMemory write-back on task completion + per-turn observation persistence
- Cognitive scaffold (plan / reflection) injected into the opening turn (ADR-018)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from robot_harness.brain.base import Brain, BrainDecision, Message, Task, ToolCallRequest
from robot_harness.brain.prompt_assembly import workspace_prompt_overlay
from robot_harness.critic.base import Critic, CriticVerdict
from robot_harness.critic.heuristic_fallback import HeuristicCritic
from robot_harness.critic.replan_policy import ReplanPolicy, SensorHeuristic
from robot_harness.errors import (
    BrainOutputInvalidError,
    ChannelError,
    CriticDisagreementError,
    CriticServiceDown,
    EmbodimentError,
    ReplanLoopExceededError,
    SafetyError,
    SkillError,
    ToolCancelledError,
    ToolError,
    ToolNotFoundError,
    ToolSchemaViolationError,
)
from robot_harness.memory.base import MemoryEntry
from robot_harness.observability.tracer import tracer
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.runtime.skill_tools import SafetyGatedToolRegistry
from robot_harness.skill.base import Skill, Subtask
from robot_harness.tools.base import BrainProfile, ToolContext, ToolRegistry, ToolResult
from robot_harness.tools.outbound import NullOutbound, OutboundHandle

_SYSTEM_PROMPT = """\
You are a robot task planner. You have access to tools that control robot hardware.
Plan step by step. Call tools one or a few at a time. Never skip the safety check tool.
When the task is complete, respond with a final message explaining the outcome.
Do NOT call tools after the task is done — just respond naturally.

Recent observations and your prior tool results are in the conversation above; use
them directly. Call the memory.query tool only to recall facts NOT in the
conversation (e.g. where an object was seen in an earlier task, why a past attempt
failed) — don't re-query for what you can already see.
"""

_NUDGE_PROMPT = """\
You ended your turn without calling any tools, and no tools have been run for this \
task yet. If completing the task requires action or verification, proceed now using \
the available tools — do not just describe a plan. If the task genuinely requires no \
tool use, restate your final answer."""


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
        max_replan:      Max critic-driven replans per task before aborting.
        max_ask_user:    Max ask_user round-trips per task before giving up.
        ask_timeout_s:   How long to wait for a user reply to an ask_user.
    """

    def __init__(
        self,
        brain: Brain,
        ctx: HarnessContext,
        max_turns: int = 20,
        critic: Critic | None = None,
        critic_interval: int = 3,
        max_replan: int = 5,
        max_ask_user: int = 5,
        ask_timeout_s: float = 300.0,
    ) -> None:
        self._brain = brain
        self._ctx = ctx
        self._max_turns = max_turns
        self._critic = critic
        self._critic_interval = critic_interval
        self._max_replan = max_replan
        self._max_ask_user = max_ask_user
        self._ask_timeout_s = ask_timeout_s

    async def run(self, task: Task, *, outbound: OutboundHandle | None = None) -> AgentResult:
        """Execute *task* until completion, give_up, or max_turns exceeded.

        ``outbound`` is the session-bound path back to the user (ADR-023): the
        Talk tools deliver through it and ask_user round-trips on it. When None
        (programmatic callers), it degrades to :class:`NullOutbound`.
        """
        outbound = outbound or NullOutbound()
        trace_id = str(uuid.uuid4())
        tracer.event(
            "agent_loop.start",
            trace_id=trace_id,
            task_id=task.task_id,
            robot_id=task.robot_id,
            description=task.description,
        )

        # The Brain plans over atomic tools AND versioned skills (skill.<name>);
        # the loop routes a skill.<name> call back to Skill.execute (ADR-025).
        brain_profile = BrainProfile(name="openai")
        # Live-capability gate (ADR-019): tools no robot can currently run — and
        # skills requiring them — never enter the Brain's planning vocabulary.
        unavailable = await self._ctx.unavailable_tool_names(trace_id)
        tool_specs = self._ctx.tool_registry.export_for_brain(
            brain_profile, exclude_names=unavailable
        )
        tool_specs += self._ctx.skill_registry.export_for_brain(
            brain_profile, unavailable_tools=unavailable
        )
        all_tool_results: list[dict[str, Any]] = []
        # The loop owns the conversation (ADR-025); it grows across turns.
        messages = self._initial_messages(task, trace_id)
        replan_count = 0
        ask_count = 0
        nudged = False
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

            decision = await self._brain.decide(
                messages, tool_specs, trace_id=trace_id, robot_id=task.robot_id
            )
            decision.trace_id = trace_id
            if decision.decision_type == "tool_call":
                self._assign_call_ids(decision, turn)
            # Append the Brain's own turn to the conversation so the next turn (and
            # the tool messages below) sees it — the assistant message must precede
            # its tool results in the native protocol.
            messages.append(self._assistant_message(decision))

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

            if decision.decision_type == "respond":
                # A text-only turn is the terminal signal — but a respond before
                # ANY tool has run is suspicious (the model may be narrating a
                # plan instead of executing it). Nudge once; a second respond,
                # or one after tools have run, is accepted as final.
                if not all_tool_results and not nudged:
                    nudged = True
                    tracer.event(
                        "agent_loop.nudge",
                        trace_id=trace_id,
                        robot_id=task.robot_id,
                        message=decision.message[:200],
                    )
                    messages.append({"role": "user", "content": _NUDGE_PROMPT})
                    continue
                tracer.event(
                    "agent_loop.respond", trace_id=trace_id, message=decision.message[:200]
                )
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
                turn_results = await self._execute_tool_calls(
                    decision.tool_calls, task, trace_id, outbound
                )
                all_tool_results.extend(turn_results)
                # Each result becomes a native role:tool message keyed by tool_call_id;
                # blobs are stripped so the raw image never enters the LLM (the small
                # artifact ref does flow through).
                for req, res in zip(decision.tool_calls, turn_results, strict=True):
                    messages.append(self._tool_message(req.call_id, res))
                await self._write_observations(task, turn, turn_results, trace_id)

                any_error = any(not r.get("success") for r in turn_results)
                if any_error:
                    sensor.error_count += 1
                else:
                    sensor.error_count = max(0, sensor.error_count - 1)

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
                    verdict, active_critic, heuristic_critic = await self._run_critic(
                        active_critic, heuristic_critic, task, trace_id
                    )

                    if verdict is not None:
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
                            # Replan = another turn (ADR-025): append the critic
                            # feedback as a user message; the next decide() replans
                            # with the full conversation in view.
                            messages.append(self._critic_feedback_message(verdict))
                            tracer.event(
                                "agent_loop.replan",
                                trace_id=trace_id,
                                replan_count=replan_count,
                                critic_state=verdict.state,
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
                ask_count += 1
                incomplete = self._ask_user_incomplete(
                    task, decision.message, turn, all_tool_results, trace_id
                )
                if ask_count > self._max_ask_user:
                    result = AgentResult(
                        task_id=task.task_id,
                        robot_id=task.robot_id,
                        outcome="give_up",
                        message="Exceeded ask_user limit without resolution",
                        turns=turn,
                        tool_results=all_tool_results,
                        trace_id=trace_id,
                    )
                    await self._write_episode(task, result, trace_id)
                    return result
                try:
                    reply = await outbound.ask(decision.message, timeout_s=self._ask_timeout_s)
                except (TimeoutError, ChannelError) as exc:
                    # No reply (timeout, no channel wired, or delivery failure):
                    # exit cleanly as incomplete carrying the unanswered question.
                    tracer.event(
                        "agent_loop.ask_user_unanswered",
                        trace_id=trace_id,
                        reason=type(exc).__name__,
                    )
                    await self._write_episode(task, incomplete, trace_id)
                    return incomplete
                # Resume the same conversation (ADR-025): the reply is a user turn.
                messages.append({"role": "user", "content": reply})
                continue

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
    # Conversation helpers (native tool-use message protocol, ADR-025)
    # ------------------------------------------------------------------

    def _initial_messages(self, task: Task, trace_id: str = "") -> list[Message]:
        """Build the opening ``[system, user]`` conversation for a task.

        The system turn is the static harness prompt plus the workspace standing
        context (MISSION.md / ROBOT.md, ADR-035) — operator-authored prose the
        Brain needs for grounding (site semantics, standing orders) that the
        harness cannot discover and that is not runtime state (which is Memory's).

        Any cognitive scaffold (a plan/reflection set before the loop) is injected
        into the opening user turn. During the task, plan/reflection updates are
        visible via their tool results already in the conversation; re-injection is
        only for the opening turn (and, later, post-compression recovery, ADR-018).
        """
        system = _SYSTEM_PROMPT
        overlay = workspace_prompt_overlay(trace_id=trace_id)
        if overlay:
            system = f"{_SYSTEM_PROMPT}\n\n{overlay}"
        sections = [f"Task: {task.description}"]
        if task.constraints:
            sections.append(f"Constraints: {'; '.join(task.constraints)}")
        scaffold = self._ctx.format_scaffold_for_injection(task.robot_id)
        if scaffold:
            sections.append(scaffold)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(sections)},
        ]

    @staticmethod
    def _assign_call_ids(decision: BrainDecision, turn: int) -> None:
        """Ensure every tool call has an id, so the assistant message and its tool
        results reference the same ``tool_call_id`` (some backends — local vLLM /
        Ollama — emit tool calls without ids; mock Brains may too).

        The backfilled id is written to BOTH copies of the tool calls: the parsed
        ``decision.tool_calls`` (which keys the role:tool results) and the raw
        ``decision.assistant_message`` (which goes back to the LLM). If only the
        former were patched, the conversation would carry an assistant turn whose
        tool_call ids don't match the tool results that follow — strict providers
        reject that history outright.
        """
        for i, tc in enumerate(decision.tool_calls):
            if not tc.call_id:
                tc.call_id = f"call_{turn}_{i}"
        am = decision.assistant_message
        if am is not None:
            wire_calls = am.get("tool_calls") or []
            if len(wire_calls) != len(decision.tool_calls):
                raise BrainOutputInvalidError(
                    f"assistant_message carries {len(wire_calls)} tool_calls but the "
                    f"decision parsed {len(decision.tool_calls)} — Brain protocol violation",
                    trace_id=decision.trace_id,
                )
            for tc, wire in zip(decision.tool_calls, wire_calls, strict=True):
                if not wire.get("id"):
                    wire["id"] = tc.call_id

    def _assistant_message(self, decision: BrainDecision) -> Message:
        """The assistant turn to append to history — the Brain's verbatim message
        when it provided one, else synthesized from the decision."""
        if decision.assistant_message is not None:
            return decision.assistant_message
        if decision.decision_type == "tool_call":
            return {
                "role": "assistant",
                "content": decision.message or None,
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {"name": tc.tool_name, "arguments": json.dumps(tc.args)},
                    }
                    for tc in decision.tool_calls
                ],
            }
        return {"role": "assistant", "content": decision.message or ""}

    def _tool_message(self, call_id: str, result: dict[str, Any]) -> Message:
        """Render one tool result as a native ``role:tool`` message. Blobs are
        stripped (the raw image never enters the LLM); the small artifact ref does.

        Failures keep their structured output too: a failed/partial verb's
        evidence, aborted_by, and state snapshot are exactly what the Brain
        needs to replan on — dropping them would leave it staring at a bare
        ``{"error": ...}`` (ISS-032)."""
        output = result.get("output") or {}
        facts = {k: v for k, v in output.items() if k not in self._OBSERVATION_BLOB_KEYS}
        if result.get("success"):
            content = json.dumps(facts, default=str)
        else:
            payload: dict[str, Any] = {
                "error": result.get("error"),
                "error_type": result.get("error_type"),
            }
            if facts:
                payload["output"] = facts
            content = json.dumps(payload, default=str)
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    @staticmethod
    def _critic_feedback_message(verdict: CriticVerdict) -> Message:
        """Fold a replan into the conversation as a critic-feedback user turn."""
        return {
            "role": "user",
            "content": (
                f"Critic feedback: progress={verdict.state} "
                f"(confidence={verdict.confidence:.2f}). Evidence: {verdict.evidence or 'none'}. "
                "Reconsider whether to continue, try a different approach, or stop."
            ),
        }

    # ------------------------------------------------------------------
    # Critic helpers
    # ------------------------------------------------------------------

    async def _run_critic(
        self,
        active_critic: Critic,
        heuristic_critic: HeuristicCritic | None,
        task: Task,
        trace_id: str,
    ) -> tuple[CriticVerdict | None, Critic, HeuristicCritic | None]:
        """Fetch a frame and call the critic, degrading instead of crashing.

        Supervision is best-effort by design (ADR-010): a camera fetch failure
        (robot offline, camera 404, malformed frame) skips this round's critic —
        traced as unsupervised — and a critic-service failure falls back to the
        timeout heuristic. Neither may abort the task.
        """
        try:
            frame = await self._ctx.get_camera_frame(task.robot_id)
        except Exception as exc:  # noqa: BLE001
            tracer.event(
                "agent_loop.critic_skipped",
                trace_id=trace_id,
                robot_id=task.robot_id,
                error=str(exc),
                warning="camera frame fetch failed — turn runs unsupervised.",
            )
            return None, active_critic, heuristic_critic
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
            except Exception as fallback_exc:  # noqa: BLE001
                tracer.event(
                    "agent_loop.critic_skipped",
                    trace_id=trace_id,
                    robot_id=task.robot_id,
                    error=str(fallback_exc),
                    warning="Heuristic critic fallback also failed — turn runs unsupervised.",
                )
                return None, active_critic, heuristic_critic
        except Exception as exc:  # noqa: BLE001
            tracer.event(
                "agent_loop.critic_skipped",
                trace_id=trace_id,
                robot_id=task.robot_id,
                error=str(exc),
                warning="Critic judge failed unexpectedly — turn runs unsupervised.",
            )
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
        """Persist each turn's structured observations to long-term memory.

        This is the durable record, recalled later on demand via the
        ``memory.query`` tool (ADR-024) — NOT the per-turn observation→Brain edge,
        which rides the native conversation the loop owns (ADR-025). Detections
        go to object memory; state / frame / verb results go to episodic. Image and
        depth blobs are stripped — only structured facts are stored. Tagged with
        ``task.task_id`` so the embedded backend's any-match tag filter keeps recall
        scoped to this task.

        (Narrowing what gets written each tick — to keep episodic semantically pure —
        is the next step, ADR-024 priority 3.)
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

    def _ask_user_incomplete(
        self,
        task: Task,
        question: str,
        turn: int,
        tool_results: list[dict[str, Any]],
        trace_id: str,
    ) -> AgentResult:
        """Build the ``incomplete`` result returned when an ask_user goes unanswered."""
        return AgentResult(
            task_id=task.task_id,
            robot_id=task.robot_id,
            outcome="incomplete",
            message=f"Awaiting user input: {question}",
            turns=turn,
            tool_results=tool_results,
            trace_id=trace_id,
        )

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def _execute_tool_calls(
        self,
        requests: list[ToolCallRequest],
        task: Task,
        trace_id: str,
        outbound: OutboundHandle,
    ) -> list[dict[str, Any]]:
        ctxs = [
            ToolContext(
                trace_id=trace_id,
                robot_id=task.robot_id,
                subtask_id=task.subtask_id,
                timeout_s=self._ctx.config.tool.default_timeout_s,
                artifact_store=self._ctx.artifact_store,
                outbound=outbound,
            )
            for _ in requests
        ]

        async def _run_one(req: ToolCallRequest, tool_ctx: ToolContext) -> dict[str, Any]:
            result = await self._invoke_tool(req, tool_ctx)
            return result.model_dump()

        calls = [
            asyncio.ensure_future(_run_one(req, c)) for req, c in zip(requests, ctxs, strict=True)
        ]
        try:
            gathered = await asyncio.gather(*calls)
        except BaseException:
            # Must-stop semantics: _invoke_tool normalizes every fault to a failed
            # result and re-raises ONLY SafetyError (enforced by its catch-all,
            # ISS-034), so what arrives here is a safety violation or a task-level
            # cancellation. Sibling in-flight calls of the same turn must not
            # keep actuating in the background while it propagates.
            await self._cancel_sibling_calls(requests, ctxs, calls, trace_id)
            raise
        return list(gathered)

    async def _cancel_sibling_calls(
        self,
        requests: list[ToolCallRequest],
        ctxs: list[ToolContext],
        calls: list[asyncio.Task[dict[str, Any]]],
        trace_id: str,
    ) -> None:
        """Cancel, reap, and actively abort the turn's in-flight sibling calls.

        Runs when one parallel call raised a safety violation. Cancellation is
        three-layered: the ToolContext cancel event (tools honoring the cancel
        protocol see it at checkpoints), asyncio task cancellation (interrupts
        the in-flight await), and — for cancellable hardware tools — the tool's
        own ``cancel()`` (a verb POSTs ``/abort``; the robot falls back to a
        safe pose). Abort failures are traced, never raised: they must not mask
        the propagating violation.
        """
        pending: list[tuple[ToolCallRequest, ToolContext]] = []
        for req, tool_ctx, call in zip(requests, ctxs, calls, strict=True):
            if call.done():
                continue
            tool_ctx.cancel()
            call.cancel()
            pending.append((req, tool_ctx))
        # Reap so no sibling exception goes unretrieved.
        await asyncio.gather(*calls, return_exceptions=True)
        if not pending:
            return
        tracer.event(
            "agent_loop.siblings_cancelled",
            trace_id=trace_id,
            tools=[req.tool_name for req, _ in pending],
            reason="safety violation in a parallel call of the same turn",
        )
        for req, tool_ctx in pending:
            # Skills have no cancel protocol; task cancellation already reached them.
            if self._ctx.skill_registry.resolve_brain_call(req.tool_name) is not None:
                continue
            try:
                tool = self._ctx.tool_registry.get(req.tool_name)
            except ToolNotFoundError:
                continue
            if not tool.is_cancellable:
                continue
            try:
                await tool.cancel(tool_ctx)
            except Exception as exc:  # noqa: BLE001 — best-effort abort, traced above
                tracer.event(
                    "agent_loop.sibling_abort_failed",
                    trace_id=trace_id,
                    tool_name=req.tool_name,
                    error=str(exc),
                )

    async def _invoke_tool(self, req: ToolCallRequest, ctx: ToolContext) -> ToolResult:
        # A skill.<name> call routes to Skill.execute; everything else is a tool.
        skill = self._ctx.skill_registry.resolve_brain_call(req.tool_name)
        if skill is not None:
            return await self._invoke_skill(skill, req, ctx)

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

        # Schema gate before anything runs: Brain-issued args missing required
        # fields are a failed call the Brain corrects — not a KeyError deep in
        # a tool, and not a "skipped" safety audit for a call that never
        # should have reached the gate (ISS-034).
        try:
            self._ctx.tool_registry.validate_args(req.tool_name, req.args)
        except ToolSchemaViolationError as exc:
            return ToolResult(
                tool_name=req.tool_name,
                trace_id=ctx.trace_id,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

        # Every tool that actuates the robot is gated behind SafetyEnvelope.
        # SafetyEnvelopeViolation is NOT caught — the refused command was never
        # dispatched (pre-dispatch semantics); the violation propagates, aborts
        # the task, and cancels the turn's sibling calls.
        registry = self._ctx.tool_registry
        if registry.requires_safety_check(req.tool_name):
            cmd = registry.build_safety_command(req.tool_name, req.args, ctx)
            if cmd is not None:
                await self._ctx.safety_envelope.check(
                    cmd,
                    trace_id=ctx.trace_id,
                    subtask_id=ctx.subtask_id,
                )
            else:
                # Hardware-bound but no harness-checkable target (e.g. a phrase
                # hint): honest "skipped" audit — never indistinguishable from
                # "checked and passed". The on-robot reflex is authoritative.
                self._ctx.safety_envelope.note_skipped(
                    tool_name=req.tool_name,
                    robot_id=ctx.robot_id,
                    trace_id=ctx.trace_id,
                    subtask_id=ctx.subtask_id,
                    reason="no harness-checkable target — on-robot reflex is authoritative",
                )

        try:
            return await asyncio.wait_for(
                tool.invoke(req.args, ctx), timeout=self._tool_deadline_s(req.args)
            )
        except TimeoutError:
            return await self._backstop_timeout(req, tool, ctx)
        except ToolCancelledError:
            return ToolResult(
                tool_name=req.tool_name,
                trace_id=ctx.trace_id,
                success=False,
                error="cancelled",
                error_type="ToolCancelledError",
            )
        except (ToolError, EmbodimentError) as exc:
            # A backend/robot fault (server unreachable, robot offline, dispatch
            # timeout) is a failed tool call the Brain can replan around — NOT a
            # crash. SafetyEnvelopeViolation is a SafetyError, not caught here, so
            # it still propagates and aborts the task.
            return ToolResult(
                tool_name=req.tool_name,
                trace_id=ctx.trace_id,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )
        except SafetyError:
            # The must-stop path: propagates, aborts the task, cancels siblings.
            raise
        except Exception as exc:  # noqa: BLE001
            # An untyped exception (tool bug, unparsed external payload) must
            # not ride the safety-violation path and crash the task without an
            # episode: normalize it to a failed result the Brain can react to,
            # with the true type preserved and the fault traced (ISS-034).
            tracer.event(
                "agent_loop.tool_unexpected_error",
                trace_id=ctx.trace_id,
                robot_id=ctx.robot_id,
                tool_name=req.tool_name,
                error=str(exc),
                error_type=type(exc).__name__,
                warning="untyped exception escaped the tool — normalized to a failed result",
            )
            return ToolResult(
                tool_name=req.tool_name,
                trace_id=ctx.trace_id,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    def _tool_deadline_s(self, args: dict[str, Any]) -> float:
        """Loop-level backstop deadline for one atomic tool call (ISS-037).

        ``ctx.timeout_s`` is advisory — only tools that self-enforce honor it —
        so a hung tool would otherwise block the turn forever. A verb's own
        execution budget (``constraints.timeout_s``, enforced on-robot and by
        the transport read timeout) may legitimately exceed the default, so the
        backstop is the larger of the two plus margin: it fires only when every
        layer below failed to.
        """
        base = self._ctx.config.tool.default_timeout_s
        constraints = (args or {}).get("constraints") or {}
        try:
            budget = float(constraints.get("timeout_s") or 0.0)
        except (TypeError, ValueError):
            budget = 0.0
        return max(base, budget + 10.0)

    async def _backstop_timeout(
        self, req: ToolCallRequest, tool: Any, ctx: ToolContext
    ) -> ToolResult:
        """A tool blew through every deadline below the loop: cancel it (for
        hardware tools that means an on-robot abort) and fail the call."""
        ctx.cancel()
        if getattr(tool, "is_cancellable", False):
            try:
                await tool.cancel(ctx)
            except Exception as exc:  # noqa: BLE001 — best-effort abort
                tracer.event(
                    "agent_loop.timeout_abort_failed",
                    trace_id=ctx.trace_id,
                    tool_name=req.tool_name,
                    error=str(exc),
                )
        deadline = self._tool_deadline_s(req.args)
        tracer.event(
            "agent_loop.tool_backstop_timeout",
            trace_id=ctx.trace_id,
            robot_id=ctx.robot_id,
            tool_name=req.tool_name,
            deadline_s=deadline,
            warning="tool exceeded the loop backstop deadline — cancelled",
        )
        return ToolResult(
            tool_name=req.tool_name,
            trace_id=ctx.trace_id,
            success=False,
            error=f"tool '{req.tool_name}' exceeded the loop backstop deadline ({deadline:.1f}s)",
            error_type="ToolTimeoutError",
        )

    async def _invoke_skill(
        self, skill: Skill, req: ToolCallRequest, ctx: ToolContext
    ) -> ToolResult:
        """Dispatch a Brain ``skill.<name>`` call to Skill.execute.

        The skill runs its internal tool calls through a
        :class:`SafetyGatedToolRegistry` bound to this call's ToolContext, so each
        hardware-bound call still passes SafetyEnvelope.check and every internal
        call stays on the task trace — a SafetyEnvelopeViolation propagates
        uncaught and aborts the task (the refused command was never dispatched).
        A ``SkillError`` becomes a failed ToolResult the Brain can react to; the
        SkillResult is surfaced as the tool output.
        """
        t0 = time.monotonic()
        args = req.args or {}
        manifest = skill.manifest
        subtask = Subtask(
            subtask_id=ctx.subtask_id or req.call_id or manifest.name,
            description=args.get("description") or manifest.description or manifest.name,
            robot_id=args.get("robot_id", ctx.robot_id),
            parameters=args.get("parameters") or {},
        )
        # A drop-in for the skill: `.get()` returns safety-gated tools rebound onto
        # the task context (trace continuity + artifact_store/outbound inheritance —
        # the Skill protocol has no trace channel, so skills mint fresh contexts),
        # everything else delegates. Not a ToolRegistry subclass (it wraps one), so
        # cast to the param type the Skill Protocol declares.
        gated_tools = cast(
            ToolRegistry,
            SafetyGatedToolRegistry(
                self._ctx.tool_registry, self._ctx.safety_envelope, parent_ctx=ctx
            ),
        )
        tracer.event(
            "agent_loop.skill_invoke",
            trace_id=ctx.trace_id,
            robot_id=subtask.robot_id,
            skill=manifest.name,
            version=manifest.version,
        )
        try:
            result = await skill.execute(subtask, gated_tools, self._ctx)
        except (SkillError, ToolError, EmbodimentError) as exc:
            # Skill failure or a backend/robot fault inside one of its tool calls:
            # surface as a failed ToolResult (true error_type preserved) so the
            # Brain can react. SafetyEnvelopeViolation is NOT in this tuple — it
            # propagates uncaught and aborts the task.
            return ToolResult(
                tool_name=req.tool_name,
                trace_id=ctx.trace_id,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except SafetyError:
            # The must-stop path: propagates, aborts the task, cancels siblings.
            raise
        except Exception as exc:  # noqa: BLE001
            # A bug in a (user-provided) skill must not crash the task through
            # the safety-violation path: normalize to a failed result, true
            # type preserved, fault traced (ISS-034).
            tracer.event(
                "agent_loop.skill_unexpected_error",
                trace_id=ctx.trace_id,
                robot_id=subtask.robot_id,
                skill=manifest.name,
                error=str(exc),
                error_type=type(exc).__name__,
                warning="untyped exception escaped the skill — normalized to a failed result",
            )
            return ToolResult(
                tool_name=req.tool_name,
                trace_id=ctx.trace_id,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        tracer.event(
            "agent_loop.skill_result",
            trace_id=ctx.trace_id,
            skill=manifest.name,
            outcome=result.outcome,
            success=result.success,
        )
        return ToolResult(
            tool_name=req.tool_name,
            trace_id=ctx.trace_id,
            success=result.success,
            output={
                "skill": result.skill_name,
                "skill_version": result.skill_version,
                "outcome": result.outcome,
                "message": result.message,
                "artifacts": result.artifacts,
            },
            error=None if result.success else (result.message or "skill failed"),
            error_type=None if result.success else "SkillError",
            latency_ms=(time.monotonic() - t0) * 1000,
        )
