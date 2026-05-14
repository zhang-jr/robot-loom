"""Long-horizon task example with Cognitive Scaffold (plan tool).

Demonstrates how PlanTool reduces token drift over ≥5 subtasks by maintaining
a structured plan that survives context compression and is re-injected each turn.

Run:  python examples/long_horizon_plan.py
"""

from __future__ import annotations

import asyncio

from robot_harness.brain.base import (
    BrainDecision,
    CriticSignal,
    ExecutionHistory,
    MemoryView,
    Task,
    ToolCallRequest,
)
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.base import BrainProfile

# ---------------------------------------------------------------------------
# Mock Brain that uses the plan tool to track 6 subtasks
# ---------------------------------------------------------------------------

_SUBTASKS = [
    "perceive_scene",
    "identify_target",
    "compute_grasp",
    "move_to_pregrasp",
    "execute_grasp",
    "place_object",
]


class LongHorizonMockBrain:
    """Simulates a Brain that writes its plan on turn 1 and executes steps."""

    def __init__(self) -> None:
        self._turn = 0

    async def decide(self, task: Task, memory_view: MemoryView, tools: list) -> BrainDecision:
        self._turn += 1

        if self._turn == 1:
            # Write the full plan on the first turn
            steps = [
                {"id": str(i + 1), "description": name, "status": "pending"}
                for i, name in enumerate(_SUBTASKS)
            ]
            return BrainDecision(
                decision_type="tool_call",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="plan",
                        args={"goal": task.description, "steps": steps},
                        call_id="plan_init",
                    )
                ],
            )

        if self._turn <= len(_SUBTASKS) + 1:
            step_idx = self._turn - 2
            if step_idx < len(_SUBTASKS):
                step_id = str(step_idx + 1)
                step_name = _SUBTASKS[step_idx]
                # Mark current step in_progress then completed
                return BrainDecision(
                    decision_type="tool_call",
                    tool_calls=[
                        ToolCallRequest(
                            tool_name="plan",
                            args={
                                "steps": [
                                    {"id": step_id, "description": step_name, "status": "completed"}
                                ],
                                "merge": True,
                            },
                            call_id=f"plan_update_{step_id}",
                        )
                    ],
                )

        # All steps done — signal completion
        return BrainDecision(
            decision_type="plan",
            plan="All 6 subtasks completed successfully.",
            message="Long-horizon task complete.",
        )

    async def replan(self, history: ExecutionHistory, critic_signal: CriticSignal) -> BrainDecision:
        return BrainDecision(decision_type="give_up", message="replan not needed in example")

    @property
    def supports_streaming(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def main() -> None:
    ctx = HarnessContext.build()
    brain = LongHorizonMockBrain()
    brain_profile = BrainProfile(name="openai", supports_native_reflection=True)

    # Inject cognitive scaffold (registers PlanTool in ToolRegistry)
    ctx.inject_cognitive_scaffold(robot_id="arm0", brain_profile=brain_profile)

    loop = AgentLoop(brain=brain, ctx=ctx, max_turns=20)
    task = Task(
        task_id="lh_demo",
        description="Pick red block from table and place in tray",
        robot_id="arm0",
    )

    result = await loop.run(task)

    print("\n=== Result ===")
    print(f"Outcome : {result.outcome}")
    print(f"Turns   : {result.turns}")
    print(f"Message : {result.message}")

    # Show final plan state from tool results
    for r in result.tool_results:
        output = r.get("output") or {}
        plan = output.get("plan")
        if plan and plan.get("steps"):
            print(f"\nFinal plan ({len(plan['steps'])} steps):")
            for s in plan["steps"]:
                marker = {"completed": "[x]", "pending": "[ ]", "failed": "[!]"}.get(
                    s["status"], "[?]"
                )
                print(f"  {marker} {s['id']}. {s['description']}")


if __name__ == "__main__":
    asyncio.run(main())
