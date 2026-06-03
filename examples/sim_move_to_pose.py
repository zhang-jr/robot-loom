"""Cartesian move_to_pose verb against a live sim agent_server (ADR-019).

The Brain issues a high-level ``robot_sdk.move_to_pose`` with a cartesian target;
SafetyEnvelope validates the pose; the verb is POSTed to the sim's
``/verb/move_to_pose`` endpoint, where the sim runs IK and drives the arm — the
harness only sees the CompletionVerdict (it never micromanages the IK / control
loop). This is the Mid-loop reactive-skill layer: the loop lives on the sim
agent_server, not in the harness.

Prerequisite — start the sim (needs mujoco for IK):

    pip install -r ../sim_shim/mujoco_shim/requirements.txt
    python ../sim_shim/mujoco_shim/server.py --port 8810

Then:

    python examples/sim_move_to_pose.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx

from robot_harness.brain.base import (
    BrainDecision,
    CriticSignal,
    ExecutionHistory,
    MemoryView,
    Task,
    ToolCallRequest,
)
from robot_harness.config.schema import EmbodimentBackendConfig, HarnessConfig
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.robot_sdk.verbs import build_robot_sdk_verb_tools

_ROBOT_ID = "robot-0"
_SIM_URL = os.environ.get("ROBOT_LOOM_SIM_URL", "http://localhost:8810")
# A reachable pre-grasp pose above the cube (x, y, z, rx, ry, rz); the 3-DOF arm
# controls position only, so orientation is ignored by the sim's IK.
_TARGET = [0.35, 0.0, 0.25, 0.0, 0.0, 0.0]


class ScriptedBrain:
    """Issue one move_to_pose verb, then declare done."""

    def __init__(self) -> None:
        self._decisions = [
            BrainDecision(
                decision_type="tool_call",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="robot_sdk.move_to_pose",
                        args={"robot_id": _ROBOT_ID, "target_pose": _TARGET},
                    )
                ],
            ),
            BrainDecision(decision_type="plan", message="reached the pre-grasp pose"),
        ]
        self._i = 0

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(self, task: Task, memory_view: MemoryView, tools: list) -> BrainDecision:  # type: ignore[type-arg]
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d

    async def replan(self, history: ExecutionHistory, critic_signal: CriticSignal) -> BrainDecision:
        return BrainDecision(decision_type="plan", message="recovered")


async def main() -> None:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            (await client.get(f"{_SIM_URL}/state")).raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Sim not reachable at {_SIM_URL}: {exc}")
        print("Start it:  python ../sim_shim/mujoco_shim/server.py --port 8810")
        return

    config = HarnessConfig(
        robot_ids=[_ROBOT_ID],
        embodiments={
            _ROBOT_ID: EmbodimentBackendConfig(
                backend="sim", sim_engine="mujoco", server_url=_SIM_URL
            )
        },
    )
    ctx = HarnessContext.build(config)
    # Verb tools wired to the sim adapter — move_to_pose hits /verb/move_to_pose.
    for tool in build_robot_sdk_verb_tools(ctx.embodiment_adapters):
        ctx.tool_registry.register(tool)

    loop = AgentLoop(ScriptedBrain(), ctx, max_turns=6)
    task = Task(
        task_id=str(uuid.uuid4()),
        description="Move the end-effector to a pre-grasp pose above the cube.",
        robot_id=_ROBOT_ID,
    )

    print(f"Sim: {_SIM_URL}")
    print(f"Target ee pose: {_TARGET[:3]}\n")
    result = await loop.run(task)

    for r in result.tool_results:
        if r.get("tool_name") == "robot_sdk.move_to_pose":
            out = r.get("output") or {}
            snap = out.get("robot_state_snapshot", {})
            print(f"  verb outcome : {out.get('outcome')}  ({out.get('evidence')})")
            print(f"  ee reached   : {snap.get('ee_position')}")
            print(f"  joints       : {[round(j, 3) for j in snap.get('joint_positions', [])]}")

    print(f"\nOutcome: {result.outcome}  turns={result.turns}  message={result.message!r}")


if __name__ == "__main__":
    asyncio.run(main())
