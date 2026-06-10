"""reactive_grasp verb against a live sim agent_server (ADR-019).

The Brain issues a single high-level ``robot_sdk.reactive_grasp`` with a target
hint. The sim agent_server runs the whole pick mid-loop internally — resolve the
object, pre-grasp, descend, close the gripper, lift — and returns one
CompletionVerdict. The harness never sees the per-step motion (that is the point
of a reactive verb: the loop lives on the robot/sim, not in the harness).

The object pose is ground truth here; a grasp/perception service (e.g. GraspNet
on the overhead RGB-D) replaces that perception step later — the action chain is
unchanged.

Prerequisite — start the sim (needs mujoco):

    pip install -r ../sim_shim/mujoco_shim/requirements.txt
    python ../sim_shim/mujoco_shim/server.py --port 8810

Then:

    python examples/sim_reactive_grasp.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx

from robot_harness.brain.base import BrainDecision, Task, ToolCallRequest
from robot_harness.config.schema import EmbodimentBackendConfig, HarnessConfig
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.robot_sdk.verbs import build_robot_sdk_verb_tools

_ROBOT_ID = "robot-0"
_SIM_URL = os.environ.get("ROBOT_LOOM_SIM_URL", "http://localhost:8810")


class ScriptedBrain:
    """Issue one reactive_grasp on the cube, then declare done."""

    def __init__(self) -> None:
        self._decisions = [
            BrainDecision(
                decision_type="tool_call",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="robot_sdk.reactive_grasp",
                        args={
                            "robot_id": _ROBOT_ID,
                            "target_hint": {"kind": "object_id", "object_id": "cube"},
                        },
                    )
                ],
            ),
            BrainDecision(decision_type="plan", message="picked up the cube"),
        ]
        self._i = 0

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(self, messages: list, tools: list) -> BrainDecision:  # type: ignore[type-arg]
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d


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
    for tool in build_robot_sdk_verb_tools(ctx.embodiment_adapters):
        ctx.tool_registry.register(tool)

    loop = AgentLoop(ScriptedBrain(), ctx, max_turns=6)
    task = Task(
        task_id=str(uuid.uuid4()),
        description="Pick up the cube.",
        robot_id=_ROBOT_ID,
    )

    print(f"Sim: {_SIM_URL}\nTask: {task.description}\n")
    result = await loop.run(task)

    for r in result.tool_results:
        if r.get("tool_name") == "robot_sdk.reactive_grasp":
            out = r.get("output") or {}
            snap = out.get("robot_state_snapshot", {})
            print(f"  verb outcome : {out.get('outcome')}  ({out.get('evidence')})")
            print(f"  grasped      : {out.get('grasped_object_id')!r}")
            print(
                f"  object pos   : {snap.get('object_position')}  gripper={snap.get('gripper_state')}"
            )

    print(f"\nOutcome: {result.outcome}  turns={result.turns}  message={result.message!r}")


if __name__ == "__main__":
    asyncio.run(main())
