"""Virtual closed loop against a live sim agent_server.

Proves the action path is wired end to end: a Brain decision drives
``robot_sdk.execute_action`` → ``SafetyEnvelope.check`` → the sim's ``/dispatch``
(``env.step``) → state read-back → Critic on a real sim camera frame → repeat.
Everything but the sim runs in-process; the sim is the sim_shim MuJoCo
agent_server reached over HTTP (ADR-021).

This example uses a deterministic *scripted* Brain and a mock Critic so it runs
offline (no LLM key). Swap in ``LiteLLMBrain`` and a real Critic adapter to make
the Brain actually plan.

Prerequisite — start the sim (see ../sim_shim):

    python ../sim_shim/mujoco_shim/server.py --port 8810
    # or: cd ../sim_shim && docker compose up mujoco-sim

Then:

    python examples/sim_closed_loop.py            # uses http://localhost:8810
    ROBOT_LOOM_SIM_URL=http://host:8810 python examples/sim_closed_loop.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx

from robot_harness.brain.base import BrainDecision, Task, ToolCallRequest
from robot_harness.config.schema import EmbodimentBackendConfig, HarnessConfig
from robot_harness.critic.base import CriticVerdict
from robot_harness.embodiment.base import Frame
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.robot_sdk.http_adapter import RobotSdkTool

_ROBOT_ID = "robot-0"
_SIM_URL = os.environ.get("ROBOT_LOOM_SIM_URL", "http://localhost:8810")


def _action(values: list[float]) -> BrainDecision:
    return BrainDecision(
        decision_type="tool_call",
        tool_calls=[
            ToolCallRequest(
                tool_name="robot_sdk.execute_action",
                args={"robot_id": _ROBOT_ID, "command_type": "joint", "values": values},
            )
        ],
    )


class ScriptedBrain:
    """Deterministic offline Brain: a few joint moves, then declare done."""

    def __init__(self) -> None:
        self._decisions = [
            _action([0.10, 0.20, 0.10]),
            _action([0.25, 0.10, 0.20]),
            _action([0.05, 0.30, 0.15]),
            BrainDecision(decision_type="respond", message="reached target configuration"),
        ]
        self._i = 0

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(self, messages: list, tools: list, **_: object) -> BrainDecision:  # type: ignore[type-arg]
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d


class MockProgressCritic:
    """Mock Critic that always reports forward progress (never aborts)."""

    async def judge(
        self, current: Frame, reference: Frame | None, task_description: str
    ) -> CriticVerdict:
        rendered = bool(current.data.get("rendered"))
        return CriticVerdict(
            state="progress",
            confidence=0.6,
            evidence=f"sim frame received (rendered={rendered})",
        )


async def main() -> None:
    # Pre-flight: the sim must be reachable.
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            (await client.get(f"{_SIM_URL}/state")).raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Sim not reachable at {_SIM_URL}: {exc}")
        print("Start it first:  python ../sim_shim/mujoco_shim/server.py --port 8810")
        return

    config = HarnessConfig(
        robot_ids=[_ROBOT_ID],
        embodiments={
            _ROBOT_ID: EmbodimentBackendConfig(
                backend="sim", sim_engine="mujoco", server_url=_SIM_URL
            )
        },
    )

    # build() constructs the sim adapter from config and populates the context.
    ctx = HarnessContext.build(config)
    # The live dispatch tool — wired to the sim adapter so it actually steps env.
    ctx.tool_registry.register(RobotSdkTool(ctx.embodiment_adapters))

    loop = AgentLoop(
        ScriptedBrain(),
        ctx,
        max_turns=10,
        critic=MockProgressCritic(),
        critic_interval=2,
    )

    task = Task(
        task_id=str(uuid.uuid4()),
        description="Move the arm through a short joint trajectory in sim.",
        robot_id=_ROBOT_ID,
    )

    print(f"Sim: {_SIM_URL}")
    print(f"Task: {task.description}\n")
    result = await loop.run(task)

    # Show that each dispatch actually advanced the sim (state read-back).
    for i, r in enumerate(result.tool_results, 1):
        state = (r.get("output") or {}).get("state")
        if state is not None:
            print(f"  step {i}: joint_positions={state.get('joint_positions')}")

    print(f"\nOutcome: {result.outcome}  turns={result.turns}  message={result.message!r}")


if __name__ == "__main__":
    asyncio.run(main())
