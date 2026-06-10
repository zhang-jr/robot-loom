"""Full closed loop shape: env_input(world_state ⊕ memory) → brain ⇌ server_tools
→ critic → loop, against a live sim.

This wires every node of the loop so it is plug-and-play once real external
services exist — only the mock perception/critic below get swapped for real
MCP/HTTP server URLs, and ScriptedBrain for LiteLLMBrain:

    world_state : robot.capture_frame (camera) + robot.get_state (proprioception)
    memory      : observations live in the native conversation; long-term recall is
                  the on-demand memory.query tool (ADR-024 / ADR-025)
    server_tools: perception.detect (mock here) + robot_sdk.reactive_grasp (sim verb)
    critic      : MockProgressCritic (mock here) → ReplanPolicy → writes episodic memory
    loop        : AgentLoop.run

Cameras are per-robot: robot-0 declares ["overhead", "wrist"], so capture_frame
is exposed and gated to those. A camera-less robot would get only get_state.

Prerequisite — start the sim (needs mujoco):

    python ../sim_shim/mujoco_shim/server.py --port 8810

Then:  python examples/sim_perception_loop.py
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import httpx

from robot_harness.brain.base import BrainDecision, Task, ToolCallRequest
from robot_harness.config.schema import EmbodimentBackendConfig, HarnessConfig
from robot_harness.critic.base import CriticVerdict
from robot_harness.embodiment.base import Frame
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.robot_sdk.observe import build_observation_tools
from robot_harness.tools.robot_sdk.verbs import build_robot_sdk_verb_tools
from robot_harness.tools.schema import ToolBackend, ToolSchema

_ROBOT_ID = "robot-0"
_SIM_URL = os.environ.get("ROBOT_LOOM_SIM_URL", "http://localhost:8810")


class MockDetectTool:
    """Stand-in perception server tool. Replace with a perception MCP/HTTP server
    (perception-vlm needs no GPU; sam3/triton need GPU). It would consume the
    image_b64 from robot.capture_frame and return real detections."""

    name = "perception.detect"
    backend: ToolBackend = "native"
    brain_visible = True
    schema = ToolSchema(
        name="perception.detect",
        description="Detect objects in a captured frame. Pass image_b64 from robot.capture_frame.",
        input_schema={
            "type": "object",
            "properties": {"image_b64": {"type": "string"}, "classes": {"type": "array"}},
            "required": ["image_b64"],
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"objects": [{"id": "cube", "label": "red cube", "confidence": 0.92}]},
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


class MockProgressCritic:
    """Stand-in critic server. Replace with a VLAC-like critic over the frame."""

    async def judge(
        self, current: Frame, reference: Frame | None, task_description: str
    ) -> CriticVerdict:
        return CriticVerdict(state="progress", confidence=0.6, evidence="scene advancing")


class ScriptedBrain:
    """Deterministic offline Brain emitting the realistic loop sequence.

    A real LiteLLMBrain would thread tool outputs (the captured image into
    detect, the detection into the grasp); here the sequence is fixed to show the
    loop shape without an LLM key."""

    def __init__(self) -> None:
        self._decisions = [
            _call("robot.capture_frame", {"robot_id": _ROBOT_ID, "camera": "overhead"}),
            _call("perception.detect", {"image_b64": "<captured-frame>"}),
            _call(
                "robot_sdk.reactive_grasp",
                {"robot_id": _ROBOT_ID, "target_hint": {"kind": "object_id", "object_id": "cube"}},
            ),
            BrainDecision(decision_type="plan", message="picked up the detected cube"),
        ]
        self._i = 0

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(self, messages: list, tools: list) -> BrainDecision:  # type: ignore[type-arg]
        d = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return d


def _call(tool: str, args: dict[str, Any]) -> BrainDecision:
    return BrainDecision(
        decision_type="tool_call", tool_calls=[ToolCallRequest(tool_name=tool, args=args)]
    )


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
                backend="sim",
                sim_engine="mujoco",
                server_url=_SIM_URL,
                cameras=["overhead", "wrist"],  # robot-0 has cameras
            )
        },
    )
    ctx = HarnessContext.build(config)
    cameras_by_robot = {r: c.cameras for r, c in config.embodiments.items()}

    # world_state tools (capability-gated) + server tools + action verbs
    for tool in build_observation_tools(ctx.embodiment_adapters, cameras_by_robot):
        ctx.tool_registry.register(tool)
    ctx.tool_registry.register(MockDetectTool())
    for tool in build_robot_sdk_verb_tools(ctx.embodiment_adapters):
        ctx.tool_registry.register(tool)

    loop = AgentLoop(
        ScriptedBrain(), ctx, max_turns=8, critic=MockProgressCritic(), critic_interval=2
    )
    task = Task(
        task_id=str(uuid.uuid4()),
        description="Find the cube and pick it up.",
        robot_id=_ROBOT_ID,
    )

    print(f"Sim: {_SIM_URL}\nTask: {task.description}\n")
    result = await loop.run(task)

    for r in result.tool_results:
        name = r.get("tool_name")
        out = r.get("output") or {}
        if name == "robot.capture_frame":
            print(
                f"  [world_state] capture_frame: format={out.get('format')} "
                f"has_depth={'depth_b64' in out} intrinsics={'intrinsics' in out}"
            )
        elif name == "perception.detect":
            print(f"  [server_tool] detect: {out.get('objects')}")
        elif name == "robot_sdk.reactive_grasp":
            print(
                f"  [server_tool] reactive_grasp: {out.get('outcome')} grasped={out.get('grasped_object_id')!r}"
            )

    print(f"\nOutcome: {result.outcome}  turns={result.turns}  message={result.message!r}")
    # memory was written by the loop (episode + critic verdict)
    print("(episode + critic verdict written to memory by the loop)")


if __name__ == "__main__":
    asyncio.run(main())
