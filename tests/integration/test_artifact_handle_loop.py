"""End-to-end: artifact handle flows capture_frame → Brain → perception.

This exercises the full data plane through AgentLoop (the path unit tests
historically bypassed by feeding image_b64 by hand):

    capture_frame  → image offloaded to ArtifactStore, returns `frame` ref
    write-back     → ref (small) lands in Memory
    query          → next turn's MemoryView carries the ref
    Brain          → reads ref from memory, calls detect(frame=ref)
    resolver       → hydrates ref → image_b64 from the store
    consumer       → receives the real bytes, never the raw image via the LLM
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from robot_harness.brain.base import BrainDecision, MemoryView, Task, ToolCallRequest
from robot_harness.embodiment.base import Frame, RobotState
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.memory.spatial_hub_adapter import SpatialHubMemory
from robot_harness.tools.middleware.artifact_resolver import ArtifactResolverMiddleware
from robot_harness.tools.middleware.base import build_chain
from robot_harness.tools.perception.mcp_bundle import PERCEPTION_DETECT_OBJECTS
from robot_harness.tools.robot_sdk.observe import CaptureFrameTool
from robot_harness.tools.schema import ToolBackend, ToolSchema

_RAW = b"\x89PNG-real-pixels-not-via-llm"


class _ImgAdapter:
    robot_id = "r0"
    robot_type = "arm"

    async def get_state(self) -> RobotState:
        return RobotState(robot_id="r0", joint_positions=[0.0])

    async def get_camera_frame(self, camera: str) -> Frame:
        return Frame(
            camera=camera,
            robot_id="r0",
            format="png",
            data={"image_b64": base64.b64encode(_RAW).decode("ascii"), "encoding": "png"},
        )

    async def dispatch(self, cmd: Any) -> Any: ...
    async def safety_check(self, cmd: Any) -> Any: ...


class _FakeDetect:
    """Stands in for the perception MCP tool; records hydrated args."""

    name = PERCEPTION_DETECT_OBJECTS
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name=PERCEPTION_DETECT_OBJECTS, description="x", input_schema={"type": "object"}
    )

    def __init__(self) -> None:
        self.seen_args: dict[str, Any] | None = None

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.seen_args = args
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"detections": [{"label": "cube", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]},
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


def _find_frame_ref(mv: MemoryView) -> dict[str, Any] | None:
    for hit in mv.episodic_hits:
        obs = (hit.get("content") or {}).get("observation") or {}
        if isinstance(obs.get("frame"), dict):
            return obs["frame"]
    return None


class _HandoffBrain:
    """Capture, then on the next turn pass the recalled frame ref to detect."""

    def __init__(self) -> None:
        self._turn = 0

    @property
    def supports_streaming(self) -> bool:
        return False

    async def decide(self, task: Task, mv: MemoryView, tools: list[Any]) -> BrainDecision:
        self._turn += 1
        if self._turn == 1:
            return BrainDecision(
                decision_type="tool_call",
                tool_calls=[
                    ToolCallRequest(tool_name="robot.capture_frame", args={"robot_id": "r0"})
                ],
            )
        ref = _find_frame_ref(mv)
        if ref is not None:
            return BrainDecision(
                decision_type="tool_call",
                tool_calls=[
                    ToolCallRequest(
                        tool_name=PERCEPTION_DETECT_OBJECTS,
                        args={"frame": ref, "prompts": ["cube"]},
                    )
                ],
            )
        return BrainDecision(decision_type="plan", message="no frame ref recalled")

    async def replan(self, history: Any, critic_signal: Any) -> BrainDecision:
        return BrainDecision(decision_type="give_up", message="n/a")


@pytest.mark.asyncio
async def test_frame_ref_flows_capture_to_perception_through_loop() -> None:
    ctx = HarnessContext.build(memory=SpatialHubMemory())
    ctx.tool_registry.register(CaptureFrameTool({"r0": _ImgAdapter()}, {"r0": ["overhead"]}))
    detect = _FakeDetect()
    ctx.tool_registry.register(build_chain(detect, [ArtifactResolverMiddleware]))

    brain = _HandoffBrain()
    task = Task(task_id="task-1", description="find the cube", robot_id="r0")
    result = await AgentLoop(brain, ctx, max_turns=4).run(task)

    # detect was reached and received the REAL bytes, hydrated from the store —
    # the raw image was never carried as image_b64 through the Brain.
    assert detect.seen_args is not None, "perception.detect was never reached"
    assert "frame" not in detect.seen_args
    assert base64.b64decode(detect.seen_args["image_b64"]) == _RAW
    assert result.outcome in {"success", "incomplete"}
