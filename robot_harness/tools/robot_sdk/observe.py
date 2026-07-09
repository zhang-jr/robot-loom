"""Observation tools — how the Brain reads a robot's current state.

Two read-only tools, both bridging to the robot's ``EmbodimentAdapter``:

- :class:`GetStateTool` (``robot.get_state``) — proprioception (joint positions,
  end-effector pose, gripper). Available for EVERY robot, including ones with no
  camera. This is the primary state channel for a heterogeneous fleet.
- :class:`CaptureFrameTool` (``robot.capture_frame``) — a camera frame
  (``image_b64`` and, for an RGB-D sim, ``depth_b64`` + intrinsics + pose). Only
  registered when the fleet has at least one camera, and gated per-robot: a call
  for a robot/camera the fleet config did not declare returns an error result.

This fills the "live frame → Brain" gap: ``EmbodimentAdapter.get_camera_frame``
is otherwise only reachable by the Critic, not by the Brain's tool vocabulary.
The Brain captures a frame, then hands the returned ``frame`` ref to an external
perception tool; the harness resolves the ref to bytes out-of-band so the raw
image never enters the LLM context or Memory. When no artifact store is wired the
tool falls back to an inline ``image_b64``. Cameras are a per-robot capability
(ADR-008 / ADR-009) — never assumed.
"""

from __future__ import annotations

import base64
from typing import Any

from robot_harness.embodiment.base import EmbodimentAdapter
from robot_harness.tools.artifacts import ARTIFACT_REF_SCHEMA, ArtifactStore
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema

# Large frame payloads are offloaded to the ArtifactStore under these output
# keys → the small ref is published under the paired ref key instead. Keeps the
# raw image/depth out of the LLM context and out of Memory.
_OFFLOAD = (("image_b64", "frame", "image"), ("depth_b64", "depth", "depth"))


async def _offload_blobs(output: dict[str, Any], store: ArtifactStore) -> dict[str, Any]:
    """Move blob fields into *store*, replacing each with a small ArtifactRef.

    Best-effort: a blob with no data is simply dropped. Subtype is taken from the
    frame's own ``encoding`` / ``depth_encoding`` when present.
    """
    for blob_key, ref_key, kind in _OFFLOAD:
        raw_b64 = output.pop(blob_key, "")
        if not raw_b64:
            continue
        subtype = output.get("encoding" if kind == "image" else "depth_encoding") or kind
        meta = {k: output[k] for k in ("camera", "width", "height", "channels") if k in output}
        ref = await store.put(base64.b64decode(raw_b64), f"{kind}/{subtype}", meta=meta)
        output[ref_key] = ref.model_dump()
    return output


def _no_adapter(tool: str, robot_id: str, ctx: ToolContext) -> ToolResult:
    return ToolResult(
        tool_name=tool,
        trace_id=ctx.trace_id,
        success=False,
        error=f"no embodiment adapter registered for robot '{robot_id}'",
        error_type="RobotOfflineError",
    )


class GetStateTool:
    """Read a robot's proprioceptive state. Works for any robot (camera or not)."""

    name = "robot.get_state"
    backend: ToolBackend = "native"
    brain_visible = True
    schema = ToolSchema(
        name="robot.get_state",
        description=(
            "Read the robot's proprioceptive state: joint positions/velocities, "
            "end-effector pose, gripper. Available for every robot, including "
            "those without a camera."
        ),
        input_schema={
            "type": "object",
            "properties": {"robot_id": {"type": "string"}},
            "required": ["robot_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "joint_positions": {"type": "array", "items": {"type": "number"}},
                "joint_velocities": {"type": "array", "items": {"type": "number"}},
                "end_effector_pose": {"type": "object"},
                "gripper_state": {"type": "number"},
            },
        },
    )

    def __init__(self, adapters: dict[str, EmbodimentAdapter]) -> None:
        self._adapters = adapters

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        robot_id = args.get("robot_id", ctx.robot_id)
        adapter = self._adapters.get(robot_id)
        if adapter is None:
            return _no_adapter(self.name, robot_id, ctx)
        state = await adapter.get_state()
        return ToolResult(
            tool_name=self.name, trace_id=ctx.trace_id, success=True, output=state.model_dump()
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


class CaptureFrameTool:
    """Capture a camera frame from a robot. Gated per-robot by declared cameras."""

    name = "robot.capture_frame"
    backend: ToolBackend = "native"
    brain_visible = True
    schema = ToolSchema(
        name="robot.capture_frame",
        description=(
            "Capture a camera frame from a robot that has one. Returns image_b64 "
            "(and depth_b64 + intrinsics + camera_pose for an RGB-D sim). Pass the "
            "result image to a perception tool. Only some robots have cameras."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "robot_id": {"type": "string"},
                "camera": {
                    "type": "string",
                    "description": "Camera name; defaults to the robot's first camera.",
                },
            },
            "required": ["robot_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "camera": {"type": "string"},
                "robot_id": {"type": "string"},
                "format": {"type": "string"},
                # With an artifact store wired, the image is offloaded and only
                # this ref is returned; pass it as `frame` to a perception tool.
                "frame": ARTIFACT_REF_SCHEMA,
                # Inline fallback when no store is wired.
                "image_b64": {"type": "string"},
            },
        },
    )

    def __init__(
        self,
        adapters: dict[str, EmbodimentAdapter],
        cameras_by_robot: dict[str, list[str]],
    ) -> None:
        self._adapters = adapters
        self._cameras_by_robot = cameras_by_robot

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    def _error(self, robot_id: str, msg: str, ctx: ToolContext) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=False,
            error=msg,
            error_type="ToolSchemaViolationError",
        )

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        robot_id = args.get("robot_id", ctx.robot_id)
        cams = self._cameras_by_robot.get(robot_id, [])
        if not cams:
            return self._error(robot_id, f"robot '{robot_id}' has no camera", ctx)
        camera = args.get("camera") or cams[0]
        if camera not in cams:
            return self._error(
                robot_id, f"robot '{robot_id}' has no camera '{camera}' (has {cams})", ctx
            )
        adapter = self._adapters.get(robot_id)
        if adapter is None:
            return _no_adapter(self.name, robot_id, ctx)
        frame = await adapter.get_camera_frame(camera)
        output: dict[str, Any] = {
            "camera": frame.camera,
            "robot_id": frame.robot_id,
            "format": frame.format,
            **frame.data,
        }
        # When a store is wired, hand large image/depth blobs to it and return
        # only a small ref — the Brain plans over the ref, never the raw bytes.
        # Without a store, fall back to the inline image (e.g. direct unit calls).
        if ctx.artifact_store is not None:
            output = await _offload_blobs(output, ctx.artifact_store)
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output=output,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


def build_observation_tools(
    adapters: dict[str, EmbodimentAdapter],
    cameras_by_robot: dict[str, list[str]],
) -> list[Any]:
    """Build the per-fleet observation tools, gated by declared cameras.

    ``robot.get_state`` is always built (proprioception is universal).
    ``robot.capture_frame`` is built only when at least one robot declares a
    camera; calls for camera-less robots return an error result.
    """
    tools: list[Any] = [GetStateTool(adapters)]
    if any(cameras_by_robot.values()):
        tools.append(CaptureFrameTool(adapters, cameras_by_robot))
    return tools
