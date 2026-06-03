"""Sim agent_server wire contract — the narrow waist between the harness and any
simulator backend (ADR-021).

Every simulator (MuJoCo / Isaac Lab / LIBERO / RoboCasa / ...) has a different
native API: gym-style ``step``/``reset``, different obs/action specs, camera
APIs, units and frame conventions. This module pins the ONE HTTP/JSON contract a
per-sim agent_server *shim* must implement so the harness adapter — and
Brain / Skill / Critic above it — stay sim-unaware. Integrating a new sim means
writing a shim that maps its native API onto these shapes, never changing
harness code.

Endpoints:

    GET  /state          -> StateResponse        (= RobotState)
    GET  /camera/{name}  -> CameraFrameResponse
    POST /dispatch       <- DispatchRequest       (= EmbodimentCommand)
                         -> DispatchResponse       (= DispatchHandle)
    POST /safety_check   <- SafetyCheckRequest     (= EmbodimentCommand)
                         -> SafetyCheckResponse    (= SafetyVerdict)
    POST /reset          -> StateResponse
    POST /scene          <- SceneRequest   -> SceneResponse
    GET  /sim_time       -> SimTimeResponse

Semantics that JSON shape alone cannot enforce (these are the conformance
targets in ``conformance.py`` — matching shapes are NOT enough):

- ``dispatch.values`` are RADIANS for ``joint``, METERS for ``cartesian`` /
  ``delta`` (see ``EmbodimentCommand`` for the per-command_type layout).
- ``dispatch`` is a high-level INTENT and returns immediately with a handle
  (ADR-019). It does NOT block until motion completes, so a conformant sim is
  NOT required to make ``/state`` reflect the target right after dispatch.
- ``reset`` returns the post-reset state and resets the sim clock toward zero.
- ``sim_time`` is non-negative and non-decreasing between successive reads.

Request / response models reuse the embodiment base models where identical, so
the wire contract and the in-process types cannot silently diverge.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from robot_harness.embodiment.base import (
    DispatchHandle,
    EmbodimentCommand,
    RobotState,
    SafetyVerdict,
)

# Reused base models — named here so the contract has a complete vocabulary.
StateResponse = RobotState
DispatchRequest = EmbodimentCommand
DispatchResponse = DispatchHandle
SafetyCheckRequest = EmbodimentCommand
SafetyCheckResponse = SafetyVerdict

CameraEncoding = Literal["png", "rgb_raw", "none"]
DepthEncoding = Literal["float32", "none"]


class CameraIntrinsics(BaseModel):
    """Pinhole intrinsics (pixels). Needed to back-project depth to a point cloud."""

    fx: float
    fy: float
    cx: float
    cy: float


class CameraPose(BaseModel):
    """World pose of the camera (``world <- camera``).

    MuJoCo / OpenGL camera convention: the camera looks down its local ``-z``,
    with ``+x`` right and ``+y`` up. ``rotation`` is the row-major 3x3 of the
    camera frame expressed in world. Consumers use this to lift a camera-frame
    point cloud into the world frame (T_wc) for world-frame grasp targets.
    """

    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = Field(default_factory=list)  # 9 floats, row-major


class CameraFrameData(BaseModel):
    """Typed shape of ``Frame.data`` for a sim camera frame.

    When ``rendered`` is True the RGB fields are present and consistent: a
    decoded ``rgb_raw`` payload has length ``width * height * channels``; a
    ``png`` payload starts with the PNG magic bytes. When False the frame is a
    descriptor (kinematic fallback or no GL backend) and ``render_error`` may
    explain why.

    Optional RGB-D fields (present when the sim renders depth):
      - ``depth_b64`` / ``depth_encoding`` — depth map. ``float32`` is a raw
        little-endian float32 array, row-major, ``width * height`` values, in
        METERS. No-hit pixels carry the far-clip distance; consumers filter
        them (e.g. drop depth beyond the workspace).
      - ``intrinsics`` — pinhole fx/fy/cx/cy for back-projection.
      - ``camera_pose`` — world<-camera transform for world-frame targets.
    """

    rendered: bool = False
    engine: str = ""
    encoding: CameraEncoding = "none"
    image_b64: str = ""
    width: int = 0
    height: int = 0
    channels: int = 3
    render_error: str = ""
    # RGB-D extensions (optional)
    depth_b64: str = ""
    depth_encoding: DepthEncoding = "none"
    intrinsics: CameraIntrinsics | None = None
    camera_pose: CameraPose | None = None


class CameraFrameResponse(BaseModel):
    """``GET /camera/{name}`` response (mirrors ``Frame`` with typed ``data``)."""

    camera: str
    robot_id: str
    data: CameraFrameData
    format: str = "none"


class SceneRequest(BaseModel):
    """``POST /scene`` body: an MJCF / USD path or inline scene description."""

    scene: str = ""


class SceneResponse(BaseModel):
    """``POST /scene`` response."""

    loaded: str = ""


class SimTimeResponse(BaseModel):
    """``GET /sim_time`` response: simulation clock in seconds."""

    sim_time: float = 0.0
