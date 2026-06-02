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

from pydantic import BaseModel

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


class CameraFrameData(BaseModel):
    """Typed shape of ``Frame.data`` for a sim camera frame.

    When ``rendered`` is True the image fields are present and consistent: a
    decoded ``rgb_raw`` payload has length ``width * height * channels``; a
    ``png`` payload starts with the PNG magic bytes. When False the frame is a
    descriptor (kinematic fallback or no GL backend) and ``render_error`` may
    explain why.
    """

    rendered: bool = False
    engine: str = ""
    encoding: CameraEncoding = "none"
    image_b64: str = ""
    width: int = 0
    height: int = 0
    channels: int = 3
    render_error: str = ""


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
