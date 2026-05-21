"""HTTP/WebSocket interface helpers for per-robot agent_server communication.

This is the on-the-wire layer of the boundary defined in ADR-019: the harness
talks to an external on-robot agent_server that owns all real-time control.
Everything in this file is a thin transport — no control logic, no calibration,
no servoing loops.

Phase 1: urllib-based stub that returns mock responses.
Phase 2: will use aiohttp or httpx with proper connection pooling.
"""

from __future__ import annotations

from typing import Any


class HttpAgentServerClient:
    """Thin client for a per-robot HTTP agent_server.

    The agent_server is an external process (not in this repo) that exposes:
    - POST /dispatch  — send an EmbodimentCommand JSON payload
    - GET  /state     — retrieve current RobotState
    - GET  /camera/{id} — retrieve latest camera frame

    Phase 1: all methods return mock data without network calls.
    """

    def __init__(self, base_url: str, robot_id: str, timeout_s: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._robot_id = robot_id
        self._timeout_s = timeout_s

    async def get_state(self) -> dict[str, Any]:
        # Phase 2: GET {base_url}/state
        return {
            "robot_id": self._robot_id,
            "joint_positions": [0.0] * 7,
            "joint_velocities": [0.0] * 7,
            "end_effector_pose": {"x": 0.5, "y": 0.0, "z": 0.3},
            "gripper_state": 0.0,
        }

    async def get_camera_frame(self, camera: str) -> dict[str, Any]:
        # Phase 2: GET {base_url}/camera/{camera}
        return {
            "camera": camera,
            "robot_id": self._robot_id,
            "data": {"mock": True},
            "format": "rgb",
        }

    async def dispatch(self, cmd: dict[str, Any]) -> dict[str, Any]:
        # Phase 2: POST {base_url}/dispatch with cmd payload
        import uuid

        return {
            "action_id": str(uuid.uuid4()),
            "estimated_duration_s": 1.5,
            "robot_id": self._robot_id,
        }

    async def safety_check(self, cmd: dict[str, Any]) -> dict[str, Any]:
        # Phase 2: POST {base_url}/safety_check with cmd payload
        return {"passed": True, "reason": "mock", "violated_rules": []}
