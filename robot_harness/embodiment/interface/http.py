"""HTTP/WebSocket interface helpers for per-robot agent_server communication.

This is the on-the-wire layer of the boundary defined in ADR-019: the harness
talks to an external on-robot agent_server that owns all real-time control.
Everything in this file is a thin transport — no control logic, no calibration,
no servoing loops.

Currently a stub that returns mock responses.
"""

from __future__ import annotations

from typing import Any


# TODO (ADR-016): real HTTP client with connection pooling (aiohttp or httpx)
# when an on-robot agent_server is wired in.
class HttpAgentServerClient:
    """Thin client for a per-robot HTTP agent_server.

    The agent_server is an external process (not in this repo) that exposes:
    - POST /dispatch       — send an EmbodimentCommand JSON payload
    - GET  /state          — retrieve current RobotState
    - GET  /camera/{id}    — retrieve latest camera frame
    - POST /safety_check   — pre-dispatch advisory check
    - POST /verb/{name}    — run an on-robot mid-loop verb, return a verdict
    - POST /abort          — cancel the in-flight action / verb
    - GET  /health         — liveness + available verbs

    Currently all methods return mock data without network calls.
    """

    def __init__(self, base_url: str, robot_id: str, timeout_s: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._robot_id = robot_id
        self._timeout_s = timeout_s

    async def get_state(self) -> dict[str, Any]:
        # TODO (ADR-016): GET {base_url}/state
        return {
            "robot_id": self._robot_id,
            "joint_positions": [0.0] * 7,
            "joint_velocities": [0.0] * 7,
            "end_effector_pose": {"x": 0.5, "y": 0.0, "z": 0.3},
            "gripper_state": 0.0,
        }

    async def get_camera_frame(self, camera: str) -> dict[str, Any]:
        # TODO (ADR-016): GET {base_url}/camera/{camera}
        return {
            "camera": camera,
            "robot_id": self._robot_id,
            "data": {"mock": True},
            "format": "rgb",
        }

    async def dispatch(self, cmd: dict[str, Any]) -> dict[str, Any]:
        # TODO (ADR-016): POST {base_url}/dispatch with cmd payload
        import uuid

        return {
            "action_id": str(uuid.uuid4()),
            "estimated_duration_s": 1.5,
            "robot_id": self._robot_id,
        }

    async def safety_check(self, cmd: dict[str, Any]) -> dict[str, Any]:
        # TODO (ADR-016): POST {base_url}/safety_check with cmd payload
        return {"passed": True, "reason": "mock", "violated_rules": []}

    async def call_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]:
        # TODO (ADR-016): POST {base_url}/verb/{verb} with the verb payload.
        # Returns a CompletionVerdict-shaped dict; the robot_sdk verb tool validates it.
        return {
            "outcome": "success",
            "evidence": f"mock {verb}",
            "robot_state_snapshot": {"robot_id": self._robot_id},
            "duration_s": 1.0,
            "aborted_by": "none",
        }

    async def abort(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        # TODO (ADR-016): POST {base_url}/abort to stop the in-flight action/verb.
        return {"aborted": True, "robot_id": self._robot_id}

    async def health(self) -> dict[str, Any]:
        # TODO (ADR-016): GET {base_url}/health
        return {"status": "ok", "robot_id": self._robot_id, "available_verbs": []}

    async def aclose(self) -> None:
        # No connection pool while this is a stub; real client (ADR-016) closes here.
        return None
