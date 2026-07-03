"""HTTP client for a real per-robot agent_server, plus an offline mock client.

This is the on-the-wire layer of the boundary defined in ADR-019: the harness
talks to an external on-robot agent_server that owns all real-time control.
Everything here is a thin transport — no control logic, no calibration, no
servoing loops.

Two clients live here, both satisfying the ``AgentServerClient`` contract:

- :class:`HttpAgentServerClient` — real ``httpx`` calls against a live
  agent_server (e.g. the om1-agent-server recipe). Transport errors are
  translated to :class:`RobotOfflineError` so the embodiment layer surfaces a
  typed, traceable failure. A custom ``transport`` (``httpx.MockTransport``) may
  be injected for tests so the real request/response path is exercised without a
  live server. Mirrors :class:`SimAgentServerClient` minus the sim-lifecycle
  endpoints.
- :class:`MockAgentServerClient` — fully offline, returns canned data, makes NO
  network call. This backs ``backend="mock"`` (and any adapter built without a
  ``server_url``): the dev / CI default that must run with no server up.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from robot_harness.errors import RobotOfflineError


class HttpAgentServerClient:
    """Thin async HTTP client for a real per-robot agent_server.

    The agent_server is an external process (not in this repo) that exposes the
    physical-executor wire contract:
    - GET  /state          — current RobotState
    - GET  /camera/{id}    — latest camera frame
    - POST /dispatch       — send an EmbodimentCommand JSON payload
    - POST /safety_check   — pre-dispatch advisory check
    - POST /verb/{name}    — run an on-robot mid-loop verb, return a verdict
    - POST /abort          — cancel the in-flight action / verb
    - GET  /health         — liveness + available verbs
    """

    def __init__(
        self,
        base_url: str,
        robot_id: str,
        *,
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._robot_id = robot_id
        self._timeout_s = timeout_s
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_s,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _offline(self, path: str, exc: Exception) -> RobotOfflineError:
        return RobotOfflineError(
            f"agent_server unreachable at {self._base_url}{path}: {exc}",
            robot_id=self._robot_id,
            module_name="embodiment.interface.http",
        )

    async def _get(self, path: str) -> dict[str, Any]:
        client = self._ensure_client()
        try:
            resp = await client.get(path)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            # ValueError covers json.JSONDecodeError: a 200 with a non-JSON body
            # (proxy error page, captive portal) is "not speaking the contract",
            # the same typed failure as unreachable.
            raise self._offline(path, exc) from exc
        return data

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._ensure_client()
        try:
            resp = await client.post(path, json=payload)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise self._offline(path, exc) from exc
        return data

    async def get_state(self) -> dict[str, Any]:
        return await self._get("/state")

    async def get_camera_frame(self, camera: str) -> dict[str, Any]:
        return await self._get(f"/camera/{camera}")

    async def dispatch(self, cmd: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/dispatch", cmd)

    async def safety_check(self, cmd: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/safety_check", cmd)

    async def call_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Invoke an on-robot mid-loop verb (ADR-019). POST /verb/{verb}.

        Returns a CompletionVerdict-shaped dict; the robot_sdk verb tool validates it.
        """
        return await self._post(f"/verb/{verb}", payload)

    async def abort(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Stop the in-flight action / verb. POST /abort."""
        return await self._post("/abort", payload or {})

    async def health(self) -> dict[str, Any]:
        return await self._get("/health")


class MockAgentServerClient:
    """Offline agent_server client — canned responses, no network call.

    Backs ``backend="mock"`` and any adapter built without a ``server_url``: the
    fleet_size=1 dev / CI default (ADR-008) that must run with no server up. Every
    verb returns a success verdict, so it exercises harness plumbing end-to-end
    without asserting any real robot behaviour.
    """

    def __init__(self, robot_id: str, *, dof: int = 6) -> None:
        self._robot_id = robot_id
        self._dof = dof

    async def get_state(self) -> dict[str, Any]:
        return {
            "robot_id": self._robot_id,
            "joint_positions": [0.0] * self._dof,
            "joint_velocities": [0.0] * self._dof,
            "end_effector_pose": {"x": 0.5, "y": 0.0, "z": 0.3},
            "gripper_state": 0.0,
        }

    async def get_camera_frame(self, camera: str) -> dict[str, Any]:
        return {
            "camera": camera,
            "robot_id": self._robot_id,
            "data": {"mock": True},
            "format": "rgb",
        }

    async def dispatch(self, cmd: dict[str, Any]) -> dict[str, Any]:
        return {
            "action_id": str(uuid.uuid4()),
            "estimated_duration_s": 1.5,
            "robot_id": self._robot_id,
        }

    async def safety_check(self, cmd: dict[str, Any]) -> dict[str, Any]:
        return {"passed": True, "reason": "mock", "violated_rules": []}

    async def call_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "outcome": "success",
            "evidence": f"mock {verb}",
            "robot_state_snapshot": {"robot_id": self._robot_id},
            "duration_s": 1.0,
            "aborted_by": "none",
        }

    async def abort(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"aborted": True, "robot_id": self._robot_id}

    async def health(self) -> dict[str, Any]:
        # No "available_verbs" key: the mock does not ADVERTISE a verb set (every
        # verb succeeds with a canned verdict), and "absent" must stay
        # distinguishable from "explicitly zero verbs" ([]) at the adapter layer.
        return {"status": "ok", "robot_id": self._robot_id}

    async def aclose(self) -> None:
        return None
