"""HTTP client for a simulator agent_server (MuJoCo / Isaac Lab).

Architectural role (ADR-021): a simulator is an *embodiment backend* — it plays
the same role as a per-robot agent_server plus hardware, not a capability tool.
The physics runtime (MuJoCo / Isaac Lab) runs as an external process; the harness
only holds this thin client. The tight/mid control loops (physics step, joint
servo) run inside the sim process, never in the harness (ADR-019).

The sim agent_server speaks the same wire contract as a real robot agent_server
(``/state`` / ``/dispatch`` / ``/camera/{name}`` / ``/safety_check`` /
``/verb/{name}`` / ``/abort`` / ``/health``) plus three sim-lifecycle endpoints
that hardware does not have:

    POST /reset         — reset the simulator to its initial state
    POST /scene         — load a scene description (MJCF / USD path or inline)
    GET  /sim_time      — current simulation clock (seconds)

A runnable reference implementation lives in
``examples/sim_servers/mujoco_agent_server.py``.
"""

from __future__ import annotations

from typing import Any

import httpx

from robot_harness.errors import RobotOfflineError


class SimAgentServerClient:
    """Thin async HTTP client for an external simulator agent_server.

    Like the real-robot ``HttpAgentServerClient``, this client performs real HTTP
    calls (a reference sim server exists to talk to). Transport errors are
    translated to ``RobotOfflineError`` so the embodiment layer surfaces a typed,
    traceable failure instead of a raw ``httpx`` exception.

    A custom ``transport`` (e.g. ``httpx.MockTransport``) may be injected for
    tests so the real request/response path is exercised without a live server.
    """

    def __init__(
        self,
        base_url: str,
        robot_id: str,
        *,
        sim_engine: str = "sim",
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._robot_id = robot_id
        self._sim_engine = sim_engine
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

    async def _get(self, path: str) -> dict[str, Any]:
        client = self._ensure_client()
        try:
            resp = await client.get(path)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            # ValueError covers json.JSONDecodeError: a 200 with a non-JSON body
            # is "not speaking the contract", the same typed failure as unreachable.
            raise RobotOfflineError(
                f"sim agent_server ({self._sim_engine}) unreachable at "
                f"{self._base_url}{path}: {exc}",
                robot_id=self._robot_id,
                module_name="embodiment.interface.sim",
            ) from exc
        return data

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._ensure_client()
        try:
            resp = await client.post(path, json=payload)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RobotOfflineError(
                f"sim agent_server ({self._sim_engine}) unreachable at "
                f"{self._base_url}{path}: {exc}",
                robot_id=self._robot_id,
                module_name="embodiment.interface.sim",
            ) from exc
        return data

    # -- shared robot agent_server contract --------------------------------

    async def get_state(self) -> dict[str, Any]:
        return await self._get("/state")

    async def get_camera_frame(self, camera: str) -> dict[str, Any]:
        return await self._get(f"/camera/{camera}")

    async def dispatch(self, cmd: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/dispatch", cmd)

    async def safety_check(self, cmd: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/safety_check", cmd)

    # -- sim-only lifecycle ------------------------------------------------

    async def reset(self) -> dict[str, Any]:
        return await self._post("/reset", {})

    async def load_scene(self, scene: str) -> dict[str, Any]:
        return await self._post("/scene", {"scene": scene})

    async def get_sim_time(self) -> dict[str, Any]:
        return await self._get("/sim_time")

    async def call_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Invoke an on-robot mid-loop verb (ADR-019). POST /verb/{verb}.

        The sim runs the verb's perception-action loop internally and returns a
        CompletionVerdict-shaped dict.
        """
        return await self._post(f"/verb/{verb}", payload)

    async def abort(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Stop the in-flight action / verb in the sim. POST /abort."""
        return await self._post("/abort", payload or {})

    async def health(self) -> dict[str, Any]:
        """Liveness + advertised verbs. GET /health (same contract as real hardware)."""
        return await self._get("/health")
