"""SimEmbodimentAdapter — simulator embodiment backend.

A simulator agent_server speaks the same wire contract as a real per-robot
agent_server, so this shares the morphology-agnostic ``AgentServerAdapter`` base
(see embodiment/agent_server_base.py) and adds only the sim-lifecycle surface
(``reset`` / ``load_scene`` / ``sim_time``) that real hardware does not have. That
surface is deliberately kept off the ``EmbodimentAdapter`` Protocol so the harness
business layer stays sim-unaware (ADR-009 / ADR-021).

Per-engine subclasses (``MujocoSimRobot`` / ``IsaacLabSimRobot``) only set the
engine label and default port — the wire contract is identical.
"""

from __future__ import annotations

from robot_harness.embodiment.agent_server_base import AgentServerAdapter
from robot_harness.embodiment.base import RobotState, RobotType
from robot_harness.embodiment.interface.sim import SimAgentServerClient


class SimEmbodimentAdapter(AgentServerAdapter):
    """EmbodimentAdapter backed by an external simulator agent_server.

    The physics step and any in-sim control loop run inside the sim process, not
    here (ADR-019); this class only transports high-level intent, samples state /
    frames, and drives the sim lifecycle.
    """

    sim_engine: str = "sim"
    default_url: str = "http://localhost:8800"

    def __init__(
        self,
        robot_id: str,
        *,
        robot_type: RobotType = "arm",
        server_url: str | None = None,
        dof: int = 6,
        scene: str = "",
        timeout_s: float = 5.0,
        client: SimAgentServerClient | None = None,
    ) -> None:
        self._scene = scene
        self._sim_client: SimAgentServerClient = client or SimAgentServerClient(
            server_url or self.default_url,
            robot_id,
            sim_engine=self.sim_engine,
            timeout_s=timeout_s,
        )
        super().__init__(robot_id, robot_type=robot_type, dof=dof, client=self._sim_client)

    # -- sim-only lifecycle (not part of EmbodimentAdapter Protocol) -------

    async def reset(self) -> RobotState:
        """Reset the simulator to its initial state and return the new state."""
        raw = await self._sim_client.reset()
        return self._to_state(raw)

    async def load_scene(self, scene: str) -> None:
        """Load a scene (MJCF / USD path or inline description) into the sim."""
        await self._sim_client.load_scene(scene)
        self._scene = scene

    async def sim_time(self) -> float:
        """Return the current simulation clock in seconds."""
        raw = await self._sim_client.get_sim_time()
        return float(raw.get("sim_time", 0.0))
