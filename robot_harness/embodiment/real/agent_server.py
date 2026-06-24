"""RealAgentServerAdapter — real-hardware backend over an HTTP agent_server.

Morphology-agnostic: a humanoid and an arm differ only by ``robot_type`` here; the
on-robot agent_server owns the body-specific control (ADR-019). The default backend
(``backend="mock"``) and the real backend (``backend="agent_server"``) both use this
class — they differ only in whether a ``server_url`` is configured.

This is the agent_server-transport member of the ``real/`` family. Robots that do
NOT speak the HTTP agent_server contract (direct CAN/Serial, vendor SDKs) get their
own sibling module here, not a morphology subclass.
"""

from __future__ import annotations

from robot_harness.embodiment.agent_server_base import AgentServerAdapter, AgentServerClient
from robot_harness.embodiment.base import RobotType
from robot_harness.embodiment.interface.http import HttpAgentServerClient


class RealAgentServerAdapter(AgentServerAdapter):
    """EmbodimentAdapter for a real per-robot HTTP agent_server."""

    default_url: str = "http://localhost:8765"

    def __init__(
        self,
        robot_id: str,
        *,
        robot_type: RobotType = "arm",
        server_url: str | None = None,
        dof: int = 6,
        timeout_s: float = 5.0,
        client: AgentServerClient | None = None,
    ) -> None:
        super().__init__(
            robot_id,
            robot_type=robot_type,
            dof=dof,
            client=client
            or HttpAgentServerClient(server_url or self.default_url, robot_id, timeout_s=timeout_s),
        )
