"""RealAgentServerAdapter — real-hardware backend over an HTTP agent_server.

Morphology-agnostic: a humanoid and an arm differ only by ``robot_type`` here; the
on-robot agent_server owns the body-specific control (ADR-019).

Client selection is by whether a ``server_url`` is configured:
- ``server_url`` set (``backend="agent_server"``) → real :class:`HttpAgentServerClient`
  making live HTTP calls (e.g. the om1-agent-server recipe).
- no ``server_url`` (``backend="mock"``, dev / CI default) → offline
  :class:`MockAgentServerClient`, canned responses, no network. This keeps the
  default runnable with nothing up while the real client talks to a live server.

This is the agent_server-transport member of the ``real/`` family. Robots that do
NOT speak the HTTP agent_server contract (direct CAN/Serial, vendor SDKs) get their
own sibling module here, not a morphology subclass.
"""

from __future__ import annotations

from robot_harness.embodiment.agent_server_base import AgentServerAdapter, AgentServerClient
from robot_harness.embodiment.base import RobotType
from robot_harness.embodiment.interface.http import HttpAgentServerClient, MockAgentServerClient


class RealAgentServerAdapter(AgentServerAdapter):
    """EmbodimentAdapter for a real per-robot HTTP agent_server."""

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
        if client is None:
            client = (
                HttpAgentServerClient(server_url, robot_id, timeout_s=timeout_s)
                if server_url
                else MockAgentServerClient(robot_id, dof=dof)
            )
        super().__init__(robot_id, robot_type=robot_type, dof=dof, client=client)
