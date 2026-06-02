"""IsaacLabSimRobot — embodiment backend for an Isaac Lab simulator agent_server.

Isaac Lab is GPU-resident and depends on Omniverse / Kit with multi-hundred-MB
assets, which squarely triggers the ADR-001 "must be an external server" test.
The harness never imports Isaac Lab; it holds only this thin client to an
external Isaac Lab agent_server that exposes the same wire contract as a real
robot agent_server plus the sim-lifecycle endpoints (ADR-021).
"""

from __future__ import annotations

from robot_harness.embodiment.sim.base import SimEmbodimentAdapter


class IsaacLabSimRobot(SimEmbodimentAdapter):
    """EmbodimentAdapter backed by an external Isaac Lab agent_server."""

    sim_engine = "isaac"
    default_url = "http://localhost:8811"
