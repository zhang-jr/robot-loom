"""MujocoSimRobot — embodiment backend for a MuJoCo simulator agent_server.

MuJoCo is CPU-based and pip-installable, but the physics step + joint servo are
a tight control loop that must run inside the sim process, never in the harness
(ADR-019). This adapter is therefore a thin client to an external MuJoCo
agent_server (see ``examples/sim_servers/mujoco_agent_server.py``), identical in
shape to a real-robot client — which is exactly what makes sim/real swappable by
config alone (ADR-009 / ADR-021).
"""

from __future__ import annotations

from robot_harness.embodiment.sim.base import SimEmbodimentAdapter


class MujocoSimRobot(SimEmbodimentAdapter):
    """EmbodimentAdapter backed by an external MuJoCo agent_server."""

    sim_engine = "mujoco"
    default_url = "http://localhost:8810"
