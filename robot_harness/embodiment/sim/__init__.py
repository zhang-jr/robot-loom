"""Simulator embodiment backends (MuJoCo / Isaac Lab).

A simulator is an embodiment backend, not a tool (ADR-021): it satisfies the
same ``EmbodimentAdapter`` Protocol as real hardware, so Brain / Skill / Critic
cannot tell whether they are driving a sim or a real robot (ADR-009). The
physics runtime is an external process; these classes are thin clients.
"""

from __future__ import annotations

from robot_harness.embodiment.sim.base import SimEmbodimentAdapter
from robot_harness.embodiment.sim.conformance import (
    ConformanceReport,
    run_sim_conformance,
)
from robot_harness.embodiment.sim.isaac_adapter import IsaacLabSimRobot
from robot_harness.embodiment.sim.mujoco_adapter import MujocoSimRobot

__all__ = [
    "ConformanceReport",
    "IsaacLabSimRobot",
    "MujocoSimRobot",
    "SimEmbodimentAdapter",
    "run_sim_conformance",
]
