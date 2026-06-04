"""Build EmbodimentAdapters from configuration.

This is the single place where the embodiment backend is chosen (mock / real
agent_server / simulator). Because every backend satisfies the same
``EmbodimentAdapter`` Protocol, swapping a robot between real hardware and a
MuJoCo / Isaac Lab sim is a config change only — no Brain / Skill / Critic code
changes (ADR-009 / ADR-021).
"""

from __future__ import annotations

from robot_harness.config.schema import EmbodimentBackendConfig, HarnessConfig
from robot_harness.embodiment.arm.generic_6dof import Generic6DofArm
from robot_harness.embodiment.base import EmbodimentAdapter
from robot_harness.embodiment.catalog import EmbodimentCatalog
from robot_harness.embodiment.sim.isaac_adapter import IsaacLabSimRobot
from robot_harness.embodiment.sim.mujoco_adapter import MujocoSimRobot
from robot_harness.errors import HardwareNotReadyError


def build_embodiment(robot_id: str, cfg: EmbodimentBackendConfig) -> EmbodimentAdapter:
    """Construct a single EmbodimentAdapter for *robot_id* from *cfg*."""
    if cfg.backend == "mock":
        return Generic6DofArm(robot_id, dof=cfg.dof)

    if cfg.backend == "agent_server":
        if not cfg.server_url:
            raise HardwareNotReadyError(
                f"embodiment backend 'agent_server' for robot '{robot_id}' requires a server_url",
                robot_id=robot_id,
                module_name="embodiment.factory",
            )
        return Generic6DofArm(robot_id, dof=cfg.dof, server_url=cfg.server_url)

    if cfg.backend == "sim":
        sim_url = cfg.server_url or None
        if cfg.sim_engine == "mujoco":
            return MujocoSimRobot(
                robot_id,
                robot_type=cfg.robot_type,
                server_url=sim_url,
                dof=cfg.dof,
                scene=cfg.scene,
                timeout_s=cfg.request_timeout_s,
            )
        if cfg.sim_engine == "isaac":
            return IsaacLabSimRobot(
                robot_id,
                robot_type=cfg.robot_type,
                server_url=sim_url,
                dof=cfg.dof,
                scene=cfg.scene,
                timeout_s=cfg.request_timeout_s,
            )

    raise HardwareNotReadyError(
        f"unknown embodiment backend '{cfg.backend}' for robot '{robot_id}'",
        robot_id=robot_id,
        module_name="embodiment.factory",
    )


def build_catalog(config: HarnessConfig) -> EmbodimentCatalog:
    """Build an EmbodimentCatalog for every robot in *config*.

    Robots without an explicit ``embodiments`` entry fall back to a mock backend
    — the fleet_size=1 dev default (ADR-008).
    """
    catalog = EmbodimentCatalog()
    for robot_id in config.robot_ids:
        cfg = config.embodiments.get(robot_id, EmbodimentBackendConfig())
        catalog.register(build_embodiment(robot_id, cfg))
    return catalog
