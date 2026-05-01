"""Unit tests for EmbodimentCatalog and Generic6DofArm mock adapter."""

from __future__ import annotations

import pytest

from robot_harness.embodiment.arm.generic_6dof import Generic6DofArm
from robot_harness.embodiment.base import EmbodimentCommand
from robot_harness.embodiment.catalog import EmbodimentCatalog
from robot_harness.errors import RobotOfflineError

# ---------------------------------------------------------------------------
# EmbodimentCatalog
# ---------------------------------------------------------------------------


def test_catalog_register_and_get() -> None:
    catalog = EmbodimentCatalog()
    arm = Generic6DofArm(robot_id="arm-0")
    catalog.register(arm)
    assert catalog.get("arm-0") is arm


def test_catalog_get_missing_raises_robot_offline() -> None:
    catalog = EmbodimentCatalog()
    with pytest.raises(RobotOfflineError, match="arm-99"):
        catalog.get("arm-99")


def test_catalog_list_robot_ids() -> None:
    catalog = EmbodimentCatalog()
    catalog.register(Generic6DofArm(robot_id="arm-0"))
    catalog.register(Generic6DofArm(robot_id="arm-1"))
    ids = catalog.list_robot_ids()
    assert set(ids) == {"arm-0", "arm-1"}


def test_catalog_contains() -> None:
    catalog = EmbodimentCatalog()
    catalog.register(Generic6DofArm(robot_id="arm-0"))
    assert "arm-0" in catalog
    assert "arm-1" not in catalog


def test_catalog_overwrite() -> None:
    catalog = EmbodimentCatalog()
    a1 = Generic6DofArm(robot_id="arm-0")
    a2 = Generic6DofArm(robot_id="arm-0")
    catalog.register(a1)
    catalog.register(a2)
    assert catalog.get("arm-0") is a2
    assert len(catalog) == 1


# ---------------------------------------------------------------------------
# Generic6DofArm
# ---------------------------------------------------------------------------


def test_arm_robot_type() -> None:
    arm = Generic6DofArm(robot_id="arm-0")
    assert arm.robot_type == "arm"
    assert arm.robot_id == "arm-0"


@pytest.mark.asyncio
async def test_arm_get_state_returns_robot_state() -> None:
    arm = Generic6DofArm(robot_id="arm-0")
    state = await arm.get_state()
    assert state.robot_id == "arm-0"
    assert len(state.joint_positions) >= 6


@pytest.mark.asyncio
async def test_arm_get_camera_frame() -> None:
    arm = Generic6DofArm(robot_id="arm-0")
    frame = await arm.get_camera_frame("wrist_camera")
    assert frame.camera == "wrist_camera"
    assert frame.robot_id == "arm-0"


@pytest.mark.asyncio
async def test_arm_dispatch_returns_handle() -> None:
    arm = Generic6DofArm(robot_id="arm-0")
    cmd = EmbodimentCommand(
        robot_id="arm-0",
        command_type="cartesian",
        values=[0.5, 0.0, 0.3, 0.0, 0.0, 0.0],
    )
    handle = await arm.dispatch(cmd)
    assert handle.robot_id == "arm-0"
    assert handle.action_id != ""
    assert handle.estimated_duration_s > 0


@pytest.mark.asyncio
async def test_arm_safety_check_passes() -> None:
    arm = Generic6DofArm(robot_id="arm-0")
    cmd = EmbodimentCommand(
        robot_id="arm-0",
        command_type="joint",
        values=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
    )
    verdict = await arm.safety_check(cmd)
    assert verdict.passed is True


# ---------------------------------------------------------------------------
# Protocol structural check
# ---------------------------------------------------------------------------


def test_generic_6dof_satisfies_embodiment_adapter_protocol() -> None:
    from robot_harness.embodiment.base import EmbodimentAdapter

    arm = Generic6DofArm(robot_id="arm-0")
    assert isinstance(arm, EmbodimentAdapter)
