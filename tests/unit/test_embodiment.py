"""Unit tests for EmbodimentCatalog and the RealAgentServerAdapter backend."""

from __future__ import annotations

import pytest

from robot_harness.embodiment.base import (
    EmbodimentAdapter,
    EmbodimentCommand,
    SupportsVerbs,
)
from robot_harness.embodiment.catalog import EmbodimentCatalog
from robot_harness.embodiment.real.agent_server import RealAgentServerAdapter
from robot_harness.errors import RobotOfflineError

# ---------------------------------------------------------------------------
# EmbodimentCatalog
# ---------------------------------------------------------------------------


def test_catalog_register_and_get() -> None:
    catalog = EmbodimentCatalog()
    arm = RealAgentServerAdapter(robot_id="arm-0")
    catalog.register(arm)
    assert catalog.get("arm-0") is arm


def test_catalog_get_missing_raises_robot_offline() -> None:
    catalog = EmbodimentCatalog()
    with pytest.raises(RobotOfflineError, match="arm-99"):
        catalog.get("arm-99")


def test_catalog_list_robot_ids() -> None:
    catalog = EmbodimentCatalog()
    catalog.register(RealAgentServerAdapter(robot_id="arm-0"))
    catalog.register(RealAgentServerAdapter(robot_id="arm-1"))
    ids = catalog.list_robot_ids()
    assert set(ids) == {"arm-0", "arm-1"}


def test_catalog_contains() -> None:
    catalog = EmbodimentCatalog()
    catalog.register(RealAgentServerAdapter(robot_id="arm-0"))
    assert "arm-0" in catalog
    assert "arm-1" not in catalog


def test_catalog_overwrite() -> None:
    catalog = EmbodimentCatalog()
    a1 = RealAgentServerAdapter(robot_id="arm-0")
    a2 = RealAgentServerAdapter(robot_id="arm-0")
    catalog.register(a1)
    catalog.register(a2)
    assert catalog.get("arm-0") is a2
    assert len(catalog) == 1


@pytest.mark.asyncio
async def test_catalog_aclose_closes_adapters() -> None:
    catalog = EmbodimentCatalog()
    catalog.register(RealAgentServerAdapter(robot_id="arm-0"))
    catalog.register(RealAgentServerAdapter(robot_id="arm-1"))
    # Stub HTTP client aclose is a no-op; this just exercises the fan-out path.
    await catalog.aclose()


# ---------------------------------------------------------------------------
# RealAgentServerAdapter
# ---------------------------------------------------------------------------


def test_adapter_defaults_to_arm() -> None:
    arm = RealAgentServerAdapter(robot_id="arm-0")
    assert arm.robot_type == "arm"
    assert arm.robot_id == "arm-0"


def test_adapter_robot_type_is_configurable() -> None:
    """Morphology is a value, not a hardcoded class attribute (方向A)."""
    humanoid = RealAgentServerAdapter(robot_id="h-0", robot_type="humanoid")
    assert humanoid.robot_type == "humanoid"


@pytest.mark.asyncio
async def test_adapter_get_state_returns_robot_state() -> None:
    arm = RealAgentServerAdapter(robot_id="arm-0")
    state = await arm.get_state()
    assert state.robot_id == "arm-0"
    assert len(state.joint_positions) >= 6


@pytest.mark.asyncio
async def test_adapter_get_camera_frame() -> None:
    arm = RealAgentServerAdapter(robot_id="arm-0")
    frame = await arm.get_camera_frame("wrist_camera")
    assert frame.camera == "wrist_camera"
    assert frame.robot_id == "arm-0"


@pytest.mark.asyncio
async def test_adapter_dispatch_returns_handle() -> None:
    arm = RealAgentServerAdapter(robot_id="arm-0")
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
async def test_adapter_safety_check_passes() -> None:
    arm = RealAgentServerAdapter(robot_id="arm-0")
    cmd = EmbodimentCommand(
        robot_id="arm-0",
        command_type="joint",
        values=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
    )
    verdict = await arm.safety_check(cmd)
    assert verdict.passed is True


@pytest.mark.asyncio
async def test_adapter_call_verb_returns_verdict_dict() -> None:
    arm = RealAgentServerAdapter(robot_id="arm-0")
    verdict = await arm.call_verb("home", {"robot_id": "arm-0"})
    assert verdict["outcome"] == "success"


# ---------------------------------------------------------------------------
# Protocol structural checks
# ---------------------------------------------------------------------------


def test_adapter_satisfies_embodiment_adapter_protocol() -> None:
    arm = RealAgentServerAdapter(robot_id="arm-0")
    assert isinstance(arm, EmbodimentAdapter)


def test_adapter_satisfies_supports_verbs() -> None:
    """Real and sim agent_server adapters both expose the verb transport (方向A)."""
    arm = RealAgentServerAdapter(robot_id="arm-0")
    assert isinstance(arm, SupportsVerbs)
