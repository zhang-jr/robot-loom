"""Unit tests for simulator embodiment backends (ADR-021).

The sim agent_server is faked with ``httpx.MockTransport`` so the real
request/response path through ``SimAgentServerClient`` is exercised without a
live server.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable

import httpx
import pytest

from robot_harness.config.schema import EmbodimentBackendConfig, HarnessConfig
from robot_harness.embodiment.base import EmbodimentAdapter, EmbodimentCommand
from robot_harness.embodiment.factory import build_catalog, build_embodiment
from robot_harness.embodiment.interface.sim import SimAgentServerClient
from robot_harness.embodiment.sim import IsaacLabSimRobot, MujocoSimRobot, run_sim_conformance
from robot_harness.errors import HardwareNotReadyError, RobotOfflineError


def _fake_sim_handler(request: httpx.Request) -> httpx.Response:
    """Minimal in-memory sim agent_server honoring the wire contract."""
    path = request.url.path
    if path == "/state":
        return httpx.Response(
            200,
            json={
                "robot_id": "sim-arm-0",
                "joint_positions": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                "joint_velocities": [0.0] * 6,
                "end_effector_pose": {"x": 0.4, "y": 0.0, "z": 0.3},
                "gripper_state": 0.0,
            },
        )
    if path.startswith("/camera/"):
        camera = path.rsplit("/", 1)[-1]
        return httpx.Response(
            200,
            json={
                "camera": camera,
                "robot_id": "sim-arm-0",
                "data": {"sim": True},
                "format": "rgb",
            },
        )
    if path == "/dispatch":
        return httpx.Response(
            200,
            json={"action_id": "act-1", "estimated_duration_s": 0.8, "robot_id": "sim-arm-0"},
        )
    if path == "/safety_check":
        return httpx.Response(200, json={"passed": True, "reason": "sim", "violated_rules": []})
    if path == "/reset":
        return httpx.Response(
            200,
            json={"robot_id": "sim-arm-0", "joint_positions": [0.0] * 6, "gripper_state": 0.0},
        )
    if path == "/scene":
        body = json.loads(request.content)
        return httpx.Response(200, json={"loaded": body.get("scene", "")})
    if path == "/sim_time":
        return httpx.Response(200, json={"sim_time": 1.25})
    return httpx.Response(404, json={"error": "not found"})


def _mujoco_with_fake_server() -> MujocoSimRobot:
    client = SimAgentServerClient(
        "http://sim.local",
        "sim-arm-0",
        sim_engine="mujoco",
        transport=httpx.MockTransport(_fake_sim_handler),
    )
    return MujocoSimRobot("sim-arm-0", client=client)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_mujoco_satisfies_embodiment_adapter_protocol() -> None:
    assert isinstance(MujocoSimRobot("sim-arm-0"), EmbodimentAdapter)


def test_isaac_satisfies_embodiment_adapter_protocol() -> None:
    assert isinstance(IsaacLabSimRobot("sim-arm-0"), EmbodimentAdapter)


def test_engine_labels_and_default_ports() -> None:
    assert MujocoSimRobot("r").sim_engine == "mujoco"
    assert MujocoSimRobot("r").default_url.endswith(":8810")
    assert IsaacLabSimRobot("r").sim_engine == "isaac"
    assert IsaacLabSimRobot("r").default_url.endswith(":8811")


# ---------------------------------------------------------------------------
# Wire contract through SimAgentServerClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_state_maps_to_robot_state() -> None:
    arm = _mujoco_with_fake_server()
    state = await arm.get_state()
    assert state.robot_id == "sim-arm-0"
    assert state.joint_positions == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
    await arm.aclose()


@pytest.mark.asyncio
async def test_get_camera_frame() -> None:
    arm = _mujoco_with_fake_server()
    frame = await arm.get_camera_frame("wrist")
    assert frame.camera == "wrist"
    assert frame.robot_id == "sim-arm-0"
    await arm.aclose()


@pytest.mark.asyncio
async def test_dispatch_returns_handle() -> None:
    arm = _mujoco_with_fake_server()
    handle = await arm.dispatch(
        EmbodimentCommand(robot_id="sim-arm-0", command_type="joint", values=[0.0] * 6)
    )
    assert handle.action_id == "act-1"
    assert handle.estimated_duration_s == 0.8
    await arm.aclose()


@pytest.mark.asyncio
async def test_safety_check_passes() -> None:
    arm = _mujoco_with_fake_server()
    verdict = await arm.safety_check(
        EmbodimentCommand(robot_id="sim-arm-0", command_type="joint", values=[0.0] * 6)
    )
    assert verdict.passed is True
    await arm.aclose()


@pytest.mark.asyncio
async def test_sim_lifecycle_reset_scene_time() -> None:
    arm = _mujoco_with_fake_server()
    state = await arm.reset()
    assert state.joint_positions == [0.0] * 6
    await arm.load_scene("table_with_cube.xml")
    assert await arm.sim_time() == 1.25
    await arm.aclose()


@pytest.mark.asyncio
async def test_unreachable_server_raises_robot_offline() -> None:
    def _down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = SimAgentServerClient(
        "http://sim.local",
        "sim-arm-0",
        sim_engine="mujoco",
        transport=httpx.MockTransport(_down),
    )
    arm = MujocoSimRobot("sim-arm-0", client=client)
    with pytest.raises(RobotOfflineError, match="unreachable"):
        await arm.get_state()
    await arm.aclose()


# ---------------------------------------------------------------------------
# Factory / config-driven selection (ADR-009)
# ---------------------------------------------------------------------------


def test_factory_builds_mujoco_from_config() -> None:
    cfg = EmbodimentBackendConfig(backend="sim", sim_engine="mujoco")
    assert isinstance(build_embodiment("r0", cfg), MujocoSimRobot)


def test_factory_builds_isaac_from_config() -> None:
    cfg = EmbodimentBackendConfig(backend="sim", sim_engine="isaac")
    assert isinstance(build_embodiment("r0", cfg), IsaacLabSimRobot)


def test_factory_agent_server_requires_url() -> None:
    cfg = EmbodimentBackendConfig(backend="agent_server")
    with pytest.raises(HardwareNotReadyError, match="server_url"):
        build_embodiment("r0", cfg)


def test_factory_sim_uses_config_server_url() -> None:
    cfg = EmbodimentBackendConfig(
        backend="sim", sim_engine="mujoco", server_url="http://gpu-box:9000"
    )
    arm = build_embodiment("r0", cfg)
    assert isinstance(arm, MujocoSimRobot)


def test_build_catalog_mixes_backends() -> None:
    config = HarnessConfig(
        fleet_size=2,
        robot_ids=["arm-real", "arm-sim"],
        embodiments={
            "arm-sim": EmbodimentBackendConfig(backend="sim", sim_engine="mujoco"),
        },
    )
    catalog = build_catalog(config)
    assert set(catalog.list_robot_ids()) == {"arm-real", "arm-sim"}
    # arm-real has no entry -> mock fallback; arm-sim -> MuJoCo backend.
    assert isinstance(catalog.get("arm-sim"), MujocoSimRobot)


# ---------------------------------------------------------------------------
# Conformance suite (semantic checks, ADR-021)
# ---------------------------------------------------------------------------


def _handler_with_camera(
    camera_data: dict[str, object],
) -> Callable[[httpx.Request], httpx.Response]:
    """Reuse the compliant fake server but override the camera frame ``data``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/camera/"):
            name = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(
                200,
                json={"camera": name, "robot_id": "sim-arm-0", "data": camera_data, "format": "x"},
            )
        return _fake_sim_handler(request)

    return handler


def _mujoco_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
) -> MujocoSimRobot:
    client = SimAgentServerClient(
        "http://sim.local",
        "sim-arm-0",
        sim_engine="mujoco",
        transport=httpx.MockTransport(handler),
    )
    return MujocoSimRobot("sim-arm-0", client=client)


@pytest.mark.asyncio
async def test_conformance_passes_against_compliant_fake() -> None:
    rgb = base64.b64encode(b"\x00" * (2 * 2 * 3)).decode("ascii")
    handler = _handler_with_camera(
        {
            "rendered": True,
            "encoding": "rgb_raw",
            "image_b64": rgb,
            "width": 2,
            "height": 2,
            "channels": 3,
        }
    )
    robot = _mujoco_with_handler(handler)
    report = await run_sim_conformance(robot)
    assert report.ok, report.summary()
    await robot.aclose()


@pytest.mark.asyncio
async def test_conformance_accepts_descriptor_camera() -> None:
    # The default fake returns a non-rendered descriptor frame — still conformant.
    robot = _mujoco_with_fake_server()
    report = await run_sim_conformance(robot)
    assert report.ok, report.summary()
    await robot.aclose()


@pytest.mark.asyncio
async def test_conformance_flags_inconsistent_render() -> None:
    # rendered=True but the payload is too short for the declared dimensions.
    bad = base64.b64encode(b"\x00" * 3).decode("ascii")
    handler = _handler_with_camera(
        {
            "rendered": True,
            "encoding": "rgb_raw",
            "image_b64": bad,
            "width": 64,
            "height": 64,
            "channels": 3,
        }
    )
    robot = _mujoco_with_handler(handler)
    report = await run_sim_conformance(robot)
    assert not report.ok
    failed = [r.name for r in report.results if not r.passed]
    assert "camera.frame_valid" in failed
    await robot.aclose()
