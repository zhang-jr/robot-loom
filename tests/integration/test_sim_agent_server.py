"""Integration test: drive a sim agent_server over REAL HTTP (no MockTransport).

Unlike ``tests/unit/test_sim_embodiment.py`` (which fakes the transport in
process), this test exercises the full path — real ``MujocoSimRobot`` →
real ``httpx`` socket → a real server process.

Two modes, selected automatically:

  • ``ROBOT_LOOM_SIM_URL`` set
        Conformance-test an already-running external sim agent_server (your own
        MuJoCo / Isaac Lab process). Example::

            ROBOT_LOOM_SIM_URL=http://gpu-box:8810 pytest \
                tests/integration/test_sim_agent_server.py -v

  • not set
        Spawn ``examples/sim_servers/mujoco_agent_server.py`` as a subprocess and
        test against it. Skipped when fastapi / uvicorn are not installed (the
        reference server's runtime deps).

The reference server runs in kinematic-fallback mode when ``mujoco`` is absent;
that is fine here — the point of this test is the real wire path against a real
process, not physics fidelity. Real high-fidelity-sim acceptance is the
``@pytest.mark.hardware`` path (ADR-011).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from robot_harness.embodiment.sim import MujocoSimRobot, run_sim_conformance

_ROBOT_ID = "sim-arm-0"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_SCRIPT = _REPO_ROOT / "examples" / "sim_servers" / "mujoco_agent_server.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_ready(base_url: str, proc: subprocess.Popen[bytes] | None, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"sim server exited early with code {proc.returncode}")
        try:
            resp = httpx.get(f"{base_url}/state", timeout=1.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.3)
    raise RuntimeError(f"sim server at {base_url} not ready within {timeout_s}s")


@pytest.fixture(scope="module")
def sim_base_url() -> Iterator[str]:
    external = os.environ.get("ROBOT_LOOM_SIM_URL")
    if external:
        base = external.rstrip("/")
        _wait_ready(base, None, timeout_s=10.0)
        yield base
        return

    pytest.importorskip("fastapi", reason="reference sim server needs fastapi")
    pytest.importorskip("uvicorn", reason="reference sim server needs uvicorn")

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(_SERVER_SCRIPT),
            "--robot-id",
            _ROBOT_ID,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
    )
    try:
        _wait_ready(base, proc, timeout_s=30.0)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.asyncio
async def test_sim_conformance_over_http(sim_base_url: str) -> None:
    """The full semantic conformance suite against a real server process.

    This is the contract-conformance target: any sim agent_server (the bundled
    reference one here, or your own via ROBOT_LOOM_SIM_URL) must pass it.
    """
    robot = MujocoSimRobot(_ROBOT_ID, server_url=sim_base_url)
    try:
        report = await run_sim_conformance(robot)
        assert report.ok, "sim conformance failed:\n" + report.summary()
    finally:
        await robot.aclose()
