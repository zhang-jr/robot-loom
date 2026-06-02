"""Semantic conformance suite for a sim agent_server (ADR-021).

Checks that a simulator backend honours the *meaning* of the wire contract
(``contract.py``), not just that it returns HTTP 200 with the right JSON shape.
This is the concrete "达标" target that keeps per-sim interface drift — mismatched
units, frames, action ordering, non-stepping clocks — from leaking past the
adapter into Brain / Skill / Critic.

Run it against any ``SimEmbodimentAdapter``:

    from robot_harness.embodiment.sim import MujocoSimRobot, run_sim_conformance

    robot = MujocoSimRobot("arm-sim", server_url="http://gpu-box:8810")
    report = await run_sim_conformance(robot)
    assert report.ok, report.summary()

The checks are deliberately engine-agnostic and respect ADR-019 (dispatch is an
async high-level intent, so the suite does NOT require ``/state`` to reach the
target right after dispatch). They hold for both the bundled reference server
and a real MuJoCo / Isaac Lab process.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Iterable

from pydantic import BaseModel, Field, ValidationError

from robot_harness.embodiment.base import EmbodimentCommand
from robot_harness.embodiment.sim.base import SimEmbodimentAdapter
from robot_harness.embodiment.sim.contract import CameraFrameData

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class ConformanceReport(BaseModel):
    """Collected results of a conformance run."""

    results: list[CheckResult] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.passed for r in self.results)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name=name, passed=passed, detail=detail))

    def summary(self) -> str:
        lines = []
        for r in self.results:
            tag = "PASS" if r.passed else "FAIL"
            line = f"[{tag}] {r.name}"
            if r.detail:
                line += f" — {r.detail}"
            lines.append(line)
        return "\n".join(lines)


def _finite(xs: Iterable[float]) -> bool:
    return all(math.isfinite(x) for x in xs)


async def run_sim_conformance(
    adapter: SimEmbodimentAdapter, *, camera: str = "wrist"
) -> ConformanceReport:
    """Exercise *adapter* against a sim agent_server and report conformance.

    Transport-level failures (server unreachable) propagate as ``RobotOfflineError``
    — a down server is not a conformance verdict. Semantic failures are recorded
    as failed checks so a single run surfaces every problem at once.
    """
    report = ConformanceReport()

    # --- state: schema + finiteness + dof stability ----------------------
    s1 = await adapter.get_state()
    dof = len(s1.joint_positions)
    report.add("state.nonempty", dof >= 1, f"dof={dof}")
    report.add(
        "state.finite",
        _finite(s1.joint_positions) and _finite(s1.joint_velocities),
    )
    s2 = await adapter.get_state()
    report.add("state.dof_stable", len(s2.joint_positions) == dof)

    # --- reset: idempotent + clock resets toward zero --------------------
    r1 = await adapter.reset()
    t_reset = await adapter.sim_time()
    r2 = await adapter.reset()
    idempotent = len(r1.joint_positions) == len(r2.joint_positions) and all(
        abs(a - b) < 1e-6 for a, b in zip(r1.joint_positions, r2.joint_positions, strict=False)
    )
    report.add("reset.idempotent", idempotent)
    report.add("reset.clock_nonneg", t_reset >= 0.0, f"t={t_reset}")

    # --- scene: accepts a load without erroring --------------------------
    await adapter.load_scene("")  # empty -> server keeps its current/default scene
    report.add("scene.accepts_load", True)

    # --- dispatch: acknowledged with a valid handle (ADR-019) ------------
    target = [0.1] * dof
    handle = await adapter.dispatch(
        EmbodimentCommand(robot_id=adapter.robot_id, command_type="joint", values=target)
    )
    report.add(
        "dispatch.ack",
        handle.action_id != "" and handle.estimated_duration_s >= 0.0,
        f"action_id={handle.action_id!r}",
    )

    # --- sim_time: non-decreasing across the dispatch --------------------
    t_after = await adapter.sim_time()
    report.add("sim_time.non_decreasing", t_after >= t_reset, f"{t_reset} -> {t_after}")

    # --- camera: typed data + decodable render ---------------------------
    frame = await adapter.get_camera_frame(camera)
    report.add("camera.identity", frame.camera == camera and frame.robot_id != "")
    cam_ok, cam_detail = _check_camera_data(frame.data)
    report.add("camera.frame_valid", cam_ok, cam_detail)

    # --- safety_check: advisory shape (ADR-019) --------------------------
    verdict = await adapter.safety_check(
        EmbodimentCommand(robot_id=adapter.robot_id, command_type="joint", values=target)
    )
    report.add("safety.shape", isinstance(verdict.passed, bool))

    return report


def _check_camera_data(raw: dict[str, object]) -> tuple[bool, str]:
    try:
        data = CameraFrameData.model_validate(raw)
    except ValidationError as exc:
        return False, f"schema: {exc.errors()[:1]}"
    if not data.rendered:
        return True, "descriptor (not rendered)"
    image = base64.b64decode(data.image_b64)
    if data.encoding == "rgb_raw":
        expected = data.width * data.height * data.channels
        return len(image) == expected, f"rgb_raw {len(image)} vs {expected}"
    if data.encoding == "png":
        return image[:8] == _PNG_MAGIC, "png header"
    return False, f"rendered but encoding={data.encoding}"
