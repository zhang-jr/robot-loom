"""Reference MuJoCo simulator agent_server.

This is a *reference external server*, not part of the harness package — it
plays the role of "per-robot agent_server + hardware" for a simulated arm
(ADR-021). The harness reaches it through ``MujocoSimRobot`` (an
``EmbodimentAdapter``), never by importing MuJoCo. The physics step lives here,
on the sim side of the boundary, never in the harness process (ADR-019).

Wire contract (same as a real robot agent_server, plus sim lifecycle):

    GET  /state            -> RobotState
    GET  /camera/{name}    -> Frame
    POST /dispatch         -> {action_id, estimated_duration_s, robot_id}
    POST /safety_check     -> {passed, reason, violated_rules}
    POST /reset            -> RobotState         (sim-only)
    POST /scene  {scene}   -> {loaded}           (sim-only)
    GET  /sim_time         -> {sim_time}         (sim-only)

Run it::

    pip install -e ".[sim-ref]"     # mujoco + fastapi + uvicorn + pillow
    python examples/sim_servers/mujoco_agent_server.py --port 8810

Then point a robot at it in workspace config.yaml::

    embodiments:
      arm-sim:
        backend: sim
        sim_engine: mujoco
        server_url: http://localhost:8810

If ``mujoco`` is not installed the server still runs in a lightweight kinematic
fallback mode so the HTTP contract can be smoke-tested in CI; install ``mujoco``
to get real physics.

Camera frames are offscreen-rendered and returned as ``data.image_b64`` (PNG when
Pillow is available, else raw RGB). On a headless host set ``MUJOCO_GL=egl`` (or
``osmesa``) so the GL backend can initialize; otherwise rendering degrades to a
descriptor with ``rendered: False``.
"""

import argparse
import base64
import uuid
from typing import Any

from robot_harness.embodiment.sim.contract import CameraFrameData

# NOTE: deliberately no ``from __future__ import annotations`` here. FastAPI
# resolves handler annotations via ``get_type_hints`` against module globals;
# with stringized annotations it cannot see the closure-local ``Request`` import
# inside build_app() and would mis-parse it as a query param (HTTP 422).

try:
    import mujoco  # type: ignore[import-not-found]

    _HAS_MUJOCO = True
except ImportError:  # pragma: no cover - depends on optional heavy dep
    _HAS_MUJOCO = False

# A minimal 6-DOF-ish arm so the server runs stand-alone with no external assets.
# Includes a fixed "wrist" camera so the offscreen render path has a viewpoint.
_DEFAULT_MJCF = """
<mujoco model="ref_arm">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>
  <worldbody>
    <light pos="0 0 2"/>
    <camera name="wrist" pos="1.2 0 0.8" xyaxes="0 -1 0 0.4 0 1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="l0" pos="0 0 0.1">
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" rgba="0.2 0.5 0.9 1"/>
      <body name="l1" pos="0 0 0.2">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" rgba="0.9 0.4 0.2 1"/>
        <body name="l2" pos="0 0 0.2">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.025" rgba="0.3 0.8 0.3 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="j0" kp="20"/>
    <position joint="j1" kp="20"/>
    <position joint="j2" kp="20"/>
  </actuator>
</mujoco>
"""

# Default offscreen render size. Kept small to keep base64 payloads light for a
# reference/dev server; a production sim server would negotiate this or stream
# frames as an MCP resource (ADR-002 / ADR-021 待验证).
_RENDER_W = 320
_RENDER_H = 240


def _encode_image(img: Any) -> tuple[str, str]:
    """Encode an HxWx3 uint8 array as (encoding, base64-string).

    Prefers PNG via Pillow when available (small payload); otherwise falls back
    to raw RGB bytes so no extra dependency is required. The field name
    ``image_b64`` matches the perception tools' image field for consistency.
    """
    try:
        import io

        from PIL import Image  # type: ignore[import-not-found]

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return "png", base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        return "rgb_raw", base64.b64encode(img.tobytes()).decode("ascii")


class _Sim:
    """Wraps MuJoCo if available, else a trivial kinematic integrator.

    The kinematic fallback exists only so the HTTP contract is testable without
    the heavy dependency; it is NOT a physics engine.
    """

    def __init__(self, robot_id: str, dof: int = 3) -> None:
        self.robot_id = robot_id
        self._dof = dof
        self._mjcf = _DEFAULT_MJCF
        self._renderer = None
        self._cube_position = [0.35, 0.0, 0.02]
        self._held_object_id: str | None = None
        if _HAS_MUJOCO:
            self._build_mujoco()
        else:
            self._qpos = [0.0] * dof
            self._qvel = [0.0] * dof
            self._time = 0.0
        self._gripper = 0.0

    def _build_mujoco(self) -> None:
        self._model = mujoco.MjModel.from_xml_string(self._mjcf)
        self._data = mujoco.MjData(self._model)
        self._dof = self._model.nq
        self._renderer = None  # bound to a model; rebuild lazily on next render

    def load_scene(self, mjcf: str) -> None:
        self._mjcf = mjcf
        if _HAS_MUJOCO:
            self._build_mujoco()
        else:
            self._qpos = [0.0] * self._dof
            self._time = 0.0

    def reset(self) -> None:
        if _HAS_MUJOCO:
            mujoco.mj_resetData(self._model, self._data)
        else:
            self._qpos = [0.0] * self._dof
            self._qvel = [0.0] * self._dof
            self._time = 0.0
        self._gripper = 0.0
        self._held_object_id = None

    def apply(self, command_type: str, values: list[float], gripper_close: bool) -> None:
        if command_type == "hand_grasp":
            self._gripper = values[0] if values else (1.0 if gripper_close else 0.0)
            return
        # joint targets are the common case; cartesian/delta would need IK that
        # belongs on the sim side — out of scope for this minimal reference.
        targets = values[: self._dof]
        if _HAS_MUJOCO:
            self._data.ctrl[: len(targets)] = targets
            for _ in range(100):  # ~0.2s of sim time at dt=0.002
                mujoco.mj_step(self._model, self._data)
        else:
            self._qpos = list(targets) + self._qpos[len(targets) :]
            self._time += 0.2

    def state(self) -> dict[str, Any]:
        if _HAS_MUJOCO:
            qpos = list(self._data.qpos[: self._dof])
            qvel = list(self._data.qvel[: self._dof])
        else:
            qpos = list(self._qpos)
            qvel = list(self._qvel)
        return {
            "robot_id": self.robot_id,
            "joint_positions": qpos,
            "joint_velocities": qvel,
            "end_effector_pose": {},
            "gripper_state": self._gripper,
            "held_object_id": self._held_object_id,
        }

    def sim_time(self) -> float:
        return float(self._data.time) if _HAS_MUJOCO else self._time

    def render(self, camera: str) -> dict[str, Any]:
        """Offscreen-render the named camera into a contract ``CameraFrameData`` dict.

        Returns a descriptor (``rendered: False``) when MuJoCo is unavailable or
        the GL backend cannot initialize (headless without EGL/OSMesa). Set the
        ``MUJOCO_GL`` env var (``egl`` / ``osmesa`` / ``glfw``) to pick a backend.
        The dict is built via the shared ``CameraFrameData`` model so this
        reference server is conformant by construction (see contract.py).
        """
        if not _HAS_MUJOCO:
            return CameraFrameData(engine="kinematic", rendered=False).model_dump()
        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model, height=_RENDER_H, width=_RENDER_W)
            cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            self._renderer.update_scene(self._data, camera=cam_id if cam_id >= 0 else -1)
            img = self._renderer.render()
        except Exception as exc:  # noqa: BLE001 - GL backend is environment-dependent
            return CameraFrameData(
                engine="mujoco", rendered=False, render_error=str(exc)
            ).model_dump()
        encoding, image_b64 = _encode_image(img)
        return CameraFrameData(
            engine="mujoco",
            rendered=True,
            encoding=encoding,
            image_b64=image_b64,
            width=_RENDER_W,
            height=_RENDER_H,
            channels=3,
        ).model_dump()

    def run_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]:
        if verb == "reactive_grasp":
            return self._reactive_grasp(payload)
        raise KeyError(verb)

    def _reactive_grasp(self, payload: dict[str, Any]) -> dict[str, Any]:
        target_hint = payload.get("target_hint") or {}
        kind = target_hint.get("kind")
        graspable = False

        if kind == "object_id" and target_hint.get("object_id") == "cube":
            graspable = True
        elif kind == "phrase" and "cube" in str(target_hint.get("phrase", "")).lower():
            graspable = True

        if not graspable:
            return {
                "outcome": "failed",
                "evidence": f"reactive_grasp could not resolve target_hint={target_hint!r}",
                "robot_state_snapshot": {
                    **self.state(),
                    "object_position": list(self._cube_position),
                },
                "duration_s": 0.4,
                "aborted_by": "none",
            }

        self._gripper = 1.0
        self._held_object_id = "cube"
        if _HAS_MUJOCO:
            for _ in range(50):
                mujoco.mj_step(self._model, self._data)
        else:
            self._time += 0.1

        final_grasp_pose = [self._cube_position[0], self._cube_position[1], 0.18, 0.0, 0.0, 0.0]
        return {
            "outcome": "success",
            "evidence": "reactive_grasp closed the gripper around the cube and lifted it.",
            "robot_state_snapshot": {
                **self.state(),
                "object_position": [self._cube_position[0], self._cube_position[1], 0.18],
            },
            "duration_s": 1.2,
            "aborted_by": "none",
            "grasped_object_id": "cube",
            "final_grasp_pose": final_grasp_pose,
        }


def build_app(robot_id: str) -> Any:
    from fastapi import FastAPI, Request

    sim = _Sim(robot_id)
    app = FastAPI(title=f"mujoco-agent-server[{robot_id}]")

    @app.get("/state")
    async def state() -> dict[str, Any]:
        return sim.state()

    @app.get("/camera/{name}")
    async def camera(name: str) -> dict[str, Any]:
        # Offscreen-render the frame on the sim side and hand the harness a Frame
        # whose ``data.image_b64`` carries the encoded image (aligned with the
        # perception tools' image field). ``format`` reports the encoding.
        data = sim.render(name)
        return {
            "camera": name,
            "robot_id": robot_id,
            "data": data,
            "format": data.get("encoding", "none"),
        }

    @app.post("/dispatch")
    async def dispatch(request: Request) -> dict[str, Any]:
        cmd = await request.json()
        sim.apply(
            cmd.get("command_type", "joint"),
            cmd.get("values", []),
            bool(cmd.get("extra", {}).get("gripper_close", False)),
        )
        return {
            "action_id": str(uuid.uuid4()),
            "estimated_duration_s": 0.2,
            "robot_id": robot_id,
        }

    @app.post("/safety_check")
    async def safety_check(request: Request) -> dict[str, Any]:
        # The harness SafetyEnvelope is the authoritative gate (ADR-007); the
        # sim reports its own view, which the harness treats as advisory.
        await request.json()
        return {"passed": True, "reason": "sim", "violated_rules": []}

    @app.post("/reset")
    async def reset() -> dict[str, Any]:
        sim.reset()
        return sim.state()

    @app.post("/verb/{verb}")
    async def verb(verb: str, request: Request) -> dict[str, Any]:
        payload = await request.json()
        try:
            return sim.run_verb(verb, payload)
        except KeyError:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail=f"unknown verb: {verb}") from None

    @app.post("/scene")
    async def scene(request: Request) -> dict[str, Any]:
        body = await request.json()
        sim.load_scene(body.get("scene") or _DEFAULT_MJCF)
        return {"loaded": body.get("scene", "<default>")}

    @app.get("/sim_time")
    async def sim_time() -> dict[str, Any]:
        return {"sim_time": sim.sim_time()}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference MuJoCo agent_server")
    parser.add_argument("--robot-id", default="arm-sim")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8810)
    args = parser.parse_args()

    import uvicorn

    mode = (
        "MuJoCo physics" if _HAS_MUJOCO else "kinematic fallback (pip install mujoco for physics)"
    )
    print(f"[mujoco_agent_server] robot_id={args.robot_id} mode={mode} on {args.host}:{args.port}")
    uvicorn.run(build_app(args.robot_id), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
