"""MuJoCo sim agent_server — a standalone shim implementing the Robot Loom
sim wire contract.

Role (Robot Loom ADR-021): a simulator is an *embodiment backend*, reached by
the harness over HTTP exactly like a real robot's per-robot agent_server. The
physics step runs HERE, inside this process — never in the harness (ADR-019).
This shim is deliberately standalone: it depends only on mujoco / fastapi and
emits the contract JSON shapes directly, so it does NOT need the robot_harness
package installed. The harness verifies conformance from the outside via
``run_sim_conformance`` (see the harness repo's
tests/integration/test_sim_agent_server.py).

Wire contract (kept byte-compatible with robot_harness.embodiment.sim.contract):

    GET  /state          -> RobotState
    GET  /camera/{name}  -> CameraFrameResponse (data.image_b64 = encoded frame)
    POST /dispatch       <- EmbodimentCommand  -> {action_id, estimated_duration_s, robot_id}
    POST /safety_check   <- EmbodimentCommand  -> {passed, reason, violated_rules}
    POST /reset          -> RobotState
    POST /scene {scene}  -> {loaded}
    GET  /sim_time       -> {sim_time}
    POST /verb/{name}    <- verb payload        -> CompletionVerdict (ADR-019)

MuJoCo is CPU physics (Apache-2.0, `pip install mujoco`). For headless rendering
set ``MUJOCO_GL=osmesa`` (pure-CPU software render, no GPU) or ``egl`` (GPU). If
mujoco is unavailable the server runs a kinematic fallback so the HTTP contract
stays testable without the dependency.
"""

# NOTE: deliberately no ``from __future__ import annotations`` here. FastAPI
# resolves handler annotations via ``get_type_hints`` against module globals;
# with stringized annotations it cannot see the closure-local ``Request`` import
# inside build_app() and would mis-parse it as a query param (HTTP 422).

import argparse
import base64
import math
import os
import time
import uuid
from typing import Any

try:
    import mujoco
    import numpy as np  # mujoco depends on numpy; only used on the IK path

    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False

# A 3-DOF arm plus a graspable cube on the floor, an overhead RGB-D camera over
# the cube, and a side "wrist" camera. The cube has a free joint (it is an
# object, not an actuated joint) — state reports only the actuated arm joints.
# An "ee" site at the arm tip is the move_to_pose IK target. Joints are limited
# so the IK solution stays in a sane workspace.
_DEFAULT_MJCF = """
<mujoco model="grasp_arm">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <default><joint damping="2" limited="true"/><position kp="150"/></default>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <camera name="wrist" pos="1.2 0 0.8" xyaxes="0 -1 0 0.4 0 1"/>
    <camera name="overhead" pos="0.35 0 1.0" xyaxes="1 0 0 0 1 0" fovy="45"/>
    <body name="l0" pos="0 0 0.1">
      <joint name="j0" type="hinge" axis="0 0 1" range="-180 180"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" rgba="0.2 0.5 0.9 1"/>
      <body name="l1" pos="0 0 0.2">
        <joint name="j1" type="hinge" axis="0 1 0" range="-150 150"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.03" rgba="0.9 0.4 0.2 1"/>
        <body name="l2" pos="0 0 0.2">
          <joint name="j2" type="hinge" axis="0 1 0" range="-150 150"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.025" rgba="0.3 0.8 0.3 1"/>
          <site name="ee" pos="0 0 0.2" size="0.01"/>
        </body>
      </body>
    </body>
    <body name="cube" pos="0.35 0 0.05">
      <freejoint/>
      <geom name="cube_g" type="box" size="0.03 0.03 0.03" rgba="0.85 0.2 0.2 1" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="j0"/>
    <position joint="j1"/>
    <position joint="j2"/>
  </actuator>
</mujoco>
"""

_RENDER_W = int(os.environ.get("RENDER_W", "320"))
_RENDER_H = int(os.environ.get("RENDER_H", "240"))

# Bent-forward "ready" arm pose — a non-singular IK seed and rest configuration.
_READY = [0.0, 0.9, 0.9]


def _encode_image(img: Any) -> tuple[str, str]:
    """Encode an HxWx3 uint8 array as (encoding, base64). PNG via Pillow if present."""
    try:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return "png", base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        return "rgb_raw", base64.b64encode(img.tobytes()).decode("ascii")


class _Sim:
    """MuJoCo-backed sim, or a trivial kinematic integrator when mujoco is absent."""

    def __init__(self, robot_id: str, dof: int = 3) -> None:
        self.robot_id = robot_id
        self._dof = dof
        self._mjcf = _DEFAULT_MJCF
        self._renderer = None
        self._grasped = -1  # body id of the currently grasped object, or -1
        self._grasp_offset: Any = None
        if _HAS_MUJOCO:
            self._build()
        else:
            self._qpos = [0.0] * dof
            self._qvel = [0.0] * dof
            self._time = 0.0
        self._gripper = 0.0

    def _build(self) -> None:
        self._model = mujoco.MjModel.from_xml_string(self._mjcf)
        self._data = mujoco.MjData(self._model)
        # Report only ACTUATED joints in state — exclude free-joint objects like
        # the cube, whose qpos would otherwise pollute joint_positions.
        self._act_qadr = []
        self._act_vadr = []
        lo, hi = [], []
        for i in range(self._model.nu):
            jnt = int(self._model.actuator_trnid[i, 0])
            self._act_qadr.append(int(self._model.jnt_qposadr[jnt]))
            self._act_vadr.append(int(self._model.jnt_dofadr[jnt]))
            r0, r1 = float(self._model.jnt_range[jnt][0]), float(self._model.jnt_range[jnt][1])
            lo.append(r0 if r0 < r1 else -3.14)
            hi.append(r1 if r0 < r1 else 3.14)
        self._jnt_lo = np.array(lo)
        self._jnt_hi = np.array(hi)
        self._ee_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self._dof = self._model.nu
        self._renderer = None
        self._rest()

    def _settle(self, steps: int) -> None:
        for _ in range(steps):
            mujoco.mj_step(self._model, self._data)

    def _rest(self) -> None:
        """Settle objects, then pin the arm to the bent-forward ready pose."""
        self._settle(200)
        if self._model.nu == len(_READY):
            for a, qi in zip(self._act_qadr, _READY, strict=True):
                self._data.qpos[a] = qi
            for a in self._act_vadr:
                self._data.qvel[a] = 0.0
            self._data.ctrl[: len(_READY)] = _READY
            mujoco.mj_forward(self._model, self._data)

    def load_scene(self, mjcf: str) -> None:
        self._mjcf = mjcf
        if _HAS_MUJOCO:
            self._build()
        else:
            self._qpos = [0.0] * self._dof
            self._time = 0.0

    def reset(self) -> None:
        self._grasped = -1
        self._grasp_offset = None
        if _HAS_MUJOCO:
            mujoco.mj_resetData(self._model, self._data)
            self._rest()
        else:
            self._qpos = [0.0] * self._dof
            self._qvel = [0.0] * self._dof
            self._time = 0.0
        self._gripper = 0.0

    def _ik(self, target: list[float], iters: int = 500, tol: float = 2e-4) -> Any:
        """Damped least-squares position IK for the ee site, seeded from READY."""
        if self._model.nu == len(_READY):
            q = np.array(_READY, dtype=float)
        else:
            q = np.array([float(self._data.qpos[a]) for a in self._act_qadr])
        tgt = np.array(target[:3], dtype=float)
        jacp = np.zeros((3, self._model.nv))
        for _ in range(iters):
            for a, qi in zip(self._act_qadr, q, strict=True):
                self._data.qpos[a] = float(qi)
            mujoco.mj_forward(self._model, self._data)
            err = tgt - self._data.site_xpos[self._ee_id]
            if float(np.linalg.norm(err)) < tol:
                break
            mujoco.mj_jacSite(self._model, self._data, jacp, None, self._ee_id)
            jac = jacp[:, self._act_vadr]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(3), err)
            q = np.clip(q + np.clip(dq, -0.05, 0.05), self._jnt_lo, self._jnt_hi)
        return q

    def move_to_pose(self, target_pose: list[float], tol: float = 0.01) -> dict[str, Any]:
        """High-level verb: drive the ee to a cartesian target via IK.

        The sim owns how the pose is reached (ADR-019). Position-only IK (the
        3-DOF arm cannot control orientation); the solution is placed
        kinematically and the ee residual decides the CompletionVerdict.
        """
        if not _HAS_MUJOCO:
            return {"outcome": "failed", "evidence": "move_to_pose needs mujoco", "duration_s": 0.0}
        if self._ee_id < 0:
            return {"outcome": "failed", "evidence": "scene has no 'ee' site", "duration_s": 0.0}
        t0 = time.monotonic()
        q = self._ik(target_pose)
        for a, qi in zip(self._act_qadr, q, strict=True):
            self._data.qpos[a] = float(qi)
        for a in self._act_vadr:
            self._data.qvel[a] = 0.0
        self._data.ctrl[: len(q)] = q
        mujoco.mj_forward(self._model, self._data)
        self._follow_if_grasped()
        ee = self._data.site_xpos[self._ee_id]
        err = float(np.linalg.norm(np.array(target_pose[:3]) - ee))
        outcome = "success" if err < tol else ("partial" if err < 5 * tol else "failed")
        return {
            "outcome": outcome,
            "evidence": f"ee within {err * 1000:.1f}mm of target",
            "robot_state_snapshot": {
                "joint_positions": [float(self._data.qpos[a]) for a in self._act_qadr],
                "ee_position": [float(x) for x in ee],
            },
            "final_pose": [float(ee[0]), float(ee[1]), float(ee[2]), 0.0, 0.0, 0.0],
            "duration_s": time.monotonic() - t0,
        }

    # -- gripper + grasp (kinematic attach model) --------------------------
    #
    # A simplified gripper: the arm has no physical fingers; closing the gripper
    # within `grasp_radius` of an object "attaches" it (the object then tracks
    # the ee). This is a deterministic stand-in for contact-rich grasping — good
    # enough to drive the pick action chain. The perception step (which object,
    # where) is resolved from ground truth here and is what a grasp/perception
    # service replaces later.

    def _obj_qadr(self, body: int) -> int:
        jadr = int(self._model.body_jntadr[body])
        return int(self._model.jnt_qposadr[jadr])

    def _obj_vadr(self, body: int) -> int:
        jadr = int(self._model.body_jntadr[body])
        return int(self._model.jnt_dofadr[jadr])

    def _follow_if_grasped(self) -> None:
        if self._grasped < 0 or self._grasp_offset is None:
            return
        ee = self._data.site_xpos[self._ee_id]
        qa = self._obj_qadr(self._grasped)
        self._data.qpos[qa : qa + 3] = ee + self._grasp_offset
        va = self._obj_vadr(self._grasped)
        self._data.qvel[va : va + 6] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def close_gripper(self, body: int = -1, grasp_radius: float = 0.12) -> bool:
        """Close the gripper; attach `body` if the ee is within grasp_radius."""
        self._gripper = 1.0
        if body >= 0:
            ee = self._data.site_xpos[self._ee_id]
            obj = self._data.xpos[body]
            if float(np.linalg.norm(ee - obj)) < grasp_radius:
                self._grasped = body
                self._grasp_offset = np.array(obj, dtype=float) - np.array(ee, dtype=float)
                return True
        return False

    def open_gripper(self) -> None:
        self._gripper = 0.0
        self._grasped = -1
        self._grasp_offset = None

    def reactive_grasp(self, target_hint: dict[str, Any]) -> dict[str, Any]:
        """Pick an object: pre-grasp -> descend -> close -> lift (ADR-019).

        The target object is resolved from `target_hint` (ground truth by body
        name; a real grasp service supplies the pose). Returns a CompletionVerdict.
        """
        if not _HAS_MUJOCO:
            return {
                "outcome": "failed",
                "evidence": "reactive_grasp needs mujoco",
                "duration_s": 0.0,
            }
        name = str(target_hint.get("object_id") or target_hint.get("object") or "cube")
        body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            return {"outcome": "failed", "evidence": f"no object '{name}'", "duration_s": 0.0}
        t0 = time.monotonic()
        self.open_gripper()
        obj0 = np.array(self._data.xpos[body], dtype=float)
        pre = (obj0 + np.array([0.0, 0.0, 0.18])).tolist()
        grasp = (obj0 + np.array([0.0, 0.0, 0.08])).tolist()
        lift = (obj0 + np.array([0.0, 0.0, 0.28])).tolist()

        self.move_to_pose(pre)
        approach = self.move_to_pose(grasp)
        grabbed = self.close_gripper(body)
        self.move_to_pose(lift)

        obj1 = np.array(self._data.xpos[body], dtype=float)
        lifted = float(obj1[2] - obj0[2])
        if grabbed and lifted > 0.05:
            outcome = "success"
        elif grabbed:
            outcome = "partial"
        else:
            outcome = "failed"
        ee = self._data.site_xpos[self._ee_id]
        return {
            "outcome": outcome,
            "evidence": (
                f"approach {approach['evidence']}; grabbed={grabbed}; lifted={lifted * 1000:.0f}mm"
            ),
            "grasped_object_id": name if grabbed else "",
            "final_grasp_pose": [float(grasp[0]), float(grasp[1]), float(grasp[2]), 0.0, 0.0, 0.0],
            "robot_state_snapshot": {
                "joint_positions": [float(self._data.qpos[a]) for a in self._act_qadr],
                "ee_position": [float(x) for x in ee],
                "object_position": [float(x) for x in obj1],
                "gripper_state": self._gripper,
            },
            "duration_s": time.monotonic() - t0,
        }

    def apply(self, command_type: str, values: list[float], gripper_close: bool) -> None:
        if command_type == "hand_grasp":
            self._gripper = values[0] if values else (1.0 if gripper_close else 0.0)
            return
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
            qpos = [float(self._data.qpos[a]) for a in self._act_qadr]
            qvel = [float(self._data.qvel[a]) for a in self._act_vadr]
        else:
            qpos = list(self._qpos)
            qvel = list(self._qvel)
        return {
            "robot_id": self.robot_id,
            "joint_positions": qpos,
            "joint_velocities": qvel,
            "end_effector_pose": {},
            "gripper_state": self._gripper,
        }

    def sim_time(self) -> float:
        return float(self._data.time) if _HAS_MUJOCO else self._time

    def render(self, camera: str) -> dict[str, Any]:
        if not _HAS_MUJOCO:
            return {"rendered": False, "engine": "kinematic", "encoding": "none"}
        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model, height=_RENDER_H, width=_RENDER_W)
            cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            cam = cam_id if cam_id >= 0 else -1
            # RGB pass.
            self._renderer.disable_depth_rendering()
            self._renderer.update_scene(self._data, camera=cam)
            rgb = self._renderer.render()
            # Depth pass (meters).
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(self._data, camera=cam)
            depth = self._renderer.render()
            self._renderer.disable_depth_rendering()
        except Exception as exc:  # noqa: BLE001 - GL backend is environment-dependent
            return {
                "rendered": False,
                "engine": "mujoco",
                "encoding": "none",
                "render_error": str(exc),
            }

        encoding, image_b64 = _encode_image(rgb)
        out: dict[str, Any] = {
            "rendered": True,
            "engine": "mujoco",
            "encoding": encoding,
            "image_b64": image_b64,
            "width": _RENDER_W,
            "height": _RENDER_H,
            "channels": 3,
        }
        # Depth is only useful with intrinsics + extrinsics, so emit all three
        # together — and only for a named camera (free-cam fallback has neither).
        if cam_id >= 0:
            fovy = float(self._model.cam_fovy[cam_id])
            focal = 0.5 * _RENDER_H / math.tan(0.5 * math.radians(fovy))
            out["depth_encoding"] = "float32"
            out["depth_b64"] = base64.b64encode(depth.astype("<f4").tobytes()).decode("ascii")
            out["intrinsics"] = {
                "fx": focal,
                "fy": focal,
                "cx": _RENDER_W / 2.0,
                "cy": _RENDER_H / 2.0,
            }
            out["camera_pose"] = {
                "position": [float(x) for x in self._data.cam_xpos[cam_id]],
                "rotation": [float(x) for x in self._data.cam_xmat[cam_id]],
            }
        return out


def build_app(robot_id: str) -> Any:
    from fastapi import FastAPI, Request

    sim = _Sim(robot_id)
    app = FastAPI(title=f"mujoco-sim[{robot_id}]")

    @app.get("/state")
    async def state() -> dict[str, Any]:
        return sim.state()

    @app.get("/camera/{name}")
    async def camera(name: str) -> dict[str, Any]:
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
        return {"action_id": str(uuid.uuid4()), "estimated_duration_s": 0.2, "robot_id": robot_id}

    @app.post("/safety_check")
    async def safety_check(request: Request) -> dict[str, Any]:
        # The harness SafetyEnvelope is authoritative (ADR-007); this is advisory.
        await request.json()
        return {"passed": True, "reason": "sim", "violated_rules": []}

    @app.post("/reset")
    async def reset() -> dict[str, Any]:
        sim.reset()
        return sim.state()

    @app.post("/scene")
    async def scene(request: Request) -> dict[str, Any]:
        body = await request.json()
        sim.load_scene(body.get("scene") or _DEFAULT_MJCF)
        return {"loaded": body.get("scene", "<default>")}

    @app.get("/sim_time")
    async def sim_time() -> dict[str, Any]:
        return {"sim_time": sim.sim_time()}

    @app.post("/verb/{name}")
    async def verb(name: str, request: Request) -> dict[str, Any]:
        # On-robot mid-loop verbs (ADR-019). The sim runs the verb internally and
        # returns a CompletionVerdict. Only move_to_pose is implemented here.
        payload = await request.json()
        if name == "move_to_pose":
            target = payload.get("target_pose") or []
            if len(target) < 3:
                return {
                    "outcome": "failed",
                    "evidence": "move_to_pose needs target_pose [x, y, z, ...]",
                    "duration_s": 0.0,
                }
            return sim.move_to_pose(list(target))
        if name == "reactive_grasp":
            return sim.reactive_grasp(payload.get("target_hint") or {})
        return {
            "outcome": "failed",
            "evidence": f"sim does not implement verb '{name}'",
            "duration_s": 0.0,
        }

    return app


# Module-level app for `uvicorn server:app`. ROBOT_ID picks the reported id.
app = build_app(os.environ.get("ROBOT_ID", "arm-sim"))


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo sim agent_server")
    parser.add_argument("--robot-id", default=os.environ.get("ROBOT_ID", "arm-sim"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8810")))
    args = parser.parse_args()

    import uvicorn

    mode = "MuJoCo physics" if _HAS_MUJOCO else "kinematic fallback (pip install mujoco)"
    print(f"[mujoco-sim] robot_id={args.robot_id} mode={mode} on {args.host}:{args.port}")
    uvicorn.run(build_app(args.robot_id), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
