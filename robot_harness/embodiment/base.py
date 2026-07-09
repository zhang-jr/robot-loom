"""EmbodimentAdapter Protocol — the harness interface to robot hardware.

Layered closed-loop boundary (see ADR-019):

    ┌──────────────────────────────────────────────────────────────┐
    │  harness (this process)                                      │
    │    • slow loop 0.5-7 Hz: brain plan / skill orchestration    │
    │    • calls dispatch(cmd) with HIGH-LEVEL intent              │
    │    • awaits completion event, NOT every control tick         │
    └──────────────────────────────────────────────────────────────┘
                              │  HTTP / WS (JSON or msgpack)
                              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  per-robot agent_server  (external process, on the robot)    │
    │    • mid loop  5-30  Hz: visual servoing, reactive grasp,    │
    │                          "approach until X" verbs            │
    │    • tight loop 100-1000 Hz: joint servo, trajectory tracking,│
    │                              hand-eye calibration, e-stop    │
    │                              reflex, IMU/force fusion        │
    └──────────────────────────────────────────────────────────────┘

What the harness DOES via this Protocol:
    • dispatch high-level commands (joint target / cartesian goal / locomotion goal / verb)
    • read latest state / camera frame (low-frequency sampling, NOT a stream)
    • run SafetyEnvelope.check() as a pre-dispatch authorization step
      (high-level constraints — pose reachability, fleet-wide conflicts);
      the per-robot agent_server is still responsible for hard safety reflexes.

What the harness MUST NOT do via this Protocol:
    • run control loops at robot frequencies (100+ Hz)
    • perform hand-eye calibration (lives on-robot, calibration data is read-only here)
    • drive every tick of a visual-servoing loop — issue a reactive verb instead
      (see robot_harness/tools/robot_sdk/, e.g. reactive_grasp / visual_servo_to)
    • assume dispatch() is synchronous to motion completion — it returns a handle

Implementations are morphology-agnostic wire clients split by transport, not by
robot_type (ADR-027): real hardware lives in embodiment/real/ (e.g. the HTTP
agent_server client) and simulators in embodiment/sim/, both sharing the
AgentServerAdapter base. robot_type is a parameter, never a subclass. An adapter
only translates the unified EmbodimentCommand into the per-robot agent_server's
wire format. Hardware-specific control logic does NOT belong here.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

RobotType = Literal["arm", "humanoid", "quadruped", "mobile"]


class Frame(BaseModel):
    """Camera or sensor frame from the robot."""

    camera: str
    robot_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    format: str = "rgb"


class RobotState(BaseModel):
    """Current kinematic state of the robot."""

    robot_id: str
    joint_positions: list[float] = Field(default_factory=list)
    joint_velocities: list[float] = Field(default_factory=list)
    end_effector_pose: dict[str, float] = Field(default_factory=dict)
    gripper_state: float = 0.0  # 0=open, 1=closed
    extra: dict[str, Any] = Field(default_factory=dict)


CommandType = Literal["joint", "cartesian", "delta", "locomotion", "hand_grasp"]


class EmbodimentCommand(BaseModel):
    """Unified command sent to the EmbodimentAdapter.dispatch().

    This is a HIGH-LEVEL intent, not a control-tick payload (ADR-019).

    command_type semantics:
      - "joint"      : target joint configuration; on-robot runtime plans the trajectory
      - "cartesian"  : target end-effector pose; on-robot runtime does IK + planning
      - "delta"      : incremental displacement from current pose
      - "locomotion" : mobile-base / leg goal (pose or twist target, NOT a velocity stream)
      - "hand_grasp" : open / close gripper at a target force or width

    For reactive verbs that need a tight perception-action loop (visual servoing,
    "approach until grasp succeeds"), do NOT model them here — expose them as
    robot_sdk tools that wrap the per-robot agent_server's verb endpoints, so
    the loop runs on-robot and the harness only observes the completion event.

    values layout per command_type (safety layer depends on this):
      - joint      : [pos_0, pos_1, ..., pos_N]  (radians, N = DOF)
      - cartesian  : [x, y, z, ...]  (meters; orientation repr is adapter-specific)
      - delta      : [dx, dy, dz, ...] (meters/radians, same length as cartesian)
      - locomotion : [x, y] or [x, y, yaw]  (map frame, meters/radians; goal pose,
                     checked against the map geofence when safety.map_bounds_m is set)
      - hand_grasp : [width_or_ratio]  (0=open, 1=closed; extra for force params)
    """

    # TODO (ADR-009): add Pydantic validator for values length per command_type
    #   once a non-arm robot_type is actually driven — today every backend goes
    #   through RealAgentServerAdapter / SimEmbodimentAdapter with robot_type="arm"
    #   default, so no per-morphology layout conflicts arise yet.
    # TODO (ADR-007): promote ``frame`` to an optional typed field when the safety
    #   layer needs coordinate-frame-aware validation for cartesian commands.
    # TODO (ADR-009): add ``locomotion_mode: Literal["pose", "twist"]`` when a
    #   mobile / quadruped adapter lands.
    robot_id: str
    command_type: CommandType
    values: list[float] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class SafetyVerdict(BaseModel):
    """Result of EmbodimentAdapter.safety_check()."""

    passed: bool
    reason: str = ""
    violated_rules: list[str] = Field(default_factory=list)


class DispatchHandle(BaseModel):
    """Handle returned by dispatch() — used to await completion."""

    robot_id: str
    action_id: str
    estimated_duration_s: float = 0.0


@runtime_checkable
class EmbodimentAdapter(Protocol):
    """Interface to a specific robot's per-robot agent_server (NOT its control layer).

    An EmbodimentAdapter is a thin client that translates EmbodimentCommand into
    the wire format consumed by an external on-robot agent_server. The agent_server
    owns all real-time control: trajectory tracking, joint servo, hand-eye
    calibration, IMU/force fusion, e-stop reflex. The adapter never runs a control
    loop in-process.

    Contract:
      • All dispatch() calls MUST be preceded by SafetyEnvelope.check() (ADR-007).
      • dispatch() returns a DispatchHandle immediately — callers await completion
        events through the handle, not by polling state at control frequency.
      • get_state() / get_camera_frame() are low-frequency sampling endpoints
        for brain reasoning; high-rate streaming belongs in a separate channel.

    Implementations are split by transport, not morphology (ADR-027):
    embodiment/real/ for hardware agent_server clients, embodiment/sim/ for
    simulators — both share the AgentServerAdapter base and take robot_type as a
    parameter rather than encoding it per subclass.
    """

    robot_id: str
    robot_type: RobotType

    async def get_camera_frame(self, camera: str) -> Frame: ...
    async def get_state(self) -> RobotState: ...
    async def dispatch(self, cmd: EmbodimentCommand) -> DispatchHandle: ...
    async def safety_check(self, cmd: EmbodimentCommand) -> SafetyVerdict: ...


@runtime_checkable
class SupportsVerbs(Protocol):
    """Optional capability: an agent_server adapter that can run on-robot verbs.

    A *verb* (``reactive_grasp`` / ``visual_servo_to`` / ``move_to_pose`` / …) is
    a mid-loop perception-action behaviour that runs ON the robot's agent_server;
    the harness issues a high-level intent and awaits a single completion verdict
    (see embodiment/base contract). ``call_verb`` is the pure transport primitive
    for that — "POST /verb/{name}, return the raw verdict dict".

    Deliberately kept OFF the core ``EmbodimentAdapter`` Protocol: verbs are
    heterogeneous per robot (some bodies implement only a subset), so "can run
    verbs" is an additive capability, not part of the contract every backend must
    satisfy. Both real-hardware and simulator agent_server adapters implement it;
    a pure mock backend does not, and verb tools fall back to a simulated verdict.

    Owned by the embodiment layer so the dependency points downward:
    ``tools/robot_sdk`` imports this to gate dispatch, not the other way around.

    ``available_verbs`` is the discovery half of the capability: which verbs the
    backend *currently* advertises (live from ``/health``, ADR-019). ``None``
    means the backend does not advertise a verb set (unknown — never prune);
    ``[]`` means it explicitly advertises zero verbs right now.
    """

    async def call_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def available_verbs(self) -> list[str] | None: ...
