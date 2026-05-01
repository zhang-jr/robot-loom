"""EmbodimentAdapter Protocol — the harness interface to robot hardware."""

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


class EmbodimentCommand(BaseModel):
    """Unified command sent to the EmbodimentAdapter.dispatch()."""

    robot_id: str
    command_type: Literal["joint", "cartesian", "delta", "locomotion", "hand_grasp"]
    values: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    """Interface to a specific robot's control layer.

    All dispatch() calls MUST be preceded by SafetyEnvelope.check().
    Implementations live in embodiment/{arm,humanoid,quadruped,mobile}/.
    """

    robot_id: str
    robot_type: RobotType

    async def get_camera_frame(self, camera: str) -> Frame: ...
    async def get_state(self) -> RobotState: ...
    async def dispatch(self, cmd: EmbodimentCommand) -> DispatchHandle: ...
    async def safety_check(self, cmd: EmbodimentCommand) -> SafetyVerdict: ...
