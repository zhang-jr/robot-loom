"""Skill Protocol and SkillManifest.

A Skill is a versioned, named composition of Tool calls.  It MUST NOT
call external servers directly — only through ToolRegistry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator, model_validator

from robot_harness.skill.safety_class import SafetyClass

if TYPE_CHECKING:
    from typing import Self

    from robot_harness.runtime.harness_context import HarnessContext
    from robot_harness.tools.base import ToolRegistry


class SkillManifest(BaseModel):
    """Versioned descriptor for a Skill — required at registration time."""

    name: str
    version: str  # SemVer e.g. "1.2.3"
    description: str = ""
    embodiment_compat: list[str] = Field(
        default_factory=list,
        description="Robot types this skill supports, e.g. ['arm', 'humanoid']",
    )
    safety_class: SafetyClass = SafetyClass.MEDIUM
    required_tools: list[str] = Field(default_factory=list)
    data_schema: dict[str, Any] = Field(default_factory=dict)
    owner: str = ""
    tags: list[str] = Field(default_factory=list)
    evolution_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def _valid_semver(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(f"version must be SemVer (X.Y.Z), got '{v}'")
        return v

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Skill name must not be empty")
        return v


class Subtask(BaseModel):
    """Unit of work assigned to a Skill."""

    subtask_id: str
    description: str
    robot_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    parent_task_id: str = ""


class SkillResult(BaseModel):
    """Outcome of a single Skill execution."""

    skill_name: str
    skill_version: str
    subtask_id: str
    success: bool
    outcome: Literal["success", "failure", "rollback", "partial"] = "success"
    message: str = ""
    artifacts: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_outcome(self) -> Self:
        if not self.success and self.outcome == "success":
            self.outcome = "failure"
        return self


class SwapHandle(BaseModel):
    """Returned by hotswap — call rollback() to revert to the previous version."""

    skill_name: str
    old_version: str
    new_version: str
    can_rollback: bool = True


@runtime_checkable
class Skill(Protocol):
    """Versioned composition of Tool calls.

    Rules:
    - Must ONLY call tools via ``tools.get(name).invoke(...)``.
    - Must NOT directly import or call external server SDKs.
    - rollback() must restore pre-execution state if feasible.
    """

    manifest: SkillManifest

    async def can_handle(self, subtask: Subtask, ctx: HarnessContext) -> bool: ...

    async def execute(
        self,
        subtask: Subtask,
        tools: ToolRegistry,
        ctx: HarnessContext,
    ) -> SkillResult: ...

    async def rollback(self, ctx: HarnessContext) -> None: ...
