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

# The Brain-facing input shape every skill call is unpacked with. A custom
# ``data_schema`` must keep this envelope: the export side hands the schema to
# the Brain verbatim, while dispatch unconditionally reads these three keys —
# a flat schema would make the Brain pass flat args that dispatch silently
# drops. Skill-specific fields belong inside ``parameters``.
SUBTASK_ENVELOPE_KEYS = frozenset({"robot_id", "description", "parameters"})


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

    @field_validator("data_schema")
    @classmethod
    def _data_schema_keeps_subtask_envelope(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            return v
        if v.get("type") != "object":
            raise ValueError(
                "data_schema must be an object-typed JSON Schema (it is exported "
                "verbatim as the Brain-facing input schema)"
            )
        extra = set(v.get("properties", {})) - SUBTASK_ENVELOPE_KEYS
        if extra:
            raise ValueError(
                "data_schema top-level properties must stay within the Subtask "
                f"envelope {sorted(SUBTASK_ENVELOPE_KEYS)}; skill-specific fields "
                f"go inside 'parameters'. Offending: {sorted(extra)}"
            )
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

    Skill selection is the Brain's job: skills enter its planning vocabulary as
    ``skill.<name>`` callables (SkillRegistry.export_for_brain) and are invoked
    by name — there is no keyword-routing entry point.
    """

    manifest: SkillManifest

    async def execute(
        self,
        subtask: Subtask,
        tools: ToolRegistry,
        ctx: HarnessContext,
    ) -> SkillResult: ...

    async def rollback(self, ctx: HarnessContext) -> None: ...
