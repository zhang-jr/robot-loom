"""Live-capability gating (ADR-019): fleet-union verb discovery → Brain export.

Covers the chain HarnessContext.unavailable_tool_names → ToolRegistry /
SkillRegistry export filtering:

  * fleet UNION semantics — a verb tool is excluded only when NO robot
    currently advertises it (fleet_size=1 degenerates to per-robot gating)
  * conservative unknowns — any non-advertising backend (None), unreachable
    /health, or non-verb adapter disables gating entirely
  * explicit-zero — every robot advertising [] gates all verb tools
  * skill layer — a skill whose required_tools depend on a gated verb is
    hidden by the same set, so the Brain cannot side-step the gate through a
    skill wrapper
"""

from __future__ import annotations

from typing import Any

import pytest

from robot_harness.errors import RobotOfflineError
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.skill.builtin.navigate_to import NavigateToSkill
from robot_harness.skill.builtin.pick import PickSkill
from robot_harness.skill.builtin.place import PlaceSkill
from robot_harness.skill.registry import SkillRegistry
from robot_harness.tools.base import BrainProfile, ToolRegistry
from robot_harness.tools.robot_sdk import (
    ROBOT_SDK_MOVE_JOINTS,
    ROBOT_SDK_MOVE_TO_POSE,
    ROBOT_SDK_REACTIVE_GRASP,
    ROBOT_SDK_VISUAL_SERVO_TO,
    VERB_TOOL_NAMES,
)


class _StubVerbAdapter:
    """Minimal SupportsVerbs stand-in: advertises a fixed verb set."""

    def __init__(self, verbs: list[str] | None, *, offline: bool = False) -> None:
        self._verbs = verbs
        self._offline = offline
        self.probes = 0

    async def call_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"outcome": "success"}

    async def available_verbs(self) -> list[str] | None:
        self.probes += 1
        if self._offline:
            raise RobotOfflineError("stub /health unreachable", robot_id="stub")
        return self._verbs


class _NoVerbAdapter:
    """An adapter without the SupportsVerbs capability (no verb methods)."""


def _ctx(adapters: dict[str, Any]) -> HarnessContext:
    ctx = HarnessContext.build()
    ctx.embodiment_adapters = adapters
    return ctx


# ---------------------------------------------------------------------------
# HarnessContext.unavailable_tool_names — fleet union
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_robot_gates_unadvertised_verbs() -> None:
    ctx = _ctx({"go2": _StubVerbAdapter(["locomote_to", "home"])})
    excluded = await ctx.unavailable_tool_names()
    assert excluded == {
        ROBOT_SDK_REACTIVE_GRASP,
        ROBOT_SDK_VISUAL_SERVO_TO,
        ROBOT_SDK_MOVE_TO_POSE,
        ROBOT_SDK_MOVE_JOINTS,
    }


@pytest.mark.asyncio
async def test_fleet_union_keeps_verbs_any_robot_advertises() -> None:
    """task.robot_id is an argument on verb calls, so the shared spec list must
    keep a verb as long as ANY robot can run it."""
    ctx = _ctx(
        {
            "go2": _StubVerbAdapter(["locomote_to", "home"]),
            "arm1": _StubVerbAdapter(
                ["reactive_grasp", "move_to_pose", "move_joints", "visual_servo_to"]
            ),
        }
    )
    assert await ctx.unavailable_tool_names() == frozenset()


@pytest.mark.asyncio
async def test_unknown_backend_disables_gating() -> None:
    """One robot not advertising a verb set (None) → availability unknowable →
    never prune (call-time verdicts stay authoritative)."""
    ctx = _ctx(
        {
            "go2": _StubVerbAdapter(["locomote_to"]),
            "arm1": _StubVerbAdapter(None),
        }
    )
    assert await ctx.unavailable_tool_names() == frozenset()


@pytest.mark.asyncio
async def test_offline_health_disables_gating() -> None:
    """Unreachable /health degrades to no gating, not a planning-time crash."""
    ctx = _ctx(
        {
            "go2": _StubVerbAdapter(["locomote_to"]),
            "arm1": _StubVerbAdapter(["reactive_grasp"], offline=True),
        }
    )
    assert await ctx.unavailable_tool_names() == frozenset()


@pytest.mark.asyncio
async def test_non_verb_adapter_disables_gating() -> None:
    ctx = _ctx({"go2": _StubVerbAdapter(["locomote_to"]), "conveyor": _NoVerbAdapter()})
    assert await ctx.unavailable_tool_names() == frozenset()


@pytest.mark.asyncio
async def test_explicit_zero_verbs_gates_everything() -> None:
    """Every robot explicitly advertising [] (all bridges down) → all verb
    tools leave the Brain's vocabulary."""
    ctx = _ctx({"go2": _StubVerbAdapter([]), "arm1": _StubVerbAdapter([])})
    assert await ctx.unavailable_tool_names() == frozenset(VERB_TOOL_NAMES)


@pytest.mark.asyncio
async def test_default_build_mock_backend_never_gates() -> None:
    """The dev/CI default (mock agent_server client) does not advertise a verb
    set, so gating stays off and every verb remains plannable end-to-end."""
    ctx = HarnessContext.build()
    assert await ctx.unavailable_tool_names() == frozenset()


@pytest.mark.asyncio
async def test_no_verb_tools_registered_skips_probes() -> None:
    """Deployments without robot_sdk verb tools must not pay the /health
    round-trip on every task start."""
    adapter = _StubVerbAdapter(["home"])
    ctx = _ctx({"r0": adapter})
    ctx.tool_registry = ToolRegistry()  # nothing gate-able registered
    assert await ctx.unavailable_tool_names() == frozenset()
    assert adapter.probes == 0


# ---------------------------------------------------------------------------
# SkillRegistry export — skills gated by the same unavailable set
# ---------------------------------------------------------------------------


def _exported_names(specs: list[dict[str, Any]]) -> set[str]:
    return {s["function"]["name"] for s in specs}


def test_skill_requiring_gated_verb_is_hidden_from_export() -> None:
    reg = SkillRegistry()
    for skill in (PickSkill(), PlaceSkill(), NavigateToSkill()):
        reg.register(skill)

    profile = BrainProfile(name="openai")
    # Arm bridge down: reactive_grasp gated → skill.pick must disappear too,
    # while skills whose required_tools are unaffected stay exported.
    specs = reg.export_for_brain(profile, unavailable_tools={ROBOT_SDK_REACTIVE_GRASP})
    names = _exported_names(specs)
    assert "skill.pick" not in names
    assert "skill.place" in names
    assert "skill.navigate_to" in names

    # No gate → everything exported (skills stay registered while hidden).
    assert "skill.pick" in _exported_names(reg.export_for_brain(profile))
