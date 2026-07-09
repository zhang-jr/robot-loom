"""Boundary guard: verb-internal / override tools must never reach Brain planning.

Grasp-pose estimation, single-step VLA inference, and low-level dispatch are
*parts of an on-robot verb*, not capabilities the Brain should orchestrate
directly. If the Brain could see them it would be tempted to hand-assemble a
slow-layer control pipeline ("detect -> estimate_pose -> dispatch this pose"),
which is exactly the cross-network control loop the layered closed-loop boundary
exists to prevent.

The contract is two-sided and both sides are asserted here:

  * These tools stay *registered and invocable* — a debug session or an explicit
    override skill can still reach them via ``registry.get(...)``.
  * They are *hidden from* ``export_for_brain`` — they are off the default
    planning vocabulary regardless of LLM backend profile.

The positive control matters as much as the negative one: the reactive verbs
(``reactive_grasp`` etc.) ARE the Brain's act surface and MUST stay visible.
If someone "fixes" the boundary by hiding the verbs too, that breaks the agent.
"""

from __future__ import annotations

import pytest

from robot_harness.tools.base import BrainProfile, ToolRegistry
from robot_harness.tools.grasp.anygrasp_adapter import AnyGraspTool
from robot_harness.tools.robot_sdk import RobotSdkTool, build_robot_sdk_verb_tools
from robot_harness.tools.vla.serving_adapter import VlaServingTool

# Sub-capabilities that are verb-internal parts or escape-hatch overrides.
# These must never appear in the Brain's planning vocabulary.
HIDDEN_TOOLS = (
    AnyGraspTool(),  # grasp.estimate_pose — consumed inside the on-robot grasp verb
    VlaServingTool(),  # vla.infer_action — single step = cross-network control loop
    RobotSdkTool(),  # robot_sdk.execute_action — low-level dispatch override
)

# The reactive verbs ARE the Brain's act surface — they must stay visible.
VERB_TOOLS = build_robot_sdk_verb_tools()

_ALL_PROFILES = (
    BrainProfile(name="openai"),
    BrainProfile(name="anthropic"),
    BrainProfile(name="mcp"),
    BrainProfile(name="litellm"),
)


def _exported_names(specs: list[dict]) -> set[str]:
    """Pull tool names out of any backend profile's export shape."""
    names: set[str] = set()
    for spec in specs:
        if "function" in spec:  # openai / litellm
            names.add(spec["function"]["name"])
        else:  # anthropic / mcp
            names.add(spec["name"])
    return names


@pytest.mark.parametrize("tool", HIDDEN_TOOLS, ids=lambda t: t.name)
def test_verb_internal_tool_is_not_brain_visible(tool: object) -> None:
    """The flag itself is set — catches a new tool that forgets to hide."""
    assert getattr(tool, "brain_visible", True) is False, (
        f"{tool.name!r} is a verb-internal/override capability and must set "
        f"brain_visible = False so the Brain cannot hand-assemble a control loop."
    )


@pytest.mark.parametrize("profile", _ALL_PROFILES, ids=lambda p: p.name)
def test_hidden_tools_absent_from_every_brain_profile(profile: BrainProfile) -> None:
    """End-to-end: hidden tools are registered+invocable but never exported."""
    reg = ToolRegistry()
    for tool in (*HIDDEN_TOOLS, *VERB_TOOLS):
        reg.register(tool)

    exported = _exported_names(reg.export_for_brain(profile))

    for tool in HIDDEN_TOOLS:
        # Registered and reachable for debug / override skills...
        assert tool.name in reg
        assert reg.get(tool.name).name == tool.name
        # ...but absent from the Brain's planning vocabulary.
        assert tool.name not in exported, (
            f"{tool.name!r} leaked into the {profile.name!r} planning vocabulary"
        )


@pytest.mark.parametrize("profile", _ALL_PROFILES, ids=lambda p: p.name)
def test_reactive_verbs_stay_visible(profile: BrainProfile) -> None:
    """Positive control: the act surface must NOT be hidden by an over-eager fix."""
    reg = ToolRegistry()
    for tool in (*HIDDEN_TOOLS, *VERB_TOOLS):
        reg.register(tool)

    exported = _exported_names(reg.export_for_brain(profile))

    for tool in VERB_TOOLS:
        assert tool.name in exported, (
            f"{tool.name!r} is a reactive verb (the Brain's act surface) and must "
            f"stay visible in the {profile.name!r} profile."
        )
