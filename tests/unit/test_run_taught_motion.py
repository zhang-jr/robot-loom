"""run_taught_motion — discovery, Brain vocabulary, and the safety boundary.

The verb's whole reason to exist is a split: the Brain gets a NAME, the
SafetyEnvelope gets every JOINT VALUE. These tests pin both halves, because
either one silently collapsing is a real failure mode — a name enum that never
populates makes the verb uncallable, and a safety hook that returns nothing
makes a taught motion the one hardware path with no pre-dispatch gate.

Safety tests use the real SafetyEnvelope (never a mock, per the safety-testing
rule) so the joint-limit check is the production one.
"""

from __future__ import annotations

from typing import Any

import pytest

from robot_harness.config.schema import SafetyConfig
from robot_harness.errors import SafetyEnvelopeViolation
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.robot_sdk import (
    ROBOT_SDK_RUN_TAUGHT_MOTION,
    RunTaughtMotionTool,
    TaughtMotionCatalog,
)
from robot_harness.tools.robot_sdk.taught_motion import parse_taught_motions

PLACE_PAYLOAD: list[dict[str, Any]] = [
    {
        "name": "place_on_tray",
        "description": "release the held object onto the tray and retreat",
        "points": [
            {"joints": [0.0, 0.5, -0.5, 0.0, 1.0, 0.0], "gripper": 2.0},
            {"joints": [0.0, 0.9, -0.7, 0.0, 1.1, 0.0], "gripper": 2.0},
            {"joints": [0.0, 0.9, -0.7, 0.0, 1.1, 0.0], "gripper": 5.0},
        ],
    }
]


def _catalog(robot_id: str = "arm1", payload: Any = None) -> TaughtMotionCatalog:
    catalog = TaughtMotionCatalog()
    catalog.replace(robot_id, parse_taught_motions(payload or PLACE_PAYLOAD))
    return catalog


def _ctx(robot_id: str = "arm1") -> ToolContext:
    return ToolContext.create(robot_id=robot_id)


# ---------------------------------------------------------------------------
# discovery / parsing
# ---------------------------------------------------------------------------


def test_parse_drops_entries_without_points() -> None:
    """A motion the envelope cannot check is one the harness will not offer —
    dropping it beats exposing a name whose trajectory is unknown."""
    motions = parse_taught_motions([{"name": "nameless_points", "points": []}, *PLACE_PAYLOAD])
    assert [m.name for m in motions] == ["place_on_tray"]


def test_parse_survives_a_malformed_entry() -> None:
    """Discovery must never take the fleet down: one bad entry costs that entry."""
    motions = parse_taught_motions(
        [{"name": "bad", "points": [{"joints": "elbow"}]}, *PLACE_PAYLOAD]
    )
    assert [m.name for m in motions] == ["place_on_tray"]


def test_parse_rejects_a_non_list_payload() -> None:
    assert parse_taught_motions({"motions": PLACE_PAYLOAD}) == []


def test_catalog_names_are_the_fleet_union() -> None:
    """The tool spec is shared across robots with robot_id as an argument, so
    pruning to one robot's names would hide another robot's vocabulary."""
    catalog = _catalog("arm1")
    catalog.replace(
        "arm2",
        parse_taught_motions([{"name": "stow_left_bin", "points": [{"joints": [0.0] * 6}]}]),
    )
    assert catalog.names() == ["place_on_tray", "stow_left_bin"]


def test_forget_drops_a_robots_motions() -> None:
    catalog = _catalog()
    catalog.forget("arm1")
    assert catalog.is_empty()
    assert catalog.names() == []


# ---------------------------------------------------------------------------
# Brain vocabulary — names in, joint angles out
# ---------------------------------------------------------------------------


def test_schema_offers_names_as_an_enum_never_joint_angles() -> None:
    tool = RunTaughtMotionTool({}, _catalog())
    props = tool.schema.input_schema["properties"]
    assert props["motion"]["enum"] == ["place_on_tray"]
    # The Brain never sees a joint value: transcribing floats is the operation
    # this verb exists to remove.
    assert "target_joints" not in props
    assert "0.9" not in str(props["motion"])


def test_schema_reflects_a_refreshed_catalog_without_re_registering() -> None:
    """Re-teaching happens on the robot; the harness must pick it up on the next
    /health, not on the next deploy."""
    catalog = _catalog()
    tool = RunTaughtMotionTool({}, catalog)
    assert tool.schema.input_schema["properties"]["motion"]["enum"] == ["place_on_tray"]

    catalog.replace(
        "arm1", parse_taught_motions([{"name": "stow_left_bin", "points": [{"joints": [0.0] * 6}]}])
    )
    assert tool.schema.input_schema["properties"]["motion"]["enum"] == ["stow_left_bin"]


def test_schema_omits_the_enum_when_nothing_is_taught() -> None:
    """An empty enum would be an unsatisfiable schema; the loop gates the tool
    out of the vocabulary entirely in this state instead."""
    tool = RunTaughtMotionTool({}, TaughtMotionCatalog())
    assert "enum" not in tool.schema.input_schema["properties"]["motion"]


def test_registry_rejects_an_unknown_motion_name_before_dispatch() -> None:
    """The enum earns its keep here: an invented name is a schema violation the
    Brain is told to fix, not a round trip to a robot that will reject it."""
    from robot_harness.errors import ToolSchemaViolationError

    registry = ToolRegistry()
    registry.register(RunTaughtMotionTool({}, _catalog()))
    registry.validate_args(
        ROBOT_SDK_RUN_TAUGHT_MOTION, {"robot_id": "arm1", "motion": "place_on_tray"}
    )
    with pytest.raises(ToolSchemaViolationError):
        registry.validate_args(
            ROBOT_SDK_RUN_TAUGHT_MOTION, {"robot_id": "arm1", "motion": "invented_motion"}
        )


# ---------------------------------------------------------------------------
# safety boundary — every waypoint checked BEFORE the first one moves
# ---------------------------------------------------------------------------


def test_safety_commands_cover_every_waypoint() -> None:
    registry = ToolRegistry()
    registry.register(RunTaughtMotionTool({}, _catalog()))
    cmds = registry.build_safety_commands(
        ROBOT_SDK_RUN_TAUGHT_MOTION, {"robot_id": "arm1", "motion": "place_on_tray"}, _ctx()
    )
    assert len(cmds) == 3
    assert {c.command_type for c in cmds} == {"joint"}
    assert cmds[0].values == [0.0, 0.5, -0.5, 0.0, 1.0, 0.0]
    assert cmds[2].values == [0.0, 0.9, -0.7, 0.0, 1.1, 0.0]


@pytest.mark.asyncio
async def test_a_later_out_of_limit_waypoint_is_refused_up_front() -> None:
    """The gate's whole value over per-point dispatch: point 3 being out of
    limits must refuse the call, not stop the arm after points 1-2 have run."""
    payload = [
        {
            "name": "bad_place",
            "points": [
                {"joints": [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]},
                {"joints": [0.0, 0.6, 0.0, 0.0, 0.0, 0.0]},
                {"joints": [0.0, 9.0, 0.0, 0.0, 0.0, 0.0]},  # beyond the limit
            ],
        }
    ]
    registry = ToolRegistry()
    registry.register(RunTaughtMotionTool({}, _catalog(payload=payload)))
    envelope = SafetyEnvelope(SafetyConfig(joint_limits_rad=[3.14] * 6))

    cmds = registry.build_safety_commands(
        ROBOT_SDK_RUN_TAUGHT_MOTION, {"robot_id": "arm1", "motion": "bad_place"}, _ctx()
    )
    with pytest.raises(SafetyEnvelopeViolation):
        for cmd in cmds:
            await envelope.check(cmd, trace_id="t", subtask_id="s")


def test_unknown_motion_yields_no_safety_command_not_a_fabricated_one() -> None:
    """Empty means 'cannot check' → the caller records a skipped audit. Inventing
    a command here would put a fake trajectory in the safety log."""
    registry = ToolRegistry()
    registry.register(RunTaughtMotionTool({}, _catalog()))
    assert (
        registry.build_safety_commands(
            ROBOT_SDK_RUN_TAUGHT_MOTION, {"robot_id": "arm1", "motion": "never_taught"}, _ctx()
        )
        == []
    )


def test_motion_taught_on_another_robot_is_not_checked_against_this_one() -> None:
    """Same name, different body: the points are per-robot, so addressing arm2
    must not validate arm1's trajectory."""
    registry = ToolRegistry()
    registry.register(RunTaughtMotionTool({}, _catalog("arm1")))
    assert (
        registry.build_safety_commands(
            ROBOT_SDK_RUN_TAUGHT_MOTION,
            {"robot_id": "arm2", "motion": "place_on_tray"},
            _ctx("arm2"),
        )
        == []
    )


def test_verb_is_hardware_bound_and_gated() -> None:
    registry = ToolRegistry()
    registry.register(RunTaughtMotionTool({}, _catalog()))
    assert registry.requires_safety_check(ROBOT_SDK_RUN_TAUGHT_MOTION)


# ---------------------------------------------------------------------------
# discovery wiring — HarnessContext.refresh_taught_motions
# ---------------------------------------------------------------------------


class _StubTaughtAdapter:
    """Minimal SupportsTaughtMotions stand-in."""

    def __init__(self, payload: Any, *, offline: bool = False) -> None:
        self._payload = payload
        self._offline = offline

    async def call_verb(self, verb: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"outcome": "success"}

    async def available_verbs(self) -> list[str] | None:
        return None

    async def taught_motions(self) -> list[dict[str, Any]]:
        if self._offline:
            from robot_harness.errors import RobotOfflineError

            raise RobotOfflineError("stub /health unreachable", robot_id="arm1")
        return self._payload


@pytest.mark.asyncio
async def test_refresh_populates_the_catalog_from_health() -> None:
    ctx = HarnessContext.build()
    ctx.embodiment_adapters = {"arm1": _StubTaughtAdapter(PLACE_PAYLOAD)}  # type: ignore[dict-item]
    await ctx.refresh_taught_motions()
    assert ctx.taught_motions is not None
    assert ctx.taught_motions.names() == ["place_on_tray"]

    # …and the registered tool sees it, because the catalog is shared by reference.
    tool = ctx.tool_registry.get(ROBOT_SDK_RUN_TAUGHT_MOTION)
    assert tool.schema.input_schema["properties"]["motion"]["enum"] == ["place_on_tray"]


@pytest.mark.asyncio
async def test_offline_robot_is_forgotten_not_left_stale() -> None:
    """Stale points are worse than none: the robot may have been re-taught while
    unreachable, and the envelope would then check a trajectory that no longer
    exists."""
    ctx = HarnessContext.build()
    ctx.embodiment_adapters = {"arm1": _StubTaughtAdapter(PLACE_PAYLOAD)}  # type: ignore[dict-item]
    await ctx.refresh_taught_motions()
    assert ctx.taught_motions is not None and not ctx.taught_motions.is_empty()

    ctx.embodiment_adapters = {"arm1": _StubTaughtAdapter(None, offline=True)}  # type: ignore[dict-item]
    await ctx.refresh_taught_motions()
    assert ctx.taught_motions.is_empty()


@pytest.mark.asyncio
async def test_backend_advertising_nothing_leaves_the_catalog_empty() -> None:
    ctx = HarnessContext.build()
    ctx.embodiment_adapters = {"arm1": _StubTaughtAdapter([])}  # type: ignore[dict-item]
    await ctx.refresh_taught_motions()
    assert ctx.taught_motions is not None
    assert ctx.taught_motions.is_empty()
