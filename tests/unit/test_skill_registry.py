"""Tests for SkillRegistry — registration, routing, hotswap, rollback."""

from __future__ import annotations

from typing import Any

import pytest

from robot_harness.errors import (
    SkillNotFoundError,
    SkillVersionMismatchError,
)
from robot_harness.skill.base import SkillManifest, SkillResult, Subtask
from robot_harness.skill.registry import SkillRegistry
from robot_harness.skill.safety_class import SafetyClass

# ---------------------------------------------------------------------------
# Minimal Skill fixture
# ---------------------------------------------------------------------------


def _make_skill(name: str = "pick", version: str = "1.0.0", tags: list[str] | None = None) -> Any:
    manifest = SkillManifest(
        name=name,
        version=version,
        description="test",
        embodiment_compat=["arm"],
        safety_class=SafetyClass.MEDIUM,
        required_tools=["echo"],
        tags=tags or [name],
    )

    class _S:
        pass

    s = _S()
    s.manifest = manifest  # type: ignore[attr-defined]

    async def can_handle(subtask: Subtask, ctx: Any) -> bool:
        return True

    async def execute(subtask: Subtask, tools: Any, ctx: Any) -> SkillResult:
        return SkillResult(
            skill_name=name,
            skill_version=version,
            subtask_id=subtask.subtask_id,
            success=True,
        )

    async def rollback(ctx: Any) -> None:
        pass

    s.can_handle = can_handle  # type: ignore[attr-defined]
    s.execute = execute  # type: ignore[attr-defined]
    s.rollback = rollback  # type: ignore[attr-defined]
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_and_get() -> None:
    reg = SkillRegistry()
    reg.register(_make_skill("pick", "1.0.0"))
    skill = reg.get("pick")
    assert skill.manifest.name == "pick"


def test_get_missing_raises() -> None:
    reg = SkillRegistry()
    with pytest.raises(SkillNotFoundError):
        reg.get("ghost")


def test_get_specific_version() -> None:
    reg = SkillRegistry()
    reg.register(_make_skill("pick", "1.0.0"))
    reg.register(_make_skill("pick", "2.0.0"))
    skill_v1 = reg.get("pick", version="1.0.0")
    assert skill_v1.manifest.version == "1.0.0"
    skill_default = reg.get("pick")
    assert skill_default.manifest.version == "2.0.0"


def test_get_nonexistent_version_raises() -> None:
    reg = SkillRegistry()
    reg.register(_make_skill("pick", "1.0.0"))
    with pytest.raises(SkillVersionMismatchError):
        reg.get("pick", version="9.9.9")


def test_hotswap_and_rollback() -> None:
    reg = SkillRegistry()
    reg.register(_make_skill("pick", "1.0.0"))
    handle = reg.hotswap(_make_skill("pick", "2.0.0"))
    assert reg.get("pick").manifest.version == "2.0.0"
    reg.rollback(handle)
    assert reg.get("pick").manifest.version == "1.0.0"


def test_rollback_with_no_prior_version_raises() -> None:
    reg = SkillRegistry()
    reg.register(_make_skill("pick", "1.0.0"))
    handle = reg.hotswap(_make_skill("pick", "2.0.0"))
    handle.old_version = "0.0.0"  # simulate missing prior
    with pytest.raises(SkillVersionMismatchError):
        reg.rollback(handle)


def test_list_returns_active_manifests() -> None:
    reg = SkillRegistry()
    reg.register(_make_skill("pick", "1.0.0"))
    reg.register(_make_skill("place", "1.0.0"))
    manifests = reg.list()
    names = {m.name for m in manifests}
    assert "pick" in names
    assert "place" in names


def test_list_filter_by_embodiment() -> None:
    reg = SkillRegistry()
    reg.register(_make_skill("pick", "1.0.0"))  # embodiment_compat=["arm"]
    manifests = reg.list(embodiment_type="humanoid")
    assert manifests == []
    manifests = reg.list(embodiment_type="arm")
    assert len(manifests) == 1


def test_route_by_keyword() -> None:
    reg = SkillRegistry()
    reg.register(_make_skill("pick", "1.0.0", tags=["pick"]))
    skill = reg.route("pick up the cup", ctx=None)
    assert skill is not None
    assert skill.manifest.name == "pick"


def test_route_no_match_returns_none() -> None:
    reg = SkillRegistry()
    reg.register(_make_skill("pick", "1.0.0", tags=["pick"]))
    skill = reg.route("navigate to kitchen", ctx=None)
    assert skill is None


def test_invalid_manifest_version_raises_on_register() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SkillManifest(
            name="bad",
            version="not-semver",
            embodiment_compat=["arm"],
            safety_class=SafetyClass.LOW,
        )
