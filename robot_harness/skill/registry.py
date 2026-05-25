"""SkillRegistry — registration, routing, versioned hotswap, and rollback."""

from __future__ import annotations

import re
from typing import Any

from robot_harness.errors import (
    SkillManifestInvalidError,
    SkillNotFoundError,
    SkillVersionMismatchError,
)
from robot_harness.observability.tracer import tracer
from robot_harness.skill.base import Skill, SkillManifest, SwapHandle


class SkillRegistry:
    """Manages versioned Skill registration and routing.

    Multiple versions of the same skill name may coexist; the most recently
    registered one is the default unless a specific version is requested.
    """

    def __init__(self) -> None:
        # name → {version → Skill}
        self._store: dict[str, dict[str, Skill]] = {}
        # name → currently active version
        self._active: dict[str, str] = {}

    def register(self, skill: Skill) -> None:
        """Register a Skill.  The new version becomes the active default."""
        m = skill.manifest
        try:
            SkillManifest.model_validate(m.model_dump())
        except Exception as exc:
            raise SkillManifestInvalidError(
                f"Skill '{m.name}' manifest invalid: {exc}",
                validation_errors=[str(exc)],
            ) from exc

        if m.name not in self._store:
            self._store[m.name] = {}
        self._store[m.name][m.version] = skill
        self._active[m.name] = m.version
        tracer.event("skill.registered", skill_name=m.name, version=m.version)

    def get(self, name: str, version: str | None = None) -> Skill:
        """Return the active (or specified) Skill version."""
        if name not in self._store:
            raise SkillNotFoundError(f"Skill '{name}' is not registered")
        versions = self._store[name]
        target = version or self._active.get(name, "")
        if target not in versions:
            raise SkillVersionMismatchError(
                f"Skill '{name}' version '{target}' not found; available: {list(versions)}"
            )
        return versions[target]

    def hotswap(self, new_skill: Skill) -> SwapHandle:
        """Atomically replace the active version; returns a rollback handle."""
        name = new_skill.manifest.name
        old_version = self._active.get(name, "")
        self.register(new_skill)
        tracer.event(
            "skill.hotswap",
            skill_name=name,
            old_version=old_version,
            new_version=new_skill.manifest.version,
        )
        return SwapHandle(
            skill_name=name,
            old_version=old_version,
            new_version=new_skill.manifest.version,
            can_rollback=bool(old_version),
        )

    def rollback(self, handle: SwapHandle) -> None:
        """Revert the active version to the one recorded in the swap handle."""
        if not handle.can_rollback:
            raise SkillNotFoundError(
                f"Cannot roll back '{handle.skill_name}': no prior version recorded"
            )
        name = handle.skill_name
        if name not in self._store or handle.old_version not in self._store[name]:
            raise SkillVersionMismatchError(
                f"Rollback target version '{handle.old_version}' not in registry"
            )
        self._active[name] = handle.old_version
        tracer.event(
            "skill.rollback",
            skill_name=name,
            reverted_to=handle.old_version,
        )

    def list(
        self,
        embodiment_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[SkillManifest]:
        """Return manifests of all active skill versions, optionally filtered."""
        manifests: list[SkillManifest] = []
        for name, versions in self._store.items():
            active_ver = self._active.get(name, "")
            if active_ver not in versions:
                continue
            m = versions[active_ver].manifest
            if embodiment_type and embodiment_type not in m.embodiment_compat:
                continue
            if tags:
                if not any(t in m.tags for t in tags):
                    continue
            manifests.append(m)
        return manifests

    def route(self, subtask_description: str, ctx: Any) -> Skill | None:
        """Simple tag/keyword-based routing; returns None if no match."""
        desc_lower = subtask_description.lower()
        for name in list(self._active):
            skill = self._store[name].get(self._active[name])
            if skill and any(
                re.search(rf"\b{re.escape(tag)}\b", desc_lower) for tag in skill.manifest.tags
            ):
                return skill
        return None

    def __len__(self) -> int:
        return len(self._store)
