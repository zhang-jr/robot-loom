"""Skill lifecycle: activate, rollback, and audit trail."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from robot_harness.observability.tracer import tracer


class SkillAuditEntry(BaseModel):
    skill_name: str
    skill_version: str
    action: Literal["register", "activate", "rollback", "execute", "fail"]
    actor: str = "harness"
    note: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class SkillAuditLog:
    """In-memory audit log for skill lifecycle events (Phase 3: persist to disk)."""

    def __init__(self) -> None:
        self._entries: list[SkillAuditEntry] = []

    def record(self, entry: SkillAuditEntry) -> None:
        self._entries.append(entry)
        tracer.event(
            "skill.audit",
            skill_name=entry.skill_name,
            version=entry.skill_version,
            action=entry.action,
            note=entry.note,
        )

    def entries(self) -> list[SkillAuditEntry]:
        return list(self._entries)
