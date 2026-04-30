"""Safety audit log — 100% persistent record of every safety check."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from robot_harness.config.paths import workspace_logs_dir
from robot_harness.observability.tracer import tracer


class AuditEntry(BaseModel):
    trace_id: str
    robot_id: str
    subtask_id: str = ""
    tool_name: str = ""
    outcome: Literal["passed", "violated"]
    violated_rules: list[str] = Field(default_factory=list)
    command_type: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class SafetyAuditLog:
    """Appends every safety verdict to a newline-delimited JSON file.

    File: ~/.robot-loom/workspace/logs/safety_audit.jsonl
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (workspace_logs_dir() / "safety_audit.jsonl")

    def record(self, entry: AuditEntry) -> None:
        tracer.event(
            "safety.audit",
            trace_id=entry.trace_id,
            robot_id=entry.robot_id,
            outcome=entry.outcome,
            violated_rules=entry.violated_rules,
        )
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")

    def tail(self, n: int = 20) -> list[AuditEntry]:
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        return [AuditEntry.model_validate(json.loads(line)) for line in lines[-n:]]
