"""Workspace standing-context assembly for the Brain's system prompt (ADR-035).

``MISSION.md`` and ``ROBOT.md`` are deployment-time operator assets (ADR-015):
standing orders / decision rules, and fleet/site prose the harness cannot
discover by itself (map-frame semantics like "the table is at [1.5, 0.0]",
no-go areas, speed etiquette). They are injected verbatim into the opening
system prompt of every task — the role a project instructions file plays for a
coding agent. Files are re-read per task, so a long-running session (heartbeat)
picks up operator edits without a restart.

Boundary — what does NOT belong in these files:

* Runtime world state. Observations / decisions / outcomes go to Memory
  (structured store + explicit query, ADR-024); splicing dynamic state into
  the prompt is the anti-pattern CLAUDE.md forbids.
* Verb / tool capability lists. Capabilities are discovered live from
  ``/health.available_verbs`` (ADR-031); a prose copy drifts the moment a
  backend changes.
* Fixed tool-call sequences. Hardcoding orderings in the prompt is the
  anti-pattern; a recurring procedure belongs in a versioned Skill.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from robot_harness.config.paths import workspace_mission_file, workspace_robot_file
from robot_harness.observability.tracer import tracer

_OVERLAY_HEADER = "Operator standing context (deployment workspace assets, injected verbatim):"

# (section title, path resolver) — order is the injection order.
_SECTIONS: tuple[tuple[str, Callable[[], Path]], ...] = (
    ("Mission (MISSION.md)", workspace_mission_file),
    ("Fleet & site notes (ROBOT.md)", workspace_robot_file),
)

# Soft budget: the overlay rides in EVERY task's system prompt. Above this we
# warn (never truncate — the files are user assets and silent mutilation would
# be worse than the cost); the operator should trim the workspace files.
OVERLAY_WARN_CHARS = 20_000


def workspace_prompt_overlay(trace_id: str = "") -> str | None:
    """Concatenate the workspace standing-context files into one prompt block.

    Missing or blank files are skipped; returns None when nothing is present
    (no workspace, fresh workspace) so callers can inject nothing at all.
    Injection is traced (which files, how many chars) — the operator can see
    from the trace exactly what standing context a task ran with.
    """
    parts: list[str] = []
    injected_files: list[str] = []
    for title, path_for in _SECTIONS:
        path = path_for()
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        parts.append(f"## {title}\n\n{text}")
        injected_files.append(path.name)
    if not parts:
        return None
    overlay = _OVERLAY_HEADER + "\n\n" + "\n\n".join(parts)
    tracer.event(
        "brain.workspace_overlay",
        trace_id=trace_id,
        files=injected_files,
        chars=len(overlay),
    )
    if len(overlay) > OVERLAY_WARN_CHARS:
        tracer.event(
            "brain.workspace_overlay_oversize",
            trace_id=trace_id,
            chars=len(overlay),
            warn_chars=OVERLAY_WARN_CHARS,
        )
    return overlay


__all__ = ["OVERLAY_WARN_CHARS", "workspace_prompt_overlay"]
