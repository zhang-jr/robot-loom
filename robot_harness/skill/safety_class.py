"""Safety classification for Skills."""

from __future__ import annotations

from enum import StrEnum


class SafetyClass(StrEnum):
    """Safety level of a Skill.

    Determines which SafetyEnvelope rule sets apply before dispatch.
    """

    LOW = "low"  # no hardware contact
    MEDIUM = "medium"  # hardware movement in open space
    HIGH = "high"  # near obstacles or humans
    CRITICAL = "critical"  # contact with environment or payload
