"""Velocity cap checker extracted from SafetyEnvelope.

Checks that joint or delta commands do not exceed the configured velocity cap.
"""

from __future__ import annotations

from robot_harness.embodiment.base import EmbodimentCommand


def check_velocity_caps(
    cmd: EmbodimentCommand,
    max_velocity_rad_s: float,
) -> list[str]:
    """Return a list of violation messages (empty = passed).

    Args:
        cmd: The command to check.
        max_velocity_rad_s: Maximum allowed absolute velocity per joint.
            Applied to ``command_type in ("joint", "delta")``.
    """
    violations: list[str] = []

    if cmd.command_type not in ("joint", "delta"):
        return violations

    for i, val in enumerate(cmd.values):
        if abs(val) > max_velocity_rad_s:
            violations.append(
                f"joint[{i}] velocity {val:.4f} rad/s > cap {max_velocity_rad_s:.4f} rad/s"
            )

    return violations
