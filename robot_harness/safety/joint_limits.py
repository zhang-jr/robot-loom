"""Joint limit checker extracted from SafetyEnvelope.

Currently a basic ±limit check per joint.
"""

from __future__ import annotations

from robot_harness.embodiment.base import EmbodimentCommand

# TODO (ADR-007): add per-robot URDF-based limits.


def check_joint_limits(
    cmd: EmbodimentCommand,
    limits_rad: list[float],
) -> list[str]:
    """Return a list of violation messages (empty = passed).

    Args:
        cmd: The command to check.
        limits_rad: Symmetric joint position limits in radians, one per joint.
            Checked only for ``command_type == "joint"``.
    """
    violations: list[str] = []

    if cmd.command_type != "joint":
        return violations

    for i, (val, lim) in enumerate(zip(cmd.values, limits_rad, strict=False)):
        if abs(val) > lim:
            violations.append(f"joint[{i}] position {val:.4f} rad exceeds limit ±{lim:.4f} rad")

    return violations
