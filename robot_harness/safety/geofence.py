"""Map-frame geofence checker for locomotion goals.

Validates that a locomotion goal pose stays inside the configured map-frame
bounding box. Yaw is not bounded — only the [x, y] position is fenced.
"""

from __future__ import annotations

from robot_harness.embodiment.base import EmbodimentCommand


def check_map_geofence(
    cmd: EmbodimentCommand,
    bounds_m: list[float],
) -> list[str]:
    """Return a list of violation messages (empty = passed).

    Args:
        cmd: The command to check. Only ``command_type == "locomotion"`` is
            inspected; values layout is ``[x, y]`` or ``[x, y, yaw]`` in the
            map frame (see EmbodimentCommand docstring).
        bounds_m: Map-frame geofence ``[xmin, ymin, xmax, ymax]`` in meters.
    """
    violations: list[str] = []

    if cmd.command_type != "locomotion":
        return violations

    if not 2 <= len(cmd.values) <= 3:
        violations.append(
            f"locomotion goal must be [x, y] or [x, y, yaw], got {len(cmd.values)} values"
        )
        return violations

    x, y = cmd.values[0], cmd.values[1]
    xmin, ymin, xmax, ymax = bounds_m
    if not xmin <= x <= xmax:
        violations.append(f"goal x={x:.3f} outside map geofence [{xmin}, {xmax}]")
    if not ymin <= y <= ymax:
        violations.append(f"goal y={y:.3f} outside map geofence [{ymin}, {ymax}]")

    return violations
