"""Multi-robot fleet coordination (ADR-008 / ADR-028).

Single-robot is the fleet_size=1 special case — there is no separate single-robot
path. The coordinator hands out exclusive leases, routes subtasks to compatible
bodies, and broadcasts real-time state; it owns no Brain and runs no control loop
(per-robot AgentLoops live in runtime/, one per leased robot, ADR-028).
"""

from __future__ import annotations

from robot_harness.fleet.coordinator import FleetCoordinator
from robot_harness.fleet.lease import LeaseManager, RobotLease
from robot_harness.fleet.routing import FleetRouter, RobotAssignment
from robot_harness.fleet.shared_bus import FleetBus, FleetEvent

__all__ = [
    "FleetBus",
    "FleetCoordinator",
    "FleetEvent",
    "FleetRouter",
    "LeaseManager",
    "RobotAssignment",
    "RobotLease",
]
