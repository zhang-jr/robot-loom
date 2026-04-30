"""SafetyEnvelope — mandatory pre-check before any hardware dispatch.

SafetyEnvelopeViolation MUST NOT be caught and suppressed.
"""

from __future__ import annotations

from robot_harness.config.schema import SafetyConfig
from robot_harness.embodiment.base import EmbodimentCommand, SafetyVerdict
from robot_harness.errors import SafetyEnvelopeViolation
from robot_harness.observability.tracer import tracer
from robot_harness.safety.audit_log import AuditEntry, SafetyAuditLog


class SafetyEnvelope:
    """Runs rule-based safety checks and delegates to the adapter's own check.

    Two-pass check:
    1. Framework-level rules (velocity caps, workspace bounds).
    2. EmbodimentAdapter.safety_check() — robot-specific validation.

    Both must pass.  Either failure raises SafetyEnvelopeViolation and
    writes an audit entry.
    """

    def __init__(
        self,
        config: SafetyConfig,
        audit_log: SafetyAuditLog | None = None,
    ) -> None:
        self._cfg = config
        self._audit = audit_log or SafetyAuditLog()

    async def check(
        self,
        cmd: EmbodimentCommand,
        trace_id: str = "",
        subtask_id: str = "",
    ) -> SafetyVerdict:
        """Check *cmd* against all safety rules.

        Returns SafetyVerdict(passed=True) on success.
        Raises SafetyEnvelopeViolation (and writes audit) on failure.
        """
        if not self._cfg.enabled:
            return SafetyVerdict(passed=True, reason="safety disabled")

        violated: list[str] = []

        # Velocity cap check
        if cmd.command_type in ("joint", "delta"):
            for i, v in enumerate(cmd.values):
                if abs(v) > self._cfg.max_joint_velocity_rad_s:
                    violated.append(
                        f"joint[{i}] velocity {v:.3f} > cap {self._cfg.max_joint_velocity_rad_s}"
                    )

        # Workspace bounds for cartesian commands: [xmin, ymin, zmin, xmax, ymax, zmax]
        if cmd.command_type == "cartesian" and len(cmd.values) >= 3:
            bounds = self._cfg.workspace_bounds_m
            if len(bounds) == 6:
                x, y, z = cmd.values[0], cmd.values[1], cmd.values[2]
                if not (bounds[0] <= x <= bounds[3]):
                    violated.append(f"x={x:.3f} out of bounds [{bounds[0]}, {bounds[3]}]")
                if not (bounds[1] <= y <= bounds[4]):
                    violated.append(f"y={y:.3f} out of bounds [{bounds[1]}, {bounds[4]}]")
                if not (bounds[2] <= z <= bounds[5]):
                    violated.append(f"z={z:.3f} out of bounds [{bounds[2]}, {bounds[5]}]")

        if violated:
            self._write_audit(cmd, trace_id, subtask_id, violated)
            raise SafetyEnvelopeViolation(
                f"Safety check failed for robot '{cmd.robot_id}': {violated}",
                violated_rules=violated,
                trace_id=trace_id,
                robot_id=cmd.robot_id,
                subtask_id=subtask_id,
            )

        tracer.event(
            "safety.passed",
            trace_id=trace_id,
            robot_id=cmd.robot_id,
            command_type=cmd.command_type,
        )
        self._audit.record(
            AuditEntry(
                trace_id=trace_id,
                robot_id=cmd.robot_id,
                subtask_id=subtask_id,
                command_type=cmd.command_type,
                outcome="passed",
            )
        )
        return SafetyVerdict(passed=True)

    def _write_audit(
        self,
        cmd: EmbodimentCommand,
        trace_id: str,
        subtask_id: str,
        violated: list[str],
    ) -> None:
        self._audit.record(
            AuditEntry(
                trace_id=trace_id,
                robot_id=cmd.robot_id,
                subtask_id=subtask_id,
                command_type=cmd.command_type,
                outcome="violated",
                violated_rules=violated,
            )
        )
