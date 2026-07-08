"""SafetyEnvelope — mandatory pre-check before any hardware dispatch.

SafetyEnvelopeViolation MUST NOT be caught and suppressed.
"""

from __future__ import annotations

from collections.abc import Mapping

from robot_harness.config.schema import SafetyConfig
from robot_harness.embodiment.base import EmbodimentAdapter, EmbodimentCommand, SafetyVerdict
from robot_harness.errors import EmbodimentError, SafetyEnvelopeViolation
from robot_harness.observability.tracer import tracer
from robot_harness.safety.audit_log import AuditEntry, SafetyAuditLog
from robot_harness.safety.geofence import check_map_geofence
from robot_harness.safety.joint_limits import check_joint_limits


class SafetyEnvelope:
    """Runs rule-based safety checks and delegates to the adapter's own check.

    Two-pass check:
    1. Framework-level rules (joint position limits, delta step cap, workspace
       bounds, map geofence). Command ``values`` semantics follow
       :class:`EmbodimentCommand` — a "joint" command carries target POSITIONS,
       never velocities; velocity constraints are enforced on-robot.
    2. ``EmbodimentAdapter.safety_check()`` — robot-specific validation, when the
       envelope was constructed with the fleet's ``adapters`` map and the command's
       robot has one.

    Both must pass.  Either failure raises SafetyEnvelopeViolation and writes an
    audit entry. A rule the configuration cannot express (no joint limits, no
    geofence, …) is recorded as an honest ``"skipped"`` — never a false
    ``"passed"`` — unless the adapter pass actually inspected the command.
    """

    def __init__(
        self,
        config: SafetyConfig,
        audit_log: SafetyAuditLog | None = None,
        adapters: Mapping[str, EmbodimentAdapter] | None = None,
    ) -> None:
        self._cfg = config
        self._audit = audit_log or SafetyAuditLog()
        self._adapters = adapters or {}

    async def check(
        self,
        cmd: EmbodimentCommand,
        trace_id: str = "",
        subtask_id: str = "",
    ) -> SafetyVerdict:
        """Check *cmd* against all safety rules.

        Returns SafetyVerdict(passed=True) on success.
        Raises SafetyEnvelopeViolation (and writes audit) on failure.
        Raises the adapter's EmbodimentError (after a "skipped" audit) when the
        robot's own check is unreachable — a recoverable robot fault, not a
        violation; the command is still never dispatched.
        """
        if not self._cfg.enabled:
            return SafetyVerdict(passed=True, reason="safety disabled")

        violated, skip_reason = self._framework_rules(cmd)
        if violated:
            self._write_audit(cmd, trace_id, subtask_id, violated)
            raise SafetyEnvelopeViolation(
                f"Safety check failed for robot '{cmd.robot_id}': {violated}",
                violated_rules=violated,
                trace_id=trace_id,
                robot_id=cmd.robot_id,
                subtask_id=subtask_id,
            )

        adapter_checked = await self._adapter_pass(cmd, trace_id, subtask_id)

        # Framework had no applicable rule AND no robot-specific check ran:
        # record the honest skip instead of a false "passed".
        if skip_reason and not adapter_checked:
            return self._skip_unchecked(cmd, trace_id, subtask_id, reason=skip_reason)

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

    def _framework_rules(self, cmd: EmbodimentCommand) -> tuple[list[str], str]:
        """Pass 1 — configuration-driven rules, one rule set per command type.

        Returns ``(violations, skip_reason)``. A non-empty ``skip_reason`` means
        no configured rule could inspect this command (honest-skip candidate);
        it is only meaningful when ``violations`` is empty. Every command type
        resolves to exactly one of: checked, violated, or skipped — a type no
        rule exists for (e.g. ``hand_grasp``) is a skip, never a silent pass
        (ISS-036).
        """
        if cmd.command_type == "joint":
            return self._rule_joint(cmd)
        if cmd.command_type == "delta":
            return self._rule_delta(cmd)
        if cmd.command_type == "cartesian":
            return self._rule_cartesian(cmd)
        if cmd.command_type == "locomotion":
            return self._rule_locomotion(cmd)
        return [], (
            f"no framework rule exists for '{cmd.command_type}' commands — "
            "on-robot runtime is authoritative"
        )

    def _rule_joint(self, cmd: EmbodimentCommand) -> tuple[list[str], str]:
        """Joint position limits: values are target joint POSITIONS (radians)."""
        limits = self._cfg.joint_limits_rad
        if not limits:
            return [], (
                "no joint limits configured — joint target unchecked, "
                "on-robot runtime limits are authoritative"
            )
        return check_joint_limits(cmd, limits), ""

    def _rule_delta(self, cmd: EmbodimentCommand) -> tuple[list[str], str]:
        """Delta step cap: values are per-axis incremental displacement."""
        cap = self._cfg.max_delta_step
        if cap <= 0:
            return [], (
                "no delta step cap configured — delta step unchecked, "
                "on-robot runtime is authoritative"
            )
        return [
            f"delta[{i}] step {v:.3f} exceeds cap ±{cap}"
            for i, v in enumerate(cmd.values)
            if abs(v) > cap
        ], ""

    def _rule_cartesian(self, cmd: EmbodimentCommand) -> tuple[list[str], str]:
        """Workspace bounds: [xmin, ymin, zmin, xmax, ymax, zmax]."""
        if len(cmd.values) < 3:
            # A target that is not at least [x, y, z] cannot be validated —
            # reject it (same policy as a malformed locomotion goal).
            return [
                f"cartesian command has {len(cmd.values)} values — expected at least [x, y, z]"
            ], ""
        bounds = self._cfg.workspace_bounds_m
        if len(bounds) != 6:
            if bounds:
                tracer.event(
                    "safety.rule_misconfigured",
                    rule="workspace_bounds_m",
                    warning=f"expected 6 values, got {len(bounds)} — rule cannot run",
                )
            return [], (
                "no valid workspace bounds configured — cartesian target "
                "unchecked, on-robot runtime is authoritative"
            )
        violated: list[str] = []
        x, y, z = cmd.values[0], cmd.values[1], cmd.values[2]
        if not (bounds[0] <= x <= bounds[3]):
            violated.append(f"x={x:.3f} out of bounds [{bounds[0]}, {bounds[3]}]")
        if not (bounds[1] <= y <= bounds[4]):
            violated.append(f"y={y:.3f} out of bounds [{bounds[1]}, {bounds[4]}]")
        if not (bounds[2] <= z <= bounds[5]):
            violated.append(f"z={z:.3f} out of bounds [{bounds[2]}, {bounds[5]}]")
        return violated, ""

    def _rule_locomotion(self, cmd: EmbodimentCommand) -> tuple[list[str], str]:
        """Map-frame geofence for locomotion goals: [xmin, ymin, xmax, ymax]."""
        map_bounds = self._cfg.map_bounds_m
        if len(map_bounds) != 4:
            if map_bounds:
                tracer.event(
                    "safety.rule_misconfigured",
                    rule="map_bounds_m",
                    warning=f"expected 4 values, got {len(map_bounds)} — rule cannot run",
                )
            return [], (
                "no map geofence configured — locomotion goal unchecked, "
                "on-robot nav stack is authoritative"
            )
        return check_map_geofence(cmd, map_bounds), ""

    async def _adapter_pass(
        self,
        cmd: EmbodimentCommand,
        trace_id: str,
        subtask_id: str,
    ) -> bool:
        """Pass 2 — the robot's own check (``EmbodimentAdapter.safety_check``).

        Returns True when the robot's check ran and passed, False when no adapter
        is registered for the command's robot. A refusal raises
        SafetyEnvelopeViolation; an unreachable robot check writes a "skipped"
        audit entry and re-raises the typed EmbodimentError (recoverable robot
        fault upstream — the command is never dispatched either way).
        """
        adapter = self._adapters.get(cmd.robot_id)
        if adapter is None:
            return False
        try:
            verdict = await adapter.safety_check(cmd)
        except EmbodimentError:
            tracer.event(
                "safety.adapter_check_unreachable",
                trace_id=trace_id,
                robot_id=cmd.robot_id,
                command_type=cmd.command_type,
                warning="robot safety_check unreachable — command refused, not dispatched",
            )
            self._audit.record(
                AuditEntry(
                    trace_id=trace_id,
                    robot_id=cmd.robot_id,
                    subtask_id=subtask_id,
                    command_type=cmd.command_type,
                    outcome="skipped",
                )
            )
            raise
        if not verdict.passed:
            rules = verdict.violated_rules or [verdict.reason or "robot safety_check refused"]
            self._write_audit(cmd, trace_id, subtask_id, rules)
            raise SafetyEnvelopeViolation(
                f"Robot safety check failed for robot '{cmd.robot_id}': {rules}",
                violated_rules=rules,
                trace_id=trace_id,
                robot_id=cmd.robot_id,
                subtask_id=subtask_id,
            )
        return True

    def note_skipped(
        self,
        *,
        tool_name: str,
        robot_id: str,
        trace_id: str = "",
        subtask_id: str = "",
        reason: str,
    ) -> None:
        """Audit a hardware-bound dispatch that produced no checkable command.

        Called by the dispatch gates when a hardware-bound tool's safety-command
        extraction returns None (e.g. ``reactive_grasp`` given only a phrase
        hint — the on-robot reflex is authoritative). The envelope never ran, and
        that fact must be auditable: without this entry, "checked and passed"
        and "never checked" would be indistinguishable after the fact.
        """
        tracer.event(
            "safety.skipped",
            trace_id=trace_id,
            robot_id=robot_id,
            tool_name=tool_name,
            warning=reason,
        )
        self._audit.record(
            AuditEntry(
                trace_id=trace_id,
                robot_id=robot_id,
                subtask_id=subtask_id,
                tool_name=tool_name,
                outcome="skipped",
            )
        )

    def _skip_unchecked(
        self,
        cmd: EmbodimentCommand,
        trace_id: str,
        subtask_id: str,
        reason: str,
    ) -> SafetyVerdict:
        """Record that no configured rule could check *cmd* — NOT a "passed"."""
        tracer.event(
            "safety.skipped",
            trace_id=trace_id,
            robot_id=cmd.robot_id,
            command_type=cmd.command_type,
            warning=reason,
        )
        self._audit.record(
            AuditEntry(
                trace_id=trace_id,
                robot_id=cmd.robot_id,
                subtask_id=subtask_id,
                command_type=cmd.command_type,
                outcome="skipped",
            )
        )
        return SafetyVerdict(passed=True, reason=reason)

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
