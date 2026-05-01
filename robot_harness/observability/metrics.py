"""Prometheus metrics stubs — Phase 1 placeholder.

Phase 3 will wire these to a real prometheus_client registry.
Keeping the API stable here so call sites don't change.
"""

from __future__ import annotations

from typing import Any


class _NoopMetric:
    def labels(self, **_kw: Any) -> _NoopMetric:
        return self

    def inc(self, amount: float = 1.0) -> None:
        pass

    def observe(self, value: float) -> None:
        pass

    def set(self, value: float) -> None:
        pass


_NOOP = _NoopMetric()

tool_invocations_total = _NOOP
tool_latency_seconds = _NOOP
skill_executions_total = _NOOP
brain_decisions_total = _NOOP
safety_checks_total = _NOOP
critic_verdicts_total = _NOOP
