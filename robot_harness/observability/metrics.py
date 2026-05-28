"""Prometheus metrics — real when prometheus_client is installed, noops otherwise.

Emits live metrics with per-robot / per-skill / per-tool label dimensions.
Export via the built-in HTTP server or mount on the FastAPI /metrics endpoint.

Install: pip install prometheus-client
"""

from __future__ import annotations

from typing import Any

try:
    from prometheus_client import Counter, Histogram, start_http_server

    _HAVE_PROMETHEUS = True
except ImportError:
    _HAVE_PROMETHEUS = False


# ---------------------------------------------------------------------------
# Metric definitions (real or noop)
# ---------------------------------------------------------------------------

if _HAVE_PROMETHEUS:
    tool_invocations_total = Counter(
        "robot_loom_tool_invocations_total",
        "Total tool invocations",
        ["tool_name", "robot_id", "outcome"],
    )
    tool_latency_seconds = Histogram(
        "robot_loom_tool_latency_seconds",
        "Tool invocation latency in seconds",
        ["tool_name", "robot_id"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    skill_executions_total = Counter(
        "robot_loom_skill_executions_total",
        "Total skill executions",
        ["skill_name", "skill_version", "robot_id", "outcome"],
    )
    brain_decisions_total = Counter(
        "robot_loom_brain_decisions_total",
        "Total Brain.decide() calls",
        ["robot_id", "decision_type"],
    )
    safety_checks_total = Counter(
        "robot_loom_safety_checks_total",
        "Total SafetyEnvelope.check() calls",
        ["robot_id", "outcome"],
    )
    critic_verdicts_total = Counter(
        "robot_loom_critic_verdicts_total",
        "Total Critic.judge() verdicts",
        ["robot_id", "state", "source"],
    )
else:

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

    tool_invocations_total = _NOOP  # type: ignore[assignment]
    tool_latency_seconds = _NOOP  # type: ignore[assignment]
    skill_executions_total = _NOOP  # type: ignore[assignment]
    brain_decisions_total = _NOOP  # type: ignore[assignment]
    safety_checks_total = _NOOP  # type: ignore[assignment]
    critic_verdicts_total = _NOOP  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# HTTP server helper
# ---------------------------------------------------------------------------


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus HTTP server exposing /metrics.

    No-ops if prometheus_client is not installed.  Logs to stderr.
    """
    if not _HAVE_PROMETHEUS:
        import sys

        print(
            f"[robot-loom] prometheus_client not installed — metrics server not started "
            f"(would listen on :{port})",
            file=sys.stderr,
        )
        return
    start_http_server(port)
    import sys

    print(
        f"[robot-loom] Prometheus metrics available at http://localhost:{port}/metrics",
        file=sys.stderr,
    )


def metrics_available() -> bool:
    """Return True when prometheus_client is installed."""
    return _HAVE_PROMETHEUS
