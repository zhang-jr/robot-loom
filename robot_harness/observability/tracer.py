"""Structured JSON tracer — Phase 1 implementation.

Emits newline-delimited JSON to stderr (or a configured file).
Phase 3 will replace the sink with OpenTelemetry spans while keeping the
same call-site API.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any


class Span:
    """Lightweight span context returned by :func:`tracer.span`."""

    def __init__(self, tracer: Tracer, name: str, attrs: dict[str, Any]) -> None:
        self._tracer = tracer
        self._name = name
        self._attrs = attrs
        self._start = time.perf_counter()
        self._extra: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Attach an attribute discovered mid-span."""
        self._extra[key] = value

    def _finish(self, outcome: str = "ok", error: str = "") -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._tracer._emit(
            {
                "span": self._name,
                "outcome": outcome,
                "latency_ms": round(elapsed_ms, 3),
                "error": error,
                **self._attrs,
                **self._extra,
            }
        )


class Tracer:
    """Singleton-style structured tracer for the harness.

    Usage::

        from robot_harness.observability.tracer import tracer

        with tracer.span("tool.invoke", tool_name="yolo", trace_id="abc") as s:
            result = await tool.invoke(args, ctx)
            s.set("outcome", "success")
    """

    def __init__(self) -> None:
        self._sink = sys.stderr

    def configure(self, sink: Any) -> None:
        """Redirect output (e.g. to a file handle or a test buffer)."""
        self._sink = sink

    def _emit(self, record: dict[str, Any]) -> None:
        record["ts"] = datetime.now(tz=UTC).isoformat()
        print(json.dumps(record, default=str), file=self._sink, flush=True)

    def event(self, name: str, **attrs: Any) -> None:
        """Emit a point-in-time event (no duration)."""
        self._emit({"event": name, **attrs})

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Context manager that emits a duration record on exit."""
        s = Span(self, name, attrs)
        try:
            yield s
            s._finish("ok")
        except Exception as exc:
            s._finish("error", error=type(exc).__name__)
            raise


tracer: Tracer = Tracer()
