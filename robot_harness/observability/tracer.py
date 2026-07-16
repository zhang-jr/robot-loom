"""Structured tracer — JSON-stderr default, upgradeable to OpenTelemetry.

Call-site API::

    from robot_harness.observability.tracer import tracer

    with tracer.span("tool.invoke", tool_name="yolo", trace_id="abc") as s:
        result = await tool.invoke(args, ctx)
        s.set("outcome", "success")

Enable OTel by calling ``tracer.configure_otel(endpoint=...)`` or via::

    tracer.init_from_config(config.observability)

When the OTLP endpoint is set, spans are exported as real OTel spans and
``event()`` adds OTel events on the current span. Without an endpoint the
tracer falls back to newline-delimited JSON on stderr.
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
    """Lightweight span returned by :func:`Tracer.span`.

    In OTel mode the ``_otel_span`` reference is set; ``set()`` writes to both
    the JSON record and the live OTel span so the output is always symmetric.
    """

    def __init__(
        self,
        tracer: Tracer,
        name: str,
        attrs: dict[str, Any],
        otel_span: Any = None,
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._attrs = attrs
        self._start = time.perf_counter()
        self._extra: dict[str, Any] = {}
        self._otel_span = otel_span

    def set(self, key: str, value: Any) -> None:
        """Attach an attribute discovered mid-span."""
        self._extra[key] = value
        if self._otel_span is not None:
            self._otel_span.set_attribute(key, str(value))

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

    Default mode: newline-delimited JSON to stderr.
    OTel mode: spans exported via OTLP when ``configure_otel`` is called.
    Both modes can be active simultaneously — JSON provides a local audit
    trail while OTel forwards to Langfuse / Jaeger.
    """

    def __init__(self) -> None:
        self._sink = sys.stderr
        self._otel_tracer: Any = None  # opentelemetry.trace.Tracer | None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, sink: Any) -> None:
        """Redirect JSON output (e.g. to a file handle or a test buffer)."""
        self._sink = sink

    def configure_otel(
        self,
        endpoint: str,
        service_name: str = "robot-loom",
    ) -> None:
        """Set up an OTLP exporter pointing at *endpoint*.

        Idempotent — calling again replaces the previous provider.
        Silently skips if the OTel SDK is not installed (unlikely since it is
        a hard dep, but guards against import errors in constrained envs).
        """
        if not endpoint:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource(attributes={"service.name": service_name})
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            trace.set_tracer_provider(provider)
            self._otel_tracer = trace.get_tracer(service_name)
        except Exception:  # noqa: BLE001
            # OTel setup failed — degrade gracefully to JSON-only mode.
            self._otel_tracer = None

    def init_from_config(self, config: Any) -> None:
        """Wire tracer from an ``ObservabilityConfig`` instance.

        Accepts the Pydantic model or any object with ``trace_sink``,
        ``trace_file``, and ``otel_endpoint`` attributes.
        """
        if getattr(config, "trace_sink", "stderr") == "file":
            path = getattr(config, "trace_file", "")
            if path:
                old = self._sink
                self._sink = open(path, "a", encoding="utf-8")
                if old is not sys.stderr:
                    old.close()
        endpoint = getattr(config, "otel_endpoint", "")
        if endpoint:
            self.configure_otel(endpoint)

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------

    def _emit(self, record: dict[str, Any]) -> None:
        record["ts"] = datetime.now(tz=UTC).isoformat()
        print(json.dumps(record, default=str, ensure_ascii=False), file=self._sink, flush=True)

    def event(self, name: str, **attrs: Any) -> None:
        """Emit a point-in-time event (no duration).

        In OTel mode the event is added to the current active span (if any)
        *and* emitted as a JSON record for local audit.
        """
        if self._otel_tracer is not None:
            try:
                from opentelemetry import trace

                current = trace.get_current_span()
                if current is not None:
                    current.add_event(name, {k: str(v) for k, v in attrs.items()})
            except Exception:  # noqa: BLE001
                pass
        self._emit({"event": name, **attrs})

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Context manager that emits a duration record on exit.

        In OTel mode, a real OTel span is created around the body; status and
        exceptions are propagated to the span.  The JSON record is still emitted
        for local audit.
        """
        if self._otel_tracer is not None:
            yield from self._span_otel(name, attrs)
        else:
            yield from self._span_json(name, attrs)

    def _span_json(self, name: str, attrs: dict[str, Any]) -> Iterator[Span]:
        s = Span(self, name, attrs)
        try:
            yield s
            s._finish("ok")
        except Exception as exc:
            s._finish("error", error=type(exc).__name__)
            raise

    def _span_otel(self, name: str, attrs: dict[str, Any]) -> Iterator[Span]:
        from opentelemetry import trace
        from opentelemetry.trace import StatusCode

        with self._otel_tracer.start_as_current_span(name) as otel_span:
            for k, v in attrs.items():
                otel_span.set_attribute(k, str(v))
            s = Span(self, name, attrs, otel_span=otel_span)
            try:
                yield s
                otel_span.set_status(StatusCode.OK)
                s._finish("ok")
            except Exception as exc:
                otel_span.set_status(StatusCode.ERROR, str(exc))
                otel_span.record_exception(exc)
                s._finish("error", error=type(exc).__name__)
                raise
        # suppress the unused import warning — trace is used via StatusCode
        _ = trace


tracer: Tracer = Tracer()
