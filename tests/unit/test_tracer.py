"""Tests for the dual-mode OTel + JSON tracer."""

from __future__ import annotations

import io
import json

import pytest

from robot_harness.observability.tracer import Tracer


@pytest.fixture()
def tracer() -> Tracer:
    t = Tracer()
    t.configure(io.StringIO())
    return t


def _records(tracer: Tracer) -> list[dict]:  # type: ignore[type-arg]
    buf = tracer._sink
    buf.seek(0)
    return [json.loads(line) for line in buf if line.strip()]


class TestJsonMode:
    def test_span_emits_record(self, tracer: Tracer) -> None:
        with tracer.span("test.op", tool="yolo") as s:
            s.set("result", "ok")
        records = _records(tracer)
        assert len(records) == 1
        r = records[0]
        assert r["span"] == "test.op"
        assert r["tool"] == "yolo"
        assert r["result"] == "ok"
        assert r["outcome"] == "ok"
        assert r["latency_ms"] >= 0

    def test_span_records_error(self, tracer: Tracer) -> None:
        with pytest.raises(ValueError):
            with tracer.span("fail.op"):
                raise ValueError("boom")
        r = _records(tracer)[0]
        assert r["outcome"] == "error"
        assert r["error"] == "ValueError"

    def test_event_emits_record(self, tracer: Tracer) -> None:
        tracer.event("something.happened", x=1, y="two")
        records = _records(tracer)
        assert records[0]["event"] == "something.happened"
        assert records[0]["x"] == 1

    def test_nested_spans(self, tracer: Tracer) -> None:
        with tracer.span("outer"):
            with tracer.span("inner"):
                pass
        records = _records(tracer)
        assert len(records) == 2
        assert records[0]["span"] == "inner"
        assert records[1]["span"] == "outer"

    def test_ts_field_present(self, tracer: Tracer) -> None:
        with tracer.span("ts.check"):
            pass
        assert "ts" in _records(tracer)[0]


class TestOtelMode:
    """OTel integration using in-memory exporter — no real backend needed."""

    def _make_otel_tracer(self) -> tuple[Tracer, list]:  # type: ignore[type-arg]
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        # Use provider directly to avoid touching the global singleton,
        # which cannot be overridden after first set.
        otel_tracer = provider.get_tracer("test")

        t = Tracer()
        t.configure(io.StringIO())
        t._otel_tracer = otel_tracer
        return t, exporter.get_finished_spans  # type: ignore[return-value]

    def test_span_creates_otel_span(self) -> None:
        tracer, get_spans = self._make_otel_tracer()
        with tracer.span("otel.op", tool="x"):
            pass
        spans = get_spans()
        assert len(spans) == 1
        assert spans[0].name == "otel.op"

    def test_span_attributes_forwarded(self) -> None:
        tracer, get_spans = self._make_otel_tracer()
        with tracer.span("otel.attrs", tool="yolo") as s:
            s.set("extra_key", "extra_val")
        span = get_spans()[0]
        assert span.attributes.get("tool") == "yolo"
        assert span.attributes.get("extra_key") == "extra_val"

    def test_span_error_sets_otel_status(self) -> None:
        from opentelemetry.trace import StatusCode

        tracer, get_spans = self._make_otel_tracer()
        with pytest.raises(RuntimeError):
            with tracer.span("otel.err"):
                raise RuntimeError("bad")
        span = get_spans()[0]
        assert span.status.status_code == StatusCode.ERROR

    def test_json_still_emitted_in_otel_mode(self) -> None:
        tracer, _ = self._make_otel_tracer()
        with tracer.span("dual.mode"):
            pass
        records = _records(tracer)
        assert len(records) == 1
        assert records[0]["span"] == "dual.mode"


class TestInitFromConfig:
    def test_no_endpoint_stays_json(self) -> None:
        from robot_harness.config.schema import ObservabilityConfig

        t = Tracer()
        t.configure(io.StringIO())
        t.init_from_config(ObservabilityConfig(otel_endpoint=""))
        assert t._otel_tracer is None

    def test_invalid_endpoint_degrades_gracefully(self) -> None:
        t = Tracer()
        t.configure(io.StringIO())
        # Bad endpoint — configure_otel should catch and degrade.
        t.configure_otel(endpoint="http://localhost:0/bad")
        # _otel_tracer may be set (OTel SDK doesn't validate eagerly) but
        # tracer must not crash during normal usage.
        with t.span("safe.op"):
            pass
