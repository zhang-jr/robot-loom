"""HTTP adapter for the VLA recipe fallback endpoint."""

from __future__ import annotations

import time
from typing import Any

import httpx

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema

VLA_INFER_ACTION = "vla.infer_action"

_ACTION_VECTOR_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 7,
    "maxItems": 7,
}

_INFER_ACTION_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "image_source": {"type": "string"},
        "instruction": {"type": "string"},
        "robot_id": {"type": "string"},
        "num_actions": {"type": "integer", "default": 1, "minimum": 1, "maximum": 16},
    },
    "required": ["image_source", "instruction", "robot_id"],
}

_INFER_ACTION_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "actions": {"type": "array", "items": _ACTION_VECTOR_SCHEMA},
        "model_id": {"type": "string"},
    },
    "required": ["actions", "model_id"],
}


class VlaHttpTool:
    """Call ``vla.infer_action`` through a simple HTTP JSON endpoint."""

    backend: ToolBackend = "http"

    def __init__(self, endpoint_url: str) -> None:
        if not endpoint_url:
            raise ValueError("VlaHttpTool requires non-empty endpoint_url")
        self._endpoint_url = endpoint_url
        self._schema = ToolSchema(
            name=VLA_INFER_ACTION,
            description="Infer robot action deltas via a VLA recipe HTTP endpoint.",
            input_schema=_INFER_ACTION_INPUT,
            output_schema=_INFER_ACTION_OUTPUT,
        )

    @property
    def name(self) -> str:
        return VLA_INFER_ACTION

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    @property
    def is_idempotent(self) -> bool:
        return False

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=ctx.timeout_s) as client:
                response = await client.post(self._endpoint_url, json=args)
                response.raise_for_status()
                output = response.json()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=self.name,
                trace_id=ctx.trace_id,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output=output,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        ctx.cancel()
