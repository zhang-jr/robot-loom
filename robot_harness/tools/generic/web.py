"""WebFetchTool — generic HTTP fetch tool (NativeTool)."""

from __future__ import annotations

import time
from typing import Any

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class WebFetchTool:
    """Fetch content from a URL and return the response body as text."""

    name = "web_fetch"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="web_fetch",
        description="Perform an HTTP GET request and return the response body.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timeout_s": {"type": "number", "default": 10},
                "headers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["url"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "status_code": {"type": "integer"},
                "body": {"type": "string"},
                "content_type": {"type": "string"},
            },
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return True

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        import asyncio
        import urllib.request

        url = args["url"]
        timeout = float(args.get("timeout_s", 10))
        headers: dict[str, str] = args.get("headers", {})
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url, headers=headers)

            def _fetch() -> tuple[int, str, str]:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read(1_048_576).decode(errors="replace")
                    return resp.status, body, resp.headers.get("Content-Type", "")

            status, body, content_type = await asyncio.to_thread(_fetch)
            latency = (time.monotonic() - t0) * 1000
            return ToolResult(
                tool_name=self.name,
                trace_id=ctx.trace_id,
                success=(200 <= status < 300),
                output={"status_code": status, "body": body, "content_type": content_type},
                latency_ms=latency,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_name=self.name,
                trace_id=ctx.trace_id,
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    async def cancel(self, ctx: ToolContext) -> None:
        ctx.cancel()
