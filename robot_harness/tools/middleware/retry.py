"""Retry middleware — exponential backoff with jitter for transient failures."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from robot_harness.errors import ToolBackendUnreachableError, ToolTimeoutError
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.middleware.base import ToolMiddleware

_RETRYABLE = (ToolBackendUnreachableError, ToolTimeoutError)


class RetryMiddleware(ToolMiddleware):
    """Retries on transient backend errors with exponential backoff + jitter.

    Only idempotent tools are retried; non-idempotent tools fail immediately.
    """

    def __init__(self, inner: Any, max_retries: int = 3, backoff_base_s: float = 1.0) -> None:
        super().__init__(inner)
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not self._inner.is_idempotent:
            return await self._inner.invoke(args, ctx)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._inner.invoke(args, ctx)
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt < self._max_retries and not ctx.is_cancelled:
                    delay = self._backoff_base_s * (2**attempt) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]
