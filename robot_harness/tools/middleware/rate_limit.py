"""RateLimitMiddleware — token-bucket rate limiter with dual (tool, robot) scopes.

Two independent buckets control call rate:
  - per-tool  : shared across all robots calling this tool (``max_rps`` calls/s)
  - per-robot : one bucket per robot_id (``max_rps_per_robot`` calls/s)

When a bucket is exhausted, ``invoke`` raises ``ToolBackendUnreachableError``
(or waits if ``block=True``).  Non-blocking mode is the default so the Brain's
replan logic can handle back-pressure without hanging the event loop.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from robot_harness.errors import ToolBackendUnreachableError
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.middleware.base import ToolMiddleware


class _TokenBucket:
    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate  # tokens per second
        self._capacity = capacity  # max burst
        self._tokens = capacity
        self._last_refill = time.monotonic()

    def consume(self) -> bool:
        """Try to consume one token.  Returns False if depleted."""
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    async def wait_and_consume(self) -> None:
        """Block until a token is available, then consume it."""
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now


class RateLimitMiddleware(ToolMiddleware):
    """Token-bucket rate limiter with per-tool and per-robot buckets."""

    def __init__(
        self,
        inner: Any,
        max_rps: float = 10.0,
        max_rps_per_robot: float = 5.0,
        burst_multiplier: float = 2.0,
        block: bool = False,
    ) -> None:
        super().__init__(inner)
        self._block = block
        self._global_bucket = _TokenBucket(rate=max_rps, capacity=max_rps * burst_multiplier)
        self._max_rps_per_robot = max_rps_per_robot
        self._burst_multiplier = burst_multiplier
        self._robot_buckets: dict[str, _TokenBucket] = {}

    def _get_robot_bucket(self, robot_id: str) -> _TokenBucket:
        if robot_id not in self._robot_buckets:
            self._robot_buckets[robot_id] = _TokenBucket(
                rate=self._max_rps_per_robot,
                capacity=self._max_rps_per_robot * self._burst_multiplier,
            )
        return self._robot_buckets[robot_id]

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if self._block:
            await self._global_bucket.wait_and_consume()
            await self._get_robot_bucket(ctx.robot_id).wait_and_consume()
        else:
            if not self._global_bucket.consume():
                raise ToolBackendUnreachableError(
                    f"Rate limit exceeded for tool '{self._inner.name}' (global)",
                    tool_name=self._inner.name,
                    trace_id=ctx.trace_id,
                    robot_id=ctx.robot_id,
                )
            if not self._get_robot_bucket(ctx.robot_id).consume():
                raise ToolBackendUnreachableError(
                    f"Rate limit exceeded for tool '{self._inner.name}' "
                    f"(robot '{ctx.robot_id}')",
                    tool_name=self._inner.name,
                    trace_id=ctx.trace_id,
                    robot_id=ctx.robot_id,
                )

        return await self._inner.invoke(args, ctx)
