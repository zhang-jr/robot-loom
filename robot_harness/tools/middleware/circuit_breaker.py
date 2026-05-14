"""CircuitBreakerMiddleware — prevent cascading failure against unreachable backends."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from robot_harness.errors import ToolBackendUnreachableError
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.middleware.base import ToolMiddleware


class _State(StrEnum):
    CLOSED = "closed"  # normal operation
    OPEN = "open"  # failing fast
    HALF_OPEN = "half_open"  # probing recovery


class CircuitBreakerMiddleware(ToolMiddleware):
    """Opens the circuit after ``failure_threshold`` consecutive backend errors.

    State machine:
      CLOSED  → (N failures) → OPEN  → (reset_timeout_s elapsed) → HALF_OPEN
      HALF_OPEN → (success)  → CLOSED
      HALF_OPEN → (failure)  → OPEN

    When OPEN, ``invoke`` raises ``ToolBackendUnreachableError`` immediately
    without hitting the inner tool.  This prevents hammering a dead backend
    and lets the Brain's retry/replan logic handle the situation.
    """

    def __init__(
        self,
        inner: Any,
        failure_threshold: int = 5,
        reset_timeout_s: float = 30.0,
    ) -> None:
        super().__init__(inner)
        self._failure_threshold = failure_threshold
        self._reset_timeout_s = reset_timeout_s
        self._state = _State.CLOSED
        self._failure_count = 0
        self._opened_at: float = 0.0

    @property
    def state(self) -> str:
        return self._state.value

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self._maybe_transition_to_half_open()

        if self._state == _State.OPEN:
            raise ToolBackendUnreachableError(
                f"Circuit OPEN for tool '{self._inner.name}' — backend in cooldown",
                tool_name=self._inner.name,
                trace_id=ctx.trace_id,
                robot_id=ctx.robot_id,
            )

        try:
            result = await self._inner.invoke(args, ctx)
            self._on_success()
            return result
        except ToolBackendUnreachableError:
            self._on_failure()
            raise

    def _maybe_transition_to_half_open(self) -> None:
        if self._state == _State.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._reset_timeout_s:
                self._state = _State.HALF_OPEN

    def _on_success(self) -> None:
        self._failure_count = 0
        self._state = _State.CLOSED

    def _on_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold or self._state == _State.HALF_OPEN:
            self._state = _State.OPEN
            self._opened_at = time.monotonic()
