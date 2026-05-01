"""Timeout-based heuristic fallback when Critic service is unavailable."""

from __future__ import annotations

import time
from typing import Literal

from robot_harness.critic.base import CriticVerdict
from robot_harness.embodiment.base import Frame
from robot_harness.observability.tracer import tracer


class HeuristicCritic:
    """Fallback critic that returns 'progress' until timeout, then 'failure'.

    Activates when CriticServiceDown is raised by the primary critic.
    Always logs a warning — operators must know this is degraded.
    """

    def __init__(self, task_timeout_s: float = 120.0) -> None:
        self._timeout = task_timeout_s
        self._start = time.monotonic()

    async def judge(
        self,
        current: Frame,
        reference: Frame | None,
        task_description: str,
    ) -> CriticVerdict:
        elapsed = time.monotonic() - self._start
        tracer.event("critic.heuristic_fallback", elapsed_s=round(elapsed, 1))

        state: Literal["progress", "completion", "failure", "unchanged"]
        if elapsed < self._timeout:
            state = "progress"
            evidence = f"heuristic: {elapsed:.0f}s elapsed, timeout at {self._timeout}s"
            confidence = 0.3
        else:
            state = "failure"
            evidence = f"heuristic: task exceeded timeout of {self._timeout}s"
            confidence = 0.5

        return CriticVerdict(
            state=state,
            confidence=confidence,
            evidence=evidence,
        )
