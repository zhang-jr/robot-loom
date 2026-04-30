"""ReplanPolicy — combines Critic verdict with sensor heuristics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from robot_harness.critic.base import CriticVerdict
from robot_harness.errors import CriticDisagreementError
from robot_harness.observability.tracer import tracer


class SensorHeuristic(BaseModel):
    """Lightweight sensor-derived signal."""

    elapsed_s: float = 0.0
    unchanged_count: int = 0
    error_count: int = 0


ReplanAction = Literal["continue", "replan", "abort"]


class ReplanPolicy:
    """Decides whether to continue, replan, or abort based on Critic + sensors.

    Logic:
    - completion → abort (task done)
    - failure + high confidence → abort
    - failure + low confidence AND sensor shows progress → raise CriticDisagreementError → replan
    - unchanged (repeated) → replan
    - progress → continue
    """

    def __init__(
        self,
        disagreement_threshold: float = 0.4,
        max_unchanged: int = 3,
    ) -> None:
        self._disagreement_threshold = disagreement_threshold
        self._max_unchanged = max_unchanged

    def decide(
        self,
        verdict: CriticVerdict,
        sensor: SensorHeuristic,
        *,
        trace_id: str = "",
    ) -> ReplanAction:
        action: ReplanAction = "continue"

        if verdict.state == "completion":
            action = "abort"
        elif verdict.state == "failure":
            if verdict.confidence >= self._disagreement_threshold:
                # Cross-check with sensor
                sensor_ok = sensor.error_count == 0 and sensor.unchanged_count < self._max_unchanged
                if sensor_ok:
                    raise CriticDisagreementError(
                        "Critic says failure but sensor heuristic disagrees — triggering replan",
                        trace_id=trace_id,
                    )
                action = "abort"
            else:
                action = "replan"
        elif verdict.state == "unchanged":
            if sensor.unchanged_count >= self._max_unchanged:
                action = "replan"
        # "progress" → continue

        tracer.event(
            "replan_policy.decision",
            critic_state=verdict.state,
            critic_confidence=verdict.confidence,
            sensor_unchanged=sensor.unchanged_count,
            action=action,
            trace_id=trace_id,
        )
        return action
