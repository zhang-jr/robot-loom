"""Unit tests -- ReplanPolicy covering all 4 critic states x consistent/conflict sensor.

4 states: progress / completion / failure / unchanged
4 state x 2 sensor conditions = 8 combinations + CriticDisagreementError path.
"""

from __future__ import annotations

import pytest

from robot_harness.critic.base import CriticVerdict
from robot_harness.critic.replan_policy import ReplanPolicy, SensorHeuristic
from robot_harness.errors import CriticDisagreementError


def _verdict(state: str, confidence: float = 0.8, evidence: str = "test") -> CriticVerdict:
    return CriticVerdict(state=state, confidence=confidence, evidence=evidence)  # type: ignore[arg-type]


def _sensor(unchanged: int = 0, errors: int = 0, elapsed: float = 0.0) -> SensorHeuristic:
    return SensorHeuristic(unchanged_count=unchanged, error_count=errors, elapsed_s=elapsed)


@pytest.fixture()
def policy() -> ReplanPolicy:
    return ReplanPolicy(disagreement_threshold=0.4, max_unchanged=3)


# ---------------------------------------------------------------------------
# completion → always abort (task done)
# ---------------------------------------------------------------------------


def test_completion_consistent_sensor_aborts(policy: ReplanPolicy) -> None:
    action = policy.decide(_verdict("completion"), _sensor())
    assert action == "abort"


def test_completion_with_errors_still_aborts(policy: ReplanPolicy) -> None:
    action = policy.decide(_verdict("completion"), _sensor(errors=5))
    assert action == "abort"


# ---------------------------------------------------------------------------
# failure high-confidence + bad sensor → abort
# ---------------------------------------------------------------------------


def test_failure_high_conf_bad_sensor_aborts(policy: ReplanPolicy) -> None:
    action = policy.decide(
        _verdict("failure", confidence=0.9),
        _sensor(errors=2),  # sensor also shows errors → no disagreement
    )
    assert action == "abort"


# ---------------------------------------------------------------------------
# failure high-confidence + good sensor → CriticDisagreementError → replan
# ---------------------------------------------------------------------------


def test_failure_high_conf_good_sensor_raises_disagreement(policy: ReplanPolicy) -> None:
    with pytest.raises(CriticDisagreementError):
        policy.decide(
            _verdict("failure", confidence=0.9),
            _sensor(unchanged=0, errors=0),  # sensor says all fine
        )


# ---------------------------------------------------------------------------
# failure low-confidence → replan regardless of sensor
# ---------------------------------------------------------------------------


def test_failure_low_conf_replans(policy: ReplanPolicy) -> None:
    action = policy.decide(
        _verdict("failure", confidence=0.1),  # below threshold
        _sensor(),
    )
    assert action == "replan"


def test_failure_low_conf_bad_sensor_still_replans(policy: ReplanPolicy) -> None:
    action = policy.decide(
        _verdict("failure", confidence=0.2),
        _sensor(errors=3),
    )
    assert action == "replan"


# ---------------------------------------------------------------------------
# unchanged → replan when exceeding max_unchanged
# ---------------------------------------------------------------------------


def test_unchanged_below_max_continues(policy: ReplanPolicy) -> None:
    action = policy.decide(_verdict("unchanged"), _sensor(unchanged=1))
    assert action == "continue"


def test_unchanged_at_max_replans(policy: ReplanPolicy) -> None:
    action = policy.decide(_verdict("unchanged"), _sensor(unchanged=3))
    assert action == "replan"


def test_unchanged_above_max_replans(policy: ReplanPolicy) -> None:
    action = policy.decide(_verdict("unchanged"), _sensor(unchanged=10))
    assert action == "replan"


# ---------------------------------------------------------------------------
# progress → continue
# ---------------------------------------------------------------------------


def test_progress_consistent_continues(policy: ReplanPolicy) -> None:
    action = policy.decide(_verdict("progress"), _sensor())
    assert action == "continue"


def test_progress_with_errors_continues(policy: ReplanPolicy) -> None:
    action = policy.decide(_verdict("progress"), _sensor(errors=1))
    assert action == "continue"
