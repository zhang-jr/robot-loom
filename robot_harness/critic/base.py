"""Critic Protocol — progress evaluation for long-horizon tasks."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from robot_harness.embodiment.base import Frame


class CriticVerdict(BaseModel):
    """Four-state judgment from the Critic."""

    state: Literal["progress", "completion", "failure", "unchanged"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    trace_id: str = ""


@runtime_checkable
class Critic(Protocol):
    """Evaluates task progress.  Must NOT be used as hard ground truth."""

    async def judge(
        self,
        current: Frame,
        reference: Frame | None,
        task_description: str,
    ) -> CriticVerdict: ...
