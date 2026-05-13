"""Cognitive Scaffold — session-scoped meta-cognitive tools for the Brain.

Plan and reflection stores that survive context compression and are re-injected
into the Brain's context window.  These are in-process, per-AgentLoop tools;
they are NOT Memory (cross-session persistence), NOT Critic (physical-world
progress judgement), and NOT Skill (capability composition).  See ADR-018.
"""

from robot_harness.tools.cognitive.base import CognitiveScaffoldStore
from robot_harness.tools.cognitive.plan_tool import PlannerStore, PlanTool
from robot_harness.tools.cognitive.reflection_tool import ReflectionStore, ReflectionTool

__all__ = [
    "CognitiveScaffoldStore",
    "PlanTool",
    "PlannerStore",
    "ReflectionStore",
    "ReflectionTool",
]
