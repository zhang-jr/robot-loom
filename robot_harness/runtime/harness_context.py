"""HarnessContext — top-level runtime context for one agent session."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from robot_harness.config.schema import HarnessConfig
from robot_harness.memory.base import Memory, NullMemory
from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.skill.registry import SkillRegistry
from robot_harness.tools.base import BrainProfile, ToolRegistry
from robot_harness.tools.cognitive.base import CognitiveScaffoldStore
from robot_harness.tools.cognitive.plan_tool import PlannerStore, PlanTool
from robot_harness.tools.cognitive.reflection_tool import ReflectionStore, ReflectionTool


@dataclass
class HarnessContext:
    """Bundles all live runtime objects for one harness session.

    Passed to Skills and the AgentLoop so they can call tools, query memory,
    and access the safety envelope without importing singletons.

    ``scaffold_stores`` is keyed by robot_id.  For fleet_size=1 the single
    robot's stores live under its robot_id.  The AgentLoop calls
    ``format_for_injection()`` on each store before Brain.decide() when the
    context has been compressed.
    """

    config: HarnessConfig
    tool_registry: ToolRegistry
    skill_registry: SkillRegistry
    safety_envelope: SafetyEnvelope
    memory: Memory
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # robot_id → list of cognitive stores (plan, reflection, …)
    scaffold_stores: dict[str, list[CognitiveScaffoldStore]] = field(default_factory=dict)

    def get_scaffold_stores(self, robot_id: str) -> list[CognitiveScaffoldStore]:
        """Return (or lazily create) the cognitive scaffold stores for a robot."""
        if robot_id not in self.scaffold_stores:
            self.scaffold_stores[robot_id] = []
        return self.scaffold_stores[robot_id]

    def inject_cognitive_scaffold(
        self,
        robot_id: str,
        brain_profile: BrainProfile,
    ) -> None:
        """Register PlanTool (and optionally ReflectionTool) for a robot.

        Registers both the NativeTool in the ToolRegistry AND adds the backing
        store to scaffold_stores so the AgentLoop can call format_for_injection().

        Call once per robot at session start.
        """
        stores = self.get_scaffold_stores(robot_id)
        store_names = {s.store_name for s in stores}

        if "plan" not in store_names:
            planner = PlannerStore()
            plan_tool = PlanTool(planner)
            self.tool_registry.register(plan_tool)
            stores.append(planner)

        if not brain_profile.supports_native_reflection and "reflection" not in store_names:
            reflection = ReflectionStore()
            reflection_tool = ReflectionTool(reflection)
            self.tool_registry.register(reflection_tool)
            stores.append(reflection)

    def format_scaffold_for_injection(self, robot_id: str) -> str | None:
        """Collect and concatenate all non-None scaffold injections for a robot."""
        parts = [
            s.format_for_injection() for s in self.get_scaffold_stores(robot_id) if s.has_content()
        ]
        non_none = [p for p in parts if p is not None]
        return "\n\n".join(non_none) if non_none else None

    @classmethod
    def build(
        cls,
        config: HarnessConfig | None = None,
        *,
        memory: Memory | None = None,
    ) -> HarnessContext:
        """Convenience factory — wires up default implementations."""
        from robot_harness.config.loader import load_config

        cfg = config or load_config()
        tool_registry = ToolRegistry()
        skill_registry = SkillRegistry()
        safety_envelope = SafetyEnvelope(cfg.safety)

        return cls(
            config=cfg,
            tool_registry=tool_registry,
            skill_registry=skill_registry,
            safety_envelope=safety_envelope,
            memory=memory or NullMemory(),
        )
