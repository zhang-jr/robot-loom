"""HarnessContext — top-level runtime context for one agent session."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from robot_harness.config.schema import HarnessConfig
from robot_harness.memory.base import Memory, NullMemory
from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.skill.registry import SkillRegistry
from robot_harness.tools.base import ToolRegistry


@dataclass
class HarnessContext:
    """Bundles all live runtime objects for one harness session.

    Passed to Skills and the AgentLoop so they can call tools, query memory,
    and access the safety envelope without importing singletons.
    """

    config: HarnessConfig
    tool_registry: ToolRegistry
    skill_registry: SkillRegistry
    safety_envelope: SafetyEnvelope
    memory: Memory
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

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
