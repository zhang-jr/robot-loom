"""HarnessContext — top-level runtime context for one agent session."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from robot_harness.config.schema import HarnessConfig
from robot_harness.embodiment.base import (
    EmbodimentAdapter,
    Frame,
    SupportsTaughtMotions,
    SupportsVerbs,
)
from robot_harness.errors import RobotOfflineError
from robot_harness.memory.base import Memory
from robot_harness.observability.tracer import tracer
from robot_harness.safety.envelope import SafetyEnvelope
from robot_harness.skill.registry import SkillRegistry
from robot_harness.tools.artifacts import ArtifactStore, InMemoryArtifactStore
from robot_harness.tools.base import BrainProfile, ToolRegistry
from robot_harness.tools.cognitive.base import CognitiveScaffoldStore
from robot_harness.tools.cognitive.plan_tool import PlannerStore, PlanTool
from robot_harness.tools.cognitive.reflection_tool import ReflectionStore, ReflectionTool
from robot_harness.tools.robot_sdk.taught_motion import TaughtMotionCatalog, parse_taught_motions


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
    # Session-scoped out-of-band store for large tool I/O (images, depth, masks).
    artifact_store: ArtifactStore = field(default_factory=InMemoryArtifactStore)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # robot_id → list of cognitive stores (plan, reflection, …)
    scaffold_stores: dict[str, list[CognitiveScaffoldStore]] = field(default_factory=dict)
    # robot_id → EmbodimentAdapter (populated at session start for each robot)
    embodiment_adapters: dict[str, EmbodimentAdapter] = field(default_factory=dict)
    # Shared by reference with RunTaughtMotionTool: refreshed from /health at task
    # start, read by the tool for both its name enum and its safety commands.
    # None when no verb tools were built (nothing reads it).
    taught_motions: TaughtMotionCatalog | None = None

    def get_adapter(self, robot_id: str) -> EmbodimentAdapter | None:
        """Return the EmbodimentAdapter for *robot_id*, or None if not registered."""
        return self.embodiment_adapters.get(robot_id)

    async def get_camera_frame(self, robot_id: str, camera: str = "") -> Frame:
        """Fetch a camera frame from the robot, or return an empty frame if no adapter.

        An empty ``camera`` resolves to the robot's first camera declared in the
        fleet config (cameras are a per-robot capability, ADR-008/ADR-009);
        "wrist" is only the last-resort fallback when the config declares none.
        """
        adapter = self.get_adapter(robot_id)
        if not camera:
            declared = self.config.embodiments.get(robot_id)
            cams = declared.cameras if declared is not None else []
            camera = cams[0] if cams else "wrist"
        if adapter is not None:
            return await adapter.get_camera_frame(camera)
        return Frame(camera=camera, robot_id=robot_id)

    async def unavailable_tool_names(self, trace_id: str = "") -> frozenset[str]:
        """Tool names to exclude from Brain exports because no robot can run them now.

        Live capability discovery for hardware-backed tools (ADR-019): probes every
        registered adapter's ``available_verbs()`` (``/health.available_verbs``) in
        parallel and takes the FLEET UNION — a verb tool is excluded only when *no*
        robot currently advertises it. The Brain plans over one shared tool-spec
        list where ``robot_id`` is an argument, so per-robot pruning would remove
        vocabulary another robot legitimately supports (fleet_size=1 degenerates to
        plain per-robot gating).

        Conservative by construction — returns ``frozenset()`` (exclude nothing,
        call-time verdicts stay authoritative) whenever availability is unknowable:
        any adapter without the ``SupportsVerbs`` capability, any backend that does
        not advertise a verb set (``available_verbs() is None``), or any unreachable
        ``/health`` (``RobotOfflineError`` — warned via tracer; the first real
        dispatch surfaces the offline error to the Brain with full context).

        Feed the result to ``ToolRegistry.export_for_brain(exclude_names=...)`` and
        ``SkillRegistry.export_for_brain(unavailable_tools=...)`` so atomic verbs
        and the skills that require them are gated by the same set.
        """
        from robot_harness.tools.robot_sdk.verbs import (
            VERB_TOOL_NAMES,
            unavailable_verb_tool_names,
        )

        if not self.embodiment_adapters:
            return frozenset()
        # No verb tools registered → nothing to gate; skip the network probes.
        if not any(name in self.tool_registry for name in VERB_TOOL_NAMES):
            return frozenset()
        verb_adapters: list[tuple[str, SupportsVerbs]] = []
        for rid, adapter in self.embodiment_adapters.items():
            if not isinstance(adapter, SupportsVerbs):
                return frozenset()
            verb_adapters.append((rid, adapter))

        async def probe(robot_id: str, adapter: SupportsVerbs) -> list[str] | None:
            try:
                return await adapter.available_verbs()
            except RobotOfflineError as exc:
                tracer.event(
                    "harness.verb_probe_offline",
                    trace_id=trace_id,
                    robot_id=robot_id,
                    reason=str(exc),
                )
                return None

        results = await asyncio.gather(*(probe(rid, a) for rid, a in verb_adapters))
        if any(r is None for r in results):
            return frozenset()
        fleet_verbs = {verb for r in results if r is not None for verb in r}
        excluded = unavailable_verb_tool_names(fleet_verbs)
        if excluded:
            tracer.event(
                "harness.verb_gate",
                trace_id=trace_id,
                robot_ids=sorted(rid for rid, _ in verb_adapters),
                fleet_available_verbs=sorted(fleet_verbs),
                excluded_tools=sorted(excluded),
            )
        return excluded

    async def refresh_taught_motions(self, trace_id: str = "") -> None:
        """Re-read every robot's taught-motion catalog from ``/health``.

        Called at task start, next to the verb gate, because both answer the
        same question ("what can this fleet do right now?") from the same
        endpoint. Kept as its own method rather than folded into
        :meth:`unavailable_tool_names` so neither one lies about what it does:
        that one computes an exclusion set, this one refreshes shared state.

        The catalog object is shared with ``RunTaughtMotionTool``, so refreshing
        it updates both the Brain's name enum and the envelope's view of the
        waypoints in one step. A robot whose ``/health`` is unreachable is
        FORGOTTEN rather than left stale: offering a name whose points we can no
        longer verify would let the envelope check a trajectory the robot may
        have since re-taught.
        """
        if self.taught_motions is None or not self.embodiment_adapters:
            return

        async def probe(robot_id: str, adapter: SupportsTaughtMotions) -> None:
            try:
                raw = await adapter.taught_motions()
            except RobotOfflineError as exc:
                tracer.event(
                    "harness.taught_motion_probe_offline",
                    trace_id=trace_id,
                    robot_id=robot_id,
                    reason=str(exc),
                )
                self.taught_motions.forget(robot_id)  # type: ignore[union-attr]
                return
            self.taught_motions.replace(robot_id, parse_taught_motions(raw))  # type: ignore[union-attr]

        capable = [
            (rid, adapter)
            for rid, adapter in self.embodiment_adapters.items()
            if isinstance(adapter, SupportsTaughtMotions)
        ]
        if not capable:
            return
        await asyncio.gather(*(probe(rid, a) for rid, a in capable))
        names = self.taught_motions.names()
        if names:
            tracer.event(
                "harness.taught_motions",
                trace_id=trace_id,
                robot_ids=sorted(rid for rid, _ in capable),
                motions=names,
            )

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
        """Convenience factory — wires up default implementations.

        Populates ``embodiment_adapters`` from ``config`` (ADR-009 / ADR-021):
        each robot's backend (mock / agent_server / sim) is constructed by the
        embodiment factory, so a robot configured ``backend: sim`` gets a live
        MuJoCo / Isaac adapter here without any business-code change.
        """
        from robot_harness.config.loader import load_config
        from robot_harness.embodiment.factory import build_catalog
        from robot_harness.skill.base import Skill
        from robot_harness.skill.builtin.follow_joint_waypoints import FollowJointWaypointsSkill
        from robot_harness.skill.builtin.navigate_to import NavigateToSkill
        from robot_harness.skill.builtin.pick import PickSkill
        from robot_harness.skill.builtin.place import PlaceSkill
        from robot_harness.tools.base import Tool
        from robot_harness.tools.memory.backend import build_memory
        from robot_harness.tools.memory.query_tool import MemoryQueryTool
        from robot_harness.tools.middleware.config import build_middleware_chain_from_config
        from robot_harness.tools.perception.mcp_bundle import register_perception_tools
        from robot_harness.tools.robot_sdk import RobotSdkTool, build_robot_sdk_verb_tools

        cfg = config or load_config()
        tool_registry = ToolRegistry()
        skill_registry = SkillRegistry()

        catalog = build_catalog(cfg)
        adapters = {rid: catalog.get(rid) for rid in catalog.list_robot_ids()}
        # The envelope's second pass (EmbodimentAdapter.safety_check, ADR-007)
        # needs the fleet's adapters, so they are built first.
        safety_envelope = SafetyEnvelope(cfg.safety, adapters=adapters)

        # Workspace-declared middleware chain (tool.default_middleware) wraps
        # every framework-registered tool (ISS-037); an empty list is a no-op.
        # ToolMiddleware forwards capability markers, so a wrapped hardware
        # tool stays safety-gated.
        specs = cfg.tool.default_middleware

        def _register(tool: Tool) -> None:
            tool_registry.register(build_middleware_chain_from_config(tool, specs))

        # External capability servers (ADR-026): a non-empty URL in
        # config.tool_servers wires the corresponding tool bundle, so pointing
        # the harness at another deployment is a workspace-config-only change.
        if cfg.tool_servers.perception:
            for perception_tool in register_perception_tools(
                tool_registry, cfg.tool_servers.perception
            ):
                if specs:
                    # Re-register wrapped in the config chain (overwrites the
                    # resolver-wrapped registration under the same name).
                    _register(perception_tool)

        mem = memory or build_memory(cfg.memory, fleet_size=cfg.fleet_size)
        # Long-term recall is an on-demand tool the Brain calls when it needs facts
        # not in recent (working-memory) context — not an every-turn auto-query
        # bypassing the registry (ADR-024 / memory-architecture invariant 4).
        _register(MemoryQueryTool(mem))

        # On-robot act surface (ADR-019): verb tools run the mid-loop perception-
        # action closed loop on the robot; execute_action is the low-level dispatch
        # skills use for gripper/joint moves. Registered here so both CLI and
        # programmatic callers get a functional act layer — and so the builtin
        # skills' ``required_tools`` resolve. Verbs dispatch to the addressed
        # robot's live/sim/mock agent_server; a robot_id outside the wired fleet
        # is a typed failure, never a simulated success.
        # Shared with run_taught_motion; filled by refresh_taught_motions() at
        # task start. Empty here, so before the first probe the verb has no valid
        # motion name — which is correct: the harness must not claim a body knows
        # a motion it has not advertised.
        taught_motions = TaughtMotionCatalog()
        for verb_tool in build_robot_sdk_verb_tools(adapters, taught_motions):
            _register(verb_tool)
        _register(RobotSdkTool(adapters))

        # Built-in skills (versioned tool compositions). The Brain sees each as a
        # ``skill.<name>`` callable via SkillRegistry.export_for_brain; the
        # AgentLoop routes the call to Skill.execute (skill_tools gates the
        # skill's internal hardware calls through SafetyEnvelope).
        builtin_skills: tuple[Skill, ...] = (
            PickSkill(),
            PlaceSkill(),
            NavigateToSkill(),
            FollowJointWaypointsSkill(),
        )
        for skill in builtin_skills:
            skill_registry.register(skill)

        return cls(
            config=cfg,
            tool_registry=tool_registry,
            skill_registry=skill_registry,
            safety_envelope=safety_envelope,
            memory=mem,
            embodiment_adapters=adapters,
            taught_motions=taught_motions,
        )
