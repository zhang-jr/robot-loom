"""PlanTool — structured goal decomposition and step tracking for the Brain.

The Brain uses this tool to decompose a complex goal into ordered steps, track
their status through execution, and recover strategic direction after context
compression.  The PlannerStore is per-AgentLoop (session-scoped); it is NOT
persisted to Memory (that is the job of EpisodicMemory after the episode ends).

Adapted from tmp/tools/planner_tool.py.
"""

from __future__ import annotations

from typing import Any, Literal

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema

StepStatus = Literal["pending", "in_progress", "completed", "failed", "skipped"]

_VALID_STATUSES: frozenset[str] = frozenset(
    {"pending", "in_progress", "completed", "failed", "skipped"}
)

_STATUS_MARKERS: dict[str, str] = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": "[x]",
    "failed": "[!]",
    "skipped": "[~]",
}


class PlannerStore:
    """In-memory plan store — one instance per AgentLoop (per robot_id bucket).

    A plan has:
      - goal: the top-level objective (string)
      - steps: ordered list of {id, description, status, depends_on?}
    """

    store_name = "plan"

    def __init__(self) -> None:
        self._goal: str = ""
        self._steps: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # CognitiveScaffoldStore protocol
    # ------------------------------------------------------------------

    def has_content(self) -> bool:
        return bool(self._goal or self._steps)

    def format_for_injection(self) -> str | None:
        """Render active (non-completed/skipped) steps for context re-injection.

        Returns None when there are no active steps — goal alone is not worth
        injecting once all work is done.
        """
        active = [s for s in self._steps if s["status"] not in ("completed", "skipped")]
        if not active:
            return None

        lines = ["[Active plan preserved across context compression]"]
        if self._goal:
            lines.append(f"Goal: {self._goal}")
        for step in active:
            marker = _STATUS_MARKERS.get(step["status"], "[?]")
            dep = f" (needs: {step['depends_on']})" if step.get("depends_on") else ""
            lines.append(f"  {marker} {step['id']}. {step['description']}{dep}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------------

    def read(self) -> dict[str, Any]:
        return {"goal": self._goal, "steps": [s.copy() for s in self._steps]}

    def write(
        self,
        goal: str | None = None,
        steps: list[dict[str, Any]] | None = None,
        merge: bool = False,
    ) -> dict[str, Any]:
        if goal is not None:
            self._goal = str(goal).strip()

        if steps is not None:
            validated = [self._validate_step(s) for s in self._dedupe(steps)]
            if not merge:
                self._steps = validated
            else:
                existing: dict[str, dict[str, Any]] = {s["id"]: s for s in self._steps}
                for step in validated:
                    sid = step["id"]
                    if sid in existing:
                        if step.get("description"):
                            existing[sid]["description"] = step["description"]
                        if step.get("status") in _VALID_STATUSES:
                            existing[sid]["status"] = step["status"]
                        if "depends_on" in step:
                            existing[sid]["depends_on"] = step["depends_on"]
                    else:
                        existing[sid] = step
                        self._steps.append(step)
                # Rebuild in original order, updated in place
                seen: set[str] = set()
                rebuilt: list[dict[str, Any]] = []
                for s in self._steps:
                    cur = existing.get(s["id"], s)
                    if cur["id"] not in seen:
                        rebuilt.append(cur)
                        seen.add(cur["id"])
                self._steps = rebuilt

        return self.read()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_step(raw: dict[str, Any]) -> dict[str, Any]:
        sid = str(raw.get("id", "")).strip() or "?"
        description = str(raw.get("description", "")).strip() or "(no description)"
        status = str(raw.get("status", "pending")).strip().lower()
        if status not in _VALID_STATUSES:
            status = "pending"
        result: dict[str, Any] = {"id": sid, "description": description, "status": status}
        dep = raw.get("depends_on")
        if dep:
            result["depends_on"] = str(dep).strip()
        return result

    @staticmethod
    def _dedupe(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the last occurrence of each id, preserving relative order."""
        last_index: dict[str, int] = {}
        for i, s in enumerate(steps):
            sid = str(s.get("id", "")).strip() or "?"
            last_index[sid] = i
        return [steps[i] for i in sorted(last_index.values())]


# ---------------------------------------------------------------------------
# NativeTool wrapper
# ---------------------------------------------------------------------------


class PlanTool:
    """NativeTool that gives the Brain read/write access to its PlannerStore.

    The store is injected at construction time by the AgentLoop / HarnessContext
    so that every Brain decision in the same session shares the same plan.
    """

    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="plan",
        description=(
            "Manage your strategic plan for the current task.  Use when you receive "
            "a complex goal that requires 3+ steps, when you need to revise your "
            "approach after a failure, or to track progress through a multi-step task.\n\n"
            "Reading: call with no parameters to see the current plan.\n\n"
            "Writing:\n"
            "- Provide 'goal' to set or update the top-level objective.\n"
            "- Provide 'steps' to define or update the execution steps.\n"
            "- merge=false (default): replace the entire step list.\n"
            "- merge=true: update specific steps by id (status changes, revisions).\n\n"
            "Each step: {id, description, status, depends_on?}\n"
            "  status: pending | in_progress | completed | failed | skipped\n\n"
            "Only ONE step should be in_progress at a time.  Mark a step 'completed' "
            "before starting the next.  Always returns the full current plan."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Top-level objective.  Omit to keep existing goal.",
                },
                "steps": {
                    "type": "array",
                    "description": "Steps to write.  Omit to read current plan.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                    "failed",
                                    "skipped",
                                ],
                            },
                            "depends_on": {"type": "string"},
                        },
                        "required": ["id", "description", "status"],
                    },
                },
                "merge": {
                    "type": "boolean",
                    "description": "true: update by id.  false (default): replace entire list.",
                    "default": False,
                },
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "summary": {"type": "object"},
            },
        },
    )

    def __init__(self, store: PlannerStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "plan"

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        goal = args.get("goal")
        steps = args.get("steps")
        merge = bool(args.get("merge", False))

        plan = self._store.write(goal=goal, steps=steps, merge=merge)

        counts: dict[str, int] = {
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
        }
        for s in plan["steps"]:
            key = s.get("status", "pending")
            counts[key] = counts.get(key, 0) + 1

        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"plan": plan, "summary": {"total": len(plan["steps"]), **counts}},
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
