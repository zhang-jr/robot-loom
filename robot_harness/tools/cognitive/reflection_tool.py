"""ReflectionTool — append-only self-evaluation log for the Brain (opt-in).

Enabled only when ``BrainProfile.supports_native_reflection`` is False.
When the LLM backend supports extended thinking natively (e.g. Claude 3.5+),
this tool is redundant and should NOT be registered.

Adapted from tmp/tools/critic_tool.py — renamed to avoid confusion with the
embodied Critic (ADR-018).  The ReflectionStore evaluates LLM reasoning quality;
the Critic judges physical-world task progress.  They must never be conflated.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema

Verdict = Literal["pass", "warn", "fail"]
_VALID_VERDICTS: frozenset[str] = frozenset({"pass", "warn", "fail"})
_VERDICT_ICONS: dict[str, str] = {"fail": "[FAIL]", "warn": "[WARN]", "pass": "[OK]"}


class ReflectionStore:
    """Append-only self-evaluation log — one instance per AgentLoop.

    Each entry captures: target_id, target_desc, verdict, reasoning, suggestion.
    Unresolved fail/warn entries survive context compression via
    ``format_for_injection()``.
    """

    store_name = "reflection"

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # CognitiveScaffoldStore protocol
    # ------------------------------------------------------------------

    def has_content(self) -> bool:
        return bool(self._entries)

    def format_for_injection(self) -> str | None:
        """Re-inject only unresolved (fail/warn) entries after compression."""
        if not self._entries:
            return None

        latest: dict[str, dict[str, Any]] = {}
        for e in self._entries:
            latest[e["target_id"]] = e

        unresolved = [e for e in latest.values() if e["verdict"] != "pass"]
        if not unresolved:
            return None

        lines = ["[Unresolved self-evaluations preserved across context compression]"]
        for e in unresolved:
            icon = _VERDICT_ICONS.get(e["verdict"], "[?]")
            suggestion = f" -> {e['suggestion']}" if e.get("suggestion") else ""
            lines.append(
                f"  {icon} [{e['target_id']}] {e['target_desc']}: {e['reasoning']}{suggestion}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Read / Append
    # ------------------------------------------------------------------

    def read(self) -> list[dict[str, Any]]:
        return [e.copy() for e in self._entries]

    def append(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        validated = self._validate(entry, seq=len(self._entries) + 1)
        self._entries.append(validated)
        return self.read()

    def last(self) -> dict[str, Any] | None:
        return self._entries[-1].copy() if self._entries else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(raw: dict[str, Any], seq: int) -> dict[str, Any]:
        target_id = str(raw.get("target_id", "")).strip() or "?"
        target_desc = str(raw.get("target_desc", "")).strip() or "(unspecified)"
        verdict = str(raw.get("verdict", "fail")).strip().lower()
        if verdict not in _VALID_VERDICTS:
            verdict = "fail"
        reasoning = str(raw.get("reasoning", "")).strip() or "(no reasoning)"
        suggestion = str(raw.get("suggestion", "")).strip()

        entry: dict[str, Any] = {
            "seq": seq,
            "target_id": target_id,
            "target_desc": target_desc,
            "verdict": verdict,
            "reasoning": reasoning,
            "ts": round(time.time(), 3),
        }
        if suggestion:
            entry["suggestion"] = suggestion
        return entry


# ---------------------------------------------------------------------------
# NativeTool wrapper
# ---------------------------------------------------------------------------


class ReflectionTool:
    """NativeTool giving the Brain read/append access to its ReflectionStore.

    Only register this tool when ``BrainProfile.supports_native_reflection``
    is False.  Backends with extended thinking (e.g. Claude 3.5+) should NOT
    have this registered — the model handles reflection internally.
    """

    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="reflection",
        description=(
            "Evaluate the quality of your own outputs or intermediate results and "
            "log the verdict.  Use after completing a plan step, after a tool call "
            "returns unexpected results, or when you detect an error.\n\n"
            "Reading: call with no parameters to see the full evaluation log.\n\n"
            "Writing: provide an 'entry' to append one self-evaluation.\n\n"
            "Verdicts:\n"
            "  pass - result is correct; proceed.\n"
            "  warn - result is usable but has a limitation; note and continue.\n"
            "  fail - result is incorrect; provide a 'suggestion' and act on it.\n\n"
            "A later 'pass' on the same target_id resolves an earlier 'fail'.\n"
            "Always returns the full evaluation log."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "entry": {
                    "type": "object",
                    "description": "Evaluation entry to append.  Omit to read the log.",
                    "properties": {
                        "target_id": {"type": "string"},
                        "target_desc": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": ["pass", "warn", "fail"],
                        },
                        "reasoning": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": ["target_id", "target_desc", "verdict", "reasoning"],
                },
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "reflections": {"type": "array"},
                "summary": {"type": "object"},
                "last": {"type": "object"},
            },
        },
    )

    def __init__(self, store: ReflectionStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "reflection"

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        entry = args.get("entry")
        if entry is not None:
            entries = self._store.append(entry)
        else:
            entries = self._store.read()

        counts: dict[str, int] = {"pass": 0, "warn": 0, "fail": 0}
        for e in entries:
            counts[e.get("verdict", "fail")] = counts.get(e.get("verdict", "fail"), 0) + 1

        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={
                "reflections": entries,
                "summary": {"total": len(entries), **counts},
                "last": self._store.last(),
            },
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
