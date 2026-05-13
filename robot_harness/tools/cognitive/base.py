"""CognitiveScaffoldStore Protocol — common interface for all cognitive stores.

Each store manages one aspect of session-scoped meta-cognition (plan, reflection,
…).  After context compression, the AgentLoop calls ``format_for_injection()`` on
every registered store and prepends the non-None results to the next Brain turn.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CognitiveScaffoldStore(Protocol):
    """Minimal interface every cognitive scaffold store must satisfy."""

    @property
    def store_name(self) -> str:
        """Short identifier shown in traces (e.g. 'plan', 'reflection')."""
        ...

    def format_for_injection(self) -> str | None:
        """Render active state for post-compression re-injection.

        Returns None when there is nothing worth injecting (empty or fully
        resolved store).  The returned string is prepended verbatim to the
        system prompt; keep it terse.
        """
        ...

    def has_content(self) -> bool:
        """Return True if the store has any content at all."""
        ...
