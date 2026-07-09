"""Export tool specs in the format expected by each Brain backend."""

from __future__ import annotations

from typing import Any

from robot_harness.tools.base import BrainProfile, BrainProfileName, ToolRegistry


def export_for_brain(registry: ToolRegistry, backend: BrainProfileName) -> list[dict[str, Any]]:
    """Return tool specs ready to be passed to the LLM backend.

    Args:
        registry: The populated ToolRegistry.
        backend: One of ``'openai'``, ``'anthropic'``, ``'mcp'``, ``'litellm'``.
    """
    profile = BrainProfile(name=backend)
    return registry.export_for_brain(profile)
