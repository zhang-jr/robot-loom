"""Navigation tools — intentionally empty (ADR-022).

This package does NOT hold a reactive ``navigate`` tool. Reactive "go there"
(close the perception-action loop, avoid obstacles, drive until arrival) is a
body verb and lives in the per-robot agent_server as ``locomote_to`` — the
harness only issues the intent and awaits the CompletionVerdict, it never holds
a navigation control client.

The only legitimate tool that may live here is a *declarative* one: an adapter
to an external VLN server that consumes an instruction and returns a navigation
plan (waypoints / path), NOT control quantities. Such a tool is Brain-visible
because it is "intent in, plan out" and reusable across embodiments.

Do not add a Brain-visible ``navigate`` that duplicates ``locomote_to``.
"""

from robot_harness.tools.navigation.mcp_bundle import (
    NAVIGATION_PLAN_PATH,
    NAVIGATION_TOOL_NAMES,
    build_navigation_tools,
    verify_server_compatibility,
)

__all__ = [
    "NAVIGATION_PLAN_PATH",
    "NAVIGATION_TOOL_NAMES",
    "build_navigation_tools",
    "verify_server_compatibility",
]
