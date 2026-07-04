"""Run one ADR-019 reactive_grasp against the LIBERO agent_server.

The harness sends one high-level language target. The external agent_server
owns the complete image -> VLA -> action -> LIBERO loop and returns one
CompletionVerdict; no per-step action crosses the harness boundary.

Start the OpenVLA MCP service and LIBERO agent_server from the recipe repository,
then run:

    set ROBOT_LOOM_LIBERO_URL=http://127.0.0.1:9000
    python examples/libero_reactive_grasp.py
"""

from __future__ import annotations

import asyncio
import os

import httpx

from robot_harness.config.schema import EmbodimentBackendConfig, HarnessConfig
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.base import ToolContext
from robot_harness.tools.robot_sdk.verbs import ROBOT_SDK_REACTIVE_GRASP, build_robot_sdk_verb_tools

ROBOT_ID = os.environ.get("ROBOT_LOOM_ROBOT_ID", "libero-sim")
AGENT_SERVER_URL = os.environ.get(
    "ROBOT_LOOM_LIBERO_URL",
    "http://127.0.0.1:9000",
)
INSTRUCTION = os.environ.get(
    "LIBERO_INSTRUCTION",
    ("pick up the black bowl between the plate and the ramekin and place it on the plate"),
)
TIMEOUT_S = float(os.environ.get("LIBERO_VERB_TIMEOUT_S", "120"))


async def main() -> None:
    async with httpx.AsyncClient(timeout=3.0) as client:
        response = await client.get(f"{AGENT_SERVER_URL}/health")
        response.raise_for_status()

    config = HarnessConfig(
        robot_ids=[ROBOT_ID],
        embodiments={
            ROBOT_ID: EmbodimentBackendConfig(
                backend="sim",
                sim_engine="mujoco",
                server_url=AGENT_SERVER_URL,
                request_timeout_s=TIMEOUT_S + 30.0,
            )
        },
    )
    harness = HarnessContext.build(config)
    for tool in build_robot_sdk_verb_tools(harness.embodiment_adapters):
        harness.tool_registry.register(tool)

    tool = harness.tool_registry.get(ROBOT_SDK_REACTIVE_GRASP)
    context = ToolContext.create(ROBOT_ID, timeout_s=TIMEOUT_S + 30.0)
    result = await tool.invoke(
        {
            "robot_id": ROBOT_ID,
            "target_hint": {"kind": "phrase", "phrase": INSTRUCTION},
            "constraints": {"timeout_s": TIMEOUT_S},
        },
        context,
    )

    print(f"tool_name: {result.tool_name}")
    print(f"trace_id:  {result.trace_id}")
    print(f"success:   {result.success}")
    if result.output is not None:
        print(f"outcome:   {result.output.get('outcome')}")
        print(f"evidence:  {result.output.get('evidence')}")
        print(f"duration:  {result.output.get('duration_s')} s")
        sim_evidence = result.output.get("sim_evidence") or {}
        print(f"steps:     {sim_evidence.get('control_steps')}")
        print(f"VLA calls: {sim_evidence.get('policy_calls')}")

    for adapter in harness.embodiment_adapters.values():
        close = getattr(adapter, "aclose", None)
        if callable(close):
            await close()


if __name__ == "__main__":
    asyncio.run(main())
