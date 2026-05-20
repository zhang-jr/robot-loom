"""Integration test against a real perception MCP server.

Skipped unless ``PERCEPTION_MCP_URL`` is set (CI without GPU stays green).
Run locally with::

    PERCEPTION_MCP_URL=http://localhost:8765 pytest tests/integration/test_perception_mcp.py -v

Expects an MCP server compliant with the perception-triton recipe.
"""

from __future__ import annotations

import base64
import os

import pytest

from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.mcp.client import MCPClientSession
from robot_harness.tools.perception.mcp_bundle import (
    PERCEPTION_DETECT_OBJECTS,
    build_perception_tools,
    verify_server_compatibility,
)

_SERVER_URL = os.environ.get("PERCEPTION_MCP_URL", "")

pytestmark = pytest.mark.skipif(
    not _SERVER_URL,
    reason="PERCEPTION_MCP_URL not set — skipping live MCP perception integration tests",
)


# A minimal 1x1 PNG (red pixel). Real servers should accept it as a valid image;
# detections will likely be empty, which is fine — we test the wire path, not
# the model's accuracy.
_TINY_PNG_B64 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63f8cfc0c000000003000100b5b9b1ff0000000049454e"
        "44ae426082"
    )
).decode("ascii")


def _ctx() -> ToolContext:
    return ToolContext.create("integration-bot", timeout_s=30.0)


@pytest.mark.asyncio
async def test_live_server_publishes_phase_a_tools() -> None:
    async with MCPClientSession(_SERVER_URL) as session:
        await verify_server_compatibility(session)


@pytest.mark.asyncio
async def test_live_detect_objects_returns_well_formed_payload() -> None:
    tools = {t.name: t for t in build_perception_tools(_SERVER_URL)}
    tool = tools[PERCEPTION_DETECT_OBJECTS]

    result = await tool.invoke(
        {
            "image_b64": _TINY_PNG_B64,
            "prompts": ["object"],
            "confidence_threshold": 0.1,
        },
        _ctx(),
    )

    assert result.success, f"detect_objects failed: {result.error}"
    output = result.output or {}
    assert "detections" in output, f"missing 'detections' in {output}"
    assert isinstance(output["detections"], list)


@pytest.mark.asyncio
async def test_live_tools_register_into_registry() -> None:
    registry = ToolRegistry()
    for perception_tool in build_perception_tools(_SERVER_URL):
        registry.register(perception_tool)
    assert len(registry) >= 3
    # Roundtrip through registry: invoke detect_objects via registry.get()
    fetched = registry.get(PERCEPTION_DETECT_OBJECTS)
    result = await fetched.invoke(
        {"image_b64": _TINY_PNG_B64, "prompts": ["object"]},
        _ctx(),
    )
    assert result.tool_name == PERCEPTION_DETECT_OBJECTS
