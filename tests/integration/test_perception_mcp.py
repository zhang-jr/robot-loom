"""Integration tests against a real perception MCP server.

Skipped unless ``PERCEPTION_MCP_URL`` is set (CI without GPU stays green).

Run locally with::

    PERCEPTION_MCP_URL=http://localhost:8766 \
        pytest tests/integration/test_perception_mcp.py -v

Any MCP server publishing the four ``perception.*`` tools with the schemas in
``robot_harness.tools.perception.mcp_bundle`` will satisfy these tests.
"""

from __future__ import annotations

import base64
import os

import pytest

from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.mcp.client import MCPClientSession
from robot_harness.tools.perception.mcp_bundle import (
    PERCEPTION_DETECT_OBJECTS,
    PERCEPTION_ESTIMATE_DEPTH,
    PERCEPTION_GROUND_PHRASE,
    PERCEPTION_SEGMENT_PROMPTABLE,
    PERCEPTION_TOOL_NAMES,
    build_perception_tools,
    verify_server_compatibility,
)

_SERVER_URL = os.environ.get("PERCEPTION_MCP_URL", "")

pytestmark = pytest.mark.skipif(
    not _SERVER_URL,
    reason="PERCEPTION_MCP_URL not set — skipping live MCP perception integration tests",
)


def _ctx() -> ToolContext:
    return ToolContext.create("integration-bot", timeout_s=60.0)


def _build_bundle() -> dict[str, object]:
    """All perception tools keyed by name."""
    return {t.name: t for t in build_perception_tools(_SERVER_URL)}


def _decode_png(b64: str) -> tuple[bytes, int, int, int]:
    """Inspect a base64 PNG: return (raw_bytes, width, height, bit_depth).

    Stdlib-only — avoids forcing Pillow as a test dependency.
    """
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "decoded bytes are not a PNG"
    # IHDR chunk starts at offset 8; payload at offset 16
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    bit_depth = raw[24]
    return raw, width, height, bit_depth


# ---------------------------------------------------------------------------
# Server catalogue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_server_publishes_all_perception_tools() -> None:
    """Server must publish detect/depth/ground/segment — all four."""
    async with MCPClientSession(_SERVER_URL) as session:
        await verify_server_compatibility(session)


# ---------------------------------------------------------------------------
# detect_objects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_detect_objects_returns_well_formed_payload(medium_png_b64: str) -> None:
    """Wire-path check: payload shape, not detection accuracy.

    A solid-grey 256x256 image will likely yield zero detections, which is
    fine — we verify the output schema.
    """
    tool = _build_bundle()[PERCEPTION_DETECT_OBJECTS]
    result = await tool.invoke(
        {
            "image_b64": medium_png_b64,
            "prompts": ["object"],
            "confidence_threshold": 0.1,
        },
        _ctx(),
    )

    assert result.success, f"detect_objects failed: {result.error}"
    output = result.output or {}
    assert "detections" in output, f"missing 'detections' in {output}"
    assert isinstance(output["detections"], list)
    for det in output["detections"]:
        assert {"label", "confidence", "bbox"}.issubset(det.keys())
        assert len(det["bbox"]) == 4


# ---------------------------------------------------------------------------
# estimate_depth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_estimate_depth_round_trip(medium_png_b64: str) -> None:
    """estimate_depth returns a 16-bit PNG depth map with shape == input."""
    tool = _build_bundle()[PERCEPTION_ESTIMATE_DEPTH]
    result = await tool.invoke({"image_b64": medium_png_b64}, _ctx())

    assert result.success, f"estimate_depth failed: {result.error}"
    output = result.output or {}

    # Required fields per schema.
    for key in ("depth_b64_png16", "shape", "scale_unit"):
        assert key in output, f"missing '{key}' in {output}"
    assert output["scale_unit"] in ("relative", "meters")

    # shape claims [h, w]; decode the depth PNG and check it matches input dimensions.
    claimed_h, claimed_w = output["shape"]
    _, png_w, png_h, bit_depth = _decode_png(output["depth_b64_png16"])
    assert (png_h, png_w) == (claimed_h, claimed_w), (
        f"shape {(claimed_h, claimed_w)} disagrees with PNG header {(png_h, png_w)}"
    )
    assert (png_h, png_w) == (256, 256), "depth shape should match input image"
    assert bit_depth == 16, f"expected 16-bit depth PNG, got bit_depth={bit_depth}"


# ---------------------------------------------------------------------------
# ground_phrase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_ground_phrase_returns_well_formed_payload(medium_png_b64: str) -> None:
    """ground_phrase returns either a Detection or null — never raises on grey image."""
    tool = _build_bundle()[PERCEPTION_GROUND_PHRASE]
    result = await tool.invoke(
        {"image_b64": medium_png_b64, "phrase": "object"},
        _ctx(),
    )

    assert result.success, f"ground_phrase failed: {result.error}"
    output = result.output or {}
    assert "detection" in output
    det = output["detection"]
    if det is not None:
        assert {"label", "confidence", "bbox"}.issubset(det.keys())
        assert len(det["bbox"]) == 4


# ---------------------------------------------------------------------------
# segment_promptable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_segment_promptable_round_trip(medium_png_b64: str) -> None:
    """Promptable segmentation returns a binary PNG mask (optionally bbox)."""
    tool = _build_bundle()[PERCEPTION_SEGMENT_PROMPTABLE]
    result = await tool.invoke(
        {"image_b64": medium_png_b64, "phrase": "object"},
        _ctx(),
    )

    assert result.success, f"segment_promptable failed: {result.error}"
    output = result.output or {}
    assert "mask_b64_png" in output, f"missing 'mask_b64_png' in {output}"

    # Mask should decode as a valid PNG. Solid-grey input may yield an empty
    # placeholder mask per server convention — accept any valid PNG size.
    _, mask_w, mask_h, _ = _decode_png(output["mask_b64_png"])
    assert mask_w >= 1 and mask_h >= 1

    if output.get("bbox") is not None:
        assert len(output["bbox"]) == 4


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_tools_register_into_registry(medium_png_b64: str) -> None:
    registry = ToolRegistry()
    for perception_tool in build_perception_tools(_SERVER_URL):
        registry.register(perception_tool)
    assert len(registry) >= len(PERCEPTION_TOOL_NAMES)
    # Roundtrip through registry: invoke detect_objects via registry.get()
    fetched = registry.get(PERCEPTION_DETECT_OBJECTS)
    result = await fetched.invoke(
        {"image_b64": medium_png_b64, "prompts": ["object"]},
        _ctx(),
    )
    assert result.tool_name == PERCEPTION_DETECT_OBJECTS
