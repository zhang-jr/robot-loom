"""Unit tests for the perception MCP bundle."""

from __future__ import annotations

import pytest

from robot_harness.errors import ToolBackendUnreachableError
from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.mcp.client import MCPTool
from robot_harness.tools.perception.mcp_bundle import (
    PERCEPTION_DETECT_OBJECTS,
    PERCEPTION_ESTIMATE_DEPTH,
    PERCEPTION_GROUND_PHRASE,
    PERCEPTION_SEGMENT_PROMPTABLE,
    PERCEPTION_TOOL_NAMES,
    build_perception_tools,
    verify_server_compatibility,
)
from tests._helpers.mcp import (
    FakeMCPClientSession,
    make_call_result,
    make_list_tools_result,
)


def _ctx() -> ToolContext:
    return ToolContext.create("robot-0", timeout_s=5.0)


# ---------------------------------------------------------------------------
# Factory output
# ---------------------------------------------------------------------------


def test_factory_returns_all_perception_tools() -> None:
    tools = build_perception_tools("http://localhost:8765")
    names = [t.name for t in tools]
    assert names == list(PERCEPTION_TOOL_NAMES)


def test_factory_tools_are_mcp_backend() -> None:
    for tool in build_perception_tools("http://localhost:8765"):
        assert isinstance(tool, MCPTool)
        assert tool.backend == "mcp"
        assert tool.server_url == "http://localhost:8765"


def test_factory_tools_are_idempotent() -> None:
    for tool in build_perception_tools("http://localhost:8765"):
        assert tool.is_idempotent is True


def test_factory_tools_register_into_registry() -> None:
    registry = ToolRegistry()
    for tool in build_perception_tools("http://localhost:8765"):
        registry.register(tool)
    for name in PERCEPTION_TOOL_NAMES:
        assert name in registry


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------


def test_detect_objects_schema_requires_image_b64_and_prompts() -> None:
    tools = {t.name: t for t in build_perception_tools("http://x")}
    schema = tools[PERCEPTION_DETECT_OBJECTS].schema.input_schema
    assert "image_b64" in schema["properties"]
    assert "prompts" in schema["properties"]
    assert set(schema["required"]) == {"image_b64", "prompts"}


def test_estimate_depth_schema_requires_image_b64() -> None:
    tools = {t.name: t for t in build_perception_tools("http://x")}
    schema = tools[PERCEPTION_ESTIMATE_DEPTH].schema.input_schema
    assert schema["required"] == ["image_b64"]
    assert schema["properties"]["output"]["enum"] == ["relative", "metric_if_available"]


def test_ground_phrase_schema_requires_image_and_phrase() -> None:
    tools = {t.name: t for t in build_perception_tools("http://x")}
    schema = tools[PERCEPTION_GROUND_PHRASE].schema.input_schema
    assert set(schema["required"]) == {"image_b64", "phrase"}


def test_segment_promptable_schema_requires_image_and_phrase() -> None:
    tools = {t.name: t for t in build_perception_tools("http://x")}
    schema = tools[PERCEPTION_SEGMENT_PROMPTABLE].schema.input_schema
    assert set(schema["required"]) == {"image_b64", "phrase"}


def test_output_schemas_present() -> None:
    """Every perception tool publishes an output schema (helps strict Brains)."""
    for tool in build_perception_tools("http://x"):
        assert tool.schema.output_schema is not None, tool.name


# ---------------------------------------------------------------------------
# End-to-end via FakeMCPClientSession
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_objects_roundtrip_through_mcp(tiny_png_b64: str) -> None:
    tools = {t.name: t for t in build_perception_tools("http://x")}
    tool = tools[PERCEPTION_DETECT_OBJECTS]

    expected_detections = [
        {
            "label": "mug",
            "confidence": 0.87,
            "bbox": [0.42, 0.31, 0.58, 0.55],
            "matched_prompt": "mug",
        }
    ]
    fake = FakeMCPClientSession(
        responses={
            PERCEPTION_DETECT_OBJECTS: make_call_result(
                structured={
                    "detections": expected_detections,
                    "latency_ms": 42.0,
                    "model_version": "sam3_decoder@trt",
                },
            )
        },
    )
    tool._session_factory = lambda _url, _t: fake  # type: ignore[method-assign,assignment]

    result = await tool.invoke(
        {"image_b64": tiny_png_b64, "prompts": ["mug"]},
        _ctx(),
    )
    assert result.success is True
    assert result.output is not None
    assert result.output["detections"] == expected_detections
    assert fake.calls[0].name == PERCEPTION_DETECT_OBJECTS
    assert fake.calls[0].args == {"image_b64": tiny_png_b64, "prompts": ["mug"]}


@pytest.mark.asyncio
async def test_ground_phrase_returns_null_detection(tiny_png_b64: str) -> None:
    tools = {t.name: t for t in build_perception_tools("http://x")}
    tool = tools[PERCEPTION_GROUND_PHRASE]
    fake = FakeMCPClientSession(
        responses={
            PERCEPTION_GROUND_PHRASE: make_call_result(
                structured={"detection": None, "latency_ms": 18.0}
            )
        },
    )
    tool._session_factory = lambda _url, _t: fake  # type: ignore[method-assign,assignment]

    result = await tool.invoke(
        {"image_b64": tiny_png_b64, "phrase": "purple unicorn"},
        _ctx(),
    )
    assert result.success is True
    assert result.output == {"detection": None, "latency_ms": 18.0}


@pytest.mark.asyncio
async def test_segment_promptable_roundtrip_through_mcp(tiny_png_b64: str) -> None:
    """Brain consumes mask_b64_png + bbox + confidence from the server."""
    tools = {t.name: t for t in build_perception_tools("http://x")}
    tool = tools[PERCEPTION_SEGMENT_PROMPTABLE]

    expected_payload = {
        "mask_b64_png": "iVBORw0KGgo=",
        "bbox": [0.10, 0.20, 0.50, 0.60],
        "confidence": 0.92,
        "latency_ms": 73.0,
        "model_version": "sam3_decoder@trt",
    }
    fake = FakeMCPClientSession(
        responses={PERCEPTION_SEGMENT_PROMPTABLE: make_call_result(structured=expected_payload)},
    )
    tool._session_factory = lambda _url, _t: fake  # type: ignore[method-assign,assignment]

    result = await tool.invoke(
        {"image_b64": tiny_png_b64, "phrase": "mug"},
        _ctx(),
    )
    assert result.success is True
    assert result.output == expected_payload


# ---------------------------------------------------------------------------
# Error propagation: server returns isError=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_objects_propagates_server_error(tiny_png_b64: str) -> None:
    """isError=True from the MCP server surfaces as ToolResult.success=False.

    The output payload is preserved so traces can inspect what the server said.
    """
    tools = {t.name: t for t in build_perception_tools("http://x")}
    tool = tools[PERCEPTION_DETECT_OBJECTS]

    server_error_payload = {
        "error_type": "TritonUnreachableError",
        "reason": "Triton at triton:8001 refused connection",
        "latency_ms": 12.0,
    }
    fake = FakeMCPClientSession(
        responses={
            PERCEPTION_DETECT_OBJECTS: make_call_result(
                structured=server_error_payload, is_error=True
            )
        },
    )
    tool._session_factory = lambda _url, _t: fake  # type: ignore[method-assign,assignment]

    result = await tool.invoke(
        {"image_b64": tiny_png_b64, "prompts": ["mug"]},
        _ctx(),
    )
    assert result.success is False
    assert result.error_type == "MCPToolError"
    assert result.error is not None
    assert "TritonUnreachableError" in result.error
    assert result.output == server_error_payload


# ---------------------------------------------------------------------------
# verify_server_compatibility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_compatibility_passes_when_all_present() -> None:
    fake = FakeMCPClientSession(
        list_tools_result=make_list_tools_result(list(PERCEPTION_TOOL_NAMES)),
    )
    async with fake as session:
        await verify_server_compatibility(session)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_verify_compatibility_raises_on_missing_tool() -> None:
    fake = FakeMCPClientSession(
        list_tools_result=make_list_tools_result(
            [PERCEPTION_DETECT_OBJECTS, PERCEPTION_ESTIMATE_DEPTH]
            # ground_phrase + segment_promptable intentionally missing
        ),
    )
    async with fake as session:
        with pytest.raises(ToolBackendUnreachableError, match="ground_phrase"):
            await verify_server_compatibility(session)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_verify_compatibility_with_custom_expected_set() -> None:
    fake = FakeMCPClientSession(
        list_tools_result=make_list_tools_result([PERCEPTION_DETECT_OBJECTS]),
    )
    async with fake as session:
        await verify_server_compatibility(
            session,  # type: ignore[arg-type]
            expected_names=(PERCEPTION_DETECT_OBJECTS,),
        )


@pytest.mark.asyncio
async def test_verify_compatibility_raises_on_missing_segment_tool() -> None:
    """Default check requires segment_promptable too."""
    fake = FakeMCPClientSession(
        list_tools_result=make_list_tools_result(
            [PERCEPTION_DETECT_OBJECTS, PERCEPTION_ESTIMATE_DEPTH, PERCEPTION_GROUND_PHRASE]
        ),
    )
    async with fake as session:
        with pytest.raises(ToolBackendUnreachableError, match="segment_promptable"):
            await verify_server_compatibility(session)  # type: ignore[arg-type]
