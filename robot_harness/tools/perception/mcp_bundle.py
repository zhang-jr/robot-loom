"""Perception tool bundle backed by an external MCP server.

The tool contracts live here so that the harness fails fast at startup if the
configured perception server doesn't actually publish the tools this harness
expects. Any MCP server publishing the same tool names + schemas is a drop-in
substitute.

Tool name namespace: ``perception.*``. Names are stable; schemas may grow
backward-compatible fields over time.
"""

from __future__ import annotations

from typing import Any

from robot_harness.errors import ToolBackendUnreachableError
from robot_harness.tools.artifacts import ARTIFACT_REF_SCHEMA
from robot_harness.tools.base import Tool, ToolRegistry
from robot_harness.tools.mcp.client import MCPClientSession, MCPTool
from robot_harness.tools.middleware.artifact_resolver import ArtifactResolverMiddleware
from robot_harness.tools.middleware.base import build_chain

# Every perception tool takes its image either inline (``image_b64``) or, when an
# artifact store is wired, as a ``frame`` ref from robot.capture_frame that the
# ArtifactResolverMiddleware hydrates into ``image_b64`` before dispatch. JSON
# Schema can't cleanly express "exactly one of" without oneOf (which strict
# backends reject), so both are optional and the description states the rule.
_IMAGE_INPUT_PROPS: dict[str, Any] = {
    "image_b64": {
        "type": "string",
        "description": "Base64-encoded image. Provide this OR `frame`, not both.",
    },
    "frame": {**ARTIFACT_REF_SCHEMA, "description": "Frame ref from robot.capture_frame."},
}

# ---------------------------------------------------------------------------
# Tool names (string constants — use these instead of inlining literals)
# ---------------------------------------------------------------------------

PERCEPTION_DETECT_OBJECTS = "perception.detect_objects"
PERCEPTION_ESTIMATE_DEPTH = "perception.estimate_depth"
PERCEPTION_GROUND_PHRASE = "perception.ground_phrase"
PERCEPTION_SEGMENT_PROMPTABLE = "perception.segment_promptable"

PERCEPTION_TOOL_NAMES: tuple[str, ...] = (
    PERCEPTION_DETECT_OBJECTS,
    PERCEPTION_ESTIMATE_DEPTH,
    PERCEPTION_GROUND_PHRASE,
    PERCEPTION_SEGMENT_PROMPTABLE,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_BBOX_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 4,
    "maxItems": 4,
    "description": "[x1, y1, x2, y2] normalized to [0, 1], top-left origin.",
}

_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "confidence": {"type": "number"},
        "bbox": _BBOX_SCHEMA,
        "matched_prompt": {"type": "string"},
    },
    "required": ["label", "confidence", "bbox"],
}

_DETECT_OBJECTS_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        **_IMAGE_INPUT_PROPS,
        "prompts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Open-vocabulary phrases (e.g. ['mug', 'drawer handle']).",
        },
        "confidence_threshold": {"type": "number", "default": 0.3, "minimum": 0.0, "maximum": 1.0},
        "max_detections": {"type": "integer", "default": 50, "minimum": 1},
    },
    "required": ["prompts"],
}

_DETECT_OBJECTS_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "detections": {"type": "array", "items": _DETECTION_SCHEMA},
        "latency_ms": {"type": "number"},
        "model_version": {"type": "string"},
    },
    "required": ["detections"],
}

_ESTIMATE_DEPTH_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        **_IMAGE_INPUT_PROPS,
        "output": {
            "type": "string",
            "enum": ["relative", "metric_if_available"],
            "default": "relative",
        },
    },
    "required": [],
}

_ESTIMATE_DEPTH_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "depth_b64_png16": {
            "type": "string",
            "description": "16-bit single-channel PNG, base64-encoded.",
        },
        "shape": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[height, width] in pixels.",
        },
        "min": {"type": "number"},
        "max": {"type": "number"},
        "scale_unit": {"type": "string", "enum": ["relative", "meters"]},
        "latency_ms": {"type": "number"},
        "model_version": {"type": "string"},
    },
    "required": ["depth_b64_png16", "shape", "scale_unit"],
}

_GROUND_PHRASE_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        **_IMAGE_INPUT_PROPS,
        "phrase": {
            "type": "string",
            "description": "Single natural-language phrase to localize (e.g. 'the red mug on the left').",
        },
    },
    "required": ["phrase"],
}

_GROUND_PHRASE_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "detection": {
            "oneOf": [_DETECTION_SCHEMA, {"type": "null"}],
            "description": "Best-matching detection, or null if no match above threshold.",
        },
        "latency_ms": {"type": "number"},
        "model_version": {"type": "string"},
    },
    "required": ["detection"],
}

_SEGMENT_PROMPTABLE_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        **_IMAGE_INPUT_PROPS,
        "phrase": {"type": "string"},
    },
    "required": ["phrase"],
}

_SEGMENT_PROMPTABLE_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mask_b64_png": {"type": "string", "description": "Binary PNG mask, base64-encoded."},
        "bbox": _BBOX_SCHEMA,
        "confidence": {"type": "number"},
        "latency_ms": {"type": "number"},
        "model_version": {"type": "string"},
    },
    "required": ["mask_b64_png"],
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_perception_tools(server_url: str) -> list[MCPTool]:
    """Construct the perception MCPTool set pointed at ``server_url``.

    Returns the four ``perception.*`` tools (detect / depth / ground / segment),
    ready to register with a :class:`ToolRegistry`.
    """
    return [
        MCPTool(
            tool_name=PERCEPTION_DETECT_OBJECTS,
            description=(
                "Open-vocabulary object detection. Takes a base64 image and a list of "
                "text prompts; returns matching detections with normalized bboxes."
            ),
            input_schema=_DETECT_OBJECTS_INPUT,
            output_schema=_DETECT_OBJECTS_OUTPUT,
            server_url=server_url,
            is_idempotent=True,
        ),
        MCPTool(
            tool_name=PERCEPTION_ESTIMATE_DEPTH,
            description=(
                "Monocular depth estimation. Returns a 16-bit depth map as a "
                "base64-encoded PNG, with relative or metric scale."
            ),
            input_schema=_ESTIMATE_DEPTH_INPUT,
            output_schema=_ESTIMATE_DEPTH_OUTPUT,
            server_url=server_url,
            is_idempotent=True,
        ),
        MCPTool(
            tool_name=PERCEPTION_GROUND_PHRASE,
            description=(
                "Localize a single natural-language phrase in an image. "
                "Returns the best-matching detection or null."
            ),
            input_schema=_GROUND_PHRASE_INPUT,
            output_schema=_GROUND_PHRASE_OUTPUT,
            server_url=server_url,
            is_idempotent=True,
        ),
        MCPTool(
            tool_name=PERCEPTION_SEGMENT_PROMPTABLE,
            description=(
                "Promptable segmentation. Returns a binary PNG mask for the region of "
                "the image matching the given phrase, plus an enclosing bbox."
            ),
            input_schema=_SEGMENT_PROMPTABLE_INPUT,
            output_schema=_SEGMENT_PROMPTABLE_OUTPUT,
            server_url=server_url,
            is_idempotent=True,
        ),
    ]


def register_perception_tools(
    registry: ToolRegistry,
    server_url: str,
    *,
    resolve_artifacts: bool = True,
) -> list[Tool]:
    """Build the perception tools and register them, resolver-wrapped by default.

    Each tool is wrapped with :class:`ArtifactResolverMiddleware` so a ``frame``
    ref from ``robot.capture_frame`` is hydrated into ``image_b64`` before the
    request reaches the external server. Pass ``resolve_artifacts=False`` to
    register the raw tools (e.g. when image bytes are always supplied inline).

    Returns the registered tools (wrapped, when applicable).
    """
    registered: list[Tool] = []
    for tool in build_perception_tools(server_url):
        wrapped: Tool = (
            build_chain(tool, [ArtifactResolverMiddleware]) if resolve_artifacts else tool
        )
        registry.register(wrapped)
        registered.append(wrapped)
    return registered


async def verify_server_compatibility(
    session: MCPClientSession,
    *,
    expected_names: tuple[str, ...] = PERCEPTION_TOOL_NAMES,
) -> None:
    """Fail-fast check that ``session``'s server publishes ``expected_names``.

    Call this at harness startup, after constructing the tools but before
    starting the agent loop. Raises :class:`ToolBackendUnreachableError` with
    the missing names if the server's catalogue is incomplete.
    """
    result = await session.list_tools()
    published = {tool.name for tool in result.tools}
    missing = [name for name in expected_names if name not in published]
    if missing:
        raise ToolBackendUnreachableError(
            f"MCP server at {session.server_url!r} is missing expected perception tools: "
            f"{missing} (published: {sorted(published)})",
            module_name="tools.perception.mcp_bundle",
        )
