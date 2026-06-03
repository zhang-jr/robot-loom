"""VLA tool bundle backed by an external MCP server."""

from __future__ import annotations

from typing import Any

from robot_harness.errors import ToolBackendUnreachableError
from robot_harness.tools.mcp.client import MCPClientSession, MCPTool

VLA_INFER_ACTION = "vla.infer_action"

VLA_TOOL_NAMES: tuple[str, ...] = (VLA_INFER_ACTION,)

_ACTION_VECTOR_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": [
        {"type": "number"},
        {"type": "number"},
        {"type": "number"},
        {"type": "number"},
        {"type": "number"},
        {"type": "number"},
        {
            "type": "number",
            "enum": [0, 1],
            "description": "Gripper command: 0 opens, 1 closes.",
        },
    ],
    "additionalItems": False,
    "minItems": 7,
    "maxItems": 7,
    "description": (
        "Single robot action: 6D cartesian EE delta plus gripper "
        "[dx, dy, dz, drx, dry, drz, gripper]. Gripper is binary: "
        "0 opens and 1 closes."
    ),
}

_INFER_ACTION_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "image_source": {
            "type": "string",
            "description": "Camera identifier, for example 'head_camera'.",
        },
        "instruction": {
            "type": "string",
            "description": "Natural-language manipulation instruction.",
        },
        "robot_id": {
            "type": "string",
            "description": "Robot instance identifier used to select the action head.",
        },
        "num_actions": {
            "type": "integer",
            "default": 1,
            "minimum": 1,
            "maximum": 16,
            "description": "Number of action steps to predict.",
        },
    },
    "required": ["image_source", "instruction", "robot_id"],
}

_INFER_ACTION_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": _ACTION_VECTOR_SCHEMA,
            "description": "Predicted action chunk.",
        },
        "model_id": {"type": "string"},
    },
    "required": ["actions", "model_id"],
}


def build_vla_tools(server_url: str) -> list[MCPTool]:
    """Construct VLA MCP tools pointed at ``server_url``."""
    return [
        MCPTool(
            tool_name=VLA_INFER_ACTION,
            description=(
                "Infer robot action deltas from an image source and language instruction "
                "using an external VLA serving runtime."
            ),
            input_schema=_INFER_ACTION_INPUT,
            output_schema=_INFER_ACTION_OUTPUT,
            server_url=server_url,
            is_idempotent=False,
        )
    ]


async def verify_server_compatibility(
    session: MCPClientSession,
    *,
    expected_names: tuple[str, ...] = VLA_TOOL_NAMES,
) -> None:
    """Fail fast if the remote MCP server does not publish expected VLA tools."""
    result = await session.list_tools()
    published = {tool.name for tool in result.tools}
    missing = [name for name in expected_names if name not in published]
    if missing:
        raise ToolBackendUnreachableError(
            f"MCP server at {session.server_url!r} is missing expected VLA tools: "
            f"{missing} (published: {sorted(published)})",
            module_name="tools.vla.mcp_bundle",
        )
