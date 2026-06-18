"""Navigation tool bundle backed by an external MCP server."""

from __future__ import annotations

from typing import Any

from robot_harness.errors import ToolBackendUnreachableError
from robot_harness.tools.mcp.client import MCPClientSession, MCPTool

NAVIGATION_PLAN_PATH = "navigation.plan_path"

NAVIGATION_TOOL_NAMES: tuple[str, ...] = (NAVIGATION_PLAN_PATH,)

_XY_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 2,
    "maxItems": 2,
    "description": "[x, y] in odom frame, meters.",
}

_WAYPOINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "xy": _XY_SCHEMA,
        "yaw": {"type": "number", "description": "Yaw in radians."},
    },
    "required": ["xy", "yaw"],
}

_PLAN_PATH_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start_xy": _XY_SCHEMA,
        "goal_xy": _XY_SCHEMA,
        "goal_name": {
            "type": "string",
            "description": "Optional semantic target name, for example 'kitchen'.",
        },
        "constraints": {
            "type": "object",
            "properties": {
                "avoid_zones": {
                    "type": "array",
                    "description": "Optional zones to avoid, e.g. [['box_a', [1, 1, 2, 2]]].",
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "max_speed_mps": {"type": "number", "minimum": 0.0},
                "instruction": {
                    "type": "string",
                    "description": "VLN instruction for image-conditioned navigation backends.",
                },
                "image_source": {
                    "type": "string",
                    "description": "RGB image path or camera frame identifier for VLN backends.",
                },
                "depth_source": {
                    "type": "string",
                    "description": "Depth image path or camera frame identifier for VLN backends.",
                },
                "current_yaw": {
                    "type": "number",
                    "description": "Current robot yaw in radians, used to place local VLN plans in odom.",
                },
                "camera_pose": {
                    "type": "array",
                    "description": "Optional 4x4 camera pose matrix.",
                },
                "camera_intrinsic": {
                    "type": "array",
                    "description": "Optional camera intrinsic matrix.",
                },
                "reset": {
                    "type": "boolean",
                    "description": "Reset episodic model state before planning.",
                },
                "look_down": {
                    "type": "boolean",
                    "description": "Request a look-down DualVLN refinement step when supported.",
                },
            },
        },
    },
    "required": ["start_xy", "goal_xy"],
}

_PLAN_PATH_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "waypoints": {"type": "array", "items": _WAYPOINT_SCHEMA},
        "duration_est_s": {"type": "number", "minimum": 0.0},
        "plan_id": {"type": "string"},
    },
    "required": ["waypoints", "duration_est_s", "plan_id"],
}

def build_navigation_tools(server_url: str) -> list[MCPTool]:
    """Construct navigation MCP tools pointed at ``server_url``."""
    return [
        MCPTool(
            tool_name=NAVIGATION_PLAN_PATH,
            description=(
                "Plan a path in odom frame from start_xy to goal_xy using an external "
                "navigation planner."
            ),
            input_schema=_PLAN_PATH_INPUT,
            output_schema=_PLAN_PATH_OUTPUT,
            server_url=server_url,
            is_idempotent=True,
        )
    ]


async def verify_server_compatibility(
    session: MCPClientSession,
    *,
    expected_names: tuple[str, ...] = NAVIGATION_TOOL_NAMES,
) -> None:
    """Fail fast if the remote MCP server does not publish expected navigation tools."""
    result = await session.list_tools()
    published = {tool.name for tool in result.tools}
    missing = [name for name in expected_names if name not in published]
    if missing:
        raise ToolBackendUnreachableError(
            f"MCP server at {session.server_url!r} is missing expected navigation tools: "
            f"{missing} (published: {sorted(published)})",
            module_name="tools.navigation.mcp_bundle",
        )
