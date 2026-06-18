"""Minimal Robot Loom VLA mock tool-call smoke test.

This bypasses the Brain and directly verifies the ToolRegistry -> Tool adapter
path for ``vla.infer_action``. Real VLA serving is consumed inside on-robot
rollout verbs, not wired to the harness by URL.

    uv run python -m examples.vla_recipe_minimal
"""

from __future__ import annotations

import asyncio
import os

from robot_harness.tools.base import ToolContext, ToolRegistry
from robot_harness.tools.vla import VlaServingTool

VLA_INFER_ACTION = "vla.infer_action"


async def main() -> None:
    registry = ToolRegistry()
    registry.register(VlaServingTool())

    tool = registry.get(VLA_INFER_ACTION)
    result = await tool.invoke(
        {
            "image_source": os.environ.get("VLA_IMAGE_SOURCE", "head_camera"),
            "instruction": os.environ.get("VLA_INSTRUCTION", "pick up the red cube"),
            "robot_id": os.environ.get("ROBOT_ID", "robot-0"),
            "num_actions": int(os.environ.get("VLA_NUM_ACTIONS", "1")),
        },
        ToolContext.create(os.environ.get("ROBOT_ID", "robot-0")),
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
