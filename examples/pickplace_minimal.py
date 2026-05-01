"""Minimal pick-and-place demo using mock tools and a real LiteLLM Brain.

Usage:
    uv run python examples/pickplace_minimal.py

Requires OPENAI_API_KEY (or set brain.model to a local endpoint in config.yaml).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

from robot_harness.brain.base import Task
from robot_harness.brain.litellm_brain import LiteLLMBrain
from robot_harness.config.schema import HarnessConfig
from robot_harness.runtime.agent_loop import AgentLoop
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema

# ---------------------------------------------------------------------------
# Mock perception tool
# ---------------------------------------------------------------------------


class DetectObjectTool:
    name = "detect_object"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="detect_object",
        description="Detect objects in the scene and return their positions.",
        input_schema={
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "description": "Name of the object to detect"},
            },
            "required": ["object_name"],
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = args["object_name"]
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={
                "found": True,
                "object": name,
                "position": {"x": 0.3, "y": 0.1, "z": 0.05},
                "confidence": 0.95,
            },
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass


# ---------------------------------------------------------------------------
# Mock grasp + place tools
# ---------------------------------------------------------------------------


class GraspObjectTool:
    name = "grasp_object"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="grasp_object",
        description="Grasp an object at a given position.",
        input_schema={
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "position": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                    },
                },
            },
            "required": ["object_name"],
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return False

    @property
    def is_cancellable(self) -> bool:
        return True

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"grasped": args["object_name"], "grip_force_n": 5.2},
        )

    async def cancel(self, ctx: ToolContext) -> None:
        ctx.cancel()


class PlaceObjectTool:
    name = "place_object"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="place_object",
        description="Place the currently held object at a target location.",
        input_schema={
            "type": "object",
            "properties": {
                "target_location": {"type": "string", "description": "Where to place the object"},
            },
            "required": ["target_location"],
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return False

    @property
    def is_cancellable(self) -> bool:
        return True

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={
                "placed_at": args["target_location"],
                "task_complete": True,
            },
        )

    async def cancel(self, ctx: ToolContext) -> None:
        ctx.cancel()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    # Build config — use local Ollama if no OpenAI key present
    model = "openai/gpt-4o-mini"
    if not os.environ.get("OPENAI_API_KEY"):
        model = "ollama/qwen2.5:7b"
        print("No OPENAI_API_KEY — falling back to ollama/qwen2.5:7b")

    config = HarnessConfig()
    config.brain.model = model

    ctx = HarnessContext.build(config)
    ctx.tool_registry.register(DetectObjectTool())
    ctx.tool_registry.register(GraspObjectTool())
    ctx.tool_registry.register(PlaceObjectTool())

    brain = LiteLLMBrain(ctx.config.brain)
    loop = AgentLoop(brain, ctx, max_turns=10)

    task = Task(
        task_id=str(uuid.uuid4()),
        description="Pick up the red cup and place it on the tray.",
        robot_id="robot-0",
    )

    print(f"Running task: {task.description}")
    result = await loop.run(task)

    print(f"\nOutcome : {result.outcome}")
    print(f"Turns   : {result.turns}")
    print(f"Message : {result.message}")


if __name__ == "__main__":
    asyncio.run(main())
