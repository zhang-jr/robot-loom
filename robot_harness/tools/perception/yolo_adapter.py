"""YoloDetectionTool — mock adapter for YOLO-based object detection server."""

from __future__ import annotations

import time
from typing import Any

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.schema import ToolBackend, ToolSchema


class YoloDetectionTool:
    """Mock YOLO detection tool.

    Phase 1: returns synthetic detections without contacting any server.
    Phase 2: will POST to an external perception server (MCP or HTTP).
    """

    name = "perception.detect_objects"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="perception.detect_objects",
        description="Detect objects in a camera frame using YOLO.",
        input_schema={
            "type": "object",
            "properties": {
                "image_source": {
                    "type": "string",
                    "description": "Camera identifier, e.g. 'wrist_camera' or 'head_camera'",
                },
                "query": {
                    "type": "string",
                    "description": "Object class or name to detect",
                },
                "confidence_threshold": {"type": "number", "default": 0.5},
            },
            "required": ["image_source", "query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "detections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "confidence": {"type": "number"},
                            "bbox_xyxy": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                        },
                    },
                },
                "count": {"type": "integer"},
            },
        },
    )

    @property
    def is_idempotent(self) -> bool:
        return True

    @property
    def is_cancellable(self) -> bool:
        return False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        t0 = time.monotonic()
        query = args.get("query", "object")
        conf_thresh = float(args.get("confidence_threshold", 0.5))

        # Synthetic detection — always returns one mock object
        detections = (
            [
                {
                    "label": query,
                    "confidence": 0.92,
                    "bbox_xyxy": [120.0, 80.0, 320.0, 280.0],
                    "center_xy": [220.0, 180.0],
                }
            ]
            if conf_thresh <= 0.92
            else []
        )

        latency = (time.monotonic() - t0) * 1000
        return ToolResult(
            tool_name=self.name,
            trace_id=ctx.trace_id,
            success=True,
            output={"detections": detections, "count": len(detections)},
            latency_ms=latency,
        )

    async def cancel(self, ctx: ToolContext) -> None:
        pass
