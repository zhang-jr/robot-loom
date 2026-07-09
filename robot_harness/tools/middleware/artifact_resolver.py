"""ArtifactResolverMiddleware — hydrate ArtifactRef inputs before dispatch.

A producing tool (e.g. ``robot.capture_frame``) returns a small
:class:`~robot_harness.tools.artifacts.ArtifactRef` instead of inlining a large
image. A consuming tool (e.g. ``perception.detect_objects``) is called with that
ref. This middleware sits in front of the consumer: just before ``invoke`` it
replaces each known ref field with the wire field the backend expects
(``frame`` → ``image_b64``, ``depth`` → ``depth_b64``), pulling the bytes from
``ctx.artifact_store``. The large binary therefore never travels through the LLM.

Keeping this as middleware (not adapter logic) follows the cross-cutting-concern
rule: the perception adapters stay unaware of the artifact data plane.
"""

from __future__ import annotations

import base64
from typing import Any

from robot_harness.errors import ToolSchemaViolationError
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.middleware.base import ToolMiddleware


class ArtifactResolverMiddleware(ToolMiddleware):
    """Resolve ArtifactRef args into the inline wire fields the backend expects."""

    # ref arg field → wire field it hydrates into (base64 string)
    _RESOLUTIONS = (("frame", "image_b64"), ("depth", "depth_b64"))

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = ctx.artifact_store
        hydrated: dict[str, Any] | None = None
        for ref_field, wire_field in self._RESOLUTIONS:
            ref = args.get(ref_field)
            if not isinstance(ref, dict) or "artifact_id" not in ref:
                continue
            if store is None:
                raise ToolSchemaViolationError(
                    f"tool '{self.name}' got a '{ref_field}' artifact ref but no artifact "
                    "store is wired to resolve it",
                    tool_name=self.name,
                )
            artifact_id = ref["artifact_id"]
            item = await store.get(artifact_id)
            if item is None:
                raise ToolSchemaViolationError(
                    f"artifact '{artifact_id}' for '{ref_field}' not found in the store "
                    "(evicted or never produced)",
                    tool_name=self.name,
                )
            if hydrated is None:
                hydrated = dict(args)
            data, _ref = item
            hydrated.pop(ref_field, None)
            hydrated[wire_field] = base64.b64encode(data).decode("ascii")

        return await self._inner.invoke(hydrated if hydrated is not None else args, ctx)
