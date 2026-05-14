"""CacheMiddleware — opt-in result cache for idempotent tools.

Only tools marked ``is_idempotent=True`` are cacheable.  Non-idempotent tools
bypass the cache unconditionally.

Cache key: ``(tool_name, sorted_args_json)``.  Results are evicted after
``ttl_s`` seconds (default 60 s) or when the cache exceeds ``max_size`` entries
(LRU eviction via ordered dict).
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any

from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.middleware.base import ToolMiddleware


def _cache_key(tool_name: str, args: dict[str, Any]) -> str:
    return f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"


class CacheMiddleware(ToolMiddleware):
    """TTL + LRU cache for idempotent tool invocations."""

    def __init__(
        self,
        inner: Any,
        ttl_s: float = 60.0,
        max_size: int = 256,
    ) -> None:
        super().__init__(inner)
        self._ttl_s = ttl_s
        self._max_size = max_size
        # key → (result, inserted_at)
        self._cache: OrderedDict[str, tuple[ToolResult, float]] = OrderedDict()

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not self._inner.is_idempotent:
            return await self._inner.invoke(args, ctx)

        key = _cache_key(self._inner.name, args)
        now = time.monotonic()

        if key in self._cache:
            result, inserted_at = self._cache[key]
            if now - inserted_at < self._ttl_s:
                self._cache.move_to_end(key)  # LRU refresh
                return result
            else:
                del self._cache[key]

        result = await self._inner.invoke(args, ctx)

        if result.success:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # evict LRU
            self._cache[key] = (result, now)

        return result

    def invalidate(self, args: dict[str, Any] | None = None) -> None:
        """Evict a specific entry or flush the entire cache."""
        if args is None:
            self._cache.clear()
        else:
            key = _cache_key(self._inner.name, args)
            self._cache.pop(key, None)

    @property
    def cache_size(self) -> int:
        return len(self._cache)
