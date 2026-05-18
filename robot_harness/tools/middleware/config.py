"""Declarative middleware chain builder.

Reads a list of :class:`~robot_harness.config.schema.MiddlewareSpec` and
wraps a :class:`~robot_harness.tools.base.Tool` with the corresponding
middleware in declaration order (first entry = outermost wrapper).

Example workspace ``config.yaml``::

    tool:
      default_middleware:
        - type: trace
        - type: timeout
        - type: retry
          params:
            max_retries: 5
            backoff_base_s: 0.5
        - type: circuit_breaker
          params:
            failure_threshold: 3
            reset_timeout_s: 60.0
        - type: rate_limit
          params:
            max_rps: 20.0
            max_rps_per_robot: 10.0
        - type: cache
          params:
            ttl_s: 120
            max_size: 512
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_harness.tools.middleware.base import ToolMiddleware
from robot_harness.tools.middleware.cache import CacheMiddleware
from robot_harness.tools.middleware.cancel import CancelMiddleware
from robot_harness.tools.middleware.circuit_breaker import CircuitBreakerMiddleware
from robot_harness.tools.middleware.rate_limit import RateLimitMiddleware
from robot_harness.tools.middleware.retry import RetryMiddleware
from robot_harness.tools.middleware.timeout import TimeoutMiddleware
from robot_harness.tools.middleware.trace import TraceMiddleware

if TYPE_CHECKING:
    from robot_harness.config.schema import MiddlewareSpec
    from robot_harness.tools.base import Tool

_TYPE_MAP: dict[str, type[ToolMiddleware]] = {
    "trace": TraceMiddleware,
    "timeout": TimeoutMiddleware,
    "retry": RetryMiddleware,
    "cancel": CancelMiddleware,
    "circuit_breaker": CircuitBreakerMiddleware,
    "cache": CacheMiddleware,
    "rate_limit": RateLimitMiddleware,
}


class UnknownMiddlewareTypeError(ValueError):
    """Raised when a spec references an unregistered middleware type."""


def build_middleware_chain_from_config(
    tool: Tool,
    specs: list[MiddlewareSpec],
) -> Tool:
    """Wrap *tool* with middleware described by *specs*.

    Specs are applied outermost-first (same convention as
    :func:`~robot_harness.tools.middleware.base.build_chain`).  An empty
    list returns the tool unchanged.

    Args:
        tool: The inner tool to wrap.
        specs: Ordered list of :class:`~robot_harness.config.schema.MiddlewareSpec`.

    Returns:
        The tool wrapped in the declared middleware chain.

    Raises:
        UnknownMiddlewareTypeError: If a spec names an unregistered type.
    """
    if not specs:
        return tool

    wrapped: Tool = tool
    for spec in reversed(specs):
        cls = _TYPE_MAP.get(spec.type)
        if cls is None:
            raise UnknownMiddlewareTypeError(
                f"Unknown middleware type '{spec.type}'. Valid types: {sorted(_TYPE_MAP)}"
            )
        kwargs: dict[str, Any] = spec.params
        wrapped = cls(wrapped, **kwargs)
    return wrapped


def register_middleware_type(name: str, cls: type[ToolMiddleware]) -> None:
    """Register a custom middleware class under *name* for use in YAML config.

    Allows workspace-level or plugin code to extend the type map without
    modifying framework source.
    """
    _TYPE_MAP[name] = cls
