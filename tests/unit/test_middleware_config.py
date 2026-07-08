"""Tests for the declarative middleware chain builder."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from robot_harness.config.schema import MiddlewareSpec, ToolConfig
from robot_harness.tools.base import ToolContext, ToolResult
from robot_harness.tools.middleware.base import ToolMiddleware
from robot_harness.tools.middleware.cache import CacheMiddleware
from robot_harness.tools.middleware.circuit_breaker import CircuitBreakerMiddleware
from robot_harness.tools.middleware.config import (
    UnknownMiddlewareTypeError,
    build_middleware_chain_from_config,
    register_middleware_type,
)
from robot_harness.tools.middleware.rate_limit import RateLimitMiddleware
from robot_harness.tools.middleware.retry import RetryMiddleware
from robot_harness.tools.middleware.timeout import TimeoutMiddleware
from robot_harness.tools.middleware.trace import TraceMiddleware
from robot_harness.tools.schema import ToolBackend, ToolSchema

# ---------------------------------------------------------------------------
# Minimal Tool fixture
# ---------------------------------------------------------------------------


class _EchoTool:
    name = "echo"
    backend: ToolBackend = "native"
    schema = ToolSchema(
        name="echo",
        description="Echo args",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    is_idempotent = True
    is_cancellable = False

    async def invoke(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(success=True, output=args, tool_name=self.name, trace_id=ctx.trace_id)

    async def cancel(self, ctx: ToolContext) -> None:
        pass


# ---------------------------------------------------------------------------
# MiddlewareSpec schema tests
# ---------------------------------------------------------------------------


class TestMiddlewareSpec:
    def test_type_only(self) -> None:
        spec = MiddlewareSpec(type="trace")
        assert spec.type == "trace"
        assert spec.params == {}

    def test_with_params(self) -> None:
        spec = MiddlewareSpec(type="retry", params={"max_retries": 5})
        assert spec.params["max_retries"] == 5

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MiddlewareSpec(type="does_not_exist")  # type: ignore[arg-type]


class TestToolConfig:
    def test_default_middleware_is_empty(self) -> None:
        cfg = ToolConfig()
        assert cfg.default_middleware == []

    def test_parse_middleware_list(self) -> None:
        cfg = ToolConfig.model_validate(
            {
                "default_middleware": [
                    {"type": "trace"},
                    {"type": "retry", "params": {"max_retries": 3}},
                ]
            }
        )
        assert len(cfg.default_middleware) == 2
        assert cfg.default_middleware[0].type == "trace"
        assert cfg.default_middleware[1].params["max_retries"] == 3


# ---------------------------------------------------------------------------
# build_middleware_chain_from_config tests
# ---------------------------------------------------------------------------


class TestBuildChain:
    def setup_method(self) -> None:
        self.tool = _EchoTool()

    def test_empty_specs_returns_tool_unchanged(self) -> None:
        result = build_middleware_chain_from_config(self.tool, [])
        assert result is self.tool

    def test_single_trace(self) -> None:
        chain = build_middleware_chain_from_config(self.tool, [MiddlewareSpec(type="trace")])
        assert isinstance(chain, TraceMiddleware)
        assert chain._inner is self.tool

    def test_single_timeout(self) -> None:
        chain = build_middleware_chain_from_config(self.tool, [MiddlewareSpec(type="timeout")])
        assert isinstance(chain, TimeoutMiddleware)

    def test_retry_with_params(self) -> None:
        chain = build_middleware_chain_from_config(
            self.tool,
            [MiddlewareSpec(type="retry", params={"max_retries": 7, "backoff_base_s": 0.1})],
        )
        assert isinstance(chain, RetryMiddleware)
        assert chain._max_retries == 7

    def test_circuit_breaker_with_params(self) -> None:
        chain = build_middleware_chain_from_config(
            self.tool,
            [MiddlewareSpec(type="circuit_breaker", params={"failure_threshold": 3})],
        )
        assert isinstance(chain, CircuitBreakerMiddleware)
        assert chain._failure_threshold == 3

    def test_cache_with_params(self) -> None:
        chain = build_middleware_chain_from_config(
            self.tool,
            [MiddlewareSpec(type="cache", params={"ttl_s": 120, "max_size": 64})],
        )
        assert isinstance(chain, CacheMiddleware)
        assert chain._ttl_s == 120

    def test_rate_limit_with_params(self) -> None:
        chain = build_middleware_chain_from_config(
            self.tool,
            [MiddlewareSpec(type="rate_limit", params={"max_rps": 20.0})],
        )
        assert isinstance(chain, RateLimitMiddleware)

    def test_ordering_outermost_first(self) -> None:
        # specs: [trace, retry] → outer=trace wraps inner=retry wraps tool
        chain = build_middleware_chain_from_config(
            self.tool,
            [MiddlewareSpec(type="trace"), MiddlewareSpec(type="retry")],
        )
        assert isinstance(chain, TraceMiddleware)
        assert isinstance(chain._inner, RetryMiddleware)
        assert chain._inner._inner is self.tool

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(UnknownMiddlewareTypeError, match="bogus"):
            build_middleware_chain_from_config(
                self.tool,
                # bypass Pydantic validation by patching the spec directly
                [MiddlewareSpec.model_construct(type="bogus", params={})],  # type: ignore[arg-type]
            )

    async def test_chain_is_invocable(self) -> None:
        chain = build_middleware_chain_from_config(
            self.tool,
            [MiddlewareSpec(type="trace"), MiddlewareSpec(type="timeout")],
        )
        ctx = ToolContext(trace_id="t1", robot_id="r0", timeout_s=5.0)
        result = await chain.invoke({"x": "hi"}, ctx)
        assert result.success

    def test_full_stack(self) -> None:
        specs = [
            MiddlewareSpec(type="trace"),
            MiddlewareSpec(type="timeout"),
            MiddlewareSpec(type="retry", params={"max_retries": 2}),
            MiddlewareSpec(type="circuit_breaker", params={"failure_threshold": 5}),
            MiddlewareSpec(type="cache", params={"ttl_s": 30}),
        ]
        chain = build_middleware_chain_from_config(self.tool, specs)
        assert isinstance(chain, TraceMiddleware)
        assert isinstance(chain._inner, TimeoutMiddleware)
        assert isinstance(chain._inner._inner, RetryMiddleware)


# ---------------------------------------------------------------------------
# register_middleware_type tests
# ---------------------------------------------------------------------------


class TestRegisterCustomType:
    def test_register_and_use(self) -> None:
        class _NoopMiddleware(ToolMiddleware):
            pass

        register_middleware_type("noop", _NoopMiddleware)
        tool = _EchoTool()
        chain = build_middleware_chain_from_config(
            tool,
            [MiddlewareSpec.model_construct(type="noop", params={})],  # type: ignore[arg-type]
        )
        assert isinstance(chain, _NoopMiddleware)


# ---------------------------------------------------------------------------
# Capability markers survive wrapping; config chain is actually applied (ISS-037)
# ---------------------------------------------------------------------------


def test_middleware_forwards_hardware_markers() -> None:
    """Wrapping must never strip hardware_bound / to_safety_command /
    brain_visible — a wrapped hardware tool that lost its markers would
    silently bypass the SafetyEnvelope gate."""
    from robot_harness.tools.base import ToolRegistry
    from robot_harness.tools.middleware.trace import TraceMiddleware
    from robot_harness.tools.robot_sdk.http_adapter import RobotSdkTool

    wrapped = TraceMiddleware(RobotSdkTool())
    assert wrapped.hardware_bound is True
    assert wrapped.brain_visible is False

    registry = ToolRegistry()
    registry.register(wrapped)
    assert registry.requires_safety_check("robot_sdk.execute_action")
    from robot_harness.tools.base import ToolContext

    cmd = registry.build_safety_command(
        "robot_sdk.execute_action",
        {"robot_id": "r0", "command_type": "cartesian", "values": [0.1, 0.2, 0.3]},
        ToolContext.create("r0"),
    )
    assert cmd is not None
    assert cmd.values == [0.1, 0.2, 0.3]


def test_harness_context_applies_default_middleware() -> None:
    """tool.default_middleware in the workspace config must actually wrap the
    framework-registered tools — a declared chain that never applies is dead
    configuration (ISS-037)."""
    from robot_harness.config.schema import HarnessConfig, MiddlewareSpec, ToolConfig
    from robot_harness.runtime.harness_context import HarnessContext
    from robot_harness.tools.middleware.trace import TraceMiddleware

    cfg = HarnessConfig(
        tool=ToolConfig(default_middleware=[MiddlewareSpec(type="trace")]),
    )
    ctx = HarnessContext.build(config=cfg)
    tool = ctx.tool_registry.get("robot_sdk.execute_action")
    assert isinstance(tool, TraceMiddleware)
    # The wrapped hardware tool stays safety-gated.
    assert ctx.tool_registry.requires_safety_check("robot_sdk.execute_action")
    assert ctx.tool_registry.requires_safety_check("robot_sdk.reactive_grasp")
