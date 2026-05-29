"""Pydantic configuration schema for Robot Loom."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BrainConfig(BaseModel):
    model: str = "openai/gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: float = 60.0
    max_replan_iterations: int = 5
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"


class MiddlewareSpec(BaseModel):
    """Declarative spec for one middleware in a tool's chain.

    ``type`` selects the middleware class; remaining fields are forwarded as
    keyword arguments to its constructor.

    Supported types:
        trace, timeout, retry, cancel, circuit_breaker, cache, rate_limit
    """

    type: Literal[
        "trace",
        "timeout",
        "retry",
        "cancel",
        "circuit_breaker",
        "cache",
        "rate_limit",
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class ToolConfig(BaseModel):
    default_timeout_s: float = 30.0
    max_retries: int = 3
    retry_backoff_base_s: float = 1.0
    default_middleware: list[MiddlewareSpec] = Field(default_factory=list)


class SafetyConfig(BaseModel):
    enabled: bool = True
    max_joint_velocity_rad_s: float = 1.0
    workspace_bounds_m: list[float] = Field(
        default_factory=lambda: [-2.0, -2.0, 0.0, 2.0, 2.0, 2.0]
    )


class MemoryConfig(BaseModel):
    """Memory backend selection.

    ``backend`` chooses where observations / decisions / outcomes are stored:

    - ``null``     — no-op store; nothing is persisted. Queries always return
      empty. Safe default when no backend is wired.
    - ``embedded`` — in-process dict-backed store. Lives in the harness process,
      so it is not shared across robots and does not survive a restart.
    - ``external`` — remote SpatialMemory-like server reached over HTTP. Shared
      across robots and durable across restarts. Requires ``server_url``.

    Only ``external`` can back a multi-robot fleet, because the embedded and null
    stores cannot be shared between robots.
    """

    backend: Literal["null", "embedded", "external"] = "embedded"
    server_url: str = ""
    request_timeout_s: float = 10.0


class ObservabilityConfig(BaseModel):
    trace_sink: Literal["stderr", "file"] = "stderr"
    trace_file: str = ""
    otel_endpoint: str = ""


class HarnessConfig(BaseModel):
    brain: BrainConfig = Field(default_factory=BrainConfig)
    tool: ToolConfig = Field(default_factory=ToolConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    fleet_size: int = 1
    robot_ids: list[str] = Field(default_factory=lambda: ["robot-0"])
