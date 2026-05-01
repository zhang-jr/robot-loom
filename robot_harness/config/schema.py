"""Pydantic configuration schema for Robot Loom."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BrainConfig(BaseModel):
    model: str = "openai/gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: float = 60.0
    max_replan_iterations: int = 5
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"


class ToolConfig(BaseModel):
    default_timeout_s: float = 30.0
    max_retries: int = 3
    retry_backoff_base_s: float = 1.0


class SafetyConfig(BaseModel):
    enabled: bool = True
    max_joint_velocity_rad_s: float = 1.0
    workspace_bounds_m: list[float] = Field(
        default_factory=lambda: [-2.0, -2.0, 0.0, 2.0, 2.0, 2.0]
    )


class ObservabilityConfig(BaseModel):
    trace_sink: Literal["stderr", "file"] = "stderr"
    trace_file: str = ""
    otel_endpoint: str = ""


class HarnessConfig(BaseModel):
    brain: BrainConfig = Field(default_factory=BrainConfig)
    tool: ToolConfig = Field(default_factory=ToolConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    fleet_size: int = 1
    robot_ids: list[str] = Field(default_factory=lambda: ["robot-0"])
