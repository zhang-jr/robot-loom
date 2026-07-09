"""Unit tests for tool_servers config wiring (ADR-026).

A non-empty URL in ``HarnessConfig.tool_servers`` must make
``HarnessContext.build`` register the corresponding tool bundle; an empty
URL must leave the registry without it.
"""

from __future__ import annotations

from robot_harness.config.schema import HarnessConfig, ToolServersConfig
from robot_harness.runtime.harness_context import HarnessContext
from robot_harness.tools.perception.mcp_bundle import PERCEPTION_TOOL_NAMES


def test_tool_servers_defaults_to_empty() -> None:
    cfg = HarnessConfig()
    assert cfg.tool_servers.perception == ""


def test_tool_servers_parses_from_config_mapping() -> None:
    cfg = HarnessConfig.model_validate(
        {"tool_servers": {"perception": "http://localhost:7100/mcp"}}
    )
    assert cfg.tool_servers.perception == "http://localhost:7100/mcp"


def test_build_without_perception_url_registers_no_perception_tools() -> None:
    ctx = HarnessContext.build(HarnessConfig())
    for name in PERCEPTION_TOOL_NAMES:
        assert name not in ctx.tool_registry


def test_build_with_perception_url_registers_perception_tools() -> None:
    cfg = HarnessConfig(tool_servers=ToolServersConfig(perception="http://localhost:7100/mcp"))
    ctx = HarnessContext.build(cfg)
    for name in PERCEPTION_TOOL_NAMES:
        assert name in ctx.tool_registry
