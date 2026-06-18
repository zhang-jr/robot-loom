"""Regression tests for tools registered but hidden from Brain planning."""

from __future__ import annotations

from robot_harness.tools.base import BrainProfile, ToolRegistry
from robot_harness.tools.vla.serving_adapter import VlaServingTool


def test_vla_single_step_tool_is_not_brain_visible() -> None:
    tool = VlaServingTool()

    assert getattr(tool, "brain_visible", True) is False


def test_brain_export_omits_brain_invisible_tools() -> None:
    registry = ToolRegistry()
    registry.register(VlaServingTool())

    specs = registry.export_for_brain(BrainProfile(name="openai"))

    exported_names = {spec["function"]["name"] for spec in specs}
    assert "vla.infer_action" not in exported_names
