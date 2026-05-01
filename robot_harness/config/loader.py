"""Load and merge HarnessConfig from YAML files and environment overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robot_harness.config.paths import workspace_config_file
from robot_harness.config.schema import HarnessConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(extra_path: Path | None = None) -> HarnessConfig:
    """Load HarnessConfig.

    Merge order (last wins): defaults → workspace config.yaml → extra_path.
    """
    data: dict[str, Any] = {}

    workspace_cfg = workspace_config_file()
    if workspace_cfg.exists():
        with workspace_cfg.open() as fh:
            loaded = yaml.safe_load(fh) or {}
        data = _deep_merge(data, loaded)

    if extra_path is not None and extra_path.exists():
        with extra_path.open() as fh:
            loaded = yaml.safe_load(fh) or {}
        data = _deep_merge(data, loaded)

    return HarnessConfig.model_validate(data)
