"""Load and merge HarnessConfig from YAML files and environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from robot_harness.config.paths import workspace_config_file, workspace_root
from robot_harness.config.schema import HarnessConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from ``path`` into ``os.environ``.

    Existing env vars are not overridden (matches python-dotenv default).
    Lines that are blank, comments, or missing ``=`` are skipped.
    Surrounding single/double quotes are stripped.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(extra_path: Path | None = None) -> HarnessConfig:
    """Load HarnessConfig.

    Side effect: also loads ``~/.robot-loom/workspace/.env`` into ``os.environ``
    so Brain/Tool backends can read API keys without manual sourcing.

    Merge order (last wins): defaults → workspace config.yaml → extra_path.
    """
    _load_dotenv(workspace_root() / ".env")

    data: dict[str, Any] = {}

    workspace_cfg = workspace_config_file()
    if workspace_cfg.exists():
        with workspace_cfg.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        data = _deep_merge(data, loaded)

    if extra_path is not None and extra_path.exists():
        with extra_path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        data = _deep_merge(data, loaded)

    return HarnessConfig.model_validate(data)
