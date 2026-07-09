"""Load and merge HarnessConfig from YAML files and environment overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

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

    Delegates to python-dotenv so inline comments (``KEY=val # note``), quoted
    values, and ``export`` prefixes are parsed correctly. ``override=False``
    keeps the process's existing env vars authoritative over the file.
    """
    if not path.exists():
        return
    load_dotenv(path, override=False)


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
