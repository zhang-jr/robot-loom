"""Framework and workspace path resolution (ADR-015).

Framework code lives under the installed package.
User assets (missions, robots, custom skills/tools) live in
``~/.robot-loom/workspace/`` (or the path set by ``ROBOT_LOOM_WORKSPACE``).
"""

from __future__ import annotations

import os
from pathlib import Path


def framework_root() -> Path:
    """Return the installed ``robot_harness`` package directory."""
    return Path(__file__).parent.parent.resolve()


def workspace_root() -> Path:
    """Return the user workspace root, creating it if absent."""
    env = os.environ.get("ROBOT_LOOM_WORKSPACE")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        root = Path.home() / ".robot-loom" / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    return root


def workspace_skills_dir() -> Path:
    return workspace_root() / "skills"


def workspace_tools_dir() -> Path:
    return workspace_root() / "tools"


def workspace_logs_dir() -> Path:
    d = workspace_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def workspace_config_file() -> Path:
    return workspace_root() / "config.yaml"


def workspace_mission_file() -> Path:
    return workspace_root() / "MISSION.md"


def workspace_robot_file() -> Path:
    return workspace_root() / "ROBOT.md"
