"""Skill manifest YAML schema validator.

CLI usage:
    robot-loom skill validate <manifest.yaml>
    robot-loom skill validate workspace/skills/pick_v2.yaml

Python usage:
    from robot_harness.skill.manifest_validator import validate_manifest_file
    errors = validate_manifest_file(path)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from robot_harness.skill.base import SkillManifest


def validate_manifest_dict(data: dict[str, Any]) -> list[str]:
    """Validate a manifest dict against SkillManifest schema.

    Returns a list of human-readable error strings (empty = valid).
    """
    try:
        SkillManifest.model_validate(data)
        return []
    except ValidationError as exc:
        return [f"{e['loc']}: {e['msg']}" for e in exc.errors()]


def validate_manifest_file(path: Path) -> list[str]:
    """Load and validate a YAML manifest file.  Returns error strings."""
    try:
        import yaml  # optional dep
    except ImportError:
        return ["pyyaml is required to validate YAML manifests: pip install pyyaml"]

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"YAML parse error: {exc}"]

    if not isinstance(raw, dict):
        return ["manifest must be a YAML mapping (dict)"]

    return validate_manifest_dict(raw)


def validate_cli(args: list[str] | None = None) -> int:
    """Entry point for ``robot-loom skill validate``.  Returns exit code."""
    argv = args if args is not None else sys.argv[1:]
    if not argv:
        print("Usage: robot-loom skill validate <manifest.yaml> [...]", file=sys.stderr)
        return 1

    all_ok = True
    for arg in argv:
        path = Path(arg)
        if not path.exists():
            print(f"[ERROR] {arg}: file not found", file=sys.stderr)
            all_ok = False
            continue
        errors = validate_manifest_file(path)
        if errors:
            print(f"[FAIL] {arg}:")
            for e in errors:
                print(f"  - {e}")
            all_ok = False
        else:
            print(f"[OK]   {arg}")

    return 0 if all_ok else 1
