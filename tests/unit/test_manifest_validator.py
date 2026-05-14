"""Unit tests for the skill manifest YAML validator."""

from __future__ import annotations

from robot_harness.skill.manifest_validator import validate_manifest_dict

_VALID = {
    "name": "pick",
    "version": "1.0.0",
    "description": "Pick skill",
    "embodiment_compat": ["arm"],
    "safety_class": "high",
    "required_tools": ["perception.detect_objects"],
}


def test_valid_manifest_returns_no_errors() -> None:
    assert validate_manifest_dict(_VALID) == []


def test_missing_name_returns_error() -> None:
    bad = {**_VALID, "name": ""}
    errors = validate_manifest_dict(bad)
    assert errors


def test_invalid_semver_returns_error() -> None:
    bad = {**_VALID, "version": "1.0"}
    errors = validate_manifest_dict(bad)
    assert errors


def test_invalid_safety_class_returns_error() -> None:
    bad = {**_VALID, "safety_class": "extreme"}
    errors = validate_manifest_dict(bad)
    assert errors


def test_missing_required_field_returns_error() -> None:
    bad = {k: v for k, v in _VALID.items() if k != "name"}
    errors = validate_manifest_dict(bad)
    assert errors


def test_extra_fields_allowed() -> None:
    extended = {**_VALID, "owner": "team-alpha", "tags": ["manipulation"]}
    assert validate_manifest_dict(extended) == []
