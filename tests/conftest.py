"""Shared pytest fixtures for the robot_harness test suite."""

from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def tmp_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ROBOT_LOOM_WORKSPACE at a per-test temp dir (autouse).

    Workspace files are user assets read at runtime — config.yaml by the config
    loader, MISSION.md / ROBOT.md by the system-prompt overlay (ADR-035). Tests
    must not depend on (or mutate) the developer's real ~/.robot-loom/workspace.
    Tests that need workspace content request this fixture and write into it.
    """
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("ROBOT_LOOM_WORKSPACE", str(ws))
    return ws


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    c = tag + payload
    return struct.pack(">I", len(payload)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)


def _make_solid_rgba_png(width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
    """Build a valid solid-color RGBA PNG of the given size using stdlib only."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    row = b"\x00" + bytes(rgba) * width  # filter byte = 0 (None), then RGBA pixels
    raw = row * height
    idat = _png_chunk(b"IDAT", zlib.compress(raw))
    iend = _png_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


@pytest.fixture(scope="session")
def tiny_png_bytes() -> bytes:
    """A valid 1x1 RGBA PNG (red pixel) as raw bytes — minimal wire-path probe."""
    return _make_solid_rgba_png(1, 1, (255, 0, 0, 255))


@pytest.fixture(scope="session")
def tiny_png_b64(tiny_png_bytes: bytes) -> str:
    """A valid 1x1 RGBA PNG (red pixel) as a base64 string — drop-in for image_b64 params."""
    return base64.b64encode(tiny_png_bytes).decode("ascii")


@pytest.fixture(scope="session")
def medium_png_bytes() -> bytes:
    """A valid 256x256 RGBA PNG (mid-grey) as raw bytes.

    Use this for integration tests against real inference servers whose
    preprocessors expect realistic input sizes (e.g. SAM3 Sam3Processor,
    Depth-Anything resize to 518x518). A 1x1 image is technically valid PNG
    but yields degenerate inference outputs.
    """
    return _make_solid_rgba_png(256, 256, (128, 128, 128, 255))


@pytest.fixture(scope="session")
def medium_png_b64(medium_png_bytes: bytes) -> str:
    """A valid 256x256 RGBA PNG (mid-grey) as base64 — drop-in for image_b64."""
    return base64.b64encode(medium_png_bytes).decode("ascii")
