"""Shared fixtures for integration tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def _bypass_proxy_for_loopback() -> Iterator[None]:
    """Exempt loopback addresses from any system-wide HTTP(S) proxy.

    Integration tests spawn servers on 127.0.0.1 and probe them with httpx,
    which trusts HTTP_PROXY / HTTPS_PROXY from the environment by default. On a
    machine with a global proxy configured, loopback requests get routed
    through the proxy and come back as 502 Bad Gateway — the local server is
    never actually reached. NO_PROXY restores direct loopback access without
    touching the proxy setup itself.
    """
    saved: dict[str, str | None] = {}
    loopback = "127.0.0.1,localhost"
    for var in ("NO_PROXY", "no_proxy"):
        saved[var] = os.environ.get(var)
        existing = saved[var]
        os.environ[var] = f"{existing},{loopback}" if existing else loopback
    yield
    for var, old in saved.items():
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old
