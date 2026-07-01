"""Tests for the agent_server wire clients (real HTTP + offline mock) and the
RealAgentServerAdapter's client selection.

The real client's request/response path is exercised with ``httpx.MockTransport``
so no live server is needed; the offline mock client makes no network call at all.
"""

from __future__ import annotations

import httpx
import pytest

from robot_harness.embodiment.interface.http import (
    HttpAgentServerClient,
    MockAgentServerClient,
)
from robot_harness.embodiment.real.agent_server import RealAgentServerAdapter
from robot_harness.errors import RobotOfflineError

# ---------------------------------------------------------------------------
# Offline mock client — canned, no network
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_client_is_offline_and_canned() -> None:
    c = MockAgentServerClient("r0", dof=6)
    state = await c.get_state()
    assert state["robot_id"] == "r0"
    assert len(state["joint_positions"]) == 6
    verdict = await c.call_verb("reactive_grasp", {"robot_id": "r0"})
    assert verdict["outcome"] == "success"
    assert (await c.abort())["aborted"] is True
    assert (await c.health())["status"] == "ok"
    await c.aclose()


# ---------------------------------------------------------------------------
# Real HTTP client — exercised through httpx.MockTransport
# ---------------------------------------------------------------------------


def _client(handler: object, robot_id: str = "r0") -> HttpAgentServerClient:
    return HttpAgentServerClient(
        "http://robot",
        robot_id,
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_http_client_call_verb_posts_to_verb_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["method"] = request.method
        return httpx.Response(200, json={"outcome": "success", "evidence": "real"})

    c = _client(handler)
    verdict = await c.call_verb("reactive_grasp", {"robot_id": "r0"})
    assert seen == {"path": "/verb/reactive_grasp", "method": "POST"}
    assert verdict["outcome"] == "success"
    await c.aclose()


@pytest.mark.asyncio
async def test_http_client_abort_posts_to_abort_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"aborted": True, "robot_id": "r0"})

    c = _client(handler)
    resp = await c.abort({"trace_id": "t1"})
    assert seen["path"] == "/abort"
    assert resp["aborted"] is True
    await c.aclose()


@pytest.mark.asyncio
async def test_http_client_translates_transport_error_to_robot_offline() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502)  # bad gateway -> raise_for_status -> HTTPError

    c = _client(handler)
    with pytest.raises(RobotOfflineError, match="unreachable"):
        await c.get_state()
    await c.aclose()


# ---------------------------------------------------------------------------
# Adapter client selection: server_url present -> real; absent -> offline mock
# ---------------------------------------------------------------------------


def test_adapter_uses_real_client_when_server_url_set() -> None:
    adapter = RealAgentServerAdapter("r0", server_url="http://robot:8765")
    assert isinstance(adapter._client, HttpAgentServerClient)


def test_adapter_uses_mock_client_when_no_server_url() -> None:
    adapter = RealAgentServerAdapter("r0")
    assert isinstance(adapter._client, MockAgentServerClient)
