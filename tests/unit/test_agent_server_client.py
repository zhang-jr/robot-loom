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
from robot_harness.embodiment.interface.sim import SimAgentServerClient
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


# ---------------------------------------------------------------------------
# available_verbs(): live capability read from /health (ADR-019)
# ---------------------------------------------------------------------------


def _health_client(payload: dict[str, object], robot_id: str = "r0") -> HttpAgentServerClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json=payload)

    return _client(handler, robot_id)


@pytest.mark.asyncio
async def test_available_verbs_reads_health_allowlist() -> None:
    # A quadruped whose arm bridge is down advertises only locomotion + home.
    client = _health_client({"status": "ok", "available_verbs": ["locomote_to", "home"]}, "go2")
    adapter = RealAgentServerAdapter("go2", client=client)
    assert await adapter.available_verbs() == ["locomote_to", "home"]
    await adapter.aclose()


@pytest.mark.asyncio
async def test_available_verbs_explicit_empty_means_zero_verbs() -> None:
    # [] is a real answer ("all capability bridges down"), distinct from unknown.
    adapter = RealAgentServerAdapter("r0", client=_health_client({"available_verbs": []}))
    assert await adapter.available_verbs() == []
    await adapter.aclose()


@pytest.mark.asyncio
async def test_available_verbs_none_when_backend_does_not_advertise() -> None:
    # The offline mock omits the key -> None ("unknown"), never prune on it.
    adapter = RealAgentServerAdapter("r0")
    assert await adapter.available_verbs() is None
    await adapter.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", [None, "home", 42, {"home": True}])
async def test_available_verbs_none_on_malformed_payload(bad_value: object) -> None:
    """A null / string / non-list value must read as 'unknown', not crash or
    char-split into a bogus allowlist."""
    adapter = RealAgentServerAdapter("r0", client=_health_client({"available_verbs": bad_value}))
    assert await adapter.available_verbs() is None
    await adapter.aclose()


@pytest.mark.asyncio
async def test_available_verbs_raises_robot_offline_on_non_json_health() -> None:
    """A 200 with a non-JSON body (proxy error page) is a typed offline error,
    not a JSONDecodeError escaping into the planning path."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway error</html>")

    adapter = RealAgentServerAdapter("r0", client=_client(handler))
    with pytest.raises(RobotOfflineError, match="unreachable"):
        await adapter.available_verbs()
    await adapter.aclose()


# ---------------------------------------------------------------------------
# Sim wire client speaks the same /health contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sim_client_implements_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "available_verbs": ["home"]})

    c = SimAgentServerClient(
        "http://sim",
        "sim-0",
        transport=httpx.MockTransport(handler),
    )
    assert (await c.health())["available_verbs"] == ["home"]
    await c.aclose()
