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


# ---------------------------------------------------------------------------
# safety_check parsing — fail closed (ISS-029)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"passed": "yes"}, {"ok": True}])
async def test_safety_check_malformed_response_fails_closed(payload: dict[str, object]) -> None:
    """safety_check is pass 2 of the mandatory pre-dispatch gate: a response
    without a boolean verdict must read as a refusal, never a default pass."""
    from robot_harness.embodiment.base import EmbodimentCommand

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/safety_check"
        return httpx.Response(200, json=payload)

    adapter = RealAgentServerAdapter("r0", client=_client(handler))
    verdict = await adapter.safety_check(
        EmbodimentCommand(robot_id="r0", command_type="cartesian", values=[0.1, 0.1, 0.1])
    )
    assert verdict.passed is False
    assert "malformed" in verdict.reason
    await adapter.aclose()


# ---------------------------------------------------------------------------
# Verb call deadline semantics (ISS-030)
# ---------------------------------------------------------------------------


from robot_harness.embodiment.interface.http import (  # noqa: E402
    DEFAULT_VERB_TIMEOUT_S,
    VERB_TIMEOUT_MARGIN_S,
    verb_read_timeout_s,
)
from robot_harness.errors import ActionDispatchTimeoutError  # noqa: E402


def test_verb_read_timeout_follows_constraints_budget() -> None:
    assert verb_read_timeout_s({"constraints": {"timeout_s": 120.0}}) == pytest.approx(
        120.0 + VERB_TIMEOUT_MARGIN_S
    )
    # Absent / malformed / non-positive budgets fall back to the schema default.
    fallback = DEFAULT_VERB_TIMEOUT_S + VERB_TIMEOUT_MARGIN_S
    assert verb_read_timeout_s({}) == pytest.approx(fallback)
    assert verb_read_timeout_s({"constraints": {"timeout_s": "soon"}}) == pytest.approx(fallback)
    assert verb_read_timeout_s({"constraints": {"timeout_s": 0}}) == pytest.approx(fallback)


@pytest.mark.asyncio
async def test_call_verb_sends_constraint_derived_read_timeout() -> None:
    """A verb blocks until the on-robot mid-loop completes: the request's read
    deadline must follow constraints.timeout_s, not the client's 5s default."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={"outcome": "success"})

    c = _client(handler)
    await c.call_verb("locomote_to", {"robot_id": "r0", "constraints": {"timeout_s": 90.0}})
    timeout = seen["timeout"]
    assert isinstance(timeout, dict)
    assert timeout["read"] == pytest.approx(90.0 + VERB_TIMEOUT_MARGIN_S)
    assert timeout["connect"] == pytest.approx(5.0)  # connect stays short
    await c.aclose()


@pytest.mark.asyncio
async def test_call_verb_read_timeout_aborts_and_raises_typed_error() -> None:
    """A verb that exceeds its budget is a dispatch timeout, NOT 'robot offline';
    the client must POST /abort so the robot's motion unwinds before the error
    surfaces (the robot accepted the verb and may still be moving)."""
    aborted: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/abort":
            import json as _json

            aborted.append(_json.loads(request.content))
            return httpx.Response(200, json={"aborted": True})
        raise httpx.ReadTimeout("verb still running", request=request)

    c = _client(handler)
    with pytest.raises(ActionDispatchTimeoutError, match="did not complete"):
        await c.call_verb("reactive_grasp", {"robot_id": "r0"})
    assert aborted and "timeout" in str(aborted[0].get("reason"))
    await c.aclose()


@pytest.mark.asyncio
async def test_sim_call_verb_read_timeout_same_semantics() -> None:
    aborted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/abort":
            aborted.append(request.url.path)
            return httpx.Response(200, json={"aborted": True})
        raise httpx.ReadTimeout("verb still running", request=request)

    c = SimAgentServerClient("http://sim", "sim-0", transport=httpx.MockTransport(handler))
    with pytest.raises(ActionDispatchTimeoutError, match="did not complete"):
        await c.call_verb("locomote_to", {"robot_id": "sim-0"})
    assert aborted == ["/abort"]
    await c.aclose()


# ---------------------------------------------------------------------------
# Malformed state / frame payloads become typed errors (ISS-034)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_camera_frame_is_typed_offline_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})  # no camera/robot_id

    adapter = RealAgentServerAdapter("r0", client=_client(handler))
    with pytest.raises(RobotOfflineError, match="malformed camera frame"):
        await adapter.get_camera_frame("wrist")
    await adapter.aclose()


@pytest.mark.asyncio
async def test_malformed_state_is_typed_offline_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"gripper_state": "closed-ish"})  # not a float

    adapter = RealAgentServerAdapter("r0", client=_client(handler))
    with pytest.raises(RobotOfflineError, match="malformed robot state"):
        await adapter.get_state()
    await adapter.aclose()
