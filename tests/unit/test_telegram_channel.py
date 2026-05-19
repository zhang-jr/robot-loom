"""Tests for TelegramChannel — uses httpx MockTransport, no real network."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from robot_harness.channels.base import ChannelResponse
from robot_harness.channels.telegram import TelegramChannel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_update(update_id: int, text: str, user_id: int = 42, chat_id: int = 99) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id * 10,
            "from": {"id": user_id, "first_name": "Test"},
            "chat": {"id": chat_id, "type": "private"},
            "text": text,
        },
    }


def _make_response(updates: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"ok": True, "result": updates},
    )


def _empty_response() -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": []})


def _send_ok_response() -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTelegramChannelInit:
    def test_raises_without_token(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # ensure TELEGRAM_BOT_TOKEN is not in env
            import os

            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            with pytest.raises(ValueError, match="bot token"):
                TelegramChannel()

    def test_uses_env_token(self) -> None:
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test:token123"}):
            ch = TelegramChannel()
            assert ch._token == "test:token123"

    def test_explicit_token(self) -> None:
        ch = TelegramChannel(bot_token="tok:abc")
        assert ch._token == "tok:abc"

    def test_channel_name(self) -> None:
        ch = TelegramChannel(bot_token="tok:x")
        assert ch.channel_name == "telegram"


class TestParseUpdate:
    def setup_method(self) -> None:
        self.ch = TelegramChannel(bot_token="tok:x", robot_id="arm-1")

    def test_text_message(self) -> None:
        update = _make_update(1, "pick up the cup")
        msg = self.ch._parse_update(update)
        assert msg is not None
        assert msg.text == "pick up the cup"
        assert msg.user_id == "42"
        assert msg.robot_id == "arm-1"
        assert msg.metadata["chat_id"] == "99"
        assert msg.metadata["update_id"] == 1

    def test_no_text_returns_none(self) -> None:
        update = {
            "update_id": 5,
            "message": {"message_id": 50, "chat": {"id": 1}, "photo": []},
        }
        assert self.ch._parse_update(update) is None

    def test_non_message_update_returns_none(self) -> None:
        update = {"update_id": 6, "channel_post": {"text": "hi"}}
        assert self.ch._parse_update(update) is None

    def test_edited_message(self) -> None:
        update = {
            "update_id": 7,
            "edited_message": {
                "message_id": 70,
                "from": {"id": 10},
                "chat": {"id": 20},
                "text": "corrected",
            },
        }
        msg = self.ch._parse_update(update)
        assert msg is not None
        assert msg.text == "corrected"

    def test_empty_text_skipped(self) -> None:
        update = _make_update(8, "   ")
        assert self.ch._parse_update(update) is None


class TestListenAndSend:
    async def test_listen_yields_messages(self) -> None:
        ch = TelegramChannel(bot_token="tok:x", robot_id="r0")

        call_count = 0

        async def mock_get_updates() -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [_make_update(1, "hello"), _make_update(2, "world")]
            await asyncio.sleep(10)  # block forever on second call
            return []

        ch._get_updates = mock_get_updates  # type: ignore[method-assign]

        # AsyncMock so that `await ch._client.aclose()` works in stop()
        ch._client = AsyncMock()
        ch._poll_task = asyncio.create_task(ch._poll_loop())

        messages = []

        async def collect() -> None:
            async for msg in ch.listen():
                messages.append(msg)
                if len(messages) >= 2:
                    await ch.stop()

        await asyncio.wait_for(collect(), timeout=2.0)
        assert len(messages) == 2
        assert messages[0].text == "hello"
        assert messages[1].text == "world"

    async def test_send_calls_api(self) -> None:
        ch = TelegramChannel(bot_token="tok:x")
        api_calls: list[tuple[str, dict[str, Any]]] = []

        async def mock_api_call(method: str, params: dict[str, Any]) -> dict[str, Any]:
            api_calls.append((method, params))
            return {"ok": True, "result": {}}

        ch._api_call = mock_api_call  # type: ignore[method-assign]
        ch._client = MagicMock()

        resp = ChannelResponse(
            channel="telegram",
            user_id="42",
            text="Task complete",
            metadata={"chat_id": "99"},
        )
        await ch.send(resp)
        assert len(api_calls) == 1
        assert api_calls[0][0] == "sendMessage"
        assert api_calls[0][1]["chat_id"] == "99"
        assert api_calls[0][1]["text"] == "Task complete"

    async def test_send_without_chat_id_is_noop(self) -> None:
        ch = TelegramChannel(bot_token="tok:x")
        api_calls: list[Any] = []
        ch._api_call = AsyncMock(side_effect=api_calls.append)  # type: ignore[method-assign]
        ch._client = MagicMock()

        resp = ChannelResponse(
            channel="telegram",
            user_id="42",
            text="No chat_id here",
            metadata={},
        )
        await ch.send(resp)  # should not raise
        assert len(api_calls) == 0

    async def test_poll_error_does_not_crash(self) -> None:
        # error_backoff_s=0 so the loop recovers immediately in tests
        ch = TelegramChannel(bot_token="tok:x", error_backoff_s=0)
        call_count = 0

        async def mock_get_updates() -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("network error")
            # Second call: put sentinel then block
            await ch._queue.put(None)
            await asyncio.sleep(10)
            return []

        ch._get_updates = mock_get_updates  # type: ignore[method-assign]
        ch._client = AsyncMock()
        ch._poll_task = asyncio.create_task(ch._poll_loop())

        # Drain the sentinel — confirms poll loop survived the error
        sentinel = await asyncio.wait_for(ch._queue.get(), timeout=2.0)
        assert sentinel is None
        ch._poll_task.cancel()
