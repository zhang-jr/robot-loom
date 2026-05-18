"""Telegram channel — long-polling bot via the Telegram Bot API.

Uses ``httpx`` (already a hard dependency) for HTTP calls so no extra
package is needed.  The bot token is read from the ``TELEGRAM_BOT_TOKEN``
environment variable or passed explicitly.

Usage::

    channel = TelegramChannel(bot_token="...", robot_id="arm-0")
    await channel.start()
    async for msg in channel.listen():
        ...  # dispatch to AgentLoop
        await channel.send(ChannelResponse(...))
    await channel.stop()

Only text messages are handled; photo/audio/sticker updates are silently
dropped so the AgentLoop sees a clean stream of task strings.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from robot_harness.channels.base import ChannelMessage, ChannelResponse
from robot_harness.observability.tracer import tracer

_API_BASE = "https://api.telegram.org/bot{token}/{method}"
_POLL_TIMEOUT = 30  # seconds — Telegram long-poll window
_HTTP_TIMEOUT = _POLL_TIMEOUT + 10  # slightly longer than poll window


class TelegramChannel:
    """Telegram Bot channel backed by HTTP long-polling.

    Args:
        bot_token: Bot API token.  Falls back to ``TELEGRAM_BOT_TOKEN`` env var.
        robot_id:  Robot identifier forwarded in every :class:`ChannelMessage`.
        poll_timeout_s: Telegram ``timeout`` parameter (seconds) per ``getUpdates``
            call.  Lower values reduce latency but increase API calls.
    """

    channel_name = "telegram"

    def __init__(
        self,
        bot_token: str | None = None,
        robot_id: str = "default",
        poll_timeout_s: int = _POLL_TIMEOUT,
        error_backoff_s: float = 5.0,
    ) -> None:
        token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            raise ValueError(
                "TelegramChannel requires a bot token — pass bot_token= or set "
                "TELEGRAM_BOT_TOKEN in the environment."
            )
        self._token = token
        self._robot_id = robot_id
        self._poll_timeout = poll_timeout_s
        self._error_backoff_s = error_backoff_s
        self._offset: int = 0
        self._queue: asyncio.Queue[ChannelMessage | None] = asyncio.Queue()
        self._poll_task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(_POLL_TIMEOUT + 10, connect=10))
        self._poll_task = asyncio.create_task(self._poll_loop())
        tracer.event("channel.started", channel="telegram", robot_id=self._robot_id)

    async def stop(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
        await self._queue.put(None)  # sentinel
        if self._client:
            await self._client.aclose()
        tracer.event("channel.stopped", channel="telegram", robot_id=self._robot_id)

    # ------------------------------------------------------------------
    # Channel Protocol
    # ------------------------------------------------------------------

    async def listen(self) -> AsyncIterator[ChannelMessage]:
        while True:
            msg = await self._queue.get()
            if msg is None:
                return
            yield msg

    async def send(self, response: ChannelResponse) -> None:
        chat_id = response.metadata.get("chat_id")
        if not chat_id:
            tracer.event(
                "channel.send_skipped",
                channel="telegram",
                reason="no chat_id in metadata",
            )
            return
        await self._api_call(
            "sendMessage",
            {"chat_id": chat_id, "text": response.text},
        )

    # ------------------------------------------------------------------
    # Internal polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while True:
            try:
                updates = await self._get_updates()
                for update in updates:
                    self._offset = update["update_id"] + 1
                    msg = self._parse_update(update)
                    if msg is not None:
                        await self._queue.put(msg)
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001
                tracer.event(
                    "channel.poll_error",
                    channel="telegram",
                    error=type(exc).__name__,
                    detail=str(exc),
                )
                await asyncio.sleep(self._error_backoff_s)  # back-off on transient errors

    async def _get_updates(self) -> list[dict[str, Any]]:
        data = await self._api_call(
            "getUpdates",
            {"offset": self._offset, "timeout": self._poll_timeout},
        )
        return cast(list[dict[str, Any]], data.get("result", []))

    async def _api_call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self._client is not None, "TelegramChannel not started"
        url = _API_BASE.format(token=self._token, method=method)
        resp = await self._client.post(url, json=params)
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def _parse_update(self, update: dict[str, Any]) -> ChannelMessage | None:
        message = update.get("message") or update.get("edited_message")
        if not message:
            return None
        text = message.get("text", "").strip()
        if not text:
            return None
        from_user = message.get("from", {})
        user_id = str(from_user.get("id", "unknown"))
        chat_id = str(message.get("chat", {}).get("id", "unknown"))
        return ChannelMessage(
            channel=self.channel_name,
            user_id=user_id,
            text=text,
            robot_id=self._robot_id,
            metadata={"chat_id": chat_id, "update_id": update["update_id"]},
        )
