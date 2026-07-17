"""VoiceGatewayChannel — thin wire client to an external voice gateway (ADR-037).

The voice gateway is a sibling-repo external server that owns the entire audio
path: device ingestion (handheld WiFi mic / phone app), transport, VAD/AEC/AGC
preprocessing, streaming ASR, and input session management (endpointing,
barge-in, speaker attribution). It delivers *finalized utterances as structured
text*; TTS synthesis and playback also live gateway-side because they couple to
the capture path (AEC, playback interruption).

This channel therefore never touches audio — it exchanges JSON text frames:

  gateway -> harness : {"type": "utterance", "text", "speaker_id", "device_id",
                        "confidence", "lang", "start_ms", "end_ms", "robot_id"?}
                       {"type": "barge_in", "speaker_id", "device_id"}
  harness -> gateway : {"type": "say", "text", "speaker_id", "level"}

The harness dials out to the gateway (it is a resident external server owning
devices — same connection direction as agent_server / perception servers). The
in-process ASR/TTS path (``VoiceChannel``) remains for direct local devices.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from robot_harness.channels.base import ChannelMessage, ChannelResponse
from robot_harness.errors import ChannelUnavailable
from robot_harness.observability.tracer import tracer


class GatewayUtterance(BaseModel):
    """One endpointed, finalized utterance delivered by the gateway."""

    type: Literal["utterance"]
    text: str
    speaker_id: str = ""
    device_id: str = ""
    confidence: float = 0.0
    lang: str = ""
    start_ms: int = 0
    end_ms: int = 0
    # Optional per-utterance robot attribution decided by the gateway's session
    # manager (speaker -> robot binding); empty falls back to the channel default.
    robot_id: str = ""


@runtime_checkable
class GatewayTransport(Protocol):
    """Bidirectional JSON-frame wire to the voice gateway (injectable for tests)."""

    def messages(self) -> AsyncIterator[dict[str, Any]]: ...

    async def send(self, payload: dict[str, Any]) -> None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class WebSocketGatewayTransport:
    """Dial-out websocket wire to the gateway with reconnect + backoff.

    Auth is a Bearer token read from the environment (``token_env``) — the URL
    is workspace config, the token is a secret (ADR-026 split). While
    disconnected, ``send`` raises :class:`ChannelUnavailable` (honest failure);
    inbound delivery resumes on reconnect.
    """

    def __init__(
        self,
        url: str,
        *,
        token_env: str = "VOICE_GATEWAY_TOKEN",
        reconnect_backoff_s: float = 3.0,
    ) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "websockets is required for the voice gateway channel. "
                "Install it with: pip install robot-loom[voice]"
            ) from exc
        self._websockets = websockets
        self._url = url
        self._token_env = token_env
        self._reconnect_backoff_s = reconnect_backoff_s
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._run_task: asyncio.Task[None] | None = None
        self._ws: Any = None

    async def start(self) -> None:
        self._run_task = asyncio.create_task(self._run())
        tracer.event("channel.voice_gateway.connecting", url=self._url)

    async def stop(self) -> None:
        if self._run_task:
            self._run_task.cancel()
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        await self._queue.put(None)  # sentinel

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise ChannelUnavailable("voice gateway is not connected", channel="voice_gateway")
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    async def _run(self) -> None:
        headers: dict[str, str] = {}
        token = os.environ.get(self._token_env, "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        while True:
            try:
                async with self._websockets.connect(self._url, additional_headers=headers) as ws:
                    self._ws = ws
                    tracer.event("channel.voice_gateway.connected", url=self._url)
                    async for raw in ws:
                        payload = self._decode(raw)
                        if payload is not None:
                            await self._queue.put(payload)
            except asyncio.CancelledError:
                self._ws = None
                return
            except Exception as exc:  # noqa: BLE001 — reconnect on any wire error (same as telegram poll loop)
                tracer.event(
                    "channel.voice_gateway.disconnected",
                    error=type(exc).__name__,
                    detail=str(exc),
                )
            self._ws = None
            await asyncio.sleep(self._reconnect_backoff_s)

    @staticmethod
    def _decode(raw: str | bytes) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            tracer.event("channel.voice_gateway.bad_frame", reason="not json")
            return None
        if not isinstance(payload, dict):
            tracer.event("channel.voice_gateway.bad_frame", reason="not an object")
            return None
        return payload


class VoiceGatewayChannel:
    """Finalized-utterance text channel over a ``GatewayTransport`` (ADR-037)."""

    channel_name = "voice_gateway"

    def __init__(
        self,
        *,
        transport: GatewayTransport,
        robot_id: str = "default",
        fallback_user_id: str = "voice-user",
    ) -> None:
        self._transport = transport
        self._robot_id = robot_id
        self._fallback_user_id = fallback_user_id

    async def start(self) -> None:
        await self._transport.start()

    async def stop(self) -> None:
        await self._transport.stop()

    async def listen(self) -> AsyncIterator[ChannelMessage]:
        async for payload in self._transport.messages():
            msg = self._handle_frame(payload)
            if msg is not None:
                yield msg

    async def send(self, response: ChannelResponse) -> None:
        await self._transport.send(
            {
                "type": "say",
                "text": response.text,
                "speaker_id": response.user_id,
                "level": response.metadata.get("level", "info"),
            }
        )
        tracer.event(
            "channel.voice_gateway.say",
            user_id=response.user_id,
            success=response.success,
        )

    def _handle_frame(self, payload: dict[str, Any]) -> ChannelMessage | None:
        kind = payload.get("type", "")
        if kind == "utterance":
            return self._utterance_message(payload)
        if kind == "barge_in":
            # Playback interruption is handled gateway-side; harness-side
            # semantics (interrupt in-flight say / Brain turn) are an ADR-037
            # open item — record only.
            tracer.event(
                "channel.voice_gateway.barge_in",
                speaker_id=str(payload.get("speaker_id", "")),
                device_id=str(payload.get("device_id", "")),
            )
            return None
        tracer.event("channel.voice_gateway.bad_frame", reason=f"unknown type {kind!r}")
        return None

    def _utterance_message(self, payload: dict[str, Any]) -> ChannelMessage | None:
        try:
            utt = GatewayUtterance.model_validate(payload)
        except ValidationError as exc:
            tracer.event(
                "channel.voice_gateway.bad_frame",
                reason="invalid utterance",
                detail=str(exc),
            )
            return None
        if not utt.text.strip():
            return None
        # speaker_id keys the ask_user round-trip (one pending question per
        # (channel, user_id) origin in ChannelManager).
        user_id = utt.speaker_id or utt.device_id or self._fallback_user_id
        tracer.event(
            "channel.voice_gateway.utterance",
            user_id=user_id,
            confidence=utt.confidence,
        )
        return ChannelMessage(
            channel=self.channel_name,
            user_id=user_id,
            text=utt.text,
            robot_id=utt.robot_id or self._robot_id,
            metadata={
                "device_id": utt.device_id,
                "confidence": utt.confidence,
                "lang": utt.lang,
                "start_ms": utt.start_ms,
                "end_ms": utt.end_ms,
            },
        )
