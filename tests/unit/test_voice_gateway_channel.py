"""Tests for the voice gateway channel (ADR-037).

Covers the finalized-utterance wire contract with an in-process fake transport —
no websocket, no gateway process. The gateway owns audio and session management;
the harness side under test only maps JSON frames <-> ChannelMessage/Response.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from robot_harness.channels.base import ChannelMessage, ChannelResponse
from robot_harness.channels.voice.gateway import VoiceGatewayChannel


class _FakeTransport:
    """GatewayTransport that replays scripted inbound frames and records sends."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self._frames = frames
        self.sent: list[dict[str, Any]] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        for f in self._frames:
            yield f

    async def send(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


def _utterance(**overrides: Any) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "type": "utterance",
        "text": "pick up the cup",
        "speaker_id": "operator-7",
        "device_id": "headset-3",
        "confidence": 0.93,
        "lang": "en",
        "start_ms": 100,
        "end_ms": 1450,
    }
    frame.update(overrides)
    return frame


async def _listen_all(ch: VoiceGatewayChannel) -> list[ChannelMessage]:
    return [m async for m in ch.listen()]


class TestVoiceGatewayChannel:
    def _build(self, frames: list[dict[str, Any]]) -> tuple[VoiceGatewayChannel, _FakeTransport]:
        transport = _FakeTransport(frames)
        ch = VoiceGatewayChannel(transport=transport, robot_id="go2-01")
        return ch, transport

    def test_channel_name(self) -> None:
        ch, _ = self._build([])
        assert ch.channel_name == "voice_gateway"

    async def test_lifecycle_drives_transport(self) -> None:
        ch, transport = self._build([])
        await ch.start()
        await ch.stop()
        assert transport.started
        assert transport.stopped

    async def test_utterance_becomes_channel_message(self) -> None:
        ch, _ = self._build([_utterance()])
        msgs = await _listen_all(ch)

        assert len(msgs) == 1
        msg = msgs[0]
        assert msg.channel == "voice_gateway"
        assert msg.user_id == "operator-7"  # speaker keys the ask_user round-trip
        assert msg.text == "pick up the cup"
        assert msg.robot_id == "go2-01"  # channel default, no per-utterance override
        assert msg.metadata["device_id"] == "headset-3"
        assert msg.metadata["confidence"] == 0.93
        assert msg.metadata["lang"] == "en"
        assert msg.metadata["start_ms"] == 100
        assert msg.metadata["end_ms"] == 1450

    async def test_per_utterance_robot_attribution_overrides_default(self) -> None:
        ch, _ = self._build([_utterance(robot_id="arm-2")])
        msgs = await _listen_all(ch)
        assert msgs[0].robot_id == "arm-2"

    async def test_user_id_falls_back_speaker_then_device(self) -> None:
        ch, _ = self._build([_utterance(speaker_id=""), _utterance(speaker_id="", device_id="")])
        msgs = await _listen_all(ch)
        assert msgs[0].user_id == "headset-3"
        assert msgs[1].user_id == "voice-user"

    async def test_barge_in_emits_no_message(self) -> None:
        # Playback interruption is gateway-local; harness-side semantics are an
        # ADR-037 open item — the frame must be absorbed, not surfaced as text.
        ch, _ = self._build([{"type": "barge_in", "speaker_id": "operator-7"}, _utterance()])
        msgs = await _listen_all(ch)
        assert len(msgs) == 1
        assert msgs[0].text == "pick up the cup"

    async def test_malformed_frames_are_dropped_stream_survives(self) -> None:
        ch, _ = self._build(
            [
                {"type": "utterance"},  # missing required text
                {"type": "utterance", "text": "   "},  # blank text
                {"no_type": True},  # unknown frame
                _utterance(text="resume patrol"),
            ]
        )
        msgs = await _listen_all(ch)
        assert [m.text for m in msgs] == ["resume patrol"]

    async def test_send_emits_say_frame(self) -> None:
        ch, transport = self._build([])
        await ch.send(
            ChannelResponse(
                channel="voice_gateway",
                user_id="operator-7",
                text="Cup secured.",
                metadata={"level": "info"},
            )
        )
        assert transport.sent == [
            {
                "type": "say",
                "text": "Cup secured.",
                "speaker_id": "operator-7",
                "level": "info",
            }
        ]
