"""VoiceChannel — Scenario A audio<->text shim over the existing Channel contract.

Implements the existing ``Channel`` Protocol (text in / text out, see
``channels/base.py``) and transcodes at the boundary:

  inbound : audio frames -> streaming ASR -> endpointed utterance -> ``ChannelMessage``
  outbound: ``ChannelResponse.text`` -> sentence split -> streaming TTS -> audio sink

The harness core (Brain / bus / AgentLoop) only ever sees text — audio never
crosses the boundary (invariant, ADR-023 Scenario A). The same class adapts to
two deployment shapes through the ``AudioSource`` / ``AudioSink`` ports:

  - edge voice gateway: source/sink are a WS to the process that owns the audio
    device — keeps the realtime VAD/ASR loop off the Brain's 1-7Hz path (principle 4).
  - direct device: source/sink wrap a local mic/speaker.

The ASR/TTS engines themselves are external servers; this class reaches them only
through the ``ASRClient`` / ``TTSClient`` thin clients (design principle 1).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from robot_harness.channels.base import ChannelMessage, ChannelResponse
from robot_harness.channels.voice.asr import ASRClient, AudioChunk
from robot_harness.channels.voice.tts import SpeechChunk, TTSClient, split_for_tts
from robot_harness.observability.tracer import tracer


@runtime_checkable
class AudioSource(Protocol):
    """Inbound audio source (local mic device, or a WS to the voice gateway)."""

    def frames(self) -> AsyncIterator[AudioChunk]: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


@runtime_checkable
class AudioSink(Protocol):
    """Outbound audio sink (local speaker device, or a WS to the voice gateway)."""

    async def play(self, chunk: SpeechChunk) -> None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class VoiceChannel:
    """audio<->text Channel shim (implements the ``Channel`` Protocol)."""

    channel_name = "voice"

    def __init__(
        self,
        *,
        source: AudioSource,
        sink: AudioSink,
        asr: ASRClient,
        tts: TTSClient,
        user_id: str = "voice-user",
        robot_id: str = "default",
        voice: str = "",
        lang: str = "",
    ) -> None:
        self._source = source
        self._sink = sink
        self._asr = asr
        self._tts = tts
        self._user_id = user_id
        self._robot_id = robot_id
        self._voice = voice
        self._lang = lang

    async def start(self) -> None:
        await self._source.start()
        await self._sink.start()

    async def stop(self) -> None:
        await self._source.stop()
        await self._sink.stop()
        await self._asr.close()
        await self._tts.close()

    async def listen(self) -> AsyncIterator[ChannelMessage]:
        """Audio -> ASR -> only final utterances become a ``ChannelMessage`` into the harness."""
        async for tr in self._asr.stream(self._source.frames(), lang=self._lang):
            if not tr.is_final:
                # Partial is UI echo only, never into the harness (invariant: no
                # high-frequency traffic in the Brain loop).
                tracer.event("channel.voice.partial", user_id=self._user_id, text=tr.text)
                continue
            tracer.event(
                "channel.voice.utterance",
                user_id=self._user_id,
                confidence=tr.confidence,
            )
            yield ChannelMessage(
                channel=self.channel_name,
                user_id=self._user_id,
                text=tr.text,
                robot_id=self._robot_id,
                metadata={
                    "lang": tr.lang or self._lang,
                    "confidence": tr.confidence,
                    "start_ms": tr.start_ms,
                    "end_ms": tr.end_ms,
                },
            )

    async def send(self, response: ChannelResponse) -> None:
        """``text`` -> sentence split -> per-fragment streaming TTS -> sink; first sentence plays fast."""
        for fragment in split_for_tts(response.text):
            async for chunk in self._tts.synthesize(fragment, voice=self._voice, lang=self._lang):
                await self._sink.play(chunk)
        tracer.event("channel.voice.spoken", user_id=response.user_id, success=response.success)
