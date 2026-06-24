"""Streaming ASR client adapter — Scenario A (user speaks to the agent, ADR-023).

The actual ASR engine is an external vendor server (CLAUDE.md design principle 1:
heavy model inference always lives out of process). This module only holds a thin
streaming client plus the audio->text transcode contract at the Channel boundary.

Invariant: streaming / VAD / incremental ASR is a high-frequency realtime loop and
MUST NOT enter the Brain's 1-7Hz decision loop (principle 4, frequency decoupling).
Only ``is_final`` (VAD-endpointed) utterances cross the boundary as a
``ChannelMessage``; partials are for UI echo only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class AudioChunk(BaseModel):
    """One raw audio frame fed to the ASR client (PCM/opus bytes + format hints)."""

    data: bytes
    sample_rate: int = 16000
    encoding: Literal["pcm16", "opus", "mulaw"] = "pcm16"
    seq: int = 0


class Transcript(BaseModel):
    """One ASR result. ``is_final`` marks a VAD-endpointed complete utterance."""

    text: str
    is_final: bool = False
    confidence: float = 0.0
    lang: str = ""
    start_ms: int = 0
    end_ms: int = 0


@runtime_checkable
class ASRClient(Protocol):
    """Thin client over an external streaming ASR server.

    Implementations wrap a vendor WS/gRPC stream. Partial transcripts MAY be
    yielded for UI echo, but only ``is_final`` ones are forwarded by
    ``VoiceChannel`` as a ``ChannelMessage``.
    """

    @property
    def provider(self) -> str: ...

    def stream(
        self,
        audio: AsyncIterator[AudioChunk],
        *,
        lang: str = "",
    ) -> AsyncIterator[Transcript]:
        """Consume an audio stream, yield (partial + final) transcripts."""
        ...

    async def transcribe(self, audio: bytes, *, lang: str = "") -> Transcript:
        """One-shot transcription (e.g. an already-complete Telegram voice clip)."""
        ...

    async def close(self) -> None:
        """Release the underlying stream connection."""
        ...
