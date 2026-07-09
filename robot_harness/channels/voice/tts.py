"""Streaming TTS client adapter + sentence chunking — Scenario A (ADR-023).

The TTS engine is an external vendor server (design principle 1). This module
only holds the text->audio transcode contract plus a sentence-chunking helper —
the latter enables "speak the first sentence while the rest is still generating"
for low-latency feedback.

Placement: the transcode hangs off the outbound stream (``channels/outbound.py``),
not the Brain. Chunking is pure (no IO, no provider knowledge) so it is unit-testable.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class SpeechChunk(BaseModel):
    """One audio frame synthesized by TTS. ``is_last`` marks the final frame of a fragment."""

    data: bytes
    sample_rate: int = 24000
    encoding: Literal["pcm16", "opus", "mp3"] = "pcm16"
    seq: int = 0
    is_last: bool = False


@runtime_checkable
class TTSClient(Protocol):
    """Thin client over an external streaming TTS server."""

    @property
    def provider(self) -> str: ...

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "",
        lang: str = "",
    ) -> AsyncIterator[SpeechChunk]:
        """Stream the audio for one text fragment."""
        ...

    async def close(self) -> None:
        """Release the underlying stream connection."""
        ...


# Sentence boundary: CJK + latin terminators and newline. Minimal but enough to
# drive speak-while-generate.
_SENTENCE_END = re.compile(r"[。！？；!?;\n]+|[.](?=\s|$)")  # noqa: RUF001


def _first_cut(text: str, min_chars: int) -> int:
    """Return the earliest boundary end whose stripped prefix is >= ``min_chars``; else 0."""
    for m in _SENTENCE_END.finditer(text):
        if len(text[: m.end()].strip()) >= min_chars:
            return m.end()
    return 0


def split_for_tts(text: str, *, min_chars: int = 12) -> list[str]:
    """Split a full text into speakable fragments at sentence boundaries (CJK + latin).

    Used to start synthesizing/playing the first sentence before the Brain output
    is complete. Pure function, no IO. Sentences shorter than ``min_chars`` are
    merged into the following one so punctuation is not spoken as single chars.
    """
    parts: list[str] = []
    buf = text
    while True:
        cut = _first_cut(buf, min_chars)
        if cut == 0:
            break
        parts.append(buf[:cut].strip())
        buf = buf[cut:]
    if buf.strip():
        parts.append(buf.strip())
    return [p for p in parts if p]


async def segment_stream(
    tokens: AsyncIterator[str],
    *,
    min_chars: int = 12,
) -> AsyncIterator[str]:
    """Consume a Brain token/text stream, yield a speakable fragment as each boundary is crossed.

    The entry point for speak-while-generate: as the upstream emits tokens, this
    flushes to TTS at sentence boundaries without waiting for the full output.
    Semantics match ``split_for_tts``: emit the shortest prefix that ends on a
    boundary and is >= ``min_chars`` (short sentences merge forward, no fragmentation).
    """
    buf = ""
    async for tok in tokens:
        buf += tok
        while True:
            cut = _first_cut(buf, min_chars)
            if cut == 0:
                break
            yield buf[:cut].strip()
            buf = buf[cut:]
    if buf.strip():
        yield buf.strip()
