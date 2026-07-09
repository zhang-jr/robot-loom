"""Tests for the voice channel shim (Scenario A, ADR-023).

Covers the two pure transcode helpers (``split_for_tts`` / ``segment_stream``)
and the ``VoiceChannel`` audio<->text wiring with in-process fakes — no vendor
ASR/TTS, no audio device. The streaming ASR/TTS engines are external and out of
scope here; this only verifies the boundary transcode contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from robot_harness.channels.base import ChannelResponse
from robot_harness.channels.voice import (
    AudioChunk,
    SpeechChunk,
    Transcript,
    VoiceChannel,
    segment_stream,
    split_for_tts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _aiter(items: list[str]) -> AsyncIterator[str]:
    for x in items:
        yield x


async def _collect(agen: AsyncIterator[str]) -> list[str]:
    return [chunk async for chunk in agen]


# ---------------------------------------------------------------------------
# split_for_tts (pure)
# ---------------------------------------------------------------------------


class TestSplitForTTS:
    def test_splits_on_cjk_boundary_and_keeps_punctuation(self) -> None:
        # min_chars=1 -> split at every boundary; punctuation preserved
        assert split_for_tts("一二三。四五六！", min_chars=1) == ["一二三。", "四五六！"]

    def test_splits_english_on_period(self) -> None:
        # period only counts as boundary when followed by space or end
        assert split_for_tts("Hello world. This is fine.", min_chars=1) == [
            "Hello world.",
            "This is fine.",
        ]

    def test_period_inside_token_not_a_boundary(self) -> None:
        # "3.14" must not split — '.' is not followed by whitespace/end
        assert split_for_tts("pi is 3.14 today", min_chars=1) == ["pi is 3.14 today"]

    def test_newline_is_a_boundary(self) -> None:
        assert split_for_tts("line one\nline two", min_chars=1) == ["line one", "line two"]

    def test_short_sentence_merges_into_next(self) -> None:
        # both pieces shorter than min_chars -> merged into one fragment
        out = split_for_tts("你好。今天天气很好啊。", min_chars=12)
        assert out == ["你好。今天天气很好啊。"]

    def test_no_boundary_returns_whole_text(self) -> None:
        assert split_for_tts("no punctuation here", min_chars=1) == ["no punctuation here"]

    def test_tail_without_boundary_is_emitted(self) -> None:
        out = split_for_tts("第一句结束了哦。还有个尾巴", min_chars=1)
        assert out == ["第一句结束了哦。", "还有个尾巴"]

    def test_empty_and_whitespace_yield_nothing(self) -> None:
        assert split_for_tts("") == []
        assert split_for_tts("   \n  ") == []


# ---------------------------------------------------------------------------
# segment_stream (streaming, must match split_for_tts semantics)
# ---------------------------------------------------------------------------


class TestSegmentStream:
    async def test_yields_sentence_as_boundary_crosses(self) -> None:
        tokens = ["一二", "三。", "四五", "六！"]
        assert await _collect(segment_stream(_aiter(tokens), min_chars=1)) == [
            "一二三。",
            "四五六！",
        ]

    async def test_holds_incomplete_buffer_then_flushes_tail(self) -> None:
        tokens = ["hello ", "world"]  # no boundary at all
        assert await _collect(segment_stream(_aiter(tokens), min_chars=1)) == ["hello world"]

    async def test_short_leading_sentence_does_not_block_later_output(self) -> None:
        # regression: a short first sentence must not stall the stream until EOF
        tokens = ["a。", "b。", "这是一段足够长的内容啦。", "再来一句也够长的内容。"]
        out = await _collect(segment_stream(_aiter(tokens), min_chars=12))
        assert out == ["a。b。这是一段足够长的内容啦。", "再来一句也够长的内容。"]

    async def test_matches_split_for_tts(self) -> None:
        text = "你好。今天天气不错呀。我们出门走走吧。"
        # feed char-by-char; streaming result must equal the batch splitter
        streamed = await _collect(segment_stream(_aiter(list(text)), min_chars=6))
        assert streamed == split_for_tts(text, min_chars=6)

    async def test_empty_stream_yields_nothing(self) -> None:
        assert await _collect(segment_stream(_aiter([]), min_chars=1)) == []


# ---------------------------------------------------------------------------
# VoiceChannel wiring (fakes, no vendor / no device)
# ---------------------------------------------------------------------------


class _FakeSource:
    """AudioSource that replays a fixed list of chunks."""

    def __init__(self, chunks: list[AudioChunk]) -> None:
        self._chunks = chunks
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def frames(self) -> AsyncIterator[AudioChunk]:
        for c in self._chunks:
            yield c


class _FakeASR:
    """ASRClient that emits a scripted transcript sequence regardless of audio."""

    provider = "fake"

    def __init__(self, transcripts: list[Transcript]) -> None:
        self._transcripts = transcripts
        self.closed = False

    async def stream(
        self, audio: AsyncIterator[AudioChunk], *, lang: str = ""
    ) -> AsyncIterator[Transcript]:
        # drain the audio so the source is actually consumed
        async for _ in audio:
            pass
        for tr in self._transcripts:
            yield tr

    async def transcribe(self, audio: bytes, *, lang: str = "") -> Transcript:
        return Transcript(text="", is_final=True)

    async def close(self) -> None:
        self.closed = True


class _FakeTTS:
    """TTSClient that records the fragments it was asked to synthesize."""

    provider = "fake"

    def __init__(self) -> None:
        self.fragments: list[str] = []
        self.closed = False

    async def synthesize(
        self, text: str, *, voice: str = "", lang: str = ""
    ) -> AsyncIterator[SpeechChunk]:
        self.fragments.append(text)
        yield SpeechChunk(data=text.encode(), is_last=True)

    async def close(self) -> None:
        self.closed = True


class _FakeSink:
    def __init__(self) -> None:
        self.played: list[SpeechChunk] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def play(self, chunk: SpeechChunk) -> None:
        self.played.append(chunk)


class TestVoiceChannel:
    def _build(
        self, transcripts: list[Transcript]
    ) -> tuple[VoiceChannel, _FakeASR, _FakeTTS, _FakeSink, _FakeSource]:
        source = _FakeSource([AudioChunk(data=b"\x00\x01")])
        sink = _FakeSink()
        asr = _FakeASR(transcripts)
        tts = _FakeTTS()
        ch = VoiceChannel(
            source=source, sink=sink, asr=asr, tts=tts, user_id="u1", robot_id="arm-1"
        )
        return ch, asr, tts, sink, source

    def test_channel_name(self) -> None:
        ch, *_ = self._build([])
        assert ch.channel_name == "voice"

    async def test_listen_yields_only_final_utterances(self) -> None:
        transcripts = [
            Transcript(text="pi", is_final=False),
            Transcript(text="pick up", is_final=False),
            Transcript(text="pick up the cup", is_final=True, confidence=0.9, lang="en"),
        ]
        ch, *_ = self._build(transcripts)
        msgs = [m async for m in ch.listen()]

        assert len(msgs) == 1
        assert msgs[0].text == "pick up the cup"
        assert msgs[0].channel == "voice"
        assert msgs[0].user_id == "u1"
        assert msgs[0].robot_id == "arm-1"
        assert msgs[0].metadata["confidence"] == 0.9
        assert msgs[0].metadata["lang"] == "en"

    async def test_send_splits_text_and_plays_each_fragment(self) -> None:
        ch, _asr, tts, sink, _src = self._build([])
        # send() uses split_for_tts default min_chars=12, so each sentence must
        # clear that bar to be spoken separately.
        resp = ChannelResponse(
            channel="voice",
            user_id="u1",
            text="第一句话写得足够长一些。第二句话也写得足够长。",
            success=True,
        )
        await ch.send(resp)

        assert tts.fragments == ["第一句话写得足够长一些。", "第二句话也写得足够长。"]
        assert len(sink.played) == 2

    async def test_start_stop_propagates_to_ports_and_clients(self) -> None:
        ch, asr, tts, sink, source = self._build([])
        await ch.start()
        await ch.stop()

        assert source.started and source.stopped
        assert sink.started and sink.stopped
        assert asr.closed and tts.closed
