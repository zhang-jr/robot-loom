"""Voice channels — Scenario A, the user speaks to the agent (ADR-023 / ADR-037).

Two deployment shapes, one invariant (the harness core only ever sees text):

- ``VoiceGatewayChannel`` — thin wire client to an external voice gateway
  (sibling repo) that owns the full audio path (ingestion, VAD/AEC, streaming
  ASR, session management, TTS playback) and delivers finalized utterances as
  structured text (ADR-037).
- ``VoiceChannel`` — direct-device path: this process owns mic/speaker and
  transcodes at the boundary via thin ASR/TTS clients to external engines.
"""

from __future__ import annotations

from robot_harness.channels.voice.asr import ASRClient, AudioChunk, Transcript
from robot_harness.channels.voice.channel import AudioSink, AudioSource, VoiceChannel
from robot_harness.channels.voice.gateway import (
    GatewayTransport,
    GatewayUtterance,
    VoiceGatewayChannel,
    WebSocketGatewayTransport,
)
from robot_harness.channels.voice.tts import (
    SpeechChunk,
    TTSClient,
    segment_stream,
    split_for_tts,
)

__all__ = [
    "ASRClient",
    "AudioChunk",
    "AudioSink",
    "AudioSource",
    "GatewayTransport",
    "GatewayUtterance",
    "SpeechChunk",
    "TTSClient",
    "Transcript",
    "VoiceChannel",
    "VoiceGatewayChannel",
    "WebSocketGatewayTransport",
    "segment_stream",
    "split_for_tts",
]
