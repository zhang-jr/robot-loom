"""Voice channel — Scenario A audio<->text shim (ADR-023).

Normalizes streaming voice access into the existing Channel text in/out contract:
inbound ASR and outbound TTS both transcode at the boundary so the harness core
only sees text. The ASR/TTS engines are always external (design principle 1); this
package holds only thin streaming clients and the device/gateway ports.
"""

from __future__ import annotations

from robot_harness.channels.voice.asr import ASRClient, AudioChunk, Transcript
from robot_harness.channels.voice.channel import AudioSink, AudioSource, VoiceChannel
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
    "SpeechChunk",
    "TTSClient",
    "Transcript",
    "VoiceChannel",
    "segment_stream",
    "split_for_tts",
]
