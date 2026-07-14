"""Tests for the ``robot-loom serve`` command wiring.

``serve`` runs the ChannelManager resident: channel messages become Tasks for
the AgentLoop and results flow back to the originating user. These tests cover
the channel construction helper; the manager routing itself is covered by
test_talk_path.py.
"""

from __future__ import annotations

import pytest

from robot_harness.channels.cli import CLIChannel
from robot_harness.channels.telegram import TelegramChannel
from robot_harness.cli import _build_channel
from robot_harness.config.schema import HarnessConfig

_CONFIG = HarnessConfig()


def test_build_channel_cli() -> None:
    ch = _build_channel("cli", "go2-01", _CONFIG)
    assert isinstance(ch, CLIChannel)
    assert ch.channel_name == "cli"
    assert ch._robot_id == "go2-01"


def test_build_channel_telegram(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:testtoken")
    ch = _build_channel("telegram", "go2-01", _CONFIG)
    assert isinstance(ch, TelegramChannel)
    assert ch.channel_name == "telegram"


def test_build_channel_telegram_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(ValueError, match="bot token"):
        _build_channel("telegram", "go2-01", _CONFIG)


def test_build_channel_voice_gateway_without_url() -> None:
    # Default config has no gateway URL — serve must refuse, not half-start.
    with pytest.raises(ValueError, match=r"voice_gateway\.url"):
        _build_channel("voice_gateway", "go2-01", _CONFIG)


def test_build_channel_voice_gateway() -> None:
    pytest.importorskip("websockets")
    from robot_harness.channels.voice.gateway import VoiceGatewayChannel

    cfg = HarnessConfig()
    cfg.channels.voice_gateway.url = "ws://gateway.local:8765/harness"
    ch = _build_channel("voice_gateway", "go2-01", cfg)
    assert isinstance(ch, VoiceGatewayChannel)
    assert ch.channel_name == "voice_gateway"


def test_build_channel_unknown() -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        _build_channel("carrier-pigeon", "go2-01", _CONFIG)
