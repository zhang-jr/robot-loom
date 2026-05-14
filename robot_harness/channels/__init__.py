"""Channels — user-facing input/output interfaces."""

from robot_harness.channels.base import Channel, ChannelMessage, ChannelResponse
from robot_harness.channels.cli import CLIChannel
from robot_harness.channels.manager import ChannelManager
from robot_harness.channels.web import WebChannel, create_app

__all__ = [
    "CLIChannel",
    "Channel",
    "ChannelManager",
    "ChannelMessage",
    "ChannelResponse",
    "WebChannel",
    "create_app",
]
