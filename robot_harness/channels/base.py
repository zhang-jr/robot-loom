"""Channel Protocol — user-facing input/output interfaces.

A Channel accepts user messages and forwards them to the AgentLoop as Tasks.
Multiple channels can share a single HarnessContext / AgentLoop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class ChannelMessage(BaseModel):
    """Inbound message from a user via any channel."""

    channel: str  # 'cli' | 'web' | 'telegram' | 'discord' | …
    user_id: str
    text: str
    robot_id: str = "default"
    metadata: dict[str, Any] = {}


class ChannelResponse(BaseModel):
    """Outbound response sent back through a channel."""

    channel: str
    user_id: str
    text: str
    success: bool = True
    metadata: dict[str, Any] = {}


@runtime_checkable
class Channel(Protocol):
    """Bidirectional user-facing interface.

    ``listen`` yields inbound messages; ``send`` delivers responses.
    The harness ChannelManager drives both ends.
    """

    @property
    def channel_name(self) -> str: ...

    def listen(self) -> AsyncIterator[ChannelMessage]: ...

    async def send(self, response: ChannelResponse) -> None: ...

    async def start(self) -> None:
        """Optional lifecycle hook — called before ``listen``."""
        ...

    async def stop(self) -> None:
        """Optional lifecycle hook — called on shutdown."""
        ...
