"""Concrete OutboundHandle backed by the ChannelManager (ADR-023).

``ChannelOutbound`` binds one originating ``(channel, user_id)`` and delegates
``report`` / ``ask`` to the manager, which owns the channels and the pending-
question registry. The AgentLoop receives an already-bound handle, so neither the
loop nor the Talk tools ever name a channel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from robot_harness.channels.manager import ChannelManager


class ChannelOrigin(BaseModel):
    """The originating address of a session — where outbound messages go back to."""

    channel: str
    user_id: str


class ChannelOutbound:
    """OutboundHandle bound to one session origin, delegating to the manager."""

    def __init__(self, manager: ChannelManager, origin: ChannelOrigin) -> None:
        self._manager = manager
        self._origin = origin

    async def report(self, text: str, level: str = "info") -> None:
        await self._manager.send(self._origin, text, level=level)

    async def ask(self, prompt: str, *, timeout_s: float) -> str:
        return await self._manager.ask(self._origin, prompt, timeout_s=timeout_s)
