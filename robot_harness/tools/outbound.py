"""OutboundHandle — the Brain's channel-agnostic path back to the user (ADR-023).

A handle is already bound to the originating session's ``(channel, user_id)`` by
the time a tool or the AgentLoop receives it, so callers never name a channel:
they only express *what to say* (``report``) or *what to ask* (``ask``). The
Channel layer owns routing; this abstraction keeps tools and the loop unaware of
whether the user is on CLI, Telegram, or the web.

``report`` is fire-and-forget (the "report" half of Talk); ``ask`` is the
round-trip "ask_user" half — it blocks until the user replies or the wait times
out. The concrete implementation lives in :mod:`robot_harness.channels.outbound`;
defining the Protocol here (in the tools layer, with no channel import) keeps the
dependency edge ``channels → tools`` one-directional.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from robot_harness.observability.tracer import tracer


@runtime_checkable
class OutboundHandle(Protocol):
    """Session-bound outbound path to the user. Channel selection is already resolved."""

    async def report(self, text: str, level: str = "info") -> None:
        """Deliver a one-way notification to the user (fire-and-forget)."""
        ...

    async def ask(self, prompt: str, *, timeout_s: float) -> str:
        """Deliver a question and block until the user's reply (or timeout).

        Raises:
            ChannelError: if the question cannot be delivered or no reply arrives.
        """
        ...


class NullOutbound:
    """Default handle when no Channel is wired (programmatic / example callers).

    ``report`` degrades to a tracer event so the message is still observable;
    ``ask`` cannot round-trip without a user on the other end, so the AgentLoop
    treats its failure as an ``incomplete`` outcome rather than crashing.
    """

    async def report(self, text: str, level: str = "info") -> None:
        tracer.event("message.sent", level=level, text=text, channel="null")

    async def ask(self, prompt: str, *, timeout_s: float) -> str:
        from robot_harness.errors import ChannelUnavailable

        raise ChannelUnavailable("no channel wired; cannot round-trip ask_user")
