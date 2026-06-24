"""ChannelManager — routes messages from all channels to the AgentLoop.

Multiple channels (CLI, Web, Telegram, …) can be registered and run
concurrently.  Each inbound ChannelMessage is normally converted to a Task and
dispatched to the AgentLoop; the result is forwarded back to the originating
channel as a ChannelResponse.

The manager also owns the **outbound** side of Talk (ADR-023): it delivers
``report`` notifications and runs the ``ask_user`` round-trip. For a round-trip,
the AgentLoop suspends on :meth:`ask`; the *next* inbound message from that same
``(channel, user_id)`` is routed to the waiting reply instead of starting a new
Task. One outstanding question per origin is allowed.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from robot_harness.brain.base import Task
from robot_harness.channels.base import Channel, ChannelMessage, ChannelResponse
from robot_harness.channels.outbound import ChannelOrigin, ChannelOutbound
from robot_harness.errors import ChannelDeliveryFailed, ChannelUnavailable
from robot_harness.observability.tracer import tracer


class ChannelManager:
    """Manages a set of channels and connects them to an AgentLoop factory.

    ``agent_loop_factory`` is a callable
    ``(task: Task, *, outbound) → Awaitable[AgentResult]`` so that the manager
    does not depend on a concrete AgentLoop class. The ``outbound`` handle is
    bound to the originating session so the loop's Talk tools reach the right user.
    """

    def __init__(self, agent_loop_factory: Any) -> None:
        self._factory = agent_loop_factory
        self._channels: dict[str, Channel] = {}
        self._running = False
        self._tasks: set[asyncio.Task[None]] = set()
        # (channel, user_id) → future awaiting the user's reply to a pending ask.
        self._pending: dict[tuple[str, str], asyncio.Future[str]] = {}

    def register(self, channel: Channel) -> None:
        self._channels[channel.channel_name] = channel

    async def start(self) -> None:
        """Start all channels and begin routing messages."""
        self._running = True
        for ch in self._channels.values():
            await ch.start()

        tasks = [self._drain(ch) for ch in self._channels.values()]
        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        self._running = False
        for ch in self._channels.values():
            await ch.stop()

    # ----- outbound (Talk) --------------------------------------------------

    async def send(self, origin: ChannelOrigin, text: str, *, level: str = "info") -> None:
        """Deliver a one-way notification to the originating user (report)."""
        ch = self._channels.get(origin.channel)
        if ch is None:
            raise ChannelUnavailable(
                f"channel '{origin.channel}' not registered", channel=origin.channel
            )
        await ch.send(
            ChannelResponse(
                channel=origin.channel,
                user_id=origin.user_id,
                text=text,
                metadata={"level": level},
            )
        )

    async def ask(self, origin: ChannelOrigin, prompt: str, *, timeout_s: float) -> str:
        """Deliver a question and block until the user's next message (ask_user).

        Only one outstanding question per origin is allowed. The reply is routed
        by :meth:`_handle` to the future registered here.

        Raises:
            ChannelDeliveryFailed: if a question is already pending for this origin.
            TimeoutError: if no reply arrives within ``timeout_s``.
        """
        key = (origin.channel, origin.user_id)
        if key in self._pending:
            raise ChannelDeliveryFailed(
                "a question is already pending for this user", channel=origin.channel
            )
        await self.send(origin, prompt)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self._pending[key] = fut
        tracer.event("channel.ask_user", channel=origin.channel, user_id=origin.user_id)
        try:
            return await asyncio.wait_for(fut, timeout_s)
        finally:
            self._pending.pop(key, None)

    # ----- inbound ----------------------------------------------------------

    async def _drain(self, channel: Channel) -> None:
        async for msg in channel.listen():
            t = asyncio.create_task(self._handle(channel, msg))
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)

    async def _handle(self, channel: Channel, msg: ChannelMessage) -> None:
        # If a question is pending for this origin, this message is the reply —
        # route it to the waiting future instead of starting a new Task.
        key = (msg.channel, msg.user_id)
        fut = self._pending.get(key)
        if fut is not None and not fut.done():
            fut.set_result(msg.text)
            return

        origin = ChannelOrigin(channel=msg.channel, user_id=msg.user_id)
        task = Task(
            task_id=str(uuid.uuid4()),
            description=msg.text,
            robot_id=msg.robot_id or "default",
        )
        tracer.event(
            "channel.message_received",
            channel=msg.channel,
            user_id=msg.user_id,
            task_id=task.task_id,
        )
        outbound = ChannelOutbound(self, origin)
        try:
            result = await self._factory(task, outbound=outbound)
            text = result.message or f"Done ({result.outcome})"
            success = result.outcome == "success"
        except Exception as exc:  # noqa: BLE001
            # Task-execution failures are reported back to the user, not crashed.
            # Channel-delivery failures below are NOT swallowed (they re-raise).
            text = f"Internal error: {exc}"
            success = False

        await channel.send(
            ChannelResponse(
                channel=msg.channel,
                user_id=msg.user_id,
                text=text,
                success=success,
            )
        )
