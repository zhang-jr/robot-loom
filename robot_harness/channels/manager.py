"""ChannelManager — routes messages from all channels to the AgentLoop.

Multiple channels (CLI, Web, Telegram, …) can be registered and run
concurrently.  Each inbound ChannelMessage is converted to a Task and
dispatched to the AgentLoop; the result is forwarded back to the originating
channel as a ChannelResponse.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from robot_harness.brain.base import Task
from robot_harness.channels.base import Channel, ChannelMessage, ChannelResponse
from robot_harness.observability.tracer import tracer


class ChannelManager:
    """Manages a set of channels and connects them to an AgentLoop factory.

    ``agent_loop_factory`` is a callable ``(task: Task) → Awaitable[AgentResult]``
    so that the manager does not depend on a concrete AgentLoop class.
    """

    def __init__(self, agent_loop_factory: Any) -> None:
        self._factory = agent_loop_factory
        self._channels: dict[str, Channel] = {}
        self._running = False
        self._tasks: set[asyncio.Task[None]] = set()

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

    async def _drain(self, channel: Channel) -> None:
        async for msg in channel.listen():
            t = asyncio.create_task(self._handle(channel, msg))
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)

    async def _handle(self, channel: Channel, msg: ChannelMessage) -> None:
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
        try:
            result = await self._factory(task)
            text = result.message or f"Done ({result.outcome})"
            success = result.outcome == "success"
        except Exception as exc:  # noqa: BLE001
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
