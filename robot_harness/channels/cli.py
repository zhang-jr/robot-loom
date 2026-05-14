"""CLI channel — reads from stdin, writes to stdout.

Intended for interactive use and local development.  Not suitable for
production multi-user deployments.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator

from robot_harness.channels.base import ChannelMessage, ChannelResponse


class CLIChannel:
    """Simple stdin/stdout channel.

    ``listen`` reads lines from stdin in a background thread (so it does not
    block the event loop).  ``send`` writes to stdout.
    """

    channel_name = "cli"

    def __init__(self, robot_id: str = "default", user_id: str = "cli_user") -> None:
        self._robot_id = robot_id
        self._user_id = user_id
        self._queue: asyncio.Queue[ChannelMessage | None] = asyncio.Queue()
        self._reader_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._reader_task = asyncio.create_task(self._read_stdin())
        print(f"[robot-loom] CLI channel ready.  Robot: {self._robot_id}", file=sys.stderr)

    async def stop(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        await self._queue.put(None)  # sentinel to unblock listen()

    async def listen(self) -> AsyncIterator[ChannelMessage]:
        while True:
            msg = await self._queue.get()
            if msg is None:
                return
            yield msg

    async def send(self, response: ChannelResponse) -> None:
        status = "OK" if response.success else "ERR"
        print(f"[{status}] {response.text}")

    async def _read_stdin(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except (EOFError, OSError):
                await self._queue.put(None)
                return
            if not line:
                await self._queue.put(None)
                return
            text = line.strip()
            if text:
                await self._queue.put(
                    ChannelMessage(
                        channel=self.channel_name,
                        user_id=self._user_id,
                        text=text,
                        robot_id=self._robot_id,
                    )
                )
