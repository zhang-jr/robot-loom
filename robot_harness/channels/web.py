"""Web channel — minimal FastAPI HTTP dashboard.

Exposes two endpoints:
  POST /task        — submit a task, returns the AgentResult JSON
  GET  /health      — liveness probe

Run with:  uvicorn robot_harness.channels.web:create_app --factory --reload

Requires: fastapi, uvicorn (optional extras, not in base deps).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from robot_harness.channels.base import ChannelMessage, ChannelResponse
from robot_harness.observability.tracer import tracer


def create_app(agent_loop_factory: Any = None) -> Any:
    """FastAPI app factory.  Pass an ``agent_loop_factory`` callable or use
    the ``/task`` endpoint as a direct JSON API without the ChannelManager."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel as _BaseModel
    except ImportError as exc:
        raise ImportError(
            "fastapi is required for the web channel.  Install it with: pip install robot-loom[web]"
        ) from exc

    app = FastAPI(title="Robot Loom", version="0.1.0")

    class TaskRequest(_BaseModel):
        description: str
        robot_id: str = "default"
        task_id: str = ""

    @app.get("/health")  # type: ignore[misc,untyped-decorator]
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/task")  # type: ignore[misc,untyped-decorator]
    async def submit_task(req: TaskRequest) -> JSONResponse:
        if agent_loop_factory is None:
            return JSONResponse({"error": "no agent_loop_factory configured"}, status_code=503)

        from robot_harness.brain.base import Task

        task = Task(
            task_id=req.task_id or str(uuid.uuid4()),
            description=req.description,
            robot_id=req.robot_id,
        )
        tracer.event("channel.web.task_received", task_id=task.task_id)
        try:
            result = await agent_loop_factory(task)
            return JSONResponse(result.model_dump())
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)}, status_code=500)

    return app


class WebChannel:
    """Adapter so WebChannel can be registered with ChannelManager.

    Uses an in-process asyncio queue bridged to the FastAPI endpoint.
    For production use, prefer running the FastAPI app standalone.
    """

    channel_name = "web"

    def __init__(self) -> None:
        self._queue: Any = None  # asyncio.Queue, set in start()

    async def start(self) -> None:
        import asyncio

        self._queue = asyncio.Queue()

    async def stop(self) -> None:
        if self._queue:
            await self._queue.put(None)

    async def listen(self) -> AsyncIterator[ChannelMessage]:
        while True:
            msg = await self._queue.get()
            if msg is None:
                return
            yield msg

    async def send(self, response: ChannelResponse) -> None:
        tracer.event("channel.web.response", user_id=response.user_id, success=response.success)

    async def push(self, msg: ChannelMessage) -> None:
        """Push a message from the FastAPI endpoint into the channel queue."""
        if self._queue:
            await self._queue.put(msg)
