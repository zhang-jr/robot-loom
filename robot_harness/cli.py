"""robot-loom CLI entry point.

Commands:
  robot-loom init                 Initialise a workspace in ~/.robot-loom/workspace/
  robot-loom run --task TEXT      Run a task with the configured Brain
  robot-loom serve                Run resident: channels (CLI / Telegram / voice gateway) → AgentLoop
  robot-loom tool list            List registered tools
  robot-loom skill list           List registered skills
  robot-loom fleet status         Show fleet robot IDs from config
  robot-loom mcp serve            Expose the harness as an MCP server (reverse direction)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_harness.runtime.harness_context import HarnessContext


def _cmd_init(args: argparse.Namespace) -> None:
    from robot_harness.config.paths import workspace_root
    from robot_harness.observability.tracer import tracer

    ws = workspace_root()
    template_dir = Path(__file__).parent.parent / "workspace_template"

    if template_dir.exists():
        for src in template_dir.rglob("*"):
            if src.is_file():
                rel = src.relative_to(template_dir)
                dst = ws / rel
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        tracer.event("cli.init", workspace=str(ws))
        print(f"Workspace initialised at: {ws}")  # noqa: T201
    else:
        (ws / "MISSION.md").write_text("# Mission\n\nDescribe the robot's mission here.\n")
        (ws / "ROBOT.md").write_text("# Robot Fleet\n\nDefine your robots here.\n")
        print(f"Workspace created at: {ws}")  # noqa: T201


def _cmd_tool_list(args: argparse.Namespace) -> None:
    ctx = _build_ctx()
    _register_generic_tools(ctx)
    schemas = ctx.tool_registry.list_schemas()
    if not schemas:
        print("No tools registered.")  # noqa: T201
        return
    for s in schemas:
        print(f"  {s.name:30s}  {s.description[:60]}")  # noqa: T201


def _cmd_skill_list(args: argparse.Namespace) -> None:
    ctx = _build_ctx()
    manifests = ctx.skill_registry.list()
    if not manifests:
        print("No skills registered.")  # noqa: T201
        return
    for m in manifests:
        print(f"  {m.name:30s}  v{m.version}  safety={m.safety_class.value}")  # noqa: T201


def _cmd_skill_validate(args: argparse.Namespace) -> None:
    from robot_harness.skill.manifest_validator import validate_cli

    sys.exit(validate_cli(args.files))


def _cmd_fleet_status(args: argparse.Namespace) -> None:
    ctx = _build_ctx()
    cfg = ctx.config
    print(f"Fleet size : {cfg.fleet_size}")  # noqa: T201
    for rid in cfg.robot_ids:
        print(f"  robot_id : {rid}")  # noqa: T201


def _cmd_run(args: argparse.Namespace) -> None:
    asyncio.run(_async_run(args))


def _cmd_serve(args: argparse.Namespace) -> None:
    try:
        asyncio.run(_async_serve(args))
    except KeyboardInterrupt:
        print("\n[robot-loom] serve stopped.", file=sys.stderr)  # noqa: T201


def _build_channel(name: str, robot_id: str, config: Any) -> Any:
    """Construct one channel by name. Telegram needs TELEGRAM_BOT_TOKEN set;
    voice_gateway needs channels.voice_gateway.url in config.yaml (ADR-037)."""
    if name == "cli":
        from robot_harness.channels.cli import CLIChannel

        return CLIChannel(robot_id=robot_id)
    if name == "telegram":
        from robot_harness.channels.telegram import TelegramChannel

        return TelegramChannel(robot_id=robot_id)
    if name == "voice_gateway":
        from robot_harness.channels.voice.gateway import (
            VoiceGatewayChannel,
            WebSocketGatewayTransport,
        )

        gw = config.channels.voice_gateway
        if not gw.url:
            raise ValueError(
                "voice_gateway channel requires channels.voice_gateway.url in config.yaml"
            )
        transport = WebSocketGatewayTransport(
            gw.url,
            token_env=gw.token_env,
            reconnect_backoff_s=gw.reconnect_backoff_s,
        )
        return VoiceGatewayChannel(transport=transport, robot_id=robot_id)
    raise ValueError(f"unknown channel: {name!r}")


async def _async_serve(args: argparse.Namespace) -> None:
    from robot_harness.brain.base import Task
    from robot_harness.brain.litellm_brain import LiteLLMBrain
    from robot_harness.channels.manager import ChannelManager
    from robot_harness.runtime.agent_loop import AgentLoop

    ctx = _build_ctx()

    channel_names = list(dict.fromkeys(args.channel or ["cli"]))
    # Host tools (shell / fs) only when every serve surface is the local
    # terminal. A Telegram bot is reachable by anyone who finds it — handing
    # it shell_run would be a remote-execution surface, same reasoning as the
    # reverse MCP server (ISS-033). Remote users get the robot vocabulary.
    if channel_names == ["cli"]:
        _register_generic_tools(ctx)

    brain = LiteLLMBrain(ctx.config.brain)
    robot_id = args.robot_id or (ctx.config.robot_ids[0] if ctx.config.robot_ids else "robot-0")

    async def agent_loop_factory(task: Task, *, outbound: Any) -> Any:
        loop = AgentLoop(brain, ctx, max_turns=args.max_turns)
        return await loop.run(task, outbound=outbound)

    manager = ChannelManager(agent_loop_factory)
    for name in channel_names:
        manager.register(_build_channel(name, robot_id, ctx.config))

    print(  # noqa: T201
        f"[robot-loom] serving channels={channel_names} robot={robot_id} "
        f"brain={ctx.config.brain.model} (Ctrl+C to stop)",
        file=sys.stderr,
    )
    try:
        await manager.start()
    finally:
        await manager.stop()


def _cmd_mcp_serve(args: argparse.Namespace) -> None:
    from robot_harness.tools.mcp.server import HarnessMCPServer

    ctx = _build_ctx()
    # Deliberately NO _register_generic_tools here: the reverse MCP server hands
    # every brain-visible tool to arbitrary external clients, and shell_run /
    # file-write tools would be an unauthenticated remote-execution surface
    # (ISS-033). External callers get the robot vocabulary, not host access.
    server = HarnessMCPServer(
        ctx.tool_registry,
        ctx.safety_envelope,
        default_timeout_s=ctx.config.tool.default_timeout_s,
    )
    # stdio transport owns stdout for the MCP protocol — tracer writes to
    # stderr, so harness logging stays out of band.
    asyncio.run(server.start(args.transport))


async def _async_run(args: argparse.Namespace) -> None:
    from robot_harness.brain.base import Task
    from robot_harness.brain.litellm_brain import LiteLLMBrain
    from robot_harness.runtime.agent_loop import AgentLoop

    ctx = _build_ctx()
    _register_generic_tools(ctx)

    brain = LiteLLMBrain(ctx.config.brain)
    loop = AgentLoop(brain, ctx, max_turns=args.max_turns)

    robot_id = args.robot_id or (ctx.config.robot_ids[0] if ctx.config.robot_ids else "robot-0")
    task = Task(
        task_id=str(uuid.uuid4()),
        description=args.task,
        robot_id=robot_id,
    )

    result = await loop.run(task)
    print(json.dumps(result.model_dump(), indent=2, default=str))  # noqa: T201
    sys.exit(0 if result.outcome == "success" else 1)


def _build_ctx() -> HarnessContext:
    from robot_harness.runtime.harness_context import HarnessContext

    return HarnessContext.build()


def _register_generic_tools(ctx: HarnessContext) -> None:
    from robot_harness.tools.generic.fs import ReadFileTool, WriteFileTool
    from robot_harness.tools.generic.shell import ShellTool

    ctx.tool_registry.register(ShellTool())
    ctx.tool_registry.register(ReadFileTool())
    ctx.tool_registry.register(WriteFileTool())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="robot-loom",
        description="Robot Loom — model- and embodiment-agnostic robot agent harness",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialise user workspace")

    run_p = sub.add_parser("run", help="Run a task")
    run_p.add_argument("--task", required=True, help="Natural-language task description")
    run_p.add_argument("--robot-id", default="", help="Robot ID (defaults to first in config)")
    run_p.add_argument("--max-turns", type=int, default=20)

    serve_run_p = sub.add_parser(
        "serve", help="Run resident: route channel messages to the AgentLoop"
    )
    serve_run_p.add_argument(
        "--channel",
        action="append",
        choices=["cli", "telegram", "voice_gateway"],
        help="Channel to serve; repeatable (default: cli). "
        "telegram reads TELEGRAM_BOT_TOKEN from the environment; voice_gateway "
        "reads channels.voice_gateway.url from config.yaml.",
    )
    serve_run_p.add_argument(
        "--robot-id",
        default="",
        help="Robot ID messages are routed to (defaults to first in config)",
    )
    serve_run_p.add_argument("--max-turns", type=int, default=20)

    tool_p = sub.add_parser("tool", help="Tool management")
    tool_sub = tool_p.add_subparsers(dest="tool_cmd")
    tool_sub.add_parser("list", help="List registered tools")

    skill_p = sub.add_parser("skill", help="Skill management")
    skill_sub = skill_p.add_subparsers(dest="skill_cmd")
    skill_sub.add_parser("list", help="List registered skills")
    validate_p = skill_sub.add_parser("validate", help="Validate skill manifest YAML files")
    validate_p.add_argument("files", nargs="+", metavar="manifest.yaml")

    fleet_p = sub.add_parser("fleet", help="Fleet management")
    fleet_sub = fleet_p.add_subparsers(dest="fleet_cmd")
    fleet_sub.add_parser("status", help="Show fleet status")

    mcp_p = sub.add_parser("mcp", help="MCP integration")
    mcp_sub = mcp_p.add_subparsers(dest="mcp_cmd")
    serve_p = mcp_sub.add_parser(
        "serve", help="Expose the harness as an MCP server (reverse direction)"
    )
    serve_p.add_argument(
        "--transport",
        default="stdio",
        help='"stdio" (default) or an HTTP bind address like "0.0.0.0:8080"',
    )

    args = parser.parse_args()

    if args.command == "init":
        _cmd_init(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "tool" and args.tool_cmd == "list":
        _cmd_tool_list(args)
    elif args.command == "skill" and args.skill_cmd == "list":
        _cmd_skill_list(args)
    elif args.command == "skill" and args.skill_cmd == "validate":
        _cmd_skill_validate(args)
    elif args.command == "fleet" and args.fleet_cmd == "status":
        _cmd_fleet_status(args)
    elif args.command == "mcp" and args.mcp_cmd == "serve":
        _cmd_mcp_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
