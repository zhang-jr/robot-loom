"""robot-loom CLI entry point.

Commands:
  robot-loom init                 Initialise a workspace in ~/.robot-loom/workspace/
  robot-loom run --task TEXT      Run a task with the configured Brain
  robot-loom tool list            List registered tools
  robot-loom skill list           List registered skills
  robot-loom fleet status         Show fleet robot IDs from config
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

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


def _cmd_fleet_status(args: argparse.Namespace) -> None:
    ctx = _build_ctx()
    cfg = ctx.config
    print(f"Fleet size : {cfg.fleet_size}")  # noqa: T201
    for rid in cfg.robot_ids:
        print(f"  robot_id : {rid}")  # noqa: T201


def _cmd_run(args: argparse.Namespace) -> None:
    asyncio.run(_async_run(args))


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

    tool_p = sub.add_parser("tool", help="Tool management")
    tool_sub = tool_p.add_subparsers(dest="tool_cmd")
    tool_sub.add_parser("list", help="List registered tools")

    skill_p = sub.add_parser("skill", help="Skill management")
    skill_sub = skill_p.add_subparsers(dest="skill_cmd")
    skill_sub.add_parser("list", help="List registered skills")

    fleet_p = sub.add_parser("fleet", help="Fleet management")
    fleet_sub = fleet_p.add_subparsers(dest="fleet_cmd")
    fleet_sub.add_parser("status", help="Show fleet status")

    args = parser.parse_args()

    if args.command == "init":
        _cmd_init(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "tool" and args.tool_cmd == "list":
        _cmd_tool_list(args)
    elif args.command == "skill" and args.skill_cmd == "list":
        _cmd_skill_list(args)
    elif args.command == "fleet" and args.fleet_cmd == "status":
        _cmd_fleet_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
