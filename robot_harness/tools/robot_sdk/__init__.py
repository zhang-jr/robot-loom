"""robot_sdk tools — harness-side clients for per-robot agent_servers.

Layered closed-loop boundary (see ADR-019):

Tools in this package are how the harness talks to a robot. They come in two
flavors, both backed by an external on-robot agent_server:

  1. Low-level dispatch tool (`robot_sdk.execute_action`)
     ----------------------------------------------------
     Wraps EmbodimentAdapter.dispatch(EmbodimentCommand). Use this when the
     harness has already computed a concrete target (joint config, cartesian
     pose, locomotion goal) and just needs the on-robot runtime to execute it.
     The on-robot runtime still owns trajectory planning and tight control.

  2. Reactive verbs / on-robot skills (future, ADR-019)
     ----------------------------------------------------
     Wraps a higher-level verb endpoint on the agent_server, e.g.
       • reactive_grasp(target_pose, hint)
       • visual_servo_to(target_pose)
       • follow_target(track_id)
       • traverse_corridor(goal_pose)
     The harness calls these like any other tool — it provides intent and
     constraints, then awaits a CompletionVerdict. The perception-action loop
     (5-30 Hz) runs on-robot; the harness never sees per-tick state.

These two flavors share a transport (HTTP/WS per ADR-016) but differ in what
the harness owns:

    | Tool flavor       | harness decides     | on-robot decides              |
    | ----------------- | ------------------- | ----------------------------- |
    | execute_action    | exact target value  | trajectory, servo, safety     |
    |                   |                     | reflex                        |
    | reactive verb     | high-level intent   | perception-action loop,       |
    |                   | + constraints       | termination criterion, plus   |
    |                   |                     | everything execute_action     |
    |                   |                     | owns                          |

What does NOT belong in this package:
    • Skill compositions across multiple robot_sdk calls — those are declarative
      skills, see robot_harness/skill/.
    • Visual-servoing loops driven from harness — that's the anti-pattern this
      boundary exists to prevent. Wrap the on-robot verb instead.
    • Capability servers (perception / grasp / memory / critic) — those live in
      tools/{perception,grasp,memory,critic}/, NOT here. robot_sdk is robot-bound;
      capability servers are reusable across robots.

Reference contrast (ADR-019):
    • OM1 / RoboClaw   — full agent loop runs on-robot; harness layer is thin.
    • sam3_navigate / sam3_grasp — capability server pattern; same protocol shape
      as our reactive verbs but the loop runs in a server instead of on-robot.
    • robot_harness    — cloud-side orchestrator; consumes both flavors above.
"""

from robot_harness.tools.robot_sdk.http_adapter import RobotSdkTool
from robot_harness.tools.robot_sdk.verbs import (
    COMPLETION_VERDICT_SCHEMA,
    ROBOT_SDK_HOME,
    ROBOT_SDK_LOCOMOTE_TO,
    ROBOT_SDK_MOVE_JOINTS,
    ROBOT_SDK_MOVE_TO_POSE,
    ROBOT_SDK_REACTIVE_GRASP,
    ROBOT_SDK_VISUAL_SERVO_TO,
    VERB_TOOL_NAMES,
    CompletionOutcome,
    CompletionVerdict,
    HomeTool,
    LocomoteToTool,
    MoveJointsTool,
    MoveToPoseTool,
    ReactiveGraspTool,
    VisualServoToTool,
    build_robot_sdk_verb_tools,
    unavailable_verb_tool_names,
)

__all__ = [
    "COMPLETION_VERDICT_SCHEMA",
    "ROBOT_SDK_HOME",
    "ROBOT_SDK_LOCOMOTE_TO",
    "ROBOT_SDK_MOVE_JOINTS",
    "ROBOT_SDK_MOVE_TO_POSE",
    "ROBOT_SDK_REACTIVE_GRASP",
    "ROBOT_SDK_VISUAL_SERVO_TO",
    "VERB_TOOL_NAMES",
    "CompletionOutcome",
    "CompletionVerdict",
    "HomeTool",
    "LocomoteToTool",
    "MoveJointsTool",
    "MoveToPoseTool",
    "ReactiveGraspTool",
    "RobotSdkTool",
    "VisualServoToTool",
    "build_robot_sdk_verb_tools",
    "unavailable_verb_tool_names",
]
