# Mission

<!--
This file (with ROBOT.md) is injected VERBATIM into the Brain's system prompt
for every task. Write standing orders here: goals, priorities, decision rules
that hold across tasks — the things a human operator would brief a new teammate
on once.

Keep OUT of this file:
- Verb/tool capability lists — the harness discovers capabilities live from
  each robot's agent_server; a prose copy goes stale the moment a backend
  changes.
- Fixed tool-call sequences — a recurring procedure belongs in a versioned
  skill, not in prose the model may misread.
- Runtime world state ("the cup is on the table") — observations belong in
  Memory, which the Brain queries on demand.
-->

Describe the robot's mission, goals, and decision rules here.

## Task Priorities

1. Safety first — always pass safety check before any hardware action
2. Pick and place objects as instructed
3. Report failures clearly

## Constraints

- Never move joints faster than 1.0 rad/s
- Stay within workspace bounds at all times
- Always verify grasp success before releasing
