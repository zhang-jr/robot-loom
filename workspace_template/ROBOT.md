# Robot Fleet

<!--
This file (with MISSION.md) is injected VERBATIM into the Brain's system prompt
for every task. Write the fleet and site facts the harness cannot discover by
itself: what each robot is, and — most importantly — the SITE SEMANTICS the
Brain needs to ground language into coordinates ("the table" → a map-frame
pose). Without site notes, navigation verbs cannot be grounded from natural
language.

Keep OUT of this file:
- Verb/tool capability lists — discovered live from each robot's agent_server.
- Deployment wiring (URLs, timeouts) — that lives in config.yaml; here goes
  the prose ABOUT the robots, not how to reach them.
- Runtime world state — observations belong in Memory.
-->

Define your robot fleet here.

## robot-0

- type: arm
- dof: 6
- gripper: parallel_jaw
- cameras: [wrist_rgb, overhead_rgb]

## Site notes

<!-- Named places in the map frame the Brain can navigate to, no-go areas,
     etiquette. Update after (re)mapping the site. Example: -->

- "charging dock" is at map pose [0.0, 0.0, 0.0]
- "the table" is at map pose [1.5, 0.0], approach from the south side
- Do not enter the elevator lobby (x > 4.0)

## Notes

- Add more robots following the same format
- robot_id must match the IDs in config.yaml
