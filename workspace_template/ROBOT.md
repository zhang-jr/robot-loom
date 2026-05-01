# Robot Fleet

Define your robot fleet here.

## robot-0

- type: arm
- dof: 6
- gripper: parallel_jaw
- cameras: [wrist_rgb, overhead_rgb]
- agent_server: http://localhost:8080

## Notes

- Add more robots following the same format
- robot_id must match the IDs in config.yaml
