# Heartbeat

Periodic health-check instructions for the harness scheduler.

## Checks

- [ ] Brain API reachable (ping LiteLLM endpoint)
- [ ] Memory server reachable
- [ ] Critic server reachable
- [ ] All registered robots reachable

## Interval

Default: 30 seconds.  Set ROBOT_LOOM_HEARTBEAT_INTERVAL_S to override.
