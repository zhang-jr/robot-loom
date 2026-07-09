"""Real-hardware embodiment backends.

The backend family for physical robots, parallel to ``sim/``. Adapters here are
split by **transport / SDK type**, not by morphology — ``robot_type`` is a value,
not a structural axis (the on-robot agent_server absorbs body-specific control).

Members:
  - ``agent_server.py`` — RealAgentServerAdapter: the default, for robots reached
    through the standard per-robot HTTP ``agent_server`` contract.
  - (future) ``canbus.py`` / ``serial.py`` / ``<vendor>_sdk.py`` — for robots that
    do NOT speak the agent_server contract and whose action-space mapping must
    happen in-harness.
"""

from robot_harness.embodiment.real.agent_server import RealAgentServerAdapter

__all__ = ["RealAgentServerAdapter"]
