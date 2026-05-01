"""EmbodimentCatalog — registry of EmbodimentAdapter instances by robot_id."""

from __future__ import annotations

from robot_harness.embodiment.base import EmbodimentAdapter
from robot_harness.errors import RobotOfflineError


class EmbodimentCatalog:
    """Maintains a map of robot_id → EmbodimentAdapter.

    Used by the Fleet coordinator and AgentLoop to resolve the adapter
    for a given robot without importing concrete adapter classes directly.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, EmbodimentAdapter] = {}

    def register(self, adapter: EmbodimentAdapter) -> None:
        """Register an adapter; overwrites existing entry for the same robot_id."""
        self._adapters[adapter.robot_id] = adapter

    def get(self, robot_id: str) -> EmbodimentAdapter:
        """Return the adapter for *robot_id* or raise RobotOfflineError."""
        try:
            return self._adapters[robot_id]
        except KeyError:
            raise RobotOfflineError(
                f"No adapter registered for robot '{robot_id}'",
                robot_id=robot_id,
            ) from None

    def list_robot_ids(self) -> list[str]:
        return list(self._adapters.keys())

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, robot_id: object) -> bool:
        return robot_id in self._adapters
