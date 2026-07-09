"""Construct a Memory backend from configuration.

The harness owns the Memory abstraction; the backing store is pluggable. This
factory selects the concrete store from ``MemoryConfig.backend`` and enforces
one consistency rule: a multi-robot fleet must use the ``external`` backend,
because the ``null`` and ``embedded`` stores cannot be shared between robots.
"""

from __future__ import annotations

from robot_harness.config.schema import MemoryConfig
from robot_harness.errors import MemoryServiceUnavailable
from robot_harness.memory.base import Memory, NullMemory
from robot_harness.observability.tracer import tracer
from robot_harness.tools.memory.remote_adapter import RemoteSpatialMemory
from robot_harness.tools.memory.spatial_hub_adapter import SpatialHubMemory


def build_memory(config: MemoryConfig, *, fleet_size: int = 1) -> Memory:
    """Build the Memory backend selected by *config*.

    Args:
        config:     The resolved ``MemoryConfig``.
        fleet_size: Number of robots in the fleet; used to validate that a
                    shareable backend is selected when more than one robot
                    is present.

    Returns:
        A Memory implementation. When ``fleet_size > 1`` is paired with a
        non-shareable backend, the unsharable store is still returned but a
        warning is emitted so the operator can correct the configuration.

    Raises:
        MemoryServiceUnavailable: ``external`` backend selected without a
            ``server_url``.
    """
    if fleet_size > 1 and config.backend != "external":
        tracer.event(
            "memory.backend_not_shareable",
            backend=config.backend,
            fleet_size=fleet_size,
            warning=(
                f"Backend '{config.backend}' is not shared across robots but "
                f"fleet_size={fleet_size}. Each robot will see only its own "
                "local memory. Use the 'external' backend for fleet-wide memory."
            ),
        )

    match config.backend:
        case "null":
            return NullMemory()
        case "embedded":
            return SpatialHubMemory()
        case "external":
            if not config.server_url:
                raise MemoryServiceUnavailable(
                    "Memory backend 'external' requires memory.server_url to be set",
                    module_name="memory",
                )
            return RemoteSpatialMemory(config.server_url, timeout_s=config.request_timeout_s)
