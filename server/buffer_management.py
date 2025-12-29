"""Buffer management and back-pressure logic for audio streaming."""

import asyncio
import logging

from server.interfaces.buffer import IBufferManager

logger = logging.getLogger(__name__)


class BufferManager(IBufferManager):
    """Manages back-pressure logic and buffer health monitoring."""

    # Buffer health thresholds
    EMERGENCY_THRESHOLD = 1  # <1 chunk
    LOW_THRESHOLD = 2  # <2 chunks
    HEALTHY_MIN = 3  # 3-4 chunks
    FULL_THRESHOLD = 5  # 5+ chunks

    # Back-pressure delay
    BACK_PRESSURE_DELAY_MS = 10

    async def apply_back_pressure(self, current_depth: int, capacity: int) -> None:
        """Apply back-pressure when buffer depth is low.

        Args:
            current_depth: Current number of chunks in buffer
            capacity: Maximum buffer capacity

        Behavior:
            - If depth < 2: Sleep 10ms to allow consumer to catch up
            - Otherwise: No delay
        """
        if current_depth < self.LOW_THRESHOLD:
            logger.debug(
                f"Applying back-pressure: depth={current_depth}, "
                f"sleeping {self.BACK_PRESSURE_DELAY_MS}ms"
            )
            await asyncio.sleep(self.BACK_PRESSURE_DELAY_MS / 1000.0)

    def should_generate(self, current_depth: int, capacity: int) -> bool:
        """Check if generation should proceed based on buffer depth.

        Args:
            current_depth: Current number of chunks in buffer
            capacity: Maximum buffer capacity

        Returns:
            True if generation should proceed, False to wait
        """
        # Don't generate if buffer is full or nearly full
        if current_depth >= capacity - 1:
            logger.debug(
                f"Buffer nearly full ({current_depth}/{capacity}), "
                "pausing generation"
            )
            return False
        return True

    def get_buffer_health(self, current_depth: int, capacity: int) -> str:
        """Get buffer health status.

        Args:
            current_depth: Current number of chunks in buffer
            capacity: Maximum buffer capacity

        Returns:
            Health status: "emergency", "low", "healthy", or "full"
        """
        if current_depth < self.EMERGENCY_THRESHOLD:
            return "emergency"
        elif current_depth < self.LOW_THRESHOLD:
            return "low"
        elif current_depth >= self.FULL_THRESHOLD:
            return "full"
        else:
            return "healthy"

    def get_utilization_percentage(self, current_depth: int, capacity: int) -> float:
        """Get buffer utilization as percentage.

        Args:
            current_depth: Current number of chunks in buffer
            capacity: Maximum buffer capacity

        Returns:
            Utilization percentage (0.0-100.0)
        """
        return (current_depth / capacity * 100.0) if capacity > 0 else 0.0
