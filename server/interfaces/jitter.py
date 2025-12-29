"""Jitter tracking interface for adaptive client buffering."""

from abc import ABC, abstractmethod


class IJitterTracker(ABC):
    """Tracks network jitter using exponential moving average (EMA)."""

    @abstractmethod
    def record_arrival(self, expected_time_ms: float, actual_time_ms: float) -> None:
        """Record chunk arrival time for jitter calculation.

        Args:
            expected_time_ms: Expected arrival time (based on chunk rate)
            actual_time_ms: Actual arrival timestamp

        Updates internal EMA with arrival variance.
        """
        pass

    @abstractmethod
    def get_jitter_ms(self) -> float:
        """Get current jitter estimate.

        Returns:
            Jitter in milliseconds (EMA of arrival variance)
        """
        pass

    @abstractmethod
    def get_recommended_buffer_size(self) -> int:
        """Get recommended buffer size based on jitter.

        Returns:
            Number of chunks to buffer (3-10 range)

        Logic:
            - Low jitter (<20ms): 3 chunks (300ms)
            - Medium jitter (20-50ms): 5 chunks (500ms)
            - High jitter (50-100ms): 7 chunks (700ms)
            - Very high jitter (>100ms): 10 chunks (1000ms)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset jitter tracking (e.g., after reconnection)."""
        pass
