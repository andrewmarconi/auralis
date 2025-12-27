"""Abstract interface for jitter tracking."""

from abc import ABC, abstractmethod


class IJitterTracker(ABC):
    """Abstract interface for jitter tracking."""

    @abstractmethod
    def record_chunk(self, expected_time: float, actual_time: float) -> None:
        """
        Record chunk delivery timing.

        Args:
            expected_time: Expected arrival time (Unix timestamp)
            actual_time: Actual arrival time (Unix timestamp)
        """
        pass

    @abstractmethod
    def record_underrun(self) -> None:
        """Record buffer underrun event."""
        pass

    @abstractmethod
    def get_current_jitter(self) -> float:
        """
        Get current mean jitter.

        Returns:
            Mean jitter in milliseconds
        """
        pass

    @abstractmethod
    def get_jitter_std(self) -> float:
        """
        Get jitter standard deviation.

        Returns:
            Standard deviation in milliseconds
        """
        pass

    @abstractmethod
    def get_underrun_rate(self) -> float:
        """
        Get underrun rate.

        Returns:
            Fraction of chunks that underran (0.0-1.0)
        """
        pass

    @abstractmethod
    def get_recommended_buffer_ms(self, confidence: float = 0.95) -> float:
        """
        Calculate recommended buffer size.

        Args:
            confidence: Confidence level (0.95 or 0.99)

        Returns:
            Recommended buffer duration in milliseconds
        """
        pass
