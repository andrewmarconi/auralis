"""Metrics and monitoring interface definitions."""

from abc import ABC, abstractmethod
from typing import Any


class IMetricsCollector(ABC):
    """Collects and aggregates performance metrics."""

    @abstractmethod
    def record_synthesis_latency(self, latency_ms: float) -> None:
        """Record synthesis latency measurement.

        Args:
            latency_ms: Synthesis time in milliseconds
        """
        pass

    @abstractmethod
    def record_network_latency(self, latency_ms: float) -> None:
        """Record network round-trip latency.

        Args:
            latency_ms: Network latency in milliseconds
        """
        pass

    @abstractmethod
    def record_end_to_end_latency(self, latency_ms: float) -> None:
        """Record total end-to-end latency.

        Args:
            latency_ms: Total latency from generation to playback (milliseconds)
        """
        pass

    @abstractmethod
    def increment_buffer_underrun(self) -> None:
        """Increment buffer underrun counter."""
        pass

    @abstractmethod
    def increment_buffer_overflow(self) -> None:
        """Increment buffer overflow counter."""
        pass

    @abstractmethod
    def increment_disconnect(self) -> None:
        """Increment client disconnect counter."""
        pass

    @abstractmethod
    def get_snapshot(self) -> dict[str, Any]:
        """Get current metrics snapshot.

        Returns:
            Dictionary with metrics data:
            - synthesis_latency_ms: {avg, p50, p95, p99, samples}
            - network_latency_ms: {avg, p50, p95, p99, samples}
            - end_to_end_latency_ms: {avg, p50, p95, p99, samples}
            - buffer_underruns: int
            - buffer_overflows: int
            - disconnects: int
            - memory_usage_mb: float
            - gc_collections: {gen0, gen1, gen2}
            - timestamp: ISO 8601 string
        """
        pass


class IMemoryMonitor(ABC):
    """Monitors memory usage and detects leaks."""

    @abstractmethod
    def start_tracking(self) -> None:
        """Start memory tracking (tracemalloc)."""
        pass

    @abstractmethod
    def stop_tracking(self) -> None:
        """Stop memory tracking."""
        pass

    @abstractmethod
    def get_current_usage_mb(self) -> float:
        """Get current process memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        pass

    @abstractmethod
    def get_gc_stats(self) -> dict[str, int]:
        """Get garbage collection statistics.

        Returns:
            Dictionary with keys: gen0, gen1, gen2 (collection counts)
        """
        pass

    @abstractmethod
    def check_for_leaks(self, threshold_mb: float = 10.0) -> bool:
        """Check if memory growth exceeds threshold.

        Args:
            threshold_mb: Maximum allowed growth in MB

        Returns:
            True if leak suspected, False otherwise
        """
        pass
