"""Abstract interface for metrics collection."""

from abc import ABC, abstractmethod


class IMetricsCollector(ABC):
    """Abstract interface for metrics collection."""

    @abstractmethod
    def record_synthesis_latency(self, latency_sec: float) -> None:
        """
        Record synthesis latency.

        Args:
            latency_sec: Latency in seconds
        """
        pass

    @abstractmethod
    def record_buffer_depth(self, client_id: str, depth: int) -> None:
        """
        Record buffer depth for client.

        Args:
            client_id: Client identifier
            depth: Buffer depth in chunks
        """
        pass

    @abstractmethod
    def record_underrun(self, client_id: str) -> None:
        """
        Record buffer underrun event.

        Args:
            client_id: Client identifier
        """
        pass

    @abstractmethod
    def update_memory_metrics(self) -> None:
        """Update all memory-related metrics."""
        pass

    @abstractmethod
    def export_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus text exposition format
        """
        pass
