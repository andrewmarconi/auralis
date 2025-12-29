"""Performance metrics collection and aggregation.

Tracks synthesis latency, network latency, buffer health, and memory usage
for monitoring and optimization.
"""

import gc
import logging
import time
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from server.interfaces.metrics import IMetricsCollector

logger = logging.getLogger(__name__)


class LatencyHistogram:
    """Tracks latency measurements with percentile calculations."""

    def __init__(self, max_samples: int = 10000):
        """Initialize latency histogram.

        Args:
            max_samples: Maximum samples to retain (circular buffer)
        """
        self.samples: deque[float] = deque(maxlen=max_samples)
        self.max_samples = max_samples

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.samples.append(latency_ms)

    def get_stats(self) -> dict[str, float | int]:
        """Get latency statistics.

        Returns:
            Dictionary with avg, p50, p95, p99, samples count
        """
        if not self.samples:
            return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "samples": 0}

        arr = np.array(list(self.samples))
        return {
            "avg": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "samples": len(self.samples),
        }


class PerformanceMetrics(IMetricsCollector):
    """Collects and aggregates performance metrics."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.synthesis_latency = LatencyHistogram()
        self.network_latency = LatencyHistogram()
        self.end_to_end_latency = LatencyHistogram()

        # Event counters
        self.buffer_underruns = 0
        self.buffer_overflows = 0
        self.disconnects = 0

        # Startup time
        self.start_time = time.time()

        logger.info("Metrics collector initialized")

    def record_synthesis_latency(self, latency_ms: float) -> None:
        """Record synthesis latency measurement.

        Args:
            latency_ms: Synthesis time in milliseconds
        """
        self.synthesis_latency.record(latency_ms)

        # Warn if exceeding target
        if latency_ms > 100.0:
            logger.warning(
                f"Synthesis latency {latency_ms:.1f}ms exceeds 100ms target"
            )

    def record_network_latency(self, latency_ms: float) -> None:
        """Record network round-trip latency.

        Args:
            latency_ms: Network latency in milliseconds
        """
        self.network_latency.record(latency_ms)

    def record_end_to_end_latency(self, latency_ms: float) -> None:
        """Record total end-to-end latency.

        Args:
            latency_ms: Total latency from generation to playback (milliseconds)
        """
        self.end_to_end_latency.record(latency_ms)

        # Warn if exceeding target
        if latency_ms > 800.0:
            logger.warning(
                f"End-to-end latency {latency_ms:.1f}ms exceeds 800ms target"
            )

    def increment_buffer_underrun(self) -> None:
        """Increment buffer underrun counter."""
        self.buffer_underruns += 1
        logger.warning(f"Buffer underrun detected (total: {self.buffer_underruns})")

    def increment_buffer_overflow(self) -> None:
        """Increment buffer overflow counter."""
        self.buffer_overflows += 1
        logger.warning(f"Buffer overflow detected (total: {self.buffer_overflows})")

    def increment_disconnect(self) -> None:
        """Increment client disconnect counter."""
        self.disconnects += 1
        logger.info(f"Client disconnect (total: {self.disconnects})")

    def get_snapshot(self) -> dict[str, Any]:
        """Get current metrics snapshot.

        Returns:
            Dictionary with all metrics data
        """
        import psutil

        # Get process memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        # Get GC stats
        gc_stats = gc.get_stats()
        gc_collections = {
            "gen0": gc_stats[0]["collections"] if gc_stats else 0,
            "gen1": gc_stats[1]["collections"] if len(gc_stats) > 1 else 0,
            "gen2": gc_stats[2]["collections"] if len(gc_stats) > 2 else 0,
        }

        return {
            "synthesis_latency_ms": self.synthesis_latency.get_stats(),
            "network_latency_ms": self.network_latency.get_stats(),
            "end_to_end_latency_ms": self.end_to_end_latency.get_stats(),
            "buffer_underruns": self.buffer_underruns,
            "buffer_overflows": self.buffer_overflows,
            "disconnects": self.disconnects,
            "memory_usage_mb": memory_mb,
            "gc_collections": gc_collections,
            "uptime_sec": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
        }
