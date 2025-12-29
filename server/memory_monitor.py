"""Memory monitoring and leak detection.

Uses tracemalloc for detailed memory tracking and psutil for process-level
memory usage monitoring.
"""

import gc
import logging
import tracemalloc
from typing import Optional

import psutil

from server.interfaces.metrics import IMemoryMonitor

logger = logging.getLogger(__name__)


class MemoryMonitor(IMemoryMonitor):
    """Monitors memory usage and detects leaks."""

    def __init__(self) -> None:
        """Initialize memory monitor."""
        self.tracking_enabled = False
        self.baseline_mb: Optional[float] = None
        self.process = psutil.Process()

        logger.info("Memory monitor initialized")

    def start_tracking(self) -> None:
        """Start memory tracking (tracemalloc)."""
        if not self.tracking_enabled:
            tracemalloc.start()
            self.tracking_enabled = True
            self.baseline_mb = self.get_current_usage_mb()
            logger.info(
                f"Memory tracking started (baseline: {self.baseline_mb:.1f}MB)"
            )

    def stop_tracking(self) -> None:
        """Stop memory tracking."""
        if self.tracking_enabled:
            tracemalloc.stop()
            self.tracking_enabled = False
            logger.info("Memory tracking stopped")

    def get_current_usage_mb(self) -> float:
        """Get current process memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        memory_info = self.process.memory_info()
        return memory_info.rss / (1024 * 1024)

    def get_gc_stats(self) -> dict[str, int]:
        """Get garbage collection statistics.

        Returns:
            Dictionary with keys: gen0, gen1, gen2 (collection counts)
        """
        stats = gc.get_stats()
        return {
            "gen0": stats[0]["collections"] if stats else 0,
            "gen1": stats[1]["collections"] if len(stats) > 1 else 0,
            "gen2": stats[2]["collections"] if len(stats) > 2 else 0,
        }

    def check_for_leaks(self, threshold_mb: float = 10.0) -> bool:
        """Check if memory growth exceeds threshold.

        Args:
            threshold_mb: Maximum allowed growth in MB

        Returns:
            True if leak suspected, False otherwise
        """
        if self.baseline_mb is None:
            logger.warning("Memory baseline not set, cannot check for leaks")
            return False

        current_mb = self.get_current_usage_mb()
        growth_mb = current_mb - self.baseline_mb

        if growth_mb > threshold_mb:
            logger.warning(
                f"Memory leak suspected: {growth_mb:.1f}MB growth "
                f"(baseline: {self.baseline_mb:.1f}MB, "
                f"current: {current_mb:.1f}MB, "
                f"threshold: {threshold_mb:.1f}MB)"
            )
            return True

        return False

    def get_memory_snapshot(self) -> dict[str, float]:
        """Get detailed memory usage snapshot.

        Returns:
            Dictionary with memory statistics
        """
        current_mb = self.get_current_usage_mb()
        growth_mb = (
            current_mb - self.baseline_mb if self.baseline_mb is not None else 0.0
        )

        snapshot = {
            "current_mb": current_mb,
            "baseline_mb": self.baseline_mb or 0.0,
            "growth_mb": growth_mb,
            "tracking_enabled": self.tracking_enabled,
        }

        # Add tracemalloc details if enabled
        if self.tracking_enabled:
            current, peak = tracemalloc.get_traced_memory()
            snapshot["traced_current_mb"] = current / (1024 * 1024)
            snapshot["traced_peak_mb"] = peak / (1024 * 1024)

        return snapshot

    def reset_baseline(self) -> None:
        """Reset memory baseline to current usage.

        Useful after initialization or major operations.
        """
        self.baseline_mb = self.get_current_usage_mb()
        logger.info(f"Memory baseline reset to {self.baseline_mb:.1f}MB")
