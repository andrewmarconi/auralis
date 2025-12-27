"""
Memory Monitoring and Leak Detection

Implements memory tracking, growth analysis, and leak detection for long-running
audio streaming sessions (8+ hours).

Tasks: T074-T079 [US3]
"""

import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Optional
import psutil
import torch
import numpy as np
from loguru import logger


@dataclass
class MemorySnapshot:
    """
    Single point-in-time memory measurement (T074).

    Attributes:
        timestamp: Unix timestamp when snapshot was taken
        rss_mb: Resident Set Size in MB (physical memory used)
        vms_mb: Virtual Memory Size in MB
        python_mb: Python-specific memory usage (via tracemalloc)
        gpu_allocated_mb: GPU memory allocated (CUDA/Metal)
        gpu_reserved_mb: GPU memory reserved (CUDA only)
    """
    timestamp: float
    rss_mb: float
    vms_mb: float
    python_mb: float = 0.0
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0

    @property
    def age_seconds(self) -> float:
        """Time elapsed since this snapshot."""
        return time.time() - self.timestamp


class MemoryGrowthTracker:
    """
    Track memory growth over time using linear regression (T075).

    Detects memory leaks by analyzing the trend in memory usage over
    a sliding window of snapshots.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize growth tracker.

        Args:
            window_size: Number of snapshots to retain for analysis
        """
        self.window_size = window_size
        self.snapshots: List[MemorySnapshot] = []

    def add_snapshot(self, snapshot: MemorySnapshot) -> None:
        """
        Add a memory snapshot to the tracking window.

        Args:
            snapshot: Memory measurement to add
        """
        self.snapshots.append(snapshot)

        # Keep only the most recent window_size snapshots
        if len(self.snapshots) > self.window_size:
            self.snapshots = self.snapshots[-self.window_size:]

    def get_growth_rate_mb_per_hour(self) -> Optional[float]:
        """
        Calculate memory growth rate using linear regression (T075).

        Returns:
            Growth rate in MB/hour, or None if insufficient data
        """
        if len(self.snapshots) < 10:
            return None

        # Extract time and memory data
        times = np.array([s.timestamp for s in self.snapshots])
        memory = np.array([s.rss_mb for s in self.snapshots])

        # Normalize time to hours from first snapshot
        times_hours = (times - times[0]) / 3600.0

        # Linear regression: memory = slope * time + intercept
        # Using numpy polyfit (degree 1 = linear)
        slope, _ = np.polyfit(times_hours, memory, 1)

        return float(slope)  # MB per hour

    def is_leak_detected(self, threshold_mb_per_hour: float = 20.0) -> bool:
        """
        Detect memory leak based on growth rate threshold (T078).

        Args:
            threshold_mb_per_hour: Maximum acceptable growth rate

        Returns:
            True if leak detected, False otherwise
        """
        growth_rate = self.get_growth_rate_mb_per_hour()

        if growth_rate is None:
            return False

        return growth_rate > threshold_mb_per_hour

    def get_projected_memory_mb(self, hours_ahead: float) -> Optional[float]:
        """
        Project future memory usage based on current growth trend.

        Args:
            hours_ahead: Hours into the future to project

        Returns:
            Projected memory in MB, or None if insufficient data
        """
        if len(self.snapshots) < 10:
            return None

        growth_rate = self.get_growth_rate_mb_per_hour()
        if growth_rate is None:
            return None

        current_memory = self.snapshots[-1].rss_mb
        projected = current_memory + (growth_rate * hours_ahead)

        return max(0.0, projected)  # Memory can't be negative


class MemoryMonitor:
    """
    Periodic memory monitoring with leak detection (T076).

    Samples memory usage at regular intervals and tracks growth trends
    to detect memory leaks during long-running streaming sessions.
    """

    def __init__(
        self,
        sample_interval_sec: float = 60.0,
        enable_tracemalloc: bool = True,
        leak_threshold_mb_per_hour: float = 20.0,
    ):
        """
        Initialize memory monitor.

        Args:
            sample_interval_sec: Time between memory snapshots
            enable_tracemalloc: Enable Python memory profiling (T077)
            leak_threshold_mb_per_hour: Leak detection threshold (T078)
        """
        self.sample_interval_sec = sample_interval_sec
        self.enable_tracemalloc = enable_tracemalloc
        self.leak_threshold_mb_per_hour = leak_threshold_mb_per_hour

        # Initialize tracking
        self.growth_tracker = MemoryGrowthTracker(window_size=100)
        self.process = psutil.Process()
        self.last_sample_time = 0.0

        # Enable tracemalloc for Python memory profiling (T077)
        if self.enable_tracemalloc:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                logger.info("✓ tracemalloc profiling enabled for memory leak detection")

        # GPU device type
        self.gpu_device = self._detect_gpu_device()

        logger.info(
            f"✓ Memory monitor initialized: "
            f"sample_interval={sample_interval_sec}s, "
            f"leak_threshold={leak_threshold_mb_per_hour}MB/hour"
        )

    def _detect_gpu_device(self) -> Optional[str]:
        """Detect GPU device type for memory monitoring (T079)."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return None

    def take_snapshot(self) -> MemorySnapshot:
        """
        Take a memory snapshot with current usage statistics.

        Returns:
            MemorySnapshot with current memory measurements
        """
        # Get process memory info
        mem_info = self.process.memory_info()
        rss_mb = mem_info.rss / 1024**2
        vms_mb = mem_info.vms / 1024**2

        # Get Python memory via tracemalloc (T077)
        python_mb = 0.0
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            python_mb = current / 1024**2

        # Get GPU memory (T079)
        gpu_allocated_mb = 0.0
        gpu_reserved_mb = 0.0

        if self.gpu_device == "cuda":
            gpu_allocated_mb = torch.cuda.memory_allocated() / 1024**2
            gpu_reserved_mb = torch.cuda.memory_reserved() / 1024**2
        elif self.gpu_device == "mps":
            # Metal doesn't expose detailed memory stats like CUDA
            # We can only get allocated memory
            try:
                gpu_allocated_mb = torch.mps.current_allocated_memory() / 1024**2
            except AttributeError:
                # Fallback for older PyTorch versions
                pass

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            python_mb=python_mb,
            gpu_allocated_mb=gpu_allocated_mb,
            gpu_reserved_mb=gpu_reserved_mb,
        )

        # Add to growth tracker
        self.growth_tracker.add_snapshot(snapshot)
        self.last_sample_time = snapshot.timestamp

        return snapshot

    def should_sample(self) -> bool:
        """
        Check if it's time to take another sample.

        Returns:
            True if sample interval has elapsed
        """
        elapsed = time.time() - self.last_sample_time
        return elapsed >= self.sample_interval_sec

    def get_memory_stats(self) -> dict:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with current memory state and growth analysis
        """
        if not self.growth_tracker.snapshots:
            snapshot = self.take_snapshot()
        else:
            snapshot = self.growth_tracker.snapshots[-1]

        growth_rate = self.growth_tracker.get_growth_rate_mb_per_hour()
        leak_detected = self.growth_tracker.is_leak_detected(
            self.leak_threshold_mb_per_hour
        )

        # Project memory 8 hours ahead (for 8-hour stability validation)
        projected_8h = self.growth_tracker.get_projected_memory_mb(8.0)

        stats = {
            "current": {
                "rss_mb": snapshot.rss_mb,
                "vms_mb": snapshot.vms_mb,
                "python_mb": snapshot.python_mb,
                "gpu_allocated_mb": snapshot.gpu_allocated_mb,
                "gpu_reserved_mb": snapshot.gpu_reserved_mb,
            },
            "growth_analysis": {
                "rate_mb_per_hour": growth_rate,
                "leak_detected": leak_detected,
                "threshold_mb_per_hour": self.leak_threshold_mb_per_hour,
                "projected_8h_mb": projected_8h,
            },
            "monitoring": {
                "snapshots_collected": len(self.growth_tracker.snapshots),
                "sample_interval_sec": self.sample_interval_sec,
                "tracemalloc_enabled": self.enable_tracemalloc,
                "gpu_device": self.gpu_device,
            },
        }

        return stats

    def check_and_log_leak(self) -> bool:
        """
        Check for memory leak and log warning if detected (T078).

        Returns:
            True if leak detected, False otherwise
        """
        if self.growth_tracker.is_leak_detected(self.leak_threshold_mb_per_hour):
            growth_rate = self.growth_tracker.get_growth_rate_mb_per_hour()
            projected_8h = self.growth_tracker.get_projected_memory_mb(8.0)

            logger.warning(
                f"⚠ Memory leak detected! "
                f"Growth rate: {growth_rate:.2f} MB/hour "
                f"(threshold: {self.leak_threshold_mb_per_hour} MB/hour). "
                f"Projected 8-hour memory: {projected_8h:.2f} MB"
            )
            return True

        return False

    def get_top_python_allocations(self, limit: int = 10) -> List[tuple]:
        """
        Get top Python memory allocations using tracemalloc (T077).

        Args:
            limit: Number of top allocations to return

        Returns:
            List of (filename, lineno, size_mb) tuples
        """
        if not self.enable_tracemalloc or not tracemalloc.is_tracing():
            return []

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        allocations = []
        for stat in top_stats[:limit]:
            allocations.append((
                stat.filename,
                stat.lineno,
                stat.size / 1024**2  # Convert to MB
            ))

        return allocations

    def cleanup(self) -> None:
        """Stop tracemalloc and cleanup resources."""
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
            logger.info("✓ tracemalloc profiling stopped")
