"""
Prometheus Performance Metrics Collection

Implements comprehensive performance monitoring for real-time audio streaming,
tracking synthesis latency, memory usage, GPU utilization, and GC statistics.

Tasks: T084-T090 [US3]
"""

import asyncio
import time
from typing import Optional
import psutil
import torch
from prometheus_client import Counter, Gauge, Histogram
from loguru import logger

from server.gc_config import RealTimeGCConfig
from server.memory_monitor import MemoryMonitor


class PrometheusMetrics:
    """
    Centralized Prometheus metrics collection for performance monitoring (T084).

    Collects and exposes metrics for:
    - Synthesis latency and throughput
    - Memory usage (RSS, GPU)
    - GC collection statistics
    - System resource utilization
    """

    def __init__(self):
        """Initialize all Prometheus metrics."""

        # Synthesis Performance Metrics (T085)
        self.synthesis_latency_seconds = Histogram(
            'auralis_synthesis_latency_seconds',
            'Time taken to synthesize audio phrase',
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
        )

        self.synthesis_total = Counter(
            'auralis_synthesis_total',
            'Total number of synthesis operations completed'
        )

        self.phrase_generation_rate_hz = Gauge(
            'auralis_phrase_generation_rate_hz',
            'Phrases generated per second (T089)'
        )

        # Memory Metrics (T086, T087)
        self.memory_usage_mb = Gauge(
            'auralis_memory_usage_mb',
            'Process memory usage in megabytes (RSS)',
            ['type']  # type: rss, vms, python
        )

        self.gpu_memory_allocated_mb = Gauge(
            'auralis_gpu_memory_allocated_mb',
            'GPU memory allocated in megabytes (T087)',
            ['device']  # device: cuda, mps
        )

        self.gpu_memory_reserved_mb = Gauge(
            'auralis_gpu_memory_reserved_mb',
            'GPU memory reserved in megabytes (CUDA only)',
            ['device']
        )

        # GC Statistics Metrics (T088, T083)
        self.gc_collections_total = Counter(
            'auralis_gc_collections_total',
            'Total garbage collection operations by generation',
            ['generation']  # generation: 0, 1, 2
        )

        self.gc_collection_time_seconds = Histogram(
            'auralis_gc_collection_time_seconds',
            'Time spent in garbage collection',
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )

        # System Resource Metrics
        self.cpu_usage_percent = Gauge(
            'auralis_cpu_usage_percent',
            'CPU utilization percentage'
        )

        self.active_connections = Gauge(
            'auralis_active_connections',
            'Number of active WebSocket connections'
        )

        self.buffer_depth_ms = Gauge(
            'auralis_buffer_depth_ms',
            'Ring buffer depth in milliseconds',
            ['client_id']
        )

        # Memory Monitor Integration
        self.memory_monitor: Optional[MemoryMonitor] = None
        self.process = psutil.Process()

        # Async collection state
        self.collection_task: Optional[asyncio.Task] = None
        self.collection_interval_sec = 5.0  # T090: Collect every 5 seconds
        self.running = False

        # Phrase generation tracking
        self._phrase_count = 0
        self._last_phrase_count = 0
        self._last_rate_update = time.time()

        logger.info("✓ Prometheus metrics initialized")

    def set_memory_monitor(self, monitor: MemoryMonitor) -> None:
        """
        Attach memory monitor for detailed memory tracking.

        Args:
            monitor: MemoryMonitor instance
        """
        self.memory_monitor = monitor
        logger.info("✓ Memory monitor attached to metrics collector")

    def record_synthesis_latency(self, latency_seconds: float) -> None:
        """
        Record synthesis operation latency (T085).

        Args:
            latency_seconds: Time taken for synthesis operation
        """
        self.synthesis_latency_seconds.observe(latency_seconds)
        self.synthesis_total.inc()

        # Track phrase generation for rate calculation
        self._phrase_count += 1

    def update_phrase_generation_rate(self) -> None:
        """
        Update phrase generation rate metric (T089).

        Calculates phrases per second based on recent activity.
        """
        now = time.time()
        elapsed = now - self._last_rate_update

        if elapsed >= 1.0:  # Update every second
            phrases_generated = self._phrase_count - self._last_phrase_count
            rate_hz = phrases_generated / elapsed

            self.phrase_generation_rate_hz.set(rate_hz)

            self._last_phrase_count = self._phrase_count
            self._last_rate_update = now

    def update_memory_metrics(self) -> None:
        """
        Update memory usage metrics (T086, T087).

        Collects RSS, VMS, Python, and GPU memory measurements.
        """
        # Process memory (T086)
        mem_info = self.process.memory_info()
        self.memory_usage_mb.labels(type='rss').set(mem_info.rss / 1024**2)
        self.memory_usage_mb.labels(type='vms').set(mem_info.vms / 1024**2)

        # Python-specific memory (from memory monitor if available)
        if self.memory_monitor:
            stats = self.memory_monitor.get_memory_stats()
            self.memory_usage_mb.labels(type='python').set(
                stats['current']['python_mb']
            )

        # GPU memory (T087, T079)
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**2
            gpu_reserved = torch.cuda.memory_reserved() / 1024**2

            self.gpu_memory_allocated_mb.labels(device='cuda').set(gpu_allocated)
            self.gpu_memory_reserved_mb.labels(device='cuda').set(gpu_reserved)

        elif torch.backends.mps.is_available():
            try:
                gpu_allocated = torch.mps.current_allocated_memory() / 1024**2
                self.gpu_memory_allocated_mb.labels(device='mps').set(gpu_allocated)
            except AttributeError:
                # Older PyTorch versions don't support MPS memory tracking
                pass

    def update_gc_metrics(self) -> None:
        """
        Update garbage collection statistics (T088, T083).

        Tracks collection counts for all 3 generations.
        """
        gc_stats = RealTimeGCConfig.get_gc_stats()

        # Update collection counters for each generation
        for gen in [0, 1, 2]:
            gen_key = f"gen{gen}"
            collections = gc_stats[gen_key]['collections']

            # Set counter to absolute value (Prometheus will handle rate calculation)
            # Note: Counters should only increase, so we track the total
            self.gc_collections_total.labels(generation=str(gen))._value.set(collections)

    def update_cpu_metrics(self) -> None:
        """Update CPU utilization metrics."""
        cpu_percent = self.process.cpu_percent()
        self.cpu_usage_percent.set(cpu_percent)

    def update_all_metrics(self) -> None:
        """
        Update all metrics in a single collection pass.

        Called periodically by async collection task (T090).
        """
        self.update_memory_metrics()
        self.update_gc_metrics()
        self.update_cpu_metrics()
        self.update_phrase_generation_rate()

        # Check for memory leaks if monitor is available
        if self.memory_monitor and self.memory_monitor.should_sample():
            self.memory_monitor.take_snapshot()
            self.memory_monitor.check_and_log_leak()

    async def start_collection(self) -> None:
        """
        Start asynchronous metrics collection (T090).

        Collects metrics every 5 seconds in background task.
        """
        if self.running:
            logger.warning("Metrics collection already running")
            return

        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info(
            f"✓ Async metrics collection started "
            f"(interval: {self.collection_interval_sec}s)"
        )

    async def stop_collection(self) -> None:
        """Stop asynchronous metrics collection."""
        self.running = False

        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass

        logger.info("✓ Metrics collection stopped")

    async def _collection_loop(self) -> None:
        """
        Background task for periodic metrics collection (T090).

        Runs every 5 seconds to update all metrics.
        """
        logger.info("Metrics collection loop started")

        while self.running:
            try:
                # Update all metrics
                self.update_all_metrics()

                # Wait for next collection interval
                await asyncio.sleep(self.collection_interval_sec)

            except asyncio.CancelledError:
                logger.info("Metrics collection loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                # Continue despite errors
                await asyncio.sleep(self.collection_interval_sec)

    def set_active_connections(self, count: int) -> None:
        """
        Update active WebSocket connections count.

        Args:
            count: Number of active connections
        """
        self.active_connections.set(count)

    def set_buffer_depth(self, client_id: str, depth_ms: float) -> None:
        """
        Update buffer depth for a specific client.

        Args:
            client_id: Client identifier
            depth_ms: Buffer depth in milliseconds
        """
        self.buffer_depth_ms.labels(client_id=client_id).set(depth_ms)

    def get_summary(self) -> dict:
        """
        Get summary of current metrics for debugging/logging.

        Returns:
            Dictionary with current metric values
        """
        summary = {
            "synthesis": {
                "total_operations": self._phrase_count,
                "generation_rate_hz": self.phrase_generation_rate_hz._value.get(),
            },
            "memory": {
                "rss_mb": self.memory_usage_mb.labels(type='rss')._value.get(),
                "cpu_percent": self.cpu_usage_percent._value.get(),
            },
            "system": {
                "active_connections": self.active_connections._value.get(),
                "collection_running": self.running,
            }
        }

        # Add GPU info if available
        if torch.cuda.is_available():
            summary["gpu"] = {
                "allocated_mb": self.gpu_memory_allocated_mb.labels(device='cuda')._value.get(),
                "reserved_mb": self.gpu_memory_reserved_mb.labels(device='cuda')._value.get(),
            }

        return summary
