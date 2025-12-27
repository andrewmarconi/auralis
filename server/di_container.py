"""Dependency injection container for Auralis component wiring."""

from typing import Protocol
from server.interfaces.buffer import IRingBuffer
from server.interfaces.jitter import IJitterTracker
from server.interfaces.synthesis import ISynthesisEngine
from server.interfaces.metrics import IMetricsCollector
from server.ring_buffer import RingBuffer
from server.synthesis_engine import SynthesisEngine


class DIContainer:
    """Dependency injection container for component wiring."""

    def __init__(self):
        # Core components
        self.ring_buffer: IRingBuffer = RingBuffer()

        # Synthesis
        self.synthesis_engine: ISynthesisEngine = SynthesisEngine()

        # Jitter tracking
        from server.buffer_management import JitterTracker

        self.jitter_tracker: IJitterTracker = JitterTracker()

        # Metrics
        from server.metrics import PrometheusMetrics

        self.metrics_collector: IMetricsCollector = PrometheusMetrics()


# Global container instance
container = DIContainer()
