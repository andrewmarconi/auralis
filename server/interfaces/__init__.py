"""Internal interfaces for Auralis server components.

Abstract Base Classes (ABCs) defining contracts for synthesis, buffering,
metrics collection, and other core server functionality.
"""

from server.interfaces.synthesis import IFluidSynthRenderer, ISynthesisEngine
from server.interfaces.buffer import IRingBuffer, IBufferManager
from server.interfaces.metrics import IMetricsCollector, IMemoryMonitor
from server.interfaces.jitter import IJitterTracker

__all__ = [
    # Synthesis interfaces
    "ISynthesisEngine",
    "IFluidSynthRenderer",
    # Buffer interfaces
    "IRingBuffer",
    "IBufferManager",
    # Metrics interfaces
    "IMetricsCollector",
    "IMemoryMonitor",
    # Client-side interfaces
    "IJitterTracker",
]
