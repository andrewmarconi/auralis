"""Abstract interfaces for Auralis components."""

from .buffer import IRingBuffer
from .jitter import IJitterTracker
from .synthesis import ISynthesisEngine
from .metrics import IMetricsCollector

__all__ = [
    "IRingBuffer",
    "IJitterTracker",
    "ISynthesisEngine",
    "IMetricsCollector",
]
