"""Auralis Server - Real-time generative ambient music streaming engine.

This module contains the FastAPI server, synthesis engine, streaming infrastructure,
and all server-side logic for the Auralis platform.
"""

from server.audio_chunk import AudioChunk
from server.buffer_management import BufferManager
from server.config import AuralisConfig, get_config
from server.fluidsynth_renderer import FluidSynthRenderer
from server.memory_monitor import MemoryMonitor
from server.metrics import PerformanceMetrics
from server.presets import get_preset, list_presets
from server.ring_buffer import RingBuffer
from server.soundfont_manager import SoundFontManager
from server.streaming_server import StreamingServer

__version__ = "2.0.0"

__all__ = [
    # Core components
    "FluidSynthRenderer",
    "SoundFontManager",
    "RingBuffer",
    "BufferManager",
    "StreamingServer",
    # Data structures
    "AudioChunk",
    # Configuration
    "AuralisConfig",
    "get_config",
    # Metrics
    "PerformanceMetrics",
    "MemoryMonitor",
    # Presets
    "get_preset",
    "list_presets",
]
