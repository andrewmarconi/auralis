"""Dependency injection container for Auralis server components.

Provides centralized management of service instances with proper lifecycle
and dependency resolution.
"""

import logging
from typing import Optional

from composition.chord_generator import ChordGenerator
from composition.melody_generator import MelodyGenerator
from server.buffer_management import BufferManager
from server.config import get_config
from server.fluidsynth_renderer import FluidSynthRenderer
from server.memory_monitor import MemoryMonitor
from server.metrics import PerformanceMetrics
from server.ring_buffer import RingBuffer
from server.soundfont_manager import SoundFontManager
from server.streaming_server import StreamingServer
from server.synthesis_engine import SynthesisEngine

logger = logging.getLogger(__name__)


class DIContainer:
    """Dependency injection container for server components."""

    def __init__(self) -> None:
        """Initialize DI container."""
        self._config = get_config()
        self._instances: dict[str, any] = {}

        logger.info("DI container initialized")

    def get_config(self):
        """Get configuration instance."""
        return self._config

    def get_fluidsynth_renderer(self) -> FluidSynthRenderer:
        """Get or create FluidSynth renderer instance."""
        if "fluidsynth_renderer" not in self._instances:
            self._instances["fluidsynth_renderer"] = FluidSynthRenderer(
                sample_rate=44100
            )
        return self._instances["fluidsynth_renderer"]

    def get_soundfont_manager(self) -> SoundFontManager:
        """Get or create SoundFont manager instance."""
        if "soundfont_manager" not in self._instances:
            renderer = self.get_fluidsynth_renderer()
            self._instances["soundfont_manager"] = SoundFontManager(renderer)
        return self._instances["soundfont_manager"]

    def get_ring_buffer(self) -> RingBuffer:
        """Get or create ring buffer instance."""
        if "ring_buffer" not in self._instances:
            self._instances["ring_buffer"] = RingBuffer(
                capacity=self._config.ring_buffer_capacity
            )
        return self._instances["ring_buffer"]

    def get_buffer_manager(self) -> BufferManager:
        """Get or create buffer manager instance."""
        if "buffer_manager" not in self._instances:
            self._instances["buffer_manager"] = BufferManager()
        return self._instances["buffer_manager"]

    def get_chord_generator(self) -> ChordGenerator:
        """Get or create chord generator instance."""
        if "chord_generator" not in self._instances:
            self._instances["chord_generator"] = ChordGenerator(sample_rate=44100)
        return self._instances["chord_generator"]

    def get_melody_generator(self) -> MelodyGenerator:
        """Get or create melody generator instance."""
        if "melody_generator" not in self._instances:
            self._instances["melody_generator"] = MelodyGenerator(sample_rate=44100)
        return self._instances["melody_generator"]

    def get_streaming_server(self) -> StreamingServer:
        """Get or create streaming server instance."""
        if "streaming_server" not in self._instances:
            ring_buffer = self.get_ring_buffer()
            synthesis_engine = self.get_synthesis_engine()
            self._instances["streaming_server"] = StreamingServer(ring_buffer, synthesis_engine)
        return self._instances["streaming_server"]

    def get_metrics(self) -> PerformanceMetrics:
        """Get or create metrics collector instance."""
        if "metrics" not in self._instances:
            self._instances["metrics"] = PerformanceMetrics()
        return self._instances["metrics"]

    def get_memory_monitor(self) -> MemoryMonitor:
        """Get or create memory monitor instance."""
        if "memory_monitor" not in self._instances:
            self._instances["memory_monitor"] = MemoryMonitor()
        return self._instances["memory_monitor"]

    def get_synthesis_engine(self) -> SynthesisEngine:
        """Get or create synthesis engine instance."""
        if "synthesis_engine" not in self._instances:
            chord_gen = self.get_chord_generator()
            melody_gen = self.get_melody_generator()
            renderer = self.get_fluidsynth_renderer()
            ring_buffer = self.get_ring_buffer()
            metrics = self.get_metrics()

            self._instances["synthesis_engine"] = SynthesisEngine(
                chord_generator=chord_gen,
                melody_generator=melody_gen,
                fluidsynth_renderer=renderer,
                ring_buffer=ring_buffer,
                metrics=metrics,
                sample_rate=44100,
            )
        return self._instances["synthesis_engine"]

    def cleanup(self) -> None:
        """Clean up all managed instances."""
        logger.info("Cleaning up DI container")

        # Clean up FluidSynth renderer
        if "fluidsynth_renderer" in self._instances:
            try:
                self._instances["fluidsynth_renderer"].cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up FluidSynth renderer: {e}")

        # Stop memory monitoring
        if "memory_monitor" in self._instances:
            try:
                self._instances["memory_monitor"].stop_tracking()
            except Exception as e:
                logger.error(f"Error stopping memory monitor: {e}")

        self._instances.clear()
        logger.info("DI container cleaned up")


# Global container instance
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Get the global DI container instance.

    Returns:
        DIContainer singleton
    """
    global _container
    if _container is None:
        _container = DIContainer()
    return _container


def cleanup_container() -> None:
    """Clean up the global DI container."""
    global _container
    if _container is not None:
        _container.cleanup()
        _container = None
