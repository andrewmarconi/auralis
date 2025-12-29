"""Integration test for FluidSynth rendering performance.

Validates that FluidSynth synthesis completes in <100ms for an 8-bar phrase.
"""

import pytest
import time

from composition.chord_generator import ChordGenerator
from composition.melody_generator import MelodyGenerator
from composition.musical_context import MusicalContext
from server.synthesis_engine import SynthesisEngine
from server.fluidsynth_renderer import FluidSynthRenderer
from server.ring_buffer import RingBuffer
from server.metrics import PerformanceMetrics


class TestFluidSynthRendering:
    """Test FluidSynth rendering integration."""

    @pytest.mark.asyncio
    async def test_phrase_rendering_latency(self):
        """Verify 8-bar phrase renders in <100ms.

        This is a critical performance target for real-time streaming.
        """
        # Setup components
        chord_gen = ChordGenerator()
        melody_gen = MelodyGenerator()
        renderer = FluidSynthRenderer()
        ring_buffer = RingBuffer(capacity=20)
        metrics = PerformanceMetrics()

        engine = SynthesisEngine(
            chord_generator=chord_gen,
            melody_generator=melody_gen,
            fluidsynth_renderer=renderer,
            ring_buffer=ring_buffer,
            metrics=metrics,
        )

        # Generate and render phrase
        context = MusicalContext.default()

        start_time = time.perf_counter()
        chords, melody = await engine.generate_phrase(context, duration_bars=8)
        audio = await engine.render_phrase(chords, melody)
        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Verify latency
        assert total_time_ms < 100.0, (
            f"Synthesis latency {total_time_ms:.1f}ms exceeds 100ms target"
        )

        # Verify audio output
        assert audio is not None
        assert audio.shape[0] == 2, "Expected stereo audio (2 channels)"
        assert audio.shape[1] > 0, "Expected non-zero audio samples"

    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Verify synthesis latency is recorded in metrics."""
        chord_gen = ChordGenerator()
        melody_gen = MelodyGenerator()
        renderer = FluidSynthRenderer()
        ring_buffer = RingBuffer(capacity=20)
        metrics = PerformanceMetrics()

        engine = SynthesisEngine(
            chord_generator=chord_gen,
            melody_generator=melody_gen,
            fluidsynth_renderer=renderer,
            ring_buffer=ring_buffer,
            metrics=metrics,
        )

        context = MusicalContext.default()

        # Generate multiple phrases
        for _ in range(5):
            chords, melody = await engine.generate_phrase(context, duration_bars=8)
            await engine.render_phrase(chords, melody)

        # Check metrics
        snapshot = metrics.get_snapshot()
        synthesis_stats = snapshot["synthesis_latency_ms"]

        assert synthesis_stats["samples"] == 5, "Expected 5 samples recorded"
        assert synthesis_stats["p95"] < 100.0, (
            f"P95 synthesis latency {synthesis_stats['p95']:.1f}ms exceeds 100ms"
        )

    @pytest.mark.asyncio
    async def test_phrase_chunking(self):
        """Verify audio is correctly chunked for streaming."""
        chord_gen = ChordGenerator()
        melody_gen = MelodyGenerator()
        renderer = FluidSynthRenderer()
        ring_buffer = RingBuffer(capacity=20)
        metrics = PerformanceMetrics()

        engine = SynthesisEngine(
            chord_generator=chord_gen,
            melody_generator=melody_gen,
            fluidsynth_renderer=renderer,
            ring_buffer=ring_buffer,
            metrics=metrics,
        )

        context = MusicalContext.default()

        # Generate and render
        chords, melody = await engine.generate_phrase(context, duration_bars=8)
        audio = await engine.render_phrase(chords, melody)

        # Chunk audio
        chunks = engine._chunk_audio(audio, chunk_size_samples=4410)

        # Verify chunks
        assert len(chunks) > 0, "Expected at least one chunk"

        for chunk in chunks:
            assert chunk.data.shape == (2, 4410) or chunk.data.shape[1] <= 4410
            assert chunk.sample_rate == 44100
            assert chunk.duration_ms == 100.0

    @pytest.mark.asyncio
    async def test_continuous_generation(self):
        """Verify continuous generation loop produces chunks."""
        chord_gen = ChordGenerator()
        melody_gen = MelodyGenerator()
        renderer = FluidSynthRenderer()
        ring_buffer = RingBuffer(capacity=20)
        metrics = PerformanceMetrics()

        engine = SynthesisEngine(
            chord_generator=chord_gen,
            melody_generator=melody_gen,
            fluidsynth_renderer=renderer,
            ring_buffer=ring_buffer,
            metrics=metrics,
        )

        # Start generation loop
        await engine.start_generation_loop()

        # Wait briefly for generation
        import asyncio
        await asyncio.sleep(0.5)

        # Stop generation
        await engine.stop_generation_loop()

        # Verify buffer has chunks
        depth = ring_buffer.get_depth()
        assert depth > 0, f"Expected chunks in buffer, got depth={depth}"
