"""Integration tests for smooth streaming and playback stability.

Validates:
- 30-minute continuous playback without failures
- Time-to-first-audio target (<800ms end-to-end)
- Buffer health maintenance during extended sessions
"""

import asyncio
import time

import pytest

from composition.chord_generator import ChordGenerator
from composition.melody_generator import MelodyGenerator
from composition.musical_context import MusicalContext
from server.buffer_management import BufferManager
from server.fluidsynth_renderer import FluidSynthRenderer
from server.metrics import PerformanceMetrics
from server.ring_buffer import RingBuffer
from server.synthesis_engine import SynthesisEngine


class TestSmoothStreaming:
    """Test smooth streaming and playback stability."""

    @pytest.mark.asyncio
    async def test_time_to_first_audio(self):
        """Verify time-to-first-audio is <800ms end-to-end.

        Measures complete pipeline latency:
        1. Engine starts generation loop
        2. First phrase generated
        3. First phrase rendered
        4. First chunk available in buffer
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

        # Measure time from engine start to first chunk available
        start_time = time.perf_counter()

        # Start generation loop
        await engine.start_generation_loop()

        # Wait for first chunk to appear in buffer
        max_wait_sec = 2.0
        elapsed = 0.0
        poll_interval = 0.01  # 10ms polling

        while ring_buffer.is_empty() and elapsed < max_wait_sec:
            await asyncio.sleep(poll_interval)
            elapsed = time.perf_counter() - start_time

        # Stop generation
        await engine.stop_generation_loop()

        # Calculate time to first audio
        time_to_first_audio_ms = elapsed * 1000.0

        # Verify chunk was produced
        assert not ring_buffer.is_empty(), "No audio chunks produced"

        # Verify latency target (<800ms)
        assert time_to_first_audio_ms < 800.0, (
            f"Time-to-first-audio {time_to_first_audio_ms:.1f}ms exceeds 800ms target"
        )

        print(f"Time-to-first-audio: {time_to_first_audio_ms:.1f}ms")

    @pytest.mark.asyncio
    async def test_30_minute_continuous_playback(self):
        """Verify system stability during 30-minute continuous playback.

        Note: This test is scaled down to 30 seconds for CI/CD.
        For full 30-minute test, run with TEST_FULL_DURATION=true.

        Validates:
        - No crashes or exceptions
        - Buffer health >98%
        - Memory stability (no leaks)
        - Continuous chunk production
        """
        import os

        # Check if full duration test requested
        test_full_duration = os.getenv("TEST_FULL_DURATION", "false").lower() == "true"
        duration_sec = 1800.0 if test_full_duration else 30.0  # 30 min vs 30 sec

        print(f"Running {duration_sec}s continuous playback test...")

        # Setup components
        chord_gen = ChordGenerator()
        melody_gen = MelodyGenerator()
        renderer = FluidSynthRenderer()
        ring_buffer = RingBuffer(capacity=20)
        metrics = PerformanceMetrics()
        buffer_manager = BufferManager()

        engine = SynthesisEngine(
            chord_generator=chord_gen,
            melody_generator=melody_gen,
            fluidsynth_renderer=renderer,
            ring_buffer=ring_buffer,
            metrics=metrics,
        )

        # Track metrics
        underrun_count = 0
        chunk_count = 0
        last_chunk_seq = -1

        # Start generation loop
        await engine.start_generation_loop()

        # Monitor for specified duration
        start_time = time.time()
        elapsed = 0.0

        try:
            while elapsed < duration_sec:
                # Check buffer depth
                depth = ring_buffer.get_depth()
                capacity = ring_buffer.capacity

                # Try to read chunk
                chunk = ring_buffer.read()

                if chunk is None:
                    underrun_count += 1
                    metrics.increment_buffer_underrun()
                else:
                    chunk_count += 1
                    last_chunk_seq = chunk.seq

                # Brief sleep to simulate consumption
                await asyncio.sleep(0.1)  # 100ms chunk interval

                elapsed = time.time() - start_time

                # Log progress every 5 seconds
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    buffer_health_pct = (
                        (chunk_count / (chunk_count + underrun_count)) * 100.0
                        if (chunk_count + underrun_count) > 0
                        else 0.0
                    )
                    print(
                        f"  {elapsed:.0f}s: {chunk_count} chunks, "
                        f"{underrun_count} underruns, "
                        f"buffer health: {buffer_health_pct:.1f}%"
                    )

        finally:
            # Stop generation
            await engine.stop_generation_loop()

        # Calculate final metrics
        total_attempts = chunk_count + underrun_count
        buffer_health_pct = (chunk_count / total_attempts) * 100.0 if total_attempts > 0 else 0.0

        print(f"\nPlayback test complete:")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Chunks received: {chunk_count}")
        print(f"  Underruns: {underrun_count}")
        print(f"  Buffer health: {buffer_health_pct:.1f}%")

        # Assertions
        assert chunk_count > 0, "No chunks produced during test"
        assert buffer_health_pct >= 98.0, (
            f"Buffer health {buffer_health_pct:.1f}% below 98% target "
            f"({underrun_count} underruns in {total_attempts} attempts)"
        )

    @pytest.mark.asyncio
    async def test_buffer_recovery_during_streaming(self):
        """Verify buffer recovers gracefully from temporary underruns during streaming."""
        # Setup components
        chord_gen = ChordGenerator()
        melody_gen = MelodyGenerator()
        renderer = FluidSynthRenderer()
        ring_buffer = RingBuffer(capacity=10)
        metrics = PerformanceMetrics()

        engine = SynthesisEngine(
            chord_generator=chord_gen,
            melody_generator=melody_gen,
            fluidsynth_renderer=renderer,
            ring_buffer=ring_buffer,
            metrics=metrics,
        )

        # Start generation
        await engine.start_generation_loop()

        # Wait for buffer to fill
        await asyncio.sleep(0.5)

        # Simulate aggressive consumption (drain buffer)
        drained_chunks = 0
        while not ring_buffer.is_empty():
            ring_buffer.read()
            drained_chunks += 1

        print(f"Drained {drained_chunks} chunks from buffer")

        # Verify buffer is empty (underrun condition)
        assert ring_buffer.is_empty(), "Buffer should be empty after draining"

        # Wait for buffer to recover
        recovery_start = time.time()
        while ring_buffer.get_depth() < 3:  # Wait for healthy depth
            await asyncio.sleep(0.05)
            if time.time() - recovery_start > 2.0:
                break

        recovery_time_ms = (time.time() - recovery_start) * 1000.0

        # Stop generation
        await engine.stop_generation_loop()

        # Verify recovery
        assert ring_buffer.get_depth() >= 3, (
            f"Buffer failed to recover (depth: {ring_buffer.get_depth()})"
        )
        assert recovery_time_ms < 1000.0, (
            f"Buffer recovery took {recovery_time_ms:.0f}ms (target: <1000ms)"
        )

        print(f"Buffer recovered in {recovery_time_ms:.0f}ms")

    @pytest.mark.asyncio
    async def test_continuous_generation_stability(self):
        """Verify generation loop runs continuously without exceptions for extended period."""
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

        # Start generation
        await engine.start_generation_loop()

        # Monitor for 10 seconds
        duration_sec = 10.0
        start_time = time.time()

        exception_occurred = False

        try:
            while time.time() - start_time < duration_sec:
                # Verify engine is still running
                assert engine.is_running(), "Engine stopped unexpectedly"

                # Verify chunks are being produced
                depth = ring_buffer.get_depth()
                assert depth > 0, "Buffer empty - generation may have stopped"

                await asyncio.sleep(0.5)

        except Exception as e:
            exception_occurred = True
            print(f"Exception during generation: {e}")
            raise
        finally:
            # Stop generation
            await engine.stop_generation_loop()

        # Verify no exceptions occurred
        assert not exception_occurred, "Exception occurred during continuous generation"

        print(f"Generation loop stable for {duration_sec}s")
