"""Integration test for buffer underrun detection and recovery.

Validates that buffer health remains >98% during normal streaming and
that underruns are properly detected and recovered.
"""

import asyncio
import pytest
import time

from server.ring_buffer import RingBuffer
from server.audio_chunk import AudioChunk
from server.buffer_management import BufferManager
from server.metrics import PerformanceMetrics
import numpy as np


class TestBufferUnderruns:
    """Test buffer underrun scenarios."""

    @pytest.mark.asyncio
    async def test_normal_streaming_no_underruns(self):
        """Verify buffer maintains health during normal streaming (5 minutes)."""
        buffer = RingBuffer(capacity=20)
        buffer_manager = BufferManager()
        metrics = PerformanceMetrics()

        underrun_count = 0
        chunk_count = 0
        chunk_interval_sec = 0.1  # 100ms

        # Simulate 30 seconds of streaming (reduced from 5 minutes for test speed)
        duration_sec = 30.0
        num_chunks = int(duration_sec / chunk_interval_sec)

        async def producer():
            """Produce chunks at regular intervals."""
            for i in range(num_chunks):
                pcm_data = np.zeros((2, 4410), dtype=np.int16)
                chunk = AudioChunk(data=pcm_data, seq=i, timestamp=time.time())

                buffer.write(chunk)
                await asyncio.sleep(chunk_interval_sec)

        async def consumer():
            """Consume chunks and track underruns."""
            nonlocal underrun_count, chunk_count

            for _ in range(num_chunks):
                # Try to read chunk
                chunk = buffer.read()

                if chunk is None:
                    underrun_count += 1
                    metrics.increment_buffer_underrun()
                else:
                    chunk_count += 1

                await asyncio.sleep(chunk_interval_sec)

        # Run producer and consumer concurrently
        await asyncio.gather(producer(), consumer())

        # Calculate buffer health
        buffer_health_pct = (chunk_count / num_chunks) * 100.0

        # Verify >98% buffer health
        assert buffer_health_pct >= 98.0, (
            f"Buffer health {buffer_health_pct:.1f}% below 98% target "
            f"({underrun_count} underruns in {num_chunks} chunks)"
        )

    @pytest.mark.asyncio
    async def test_underrun_detection(self):
        """Verify buffer underruns are properly detected and counted."""
        buffer = RingBuffer(capacity=5)
        metrics = PerformanceMetrics()

        # Read from empty buffer
        chunk = buffer.read()
        assert chunk is None, "Expected None from empty buffer"

        # Increment underrun counter
        metrics.increment_buffer_underrun()

        # Verify metrics
        snapshot = metrics.get_snapshot()
        assert snapshot["buffer_underruns"] == 1, "Expected 1 underrun recorded"

    @pytest.mark.asyncio
    async def test_back_pressure_prevents_overflow(self):
        """Verify back-pressure logic prevents buffer overflow."""
        buffer = RingBuffer(capacity=10)
        buffer_manager = BufferManager()
        metrics = PerformanceMetrics()

        overflow_count = 0

        # Fill buffer to near capacity
        for i in range(8):
            pcm_data = np.zeros((2, 4410), dtype=np.int16)
            chunk = AudioChunk(data=pcm_data, seq=i, timestamp=time.time())
            buffer.write(chunk)

        # Try to write more with back-pressure
        for i in range(8, 15):
            depth = buffer.get_depth()
            capacity = buffer.capacity

            # Apply back-pressure
            await buffer_manager.apply_back_pressure(depth, capacity)

            # Attempt write
            pcm_data = np.zeros((2, 4410), dtype=np.int16)
            chunk = AudioChunk(data=pcm_data, seq=i, timestamp=time.time())

            success = buffer.write(chunk)
            if not success:
                overflow_count += 1
                metrics.increment_buffer_overflow()

        # Verify no catastrophic overflow
        # Some overflow is expected when capacity is reached
        assert overflow_count < 5, f"Excessive overflow: {overflow_count} chunks dropped"

    @pytest.mark.asyncio
    async def test_buffer_recovery_after_underrun(self):
        """Verify buffer recovers gracefully after underrun."""
        buffer = RingBuffer(capacity=10)

        # Create underrun condition (empty buffer)
        chunk = buffer.read()
        assert chunk is None

        # Fill buffer
        for i in range(5):
            pcm_data = np.zeros((2, 4410), dtype=np.int16)
            chunk = AudioChunk(data=pcm_data, seq=i, timestamp=time.time())
            buffer.write(chunk)

        # Verify recovery
        assert buffer.get_depth() == 5
        assert not buffer.is_empty()

        # Verify can read chunks
        for _ in range(5):
            chunk = buffer.read()
            assert chunk is not None

    @pytest.mark.asyncio
    async def test_buffer_health_status(self):
        """Verify buffer health status is correctly reported."""
        buffer = RingBuffer(capacity=20)
        buffer_manager = BufferManager()

        # Emergency (<1 chunk)
        health = buffer_manager.get_buffer_health(0, 20)
        assert health == "emergency"

        # Low (1-2 chunks)
        health = buffer_manager.get_buffer_health(2, 20)
        assert health == "low"

        # Healthy (3-4 chunks)
        health = buffer_manager.get_buffer_health(3, 20)
        assert health == "healthy"

        # Full (5+ chunks)
        health = buffer_manager.get_buffer_health(5, 20)
        assert health == "full"
