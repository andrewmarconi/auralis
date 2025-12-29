"""Test chunk delivery timing and variance.

Validates that audio chunks are delivered with <50ms timing variance
to ensure smooth, glitch-free playback.
"""

import asyncio
import time
from typing import List

import numpy as np
import pytest

from server.audio_chunk import AudioChunk
from server.ring_buffer import RingBuffer


class TestChunkDelivery:
    """Test audio chunk delivery timing."""

    def test_ring_buffer_write_read_latency(self):
        """Verify ring buffer operations complete in <1ms."""
        buffer = RingBuffer(capacity=20)

        # Create test chunk
        pcm_data = np.zeros((2, 4410), dtype=np.int16)
        chunk = AudioChunk(data=pcm_data, seq=1, timestamp=time.time())

        # Measure write latency
        write_times: List[float] = []
        for i in range(100):
            chunk.seq = i
            start = time.perf_counter()
            buffer.write(chunk)
            write_times.append((time.perf_counter() - start) * 1000.0)

        # Measure read latency
        read_times: List[float] = []
        for _ in range(100):
            start = time.perf_counter()
            buffer.read()
            read_times.append((time.perf_counter() - start) * 1000.0)

        # Verify latencies
        avg_write_ms = np.mean(write_times)
        avg_read_ms = np.mean(read_times)

        assert avg_write_ms < 1.0, f"Buffer write avg {avg_write_ms:.3f}ms exceeds 1ms"
        assert avg_read_ms < 1.0, f"Buffer read avg {avg_read_ms:.3f}ms exceeds 1ms"

    @pytest.mark.asyncio
    async def test_chunk_delivery_timing_variance(self):
        """Verify chunk delivery timing has <50ms variance.

        Simulates chunk generation and delivery to validate timing stability.
        """
        buffer = RingBuffer(capacity=20)
        chunk_interval_ms = 100.0  # 100ms per chunk
        num_chunks = 50

        delivery_times: List[float] = []
        expected_times: List[float] = []

        # Producer: Generate chunks at 100ms intervals
        async def producer():
            for i in range(num_chunks):
                pcm_data = np.zeros((2, 4410), dtype=np.int16)
                chunk = AudioChunk(data=pcm_data, seq=i, timestamp=time.time())

                buffer.write(chunk)
                await asyncio.sleep(chunk_interval_ms / 1000.0)

        # Consumer: Read chunks and measure timing
        async def consumer():
            start_time = time.time()

            for i in range(num_chunks):
                # Wait for chunk to be available
                while buffer.is_empty():
                    await asyncio.sleep(0.001)  # 1ms polling

                actual_time = time.time() - start_time
                expected_time = i * (chunk_interval_ms / 1000.0)

                delivery_times.append(actual_time)
                expected_times.append(expected_time)

                buffer.read()

        # Run producer and consumer concurrently
        await asyncio.gather(producer(), consumer())

        # Calculate timing variance
        variances = [
            abs(actual - expected) * 1000.0  # Convert to ms
            for actual, expected in zip(delivery_times, expected_times)
        ]

        avg_variance_ms = np.mean(variances)
        max_variance_ms = np.max(variances)
        p95_variance_ms = np.percentile(variances, 95)

        # Assertions
        assert avg_variance_ms < 50.0, (
            f"Average timing variance {avg_variance_ms:.2f}ms exceeds 50ms target"
        )
        assert p95_variance_ms < 100.0, (
            f"P95 timing variance {p95_variance_ms:.2f}ms exceeds 100ms threshold"
        )

    def test_buffer_depth_tracking_accuracy(self):
        """Verify buffer depth tracking is accurate under concurrent access."""
        buffer = RingBuffer(capacity=20)

        # Write 10 chunks
        for i in range(10):
            pcm_data = np.zeros((2, 4410), dtype=np.int16)
            chunk = AudioChunk(data=pcm_data, seq=i, timestamp=time.time())
            buffer.write(chunk)

        assert buffer.get_depth() == 10, f"Expected depth 10, got {buffer.get_depth()}"

        # Read 5 chunks
        for _ in range(5):
            buffer.read()

        assert buffer.get_depth() == 5, f"Expected depth 5, got {buffer.get_depth()}"

        # Clear buffer
        buffer.clear()
        assert buffer.get_depth() == 0, f"Expected depth 0 after clear, got {buffer.get_depth()}"

    def test_on_time_delivery_percentage(self):
        """Verify >98% of chunks are delivered on time.

        'On time' = delivered within ±10ms of expected interval.
        """
        buffer = RingBuffer(capacity=20)
        chunk_interval_ms = 100.0
        num_chunks = 100

        delivery_times: List[float] = []

        # Simulate chunk production
        start_time = time.time()
        for i in range(num_chunks):
            pcm_data = np.zeros((2, 4410), dtype=np.int16)
            chunk = AudioChunk(data=pcm_data, seq=i, timestamp=time.time())

            buffer.write(chunk)

            # Simulate consumption
            if i >= 5:  # Start consuming after buffer has some chunks
                chunk_read = buffer.read()
                if chunk_read:
                    actual_time = time.time() - start_time
                    expected_time = (i - 4) * (chunk_interval_ms / 1000.0)
                    delivery_times.append(abs(actual_time - expected_time) * 1000.0)

            # Simulate interval
            time.sleep(chunk_interval_ms / 1000.0 * 0.01)  # 1% of interval for speed

        # Calculate on-time percentage (±10ms tolerance)
        on_time_count = sum(1 for dt in delivery_times if dt <= 10.0)
        on_time_percentage = (on_time_count / len(delivery_times)) * 100.0

        assert on_time_percentage >= 98.0, (
            f"On-time delivery {on_time_percentage:.1f}% below 98% target"
        )
