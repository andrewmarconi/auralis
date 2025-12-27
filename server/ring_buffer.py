"""
Ring Buffer for Real-Time Audio Streaming

Thread-safe circular buffer for continuous audio data flow between
synthesis and streaming components.
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from server.buffer_management import JitterTracker


class RingBuffer:
    """
    Thread-safe ring buffer for audio streaming.

    Manages a circular buffer for PCM audio data with atomic read/write operations.
    Designed for real-time audio with <100ms latency requirements.

    Attributes:
        capacity_samples: Total buffer size in samples (stereo)
        sample_rate: Audio sample rate (44100 Hz)
        chunk_size: Size of each audio chunk in samples
    """

    def __init__(
        self,
        capacity_samples: int = 88200,  # 2 seconds at 44.1kHz
        sample_rate: int = 44100,
        chunk_size: int = 4410,  # 100ms chunks
    ) -> None:
        """
        Initialize ring buffer with pre-allocated numpy array.

        Args:
            capacity_samples: Buffer capacity in samples (default: 2 seconds)
            sample_rate: Audio sampling rate in Hz
            chunk_size: Chunk size in samples (default: 100ms at 44.1kHz)
        """
        if capacity_samples % chunk_size != 0:
            raise ValueError("Capacity must be multiple of chunk_size")

        self.capacity_samples = capacity_samples
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Pre-allocate buffer (stereo float32) - prevents GC pauses
        self.buffer_data: NDArray[np.float32] = np.zeros((2, capacity_samples), dtype=np.float32)

        # Atomic cursor positions
        self.write_position = 0
        self.read_position = 0

        # Thread synchronization
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    def write_chunk(self, chunk: np.ndarray) -> int:
        """
        Write audio chunk to buffer.

        Args:
            chunk: Audio data (int16, stereo interleaved)

        Returns:
            Chunk ID (write position)

        Raises:
            BufferFullError: If buffer is full and cannot accept writes
        """
        # Convert float32 chunk to int16 stereo if needed
        if chunk.dtype == np.float32:
            audio_data = (chunk * 32767).astype(np.int16)
        elif chunk.ndim == 2 and chunk.shape[0] == 2:
            # Already in stereo float32 format, convert to interleaved int16
            audio_data = np.empty((chunk.shape[1] * 2,), dtype=np.int16)
            audio_data[0::2] = (chunk[0] * 32767).astype(np.int16)
            audio_data[1::2] = (chunk[1] * 32767).astype(np.int16)
        else:
            # Assume already in correct format (int16 interleaved)
            audio_data = chunk

        return self._write_internal(audio_data)

    def _write_internal(self, audio_data: np.ndarray) -> int:
        """Internal write method that handles actual buffer writing."""
        if audio_data.ndim != 1:
            raise ValueError("Audio data for _write_internal must be 1D interleaved")

        num_samples = audio_data.shape[0]
        num_channels = 2
        num_stereo_samples = num_samples // num_channels
        num_chunks = num_stereo_samples // self.chunk_size

        chunk_id = self.write_position

        for i in range(num_chunks):
            start_sample = i * self.chunk_size * 2
            end_sample = (i + 1) * self.chunk_size * 2
            chunk_1d = audio_data[start_sample:end_sample]
            # Convert 1D interleaved to 2D stereo
            chunk_data = chunk_1d.reshape(-1, 2).T.astype(np.float32) / 32767.0
            chunk_data = chunk_data.reshape(2, -1)

            with self._lock:
                # Write to buffer
                write_idx = self.write_position
                end_idx = write_idx + self.chunk_size

                if end_idx <= self.capacity_samples:
                    # Contiguous write
                    self.buffer_data[:, write_idx:end_idx] = chunk_data
                else:
                    # Wrap-around write
                    first_part = self.capacity_samples - write_idx
                    self.buffer_data[:, write_idx:] = chunk_data[:, :first_part]
                    self.buffer_data[:, :end_idx - self.capacity_samples] = chunk_data[:, first_part:]

                # Update write cursor
                self.write_position = end_idx % self.capacity_samples

                # Signal readers
                self._not_empty.notify()

        return chunk_id

    def read(self, num_samples: Optional[int] = None) -> Optional[NDArray[np.float32]]:
        """
        Read audio chunk from buffer.

        Args:
            num_samples: Number of samples to read (default: chunk_size)

        Returns:
            Stereo audio array, shape (2, num_samples), or None if insufficient data
        """
        if num_samples is None:
            num_samples = self.chunk_size

        with self._lock:
            # Check for underflow
            available_data = self._available_read_data()
            if available_data < num_samples:
                return None  # Buffer underflow

            # Read from buffer (circular)
            read_idx = self.read_position
            end_idx = read_idx + num_samples

            if end_idx <= self.capacity_samples:
                # Contiguous read
                audio_data = self.buffer_data[:, read_idx:end_idx].copy()
            else:
                # Wrap-around read
                first_part = self.capacity_samples - read_idx
                audio_data = np.zeros((2, num_samples), dtype=np.float32)
                audio_data[:, :first_part] = self.buffer_data[:, read_idx:]
                audio_data[:, first_part:] = self.buffer_data[:, : num_samples - first_part]

            # Update read cursor
            self.read_position = (read_idx + num_samples) % self.capacity_samples

            # Signal writers
            self._not_full.notify()

        return self.write_position

    def read_chunk(self) -> Optional[np.ndarray]:
        """
        Read next audio chunk from buffer.

        Returns:
            Audio chunk or None if no data available

        Raises:
            BufferUnderrunError: If buffer is empty when read expected
        """
        with self._lock:
            # Check for underflow
            available_data = self._available_read_data()
            if available_data < self.chunk_size:
                # Not enough data for a full chunk
                if available_data > 0:
                    # Return partial chunk as underrun mitigation
                    num_samples = available_data
                    read_idx = self.read_position
                    end_idx = read_idx + num_samples

                    if end_idx <= self.capacity_samples:
                        audio_data = self.buffer_data[:, read_idx:end_idx].copy()
                    else:
                        first_part = self.capacity_samples - read_idx
                        audio_data = np.zeros((2, num_samples), dtype=np.float32)
                        audio_data[:, :first_part] = self.buffer_data[:, read_idx:]
                        audio_data[:, first_part:] = self.buffer_data[:, : num_samples - first_part]

                    self.read_position = (read_idx + num_samples) % self.capacity_samples
                    self._not_full.notify()
                    return audio_data.astype(np.int16)
                return None

            # Read full chunk
            read_idx = self.read_position
            end_idx = read_idx + self.chunk_size

            if end_idx <= self.capacity_samples:
                audio_data = self.buffer_data[:, read_idx:end_idx].copy()
            else:
                first_part = self.capacity_samples - read_idx
                audio_data = np.zeros((2, self.chunk_size), dtype=np.float32)
                audio_data[:, :first_part] = self.buffer_data[:, read_idx:]
                audio_data[:, first_part:] = self.buffer_data[:, : self.chunk_size - first_part]

            # Convert to int16 interleaved format
            audio_int16 = np.empty((self.chunk_size * 2,), dtype=np.int16)
            audio_int16[0::2] = (audio_data[0] * 32767).astype(np.int16)
            audio_int16[1::2] = (audio_data[1] * 32767).astype(np.int16)

            # Update read cursor
            self.read_position = (read_idx + self.chunk_size) % self.capacity_samples

            # Signal writers
            self._not_full.notify()

            return audio_int16

    def get_capacity(self) -> int:
        """
        Get buffer capacity.

        Returns:
            Maximum number of chunks buffer can hold
        """
        with self._lock:
            return self.capacity_samples // self.chunk_size

    def read_blocking(
        self, num_samples: Optional[int] = None, timeout: Optional[float] = None
    ) -> Optional[NDArray[np.float32]]:
        """
        Read audio chunk with blocking wait if insufficient data.

        Args:
            num_samples: Number of samples to read (default: chunk_size)
            timeout: Maximum wait time in seconds (None = wait forever)

        Returns:
            Stereo audio array or None if timeout
        """
        if num_samples is None:
            num_samples = self.chunk_size

        with self._lock:
            # Wait for data availability
            while self._available_read_data() < num_samples:
                if not self._not_empty.wait(timeout):
                    return None  # Timeout

            # Perform read (same logic as read())
            read_idx = self.read_position
            end_idx = read_idx + num_samples

            if end_idx <= self.capacity_samples:
                audio_data = self.buffer_data[:, read_idx:end_idx].copy()
            else:
                first_part = self.capacity_samples - read_idx
                audio_data = np.zeros((2, num_samples), dtype=np.float32)
                audio_data[:, :first_part] = self.buffer_data[:, read_idx:]
                audio_data[:, first_part:] = self.buffer_data[:, : num_samples - first_part]

            self.read_position = (read_idx + num_samples) % self.capacity_samples
            self._not_full.notify()

        return audio_data

    def _available_read_data(self) -> int:
        """Calculate available samples for reading."""
        return (self.write_position - self.read_position) % self.capacity_samples

    def _available_write_space(self) -> int:
        """Calculate available space for writing."""
        return self.capacity_samples - self._available_read_data() - 1

    def get_buffer_depth_ms(self) -> float:
        """
        Get current buffer depth in milliseconds.

        Returns:
            Buffer fill level in milliseconds
        """
        with self._lock:
            samples = self._available_read_data()
            return (samples / self.sample_rate) * 1000

    def clear(self) -> None:
        """Clear buffer and reset cursors."""
        with self._lock:
            self.write_position = 0
            self.read_position = 0
            self.buffer_data.fill(0.0)
            self._not_full.notify_all()

    def get_depth(self) -> int:
        """
        Get current buffer depth in chunks.

        Returns:
            Number of chunks available for reading
        """
        with self._lock:
            samples = self._available_read_data()
            return samples // self.chunk_size


@dataclass
class BufferTier:
    """Configuration for a specific buffer tier."""

    target_ms: int
    description: str


class AdaptiveRingBuffer:
    """
    Ring buffer with adaptive tier-based buffering.

    Dynamically adjusts buffer size based on observed network jitter and
    underrun rate to maintain smooth audio playback while minimizing latency.

    Attributes:
        sample_rate: Audio sample rate (Hz)
        chunk_duration_ms: Duration of each chunk in milliseconds
        current_tier: Current buffer tier ("minimal", "normal", "stable", "defensive")
        jitter_tracker: JitterTracker instance for jitter measurement
    """

    # Class-level tier definitions
    TIERS: Dict[str, BufferTier] = {
        "minimal": BufferTier(target_ms=500, description="Stable network, low latency"),
        "normal": BufferTier(target_ms=1000, description="Default for new connections"),
        "stable": BufferTier(target_ms=2000, description="Occasional jitter"),
        "defensive": BufferTier(target_ms=3000, description="High jitter/unstable network"),
    }

    def __init__(
        self, sample_rate: int = 44100, chunk_duration_ms: int = 100, initial_tier: str = "normal"
    ):
        """
        Initialize adaptive ring buffer.

        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_duration_ms: Duration of each chunk in milliseconds
            initial_tier: Initial buffer tier (default: "normal")
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.samples_per_chunk = int((chunk_duration_ms / 1000) * sample_rate)

        # Current tier state
        self.current_tier = initial_tier
        self.target_buffer_ms = self.TIERS[initial_tier].target_ms

        # Buffer allocation (max size = defensive tier)
        max_buffer_ms = self.TIERS["defensive"].target_ms
        self.max_chunks = int(max_buffer_ms / chunk_duration_ms)
        capacity_samples = self.max_chunks * self.samples_per_chunk

        # Underlying ring buffer
        self._buffer = RingBuffer(
            capacity_samples=capacity_samples,
            sample_rate=sample_rate,
            chunk_size=self.samples_per_chunk,
        )

        # Jitter tracking
        self.jitter_tracker = JitterTracker(window_size=50, alpha=0.1, tier_adjustment_interval=50)

        # Timing tracking for jitter measurement
        self.expected_next_chunk_time: Optional[float] = None
        self.chunks_written = 0
        self.chunks_read = 0

        # Thread safety
        self._lock = threading.Lock()

    def write_chunk(self, chunk: np.ndarray) -> int:
        """
        Write audio chunk to buffer.

        Args:
            chunk: Audio data (int16, stereo interleaved or 2D stereo)

        Returns:
            Chunk ID (write position)
        """
        chunk_id = self._buffer.write_chunk(chunk)
        self.chunks_written += 1
        return chunk_id

    def read_chunk(self) -> Optional[np.ndarray]:
        """
        Read next audio chunk from buffer with jitter tracking.

        Returns:
            Audio chunk or None if no data available
        """
        actual_time = time.time()

        # Track jitter if we have an expected time
        if self.expected_next_chunk_time is not None:
            self.jitter_tracker.record_chunk(
                expected_time=self.expected_next_chunk_time, actual_time=actual_time
            )

        # Attempt read
        chunk = self._buffer.read_chunk()

        if chunk is None:
            # Buffer underrun
            self.jitter_tracker.record_underrun()
            return None

        # Update timing for next chunk
        self.expected_next_chunk_time = actual_time + (self.chunk_duration_ms / 1000.0)
        self.chunks_read += 1

        # Check if we should adjust tier
        if self.chunks_read % 50 == 0:
            self.adjust_tier()

        return chunk

    def adjust_tier(self) -> None:
        """
        Adjust buffer tier based on jitter and underrun rate.

        Tier Promotion Rules:
        - Promote if underrun_rate > 5% OR jitter > 30ms
        - Demote if underrun_rate < 1% AND jitter < 15ms

        State transitions:
        minimal ↔ normal ↔ stable ↔ defensive
        """
        with self._lock:
            underrun_rate = self.jitter_tracker.get_underrun_rate()
            mean_jitter_ms = self.jitter_tracker.get_current_jitter()

            old_tier = self.current_tier

            # Tier escalation logic
            if underrun_rate > 0.05 or mean_jitter_ms > 30:
                # High underrun rate or high jitter → escalate
                if self.current_tier == "minimal":
                    self.current_tier = "normal"
                elif self.current_tier == "normal":
                    self.current_tier = "stable"
                elif self.current_tier == "stable":
                    self.current_tier = "defensive"
            elif underrun_rate < 0.01 and mean_jitter_ms < 15:
                # Low underrun rate and low jitter → de-escalate
                if self.current_tier == "defensive":
                    self.current_tier = "stable"
                elif self.current_tier == "stable":
                    self.current_tier = "normal"
                elif self.current_tier == "normal":
                    self.current_tier = "minimal"

            # Update target buffer size
            if self.current_tier != old_tier:
                self.target_buffer_ms = self.TIERS[self.current_tier].target_ms

    def get_buffer_health(self) -> dict:
        """
        Get comprehensive buffer health metrics.

        Returns:
            Dictionary containing:
            - tier: Current buffer tier
            - tier_description: Description of current tier
            - target_buffer_ms: Target buffer size in milliseconds
            - current_depth_ms: Current buffer depth in milliseconds
            - current_depth_chunks: Current buffer depth in chunks
            - jitter_mean_ms: Mean jitter in milliseconds
            - jitter_std_ms: Jitter standard deviation in milliseconds
            - underrun_rate: Fraction of chunks that underran (0.0-1.0)
            - underrun_count: Total number of underruns
            - chunks_read: Total chunks read
            - chunks_written: Total chunks written
            - recommended_buffer_ms: Statistically recommended buffer size
        """
        with self._lock:
            depth_ms = self._buffer.get_buffer_depth_ms()
            depth_chunks = self._buffer.get_depth()

            return {
                "tier": self.current_tier,
                "tier_description": self.TIERS[self.current_tier].description,
                "target_buffer_ms": self.target_buffer_ms,
                "current_depth_ms": depth_ms,
                "current_depth_chunks": depth_chunks,
                "jitter_mean_ms": self.jitter_tracker.get_current_jitter(),
                "jitter_std_ms": self.jitter_tracker.get_jitter_std(),
                "underrun_rate": self.jitter_tracker.get_underrun_rate(),
                "underrun_count": self.jitter_tracker.underrun_count,
                "chunks_read": self.chunks_read,
                "chunks_written": self.chunks_written,
                "recommended_buffer_ms": self.jitter_tracker.get_recommended_buffer_ms(
                    confidence=0.95
                ),
            }

    def get_depth(self) -> int:
        """
        Get current buffer depth in chunks.

        Returns:
            Number of chunks available for reading
        """
        return self._buffer.get_depth()

    def get_capacity(self) -> int:
        """
        Get buffer capacity in chunks.

        Returns:
            Maximum number of chunks buffer can hold
        """
        return self._buffer.get_capacity()

    def clear(self) -> None:
        """Clear all buffered data and reset state."""
        with self._lock:
            self._buffer.clear()
            self.chunks_read = 0
            self.chunks_written = 0
            self.expected_next_chunk_time = None
            self.jitter_tracker = JitterTracker(
                window_size=50, alpha=0.1, tier_adjustment_interval=50
            )
