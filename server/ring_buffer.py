"""
Ring Buffer for Real-Time Audio Streaming

Thread-safe circular buffer for continuous audio data flow between
synthesis and streaming components.
"""

import threading
from typing import Optional

import numpy as np
from numpy.typing import NDArray


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
        self.buffer_data: NDArray[np.float32] = np.zeros(
            (2, capacity_samples), dtype=np.float32
        )

        # Atomic cursor positions
        self.write_position = 0
        self.read_position = 0

        # Thread synchronization
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    def write(self, audio_data: NDArray[np.float32]) -> bool:
        """
        Write audio chunk to buffer.

        Args:
            audio_data: Stereo audio array, shape (2, num_samples), float32 [-1, 1]

        Returns:
            True if write successful, False if buffer overflow
        """
        if audio_data.shape[0] != 2:
            raise ValueError("Audio data must be stereo (2 channels)")

        num_samples = audio_data.shape[1]

        with self._lock:
            # Check for overflow
            available_space = self._available_write_space()
            if available_space < num_samples:
                return False  # Buffer overflow

            # Write to buffer (circular)
            write_idx = self.write_position
            end_idx = write_idx + num_samples

            if end_idx <= self.capacity_samples:
                # Contiguous write
                self.buffer_data[:, write_idx:end_idx] = audio_data
            else:
                # Wrap-around write
                first_part = self.capacity_samples - write_idx
                self.buffer_data[:, write_idx:] = audio_data[:, :first_part]
                self.buffer_data[:, :num_samples - first_part] = audio_data[:, first_part:]

            # Update write cursor
            self.write_position = (write_idx + num_samples) % self.capacity_samples

            # Signal readers
            self._not_empty.notify()

        return True

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
                audio_data[:, first_part:] = self.buffer_data[:, :num_samples - first_part]

            # Update read cursor
            self.read_position = (read_idx + num_samples) % self.capacity_samples

            # Signal writers
            self._not_full.notify()

        return audio_data

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
                audio_data[:, first_part:] = self.buffer_data[:, :num_samples - first_part]

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
