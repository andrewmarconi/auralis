"""Thread-safe ring buffer for audio chunk streaming.

Implements a circular buffer with atomic read/write operations using
threading.Lock for GIL-compatible synchronization.
"""

import logging
import threading
from typing import Optional

from server.audio_chunk import AudioChunk
from server.interfaces.buffer import IRingBuffer

logger = logging.getLogger(__name__)


class RingBuffer(IRingBuffer):
    """Thread-safe ring buffer for audio chunks.

    Uses pre-allocated NumPy-style storage with atomic cursors for
    concurrent producer/consumer access.
    """

    def __init__(self, capacity: int = 20):
        """Initialize ring buffer.

        Args:
            capacity: Maximum number of chunks to buffer (default 20 = 2 seconds)
        """
        if capacity < 1:
            raise ValueError(f"Capacity must be >= 1, got {capacity}")

        self.capacity = capacity
        self.buffer: list[Optional[AudioChunk]] = [None] * capacity
        self.read_cursor = 0
        self.write_cursor = 0
        self.count = 0  # Number of buffered chunks
        self.lock = threading.Lock()

        logger.info(f"RingBuffer initialized with capacity={capacity} chunks")

    def write(self, chunk: AudioChunk) -> bool:
        """Add chunk to buffer.

        Args:
            chunk: Audio chunk to write

        Returns:
            True if written, False if buffer full

        Thread-safe: Uses internal lock
        """
        with self.lock:
            if self.count >= self.capacity:
                logger.warning(
                    f"Buffer full (capacity={self.capacity}), dropping chunk {chunk.seq}"
                )
                return False

            self.buffer[self.write_cursor] = chunk
            self.write_cursor = (self.write_cursor + 1) % self.capacity
            self.count += 1

            logger.debug(
                f"Wrote chunk {chunk.seq} at position {self.write_cursor-1}, "
                f"depth={self.count}/{self.capacity}"
            )
            return True

    def read(self) -> Optional[AudioChunk]:
        """Remove chunk from buffer.

        Returns:
            AudioChunk if available, None if empty

        Thread-safe: Uses internal lock
        """
        with self.lock:
            if self.count == 0:
                logger.debug("Buffer empty, no chunk to read")
                return None

            chunk = self.buffer[self.read_cursor]
            self.buffer[self.read_cursor] = None  # Clear reference
            self.read_cursor = (self.read_cursor + 1) % self.capacity
            self.count -= 1

            if chunk:
                logger.debug(
                    f"Read chunk {chunk.seq}, remaining depth={self.count}/{self.capacity}"
                )

            return chunk

    def get_depth(self) -> int:
        """Return current buffer depth (number of buffered chunks).

        Returns:
            Integer in range [0, capacity]
        """
        with self.lock:
            return self.count

    def is_full(self) -> bool:
        """Check if buffer is full (write would fail).

        Returns:
            True if full, False otherwise
        """
        with self.lock:
            return self.count >= self.capacity

    def is_empty(self) -> bool:
        """Check if buffer is empty (read would return None).

        Returns:
            True if empty, False otherwise
        """
        with self.lock:
            return self.count == 0

    def clear(self) -> None:
        """Remove all chunks from buffer.

        Thread-safe: Uses internal lock
        """
        with self.lock:
            self.buffer = [None] * self.capacity
            self.read_cursor = 0
            self.write_cursor = 0
            old_count = self.count
            self.count = 0
            logger.info(f"Buffer cleared ({old_count} chunks removed)")

    def get_utilization(self) -> float:
        """Get buffer utilization percentage.

        Returns:
            Utilization as float 0.0-1.0
        """
        with self.lock:
            return self.count / self.capacity if self.capacity > 0 else 0.0

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"RingBuffer(capacity={self.capacity}, depth={self.count}, "
            f"utilization={self.get_utilization():.1%})"
        )
