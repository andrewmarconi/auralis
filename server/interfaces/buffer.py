"""Buffer interface definitions for audio chunk management."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from server.audio_chunk import AudioChunk


class IRingBuffer(ABC):
    """Thread-safe ring buffer for audio chunks."""

    @abstractmethod
    def write(self, chunk: "AudioChunk") -> bool:
        """Add chunk to buffer.

        Args:
            chunk: Audio chunk to write

        Returns:
            True if written, False if buffer full

        Thread-safe: Uses internal lock
        """
        pass

    @abstractmethod
    def read(self) -> Optional["AudioChunk"]:
        """Remove chunk from buffer.

        Returns:
            AudioChunk if available, None if empty

        Thread-safe: Uses internal lock
        """
        pass

    @abstractmethod
    def get_depth(self) -> int:
        """Return current buffer depth (number of buffered chunks).

        Returns:
            Integer in range [0, capacity]
        """
        pass

    @abstractmethod
    def is_full(self) -> bool:
        """Check if buffer is full (write would fail).

        Returns:
            True if full, False otherwise
        """
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if buffer is empty (read would return None).

        Returns:
            True if empty, False otherwise
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all chunks from buffer.

        Thread-safe: Uses internal lock
        """
        pass


class IBufferManager(ABC):
    """Manages back-pressure logic and buffer health monitoring."""

    @abstractmethod
    async def apply_back_pressure(self, current_depth: int, capacity: int) -> None:
        """Apply back-pressure when buffer depth is low.

        Args:
            current_depth: Current number of chunks in buffer
            capacity: Maximum buffer capacity

        Behavior:
            - If depth < 2: Sleep 10ms to allow consumer to catch up
            - Otherwise: No delay
        """
        pass

    @abstractmethod
    def should_generate(self, current_depth: int, capacity: int) -> bool:
        """Check if generation should proceed based on buffer depth.

        Args:
            current_depth: Current number of chunks in buffer
            capacity: Maximum buffer capacity

        Returns:
            True if generation should proceed, False to wait
        """
        pass

    @abstractmethod
    def get_buffer_health(self, current_depth: int, capacity: int) -> str:
        """Get buffer health status.

        Args:
            current_depth: Current number of chunks in buffer
            capacity: Maximum buffer capacity

        Returns:
            Health status: "emergency", "low", "healthy", or "full"
        """
        pass
