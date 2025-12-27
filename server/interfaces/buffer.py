"""Abstract interface for ring buffer implementations."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class IRingBuffer(ABC):
    """Abstract interface for ring buffer implementations."""

    @abstractmethod
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
        pass

    @abstractmethod
    def read_chunk(self) -> Optional[np.ndarray]:
        """
        Read next audio chunk from buffer.

        Returns:
            Audio chunk or None if no data available

        Raises:
            BufferUnderrunError: If buffer is empty when read expected
        """
        pass

    @abstractmethod
    def get_depth(self) -> int:
        """
        Get current buffer depth.

        Returns:
            Number of chunks available for reading
        """
        pass

    @abstractmethod
    def get_capacity(self) -> int:
        """
        Get buffer capacity.

        Returns:
            Maximum number of chunks buffer can hold
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all buffered data."""
        pass
