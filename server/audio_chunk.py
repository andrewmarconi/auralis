"""Audio chunk data structure for streaming."""

import base64
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class AudioChunk:
    """Represents a chunk of PCM audio data with metadata.

    Attributes:
        data: Stereo PCM audio, shape (2, num_samples), dtype int16
        seq: Sequence number for ordering
        timestamp: Generation timestamp (seconds since epoch)
        sample_rate: Sample rate in Hz (default 44100)
        duration_ms: Chunk duration in milliseconds
    """

    data: np.ndarray  # Shape: (2, num_samples), dtype: int16
    seq: int
    timestamp: float
    sample_rate: int = 44100
    duration_ms: float = 100.0

    def to_base64(self) -> str:
        """Encode PCM data as base64 string.

        Returns:
            Base64-encoded string of stereo INTERLEAVED int16 PCM data (L,R,L,R,...)
        """
        # Ensure data is int16
        if self.data.dtype != np.int16:
            # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
            data_int16 = (self.data * 32767).astype(np.int16)
        else:
            data_int16 = self.data

        # Interleave stereo channels: (2, N) → (N*2,) as [L,R,L,R,L,R,...]
        # data_int16 is shape (2, num_samples) - need to interleave
        left_channel = data_int16[0]  # First row
        right_channel = data_int16[1]  # Second row

        # Create interleaved array
        interleaved = np.empty(left_channel.size + right_channel.size, dtype=np.int16)
        interleaved[0::2] = left_channel  # Every other element starting at 0
        interleaved[1::2] = right_channel  # Every other element starting at 1

        # Flatten to bytes and encode
        data_bytes = interleaved.tobytes()
        return base64.b64encode(data_bytes).decode("ascii")

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary with chunk metadata and base64 PCM data
        """
        return {
            "type": "audio_chunk",
            "seq": self.seq,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "sample_rate": self.sample_rate,
            "duration_ms": self.duration_ms,
            "channels": 2,
            "data": self.to_base64(),
            "num_samples": self.data.shape[1] if self.data.ndim == 2 else len(self.data),
        }

    @classmethod
    def from_float32(
        cls,
        float_data: np.ndarray,
        seq: int,
        timestamp: float,
        sample_rate: int = 44100,
        duration_ms: float = 100.0,
    ) -> "AudioChunk":
        """Create AudioChunk from float32 data.

        Args:
            float_data: Stereo audio, shape (2, num_samples), dtype float32, range [-1.0, 1.0]
            seq: Sequence number
            timestamp: Generation timestamp
            sample_rate: Sample rate in Hz
            duration_ms: Chunk duration in milliseconds

        Returns:
            AudioChunk with int16 PCM data
        """
        # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
        int16_data = (np.clip(float_data, -1.0, 1.0) * 32767).astype(np.int16)

        return cls(
            data=int16_data,
            seq=seq,
            timestamp=timestamp,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
        )
