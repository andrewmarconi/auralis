"""Abstract interface for audio synthesis engine."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import numpy as np


class ISynthesisEngine(ABC):
    """Abstract interface for audio synthesis engine."""

    @abstractmethod
    def render_phrase(
        self,
        chords: List[Tuple[int, int, str]],
        melody: List[Tuple[int, int, float, float]],
        duration_sec: float,
    ) -> np.ndarray:
        """
        Render musical phrase to audio.

        Args:
            chords: List of (onset_sample, root_midi, chord_type)
            melody: List of (onset_sample, pitch_midi, velocity, duration_sec)
            duration_sec: Total phrase duration

        Returns:
            Stereo audio array (2, num_samples), int16

        Raises:
            SynthesisError: If rendering fails
        """
        pass

    @abstractmethod
    def get_device_info(self) -> Dict:
        """
        Get GPU device information.

        Returns:
            Dictionary with device details (type, name, memory, etc.)
        """
        pass

    @abstractmethod
    def get_render_stats(self) -> Dict:
        """
        Get rendering performance statistics.

        Returns:
            Dictionary with latency, throughput, etc.
        """
        pass

    @abstractmethod
    def clear_gpu_cache(self) -> None:
        """Clear GPU memory cache (CUDA/MPS)."""
        pass
