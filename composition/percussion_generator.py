"""Percussion generator (placeholder for post-MVP).

Ambient percussion is out of scope for MVP. This module raises
NotImplementedError until percussion is added in a future phase.
"""

import logging

from composition.musical_context import MusicalContext

logger = logging.getLogger(__name__)


class PercussionGenerator:
    """Generates sparse ambient percussion patterns (post-MVP)."""

    def __init__(self, sample_rate: int = 44100):
        """Initialize percussion generator.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        logger.info("Percussion generator initialized (placeholder)")

    def generate(self, context: MusicalContext, duration_bars: int = 8) -> None:
        """Generate percussion pattern (not implemented).

        Args:
            context: Musical parameters
            duration_bars: Number of bars

        Raises:
            NotImplementedError: Percussion is post-MVP
        """
        raise NotImplementedError(
            "Percussion generation is out of scope for MVP. "
            "This feature will be added in a future release."
        )
