"""
Sparse Ambient Percussion Generator

Generates sparse percussion events for ambient music:
- Kick drums: Deep, subtle low-frequency pulses
- Swells: Granular textured events using filtered noise
"""

from typing import List, Tuple
import random

from loguru import logger


class PercussionGenerator:
    """
    Generator for sparse ambient percussion events.

    Creates subtle rhythmic anchors and textural swells
    without overwhelming the ambient atmosphere.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize percussion generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        logger.debug("Percussion generator initialized")

    def generate_percussion(
        self,
        num_bars: int,
        bpm: int,
        sample_rate: int,
        intensity: float = 0.5,
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float, float]]]:
        """
        Generate percussion events for a musical phrase.

        Args:
            num_bars: Number of bars in phrase
            bpm: Beats per minute
            sample_rate: Audio sample rate (Hz)
            intensity: Performance intensity (0.0-1.0)

        Returns:
            Tuple of (kick_events, swell_events):
                - kick_events: List of (onset_sample, velocity)
                - swell_events: List of (onset_sample, duration_sec, velocity)
        """
        kicks = self._generate_kicks(num_bars, bpm, sample_rate, intensity)
        swells = self._generate_swells(num_bars, bpm, sample_rate, intensity)

        logger.debug(
            f"Generated {len(kicks)} kicks, {len(swells)} swells "
            f"for {num_bars} bars at {bpm} BPM"
        )

        return kicks, swells

    def _generate_kicks(
        self,
        num_bars: int,
        bpm: int,
        sample_rate: int,
        intensity: float,
    ) -> List[Tuple[int, float]]:
        """
        Generate sparse kick drum events.

        Args:
            num_bars: Number of bars in phrase
            bpm: Beats per minute
            sample_rate: Audio sample rate (Hz)
            intensity: Performance intensity (0.0-1.0)

        Returns:
            List of (onset_sample, velocity) tuples
        """
        # Only generate kicks if intensity is above threshold
        if intensity < 0.3:
            return []

        kicks = []
        samples_per_beat = int(60.0 * sample_rate / bpm)
        samples_per_bar = samples_per_beat * 4  # 4/4 time signature

        # Generate very sparse kicks (every 4-8 bars for ambient feel)
        kick_interval_bars = self.rng.randint(4, 8)

        for bar_idx in range(0, num_bars, kick_interval_bars):
            # Randomly skip some kicks for more variety (30% chance)
            if self.rng.random() < 0.3:
                continue

            # Place kick at start of bar with significant humanization
            humanize_samples = self.rng.randint(-100, 100)  # ±2ms timing variation
            onset_sample = max(0, (bar_idx * samples_per_bar) + humanize_samples)

            # Velocity based on intensity (softer, more subtle)
            velocity = 0.4 + (intensity * 0.3)
            # Add slight velocity variation
            velocity += self.rng.uniform(-0.1, 0.1)
            velocity = max(0.3, min(0.8, velocity))

            kicks.append((onset_sample, velocity))

        return kicks

    def _generate_swells(
        self,
        num_bars: int,
        bpm: int,
        sample_rate: int,
        intensity: float,
    ) -> List[Tuple[int, float, float]]:
        """
        Generate granular swell events (filtered noise textures).

        Args:
            num_bars: Number of bars in phrase
            bpm: Beats per minute
            sample_rate: Audio sample rate (Hz)
            intensity: Performance intensity (0.0-1.0)

        Returns:
            List of (onset_sample, duration_sec, velocity) tuples
        """
        swells = []
        samples_per_beat = int(60.0 * sample_rate / bpm)
        samples_per_bar = samples_per_beat * 4
        total_duration_sec = (num_bars * samples_per_bar) / sample_rate

        # Number of swells based on intensity (0-2 per phrase, more sparse)
        num_swells = int(intensity * 2) if intensity > 0.3 else 0

        if num_swells == 0:
            return []

        # Place swells more randomly throughout phrase (no strict regions)
        for i in range(num_swells):
            # Completely random placement
            onset_sec = self.rng.uniform(0, total_duration_sec * 0.8)
            onset_sample = int(onset_sec * sample_rate)

            # Very varied swell durations: 3-8 seconds
            duration_sec = self.rng.uniform(3.0, 8.0)

            # Velocity based on intensity (softer, more subtle)
            velocity = intensity * 0.5
            # Add variation
            velocity += self.rng.uniform(-0.15, 0.15)
            velocity = max(0.15, min(0.6, velocity))

            swells.append((onset_sample, duration_sec, velocity))

        return swells
