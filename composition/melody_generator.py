"""
Constraint-Based Melody Generator

Generates ambient melodies that conform to harmonic constraints.
"""

import random
from typing import List, Tuple

import numpy as np


# Scale intervals (A natural minor)
A_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]  # Relative to root

# Chord tone mappings (intervals from root)
CHORD_INTERVALS = {
    "i": [0, 3, 7],  # minor triad (root, minor 3rd, perfect 5th)
    "iv": [5, 8, 0],  # iv chord (4th, minor 6th, root)
    "V": [7, 11, 2],  # V chord (5th, major 7th, 2nd)
    "VI": [8, 0, 3],  # VI chord (6th, root, minor 3rd)
    "III": [3, 7, 10],  # III chord (minor 3rd, 5th, minor 7th)
}


class MelodyPhrase:
    """
    Series of MIDI notes conforming to harmonic constraints.

    Attributes:
        notes: List of (onset_sec, pitch_midi, velocity, duration_sec) tuples
        bars: Number of bars in phrase
        scale_intervals: Scale intervals used (relative to root)
    """

    def __init__(
        self,
        notes: List[Tuple[float, int, float, float]],
        bars: int = 8,
        scale_intervals: List[int] = A_MINOR_SCALE,
    ):
        """
        Create melody phrase.

        Args:
            notes: List of (onset_sec, pitch_midi, velocity, duration_sec)
            bars: Number of bars in phrase
            scale_intervals: Scale intervals (relative to root)
        """
        self.notes = notes
        self.bars = bars
        self.scale_intervals = scale_intervals

    def to_sample_events(self, sample_rate: int = 44100) -> List[Tuple[int, int, float, float]]:
        """
        Convert timing to sample-accurate events.

        Args:
            sample_rate: Audio sample rate

        Returns:
            List of (onset_sample, pitch_midi, velocity, duration_sec) tuples
        """
        events = []
        for onset_sec, pitch_midi, velocity, duration_sec in self.notes:
            onset_sample = int(onset_sec * sample_rate)
            events.append((onset_sample, pitch_midi, velocity, duration_sec))
        return events


class ConstrainedMelodyGenerator:
    """
    Generates melodies constrained to chord harmony.

    Constraint distribution:
    - 70% chord tones (strong harmonic fit)
    - 25% scale tones (passing notes)
    - 5% chromatic (color tones)
    """

    def __init__(
        self,
        scale_intervals: List[int] = A_MINOR_SCALE,
        root_midi: int = 57,  # A3
        chord_tone_prob: float = 0.7,
        scale_tone_prob: float = 0.25,
        chromatic_prob: float = 0.05,
    ):
        """
        Initialize melody generator.

        Args:
            scale_intervals: Scale intervals relative to root
            root_midi: Root MIDI note
            chord_tone_prob: Probability of chord tone selection
            scale_tone_prob: Probability of scale tone selection
            chromatic_prob: Probability of chromatic passing tone
        """
        self.scale_intervals = scale_intervals
        self.root_midi = root_midi
        self.chord_tone_prob = chord_tone_prob
        self.scale_tone_prob = scale_tone_prob
        self.chromatic_prob = chromatic_prob

        # Validate probabilities
        total_prob = chord_tone_prob + scale_tone_prob + chromatic_prob
        if not np.isclose(total_prob, 1.0):
            raise ValueError(f"Probabilities must sum to 1.0, got {total_prob}")

    def generate_melody(
        self,
        chord_progression: List[Tuple[int, int, str]],
        duration_sec: float,
        bpm: int = 70,
        intensity: float = 0.5,
        complexity: float = 0.5,
    ) -> MelodyPhrase:
        """
        Generate melody for chord progression.

        Args:
            chord_progression: List of (onset_sample, root_midi, chord_type)
            duration_sec: Total phrase duration in seconds
            bpm: Tempo for note density calculation
            intensity: Control density and range (0.0-1.0)
            complexity: Control harmonic adherence vs chromaticism (0.0-1.0)

        Returns:
            MelodyPhrase with generated notes
        """
        notes = []

        # Adjust probabilities based on complexity
        # Low complexity: more chord tones, high: more chromatic
        chord_tone_prob = 0.7 - complexity * 0.4
        scale_tone_prob = 0.25 + complexity * 0.2
        chromatic_prob = 0.05 + complexity * 0.2

        # Normalize
        total = chord_tone_prob + scale_tone_prob + chromatic_prob
        self.chord_tone_prob = chord_tone_prob / total
        self.scale_tone_prob = scale_tone_prob / total
        self.chromatic_prob = chromatic_prob / total

        # Calculate note density based on intensity
        # Low intensity: ~0.5 notes/second (sparse)
        # High intensity: ~2.5 notes/second (continuous)
        notes_per_second = 0.5 + (intensity * 2.0)

        # Generate notes throughout duration
        current_time = 0.0
        while current_time < duration_sec:
            # Find active chord at current time
            active_chord = self._get_active_chord(current_time, chord_progression, duration_sec)

            # Generate note pitch constrained to harmony
            pitch_midi = self._generate_constrained_pitch(active_chord, intensity)

            # Note duration (ambient = longer, varied notes)
            # Low intensity: 1-4 second notes, high: 0.5-2 second notes
            if intensity < 0.3:
                duration = random.uniform(2.0, 5.0)  # Very long, sustained
            elif intensity < 0.6:
                duration = random.uniform(1.0, 3.0)  # Medium sustained
            else:
                duration = random.uniform(0.5, 2.0)  # Shorter, more active

            # Velocity (ambient = softer dynamics with variation)
            base_velocity = random.uniform(0.25, 0.5)
            velocity = base_velocity * (0.7 + intensity * 0.3)

            notes.append((current_time, pitch_midi, velocity, duration))

            # Inter-onset interval with MUCH more variation for organic, non-repetitive feel
            # Base interval from note density
            base_interval = 1.0 / notes_per_second

            # Add substantial random variation (±50-150%)
            variation_factor = random.uniform(0.5, 2.5)
            interval = base_interval * variation_factor

            # Occasionally add longer pauses for breathing room
            if random.random() < 0.2:  # 20% chance
                interval *= random.uniform(1.5, 3.0)

            interval = max(0.2, interval)  # Minimum 200ms
            current_time += interval

        # Calculate bars from duration and BPM
        seconds_per_bar = (60.0 / bpm) * 4
        bars = int(duration_sec / seconds_per_bar)

        return MelodyPhrase(notes, bars, self.scale_intervals)

    def _get_active_chord(
        self,
        time_sec: float,
        chord_progression: List[Tuple[int, int, str]],
        duration_sec: float,
    ) -> str:
        """
        Get active chord at specific time.

        Args:
            time_sec: Time in seconds
            chord_progression: List of (onset_sample, root_midi, chord_type)
            duration_sec: Total duration

        Returns:
            Chord symbol active at time_sec
        """
        if not chord_progression:
            return "i"  # Default to tonic

        # Convert time to samples for comparison
        sample_rate = 44100  # Standard sample rate
        time_samples = int(time_sec * sample_rate)

        # Find the chord active at this time
        # Chords are sorted by onset time
        active_chord = chord_progression[0][2]  # Start with first chord

        for onset_sample, root_midi, chord_type in chord_progression:
            if onset_sample <= time_samples:
                active_chord = chord_type
            else:
                # We've passed the current time
                break

        return active_chord

    def _generate_constrained_pitch(self, chord_symbol: str, intensity: float) -> int:
        """
        Generate MIDI pitch constrained to chord harmony.

        Args:
            chord_symbol: Current chord (e.g., "i", "iv")
            intensity: Control range and chromaticism

        Returns:
            MIDI pitch number
        """
        # Sample note type based on probabilities
        rand = random.random()

        if rand < self.chord_tone_prob:
            # Chord tone (70% probability)
            intervals = CHORD_INTERVALS.get(chord_symbol, [0, 3, 7])
            interval = random.choice(intervals)
        elif rand < (self.chord_tone_prob + self.scale_tone_prob):
            # Scale tone (25% probability)
            interval = random.choice(self.scale_intervals)
        else:
            # Chromatic passing tone (5% probability)
            interval = random.randint(0, 11)

        # Octave selection based on intensity
        # Low intensity: narrow range (1 octave)
        # High intensity: wider range (2-3 octaves)
        octave_range = 1 + int(intensity * 2)
        octave_offset = random.randint(0, octave_range) * 12

        # Construct MIDI pitch
        pitch_midi = self.root_midi + interval + octave_offset

        # Clamp to valid MIDI range
        pitch_midi = max(21, min(127, pitch_midi))

        return pitch_midi
