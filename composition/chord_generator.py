"""
Markov Chain Chord Progression Generator

Generates ambient-optimized chord progressions using bigram Markov chains.
"""

import random
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


# Ambient chord vocabulary (Roman numeral notation for A minor)
AMBIENT_CHORDS = ["i", "iv", "V", "VI", "III"]

# Ambient-biased transition matrix (probabilities)
# Rows: current chord, Columns: next chord
# Designed for slow, contemplative progressions
AMBIENT_TRANSITION_MATRIX = np.array([
    # i    iv    V    VI   III
    [0.2, 0.3, 0.1, 0.3, 0.1],  # from i (tonic)
    [0.4, 0.1, 0.2, 0.2, 0.1],  # from iv (subdominant)
    [0.5, 0.1, 0.1, 0.2, 0.1],  # from V (dominant)
    [0.3, 0.2, 0.1, 0.2, 0.2],  # from VI
    [0.4, 0.2, 0.1, 0.2, 0.1],  # from III
], dtype=np.float32)


class ChordProgression:
    """
    8-bar harmonic sequence for ambient music generation.

    Attributes:
        chords: List of chord symbols (e.g., ["i", "iv", "V", "VI"])
        length_bars: Phrase length in bars (default: 8)
        root_midi: Root MIDI note for key reference
    """

    def __init__(
        self,
        chords: List[str],
        length_bars: int = 8,
        root_midi: int = 57,  # A3
    ):
        """
        Create chord progression.

        Args:
            chords: List of chord symbols
            length_bars: Number of bars in progression
            root_midi: MIDI note for root (default A3)
        """
        self.chords = chords
        self.length_bars = length_bars
        self.root_midi = root_midi

    def to_midi_events(
        self, bpm: int = 70, sample_rate: int = 44100
    ) -> List[Tuple[int, int, str]]:
        """
        Convert chord progression to MIDI event list with sample-accurate timing.

        Args:
            bpm: Tempo in beats per minute
            sample_rate: Audio sample rate

        Returns:
            List of (onset_sample, root_midi, chord_type) tuples
        """
        # Calculate samples per bar (4/4 time)
        seconds_per_bar = (60.0 / bpm) * 4  # 4 beats per bar
        samples_per_bar = int(seconds_per_bar * sample_rate)

        events = []
        for bar_idx, chord_symbol in enumerate(self.chords):
            onset_sample = bar_idx * samples_per_bar
            events.append((onset_sample, self.root_midi, chord_symbol))

        return events


class ChordProgressionGenerator:
    """
    Markov chain generator for ambient chord progressions.

    Uses bigram (order-2) Markov chain with ambient-biased transitions.
    """

    def __init__(
        self,
        transition_matrix: NDArray[np.float32] = AMBIENT_TRANSITION_MATRIX,
        chord_vocabulary: List[str] = AMBIENT_CHORDS,
    ):
        """
        Initialize chord generator.

        Args:
            transition_matrix: Probability matrix for chord transitions
            chord_vocabulary: Available chord symbols
        """
        self.transition_matrix = transition_matrix
        self.chord_vocabulary = chord_vocabulary
        self.num_chords = len(chord_vocabulary)

        # Validate transition matrix
        if transition_matrix.shape != (self.num_chords, self.num_chords):
            raise ValueError(
                f"Transition matrix shape {transition_matrix.shape} "
                f"must match vocabulary size {self.num_chords}"
            )

        # Ensure rows sum to 1.0
        row_sums = transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Transition matrix rows must sum to 1.0")

    def generate_progression(
        self, length_bars: int = 8, root_midi: int = 57
    ) -> ChordProgression:
        """
        Generate chord progression using Markov chain.

        Args:
            length_bars: Number of bars to generate
            root_midi: MIDI note for key root

        Returns:
            ChordProgression instance with generated chords
        """
        if length_bars < 1:
            raise ValueError("Progression length must be at least 1 bar")

        chords = []

        # Start with tonic (i)
        current_idx = 0  # Index of "i" chord
        chords.append(self.chord_vocabulary[current_idx])

        # Generate remaining chords
        for _ in range(length_bars - 1):
            # Get transition probabilities for current chord
            probabilities = self.transition_matrix[current_idx]

            # Sample next chord
            next_idx = np.random.choice(self.num_chords, p=probabilities)
            chords.append(self.chord_vocabulary[next_idx])

            # Update current chord
            current_idx = next_idx

        return ChordProgression(chords, length_bars, root_midi)

    def generate_with_constraints(
        self,
        length_bars: int = 8,
        root_midi: int = 57,
        start_chord: str = "i",
        end_chord: str = "i",
    ) -> ChordProgression:
        """
        Generate progression with start/end constraints.

        Args:
            length_bars: Number of bars to generate
            root_midi: MIDI note for key root
            start_chord: Required first chord
            end_chord: Required last chord

        Returns:
            ChordProgression with constrained start/end
        """
        if length_bars < 2:
            raise ValueError("Constrained progression requires at least 2 bars")

        if start_chord not in self.chord_vocabulary:
            raise ValueError(f"Start chord '{start_chord}' not in vocabulary")
        if end_chord not in self.chord_vocabulary:
            raise ValueError(f"End chord '{end_chord}' not in vocabulary")

        # Generate middle section
        middle_length = length_bars - 2
        if middle_length > 0:
            middle_progression = self.generate_progression(middle_length, root_midi)
            chords = [start_chord] + middle_progression.chords + [end_chord]
        else:
            chords = [start_chord, end_chord]

        return ChordProgression(chords, length_bars, root_midi)
