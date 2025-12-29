"""Markov chain chord progression generator.

Generates harmonically coherent chord progressions using a bigram Markov
chain (order 2) with modal constraints.
"""

import logging
import random
from dataclasses import dataclass
from typing import List, Literal

from composition.musical_context import MusicalContext

logger = logging.getLogger(__name__)

ChordType = Literal["major", "minor", "sus2", "sus4", "add9", "maj7"]


@dataclass
class ChordEvent:
    """Single chord event in a progression."""

    onset_time: int  # Sample offset from phrase start
    root_pitch: int  # MIDI note number
    chord_type: ChordType


@dataclass
class ChordProgression:
    """Generated sequence of chord changes."""

    chords: List[ChordEvent]
    duration_samples: int
    bpm: float
    mode: str


class ChordGenerator:
    """Generates chord progressions using Markov chains."""

    # Modal scale degrees (relative to root)
    MODES = {
        "aeolian": [0, 2, 3, 5, 7, 8, 10],  # Natural minor
        "dorian": [0, 2, 3, 5, 7, 9, 10],  # Dorian mode
        "lydian": [0, 2, 4, 6, 7, 9, 11],  # Lydian mode
        "phrygian": [0, 1, 3, 5, 7, 8, 10],  # Phrygian mode
    }

    # Chord type preferences by mode
    CHORD_TYPES_BY_MODE = {
        "aeolian": ["minor", "major", "sus4"],
        "dorian": ["minor", "sus2", "major"],
        "lydian": ["major", "maj7", "add9"],
        "phrygian": ["minor", "sus2", "sus4"],
    }

    # Markov transition probabilities (scale degree → next scale degree)
    # Weighted toward common ambient progressions (I-IV-V, I-VI-III-VII, etc.)
    TRANSITION_WEIGHTS = {
        0: {0: 0.3, 3: 0.2, 5: 0.2, 7: 0.15, 2: 0.1, 10: 0.05},  # I → I, IV, V, VI, III, VII
        2: {5: 0.3, 7: 0.25, 0: 0.2, 3: 0.15, 10: 0.1},  # III → V, VI, I, IV, VII
        3: {0: 0.3, 5: 0.25, 7: 0.2, 2: 0.15, 10: 0.1},  # IV → I, V, VI, III, VII
        5: {0: 0.35, 3: 0.25, 7: 0.2, 2: 0.15, 10: 0.05},  # V → I, IV, VI, III, VII
        7: {0: 0.3, 3: 0.25, 5: 0.2, 2: 0.15, 10: 0.1},  # VI → I, IV, V, III, VII
        8: {0: 0.4, 7: 0.25, 5: 0.2, 3: 0.15},  # VII♭ → I, VI, V, IV
        10: {0: 0.4, 3: 0.25, 5: 0.2, 7: 0.15},  # VII → I, IV, V, VI
    }

    def __init__(self, sample_rate: int = 44100):
        """Initialize chord generator.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        logger.info("Chord generator initialized")

    def generate(
        self, context: MusicalContext, duration_bars: int = 8
    ) -> ChordProgression:
        """Generate chord progression.

        Args:
            context: Musical parameters (key, mode, BPM, intensity)
            duration_bars: Number of bars to generate (8 or 16)

        Returns:
            ChordProgression with generated chords
        """
        # Calculate duration in samples
        beats_per_bar = 4
        total_beats = duration_bars * beats_per_bar
        seconds_per_beat = 60.0 / context.bpm
        duration_sec = total_beats * seconds_per_beat
        duration_samples = int(duration_sec * self.sample_rate)

        # Generate chord changes based on intensity
        # Lower intensity = fewer chord changes (longer sustain)
        # Higher intensity = more changes
        num_chords = max(2, int(duration_bars / 2 * (0.5 + context.intensity)))

        # Get scale degrees for mode
        scale_degrees = self.MODES[context.mode]
        chord_types = self.CHORD_TYPES_BY_MODE[context.mode]

        # Generate chord events using Markov chain
        chords: List[ChordEvent] = []
        samples_per_chord = duration_samples // num_chords

        # Start on tonic (degree 0)
        current_degree = 0

        for i in range(num_chords):
            # Calculate root pitch from current degree
            root_pitch = context.key + current_degree

            # Select chord type appropriate for mode
            chord_type = random.choice(chord_types)

            # Calculate onset time
            onset_time = i * samples_per_chord

            chords.append(
                ChordEvent(
                    onset_time=onset_time, root_pitch=root_pitch, chord_type=chord_type
                )
            )

            # Select next degree using Markov transition probabilities
            if i < num_chords - 1:  # Don't transition after last chord
                current_degree = self._select_next_degree(current_degree, scale_degrees)


        logger.debug(
            f"Generated {len(chords)} chords for {duration_bars} bars "
            f"in {context.mode} mode"
        )

        return ChordProgression(
            chords=chords,
            duration_samples=duration_samples,
            bpm=context.bpm,
            mode=context.mode,
        )

    def _select_next_degree(self, current_degree: int, scale_degrees: List[int]) -> int:
        """Select next scale degree using Markov transition probabilities.

        Args:
            current_degree: Current scale degree
            scale_degrees: Valid degrees for the mode

        Returns:
            Next scale degree
        """
        # Get transition weights for current degree (default to uniform if not in table)
        if current_degree in self.TRANSITION_WEIGHTS:
            weights = self.TRANSITION_WEIGHTS[current_degree]

            # Filter to only valid degrees in current mode
            valid_transitions = {deg: w for deg, w in weights.items() if deg in scale_degrees}

            if valid_transitions:
                # Weighted random choice
                degrees = list(valid_transitions.keys())
                probabilities = list(valid_transitions.values())
                return random.choices(degrees, weights=probabilities)[0]

        # Fallback: random choice from scale degrees
        return random.choice(scale_degrees)
