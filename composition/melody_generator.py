"""Constraint-based melody generator.

Generates sparse ambient melodies with 70% chord tones, 25% scale notes,
and 5% chromatic passing tones.
"""

import logging
import random
from dataclasses import dataclass
from typing import List

from composition.chord_generator import ChordEvent, ChordProgression
from composition.musical_context import MusicalContext

logger = logging.getLogger(__name__)


@dataclass
class NoteEvent:
    """Single note event in a melody."""

    onset_time: int  # Sample offset from phrase start
    pitch: int  # MIDI note number
    velocity: int  # Note velocity (20-100)
    duration: int  # Note length in samples


@dataclass
class MelodyPhrase:
    """Generated melodic line."""

    notes: List[NoteEvent]
    duration_samples: int
    chord_context: ChordProgression


class MelodyGenerator:
    """Generates constraint-based melodies."""

    # Chord intervals (semitones from root)
    CHORD_INTERVALS = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],
        "add9": [0, 2, 4, 7],
        "maj7": [0, 4, 7, 11],
    }

    # Modal scale degrees
    MODES = {
        "aeolian": [0, 2, 3, 5, 7, 8, 10],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "lydian": [0, 2, 4, 6, 7, 9, 11],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
    }

    def __init__(self, sample_rate: int = 44100):
        """Initialize melody generator.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        logger.info("Melody generator initialized")

    def generate(
        self, context: MusicalContext, chords: ChordProgression
    ) -> MelodyPhrase:
        """Generate melody phrase.

        Args:
            context: Musical parameters
            chords: Chord progression for harmonic context

        Returns:
            MelodyPhrase with generated notes
        """
        # Calculate note probability based on intensity
        # Lower intensity = sparser melody
        note_probability = 0.3 + (context.intensity * 0.5)  # 0.3 to 0.8

        # Estimate number of potential note positions
        beats_per_second = context.bpm / 60.0
        note_grid_hz = beats_per_second * 2  # Eighth note grid
        max_notes = int(chords.duration_samples / self.sample_rate * note_grid_hz)

        notes: List[NoteEvent] = []
        last_pitch: int | None = None  # Track for stepwise motion

        # Generate melodic contour curve (arch shape: rise then fall)
        # This creates intentional melodic direction
        contour_curve = self._generate_melodic_contour(max_notes)

        for i in range(max_notes):
            # Skip note based on probability (sparse texture)
            if random.random() > note_probability:
                continue

            # Calculate onset time
            onset_time = int(i * (chords.duration_samples / max_notes))

            # Find active chord at this time
            active_chord = self._get_active_chord(onset_time, chords)

            if active_chord is None:
                continue

            # Select pitch based on constraint distribution
            pitch = self._select_pitch(active_chord, context, last_pitch, contour_curve[i])
            last_pitch = pitch

            # Calculate velocity using dynamic curve (phrase shaping)
            velocity = self._calculate_velocity(i, max_notes, context.intensity)

            # Note duration with some variation (0.5 to 1.5 seconds for ambient)
            duration_samples = int(
                random.uniform(0.5, 1.5) * self.sample_rate
            )

            notes.append(
                NoteEvent(
                    onset_time=onset_time,
                    pitch=pitch,
                    velocity=velocity,
                    duration=duration_samples,
                )
            )

        logger.debug(
            f"Generated {len(notes)} notes for melody "
            f"(probability={note_probability:.2f})"
        )

        return MelodyPhrase(
            notes=notes,
            duration_samples=chords.duration_samples,
            chord_context=chords,
        )

    def _get_active_chord(
        self, onset_time: int, chords: ChordProgression
    ) -> ChordEvent | None:
        """Find active chord at given time.

        Args:
            onset_time: Sample offset
            chords: Chord progression

        Returns:
            Active ChordEvent or None
        """
        active_chord: ChordEvent | None = None

        for chord in chords.chords:
            if chord.onset_time <= onset_time:
                active_chord = chord
            else:
                break

        return active_chord

    def _select_pitch(
        self, chord: "ChordEvent", context: MusicalContext, last_pitch: int | None, contour_value: float
    ) -> int:
        """Select pitch based on constraint distribution and melodic contour.

        70% chord tones, 25% scale notes, 5% chromatic.
        Prefers stepwise motion when last_pitch is provided.

        Args:
            chord: Active chord
            context: Musical context
            last_pitch: Previous note pitch (for stepwise motion)
            contour_value: Melodic contour direction (-1 to 1, negative=descend, positive=ascend)

        Returns:
            MIDI pitch
        """
        from composition.chord_generator import ChordEvent

        rand = random.random()

        # Base octave selection influenced by melodic contour
        if contour_value > 0.5:
            octave_offset = 24  # Higher octave
        elif contour_value > 0:
            octave_offset = 12  # Middle octave
        else:
            octave_offset = 0  # Lower octave

        if rand < 0.70:
            # 70% chord tones
            intervals = self.CHORD_INTERVALS[chord.chord_type]
            interval = random.choice(intervals)
            pitch = chord.root_pitch + interval + octave_offset

        elif rand < 0.95:
            # 25% scale notes
            scale_degrees = self.MODES[context.mode]
            degree = random.choice(scale_degrees)
            pitch = context.key + degree + octave_offset

        else:
            # 5% chromatic passing tones
            pitch = context.key + random.randint(0, 11) + octave_offset

        # Prefer stepwise motion (within 3 semitones) when possible
        if last_pitch is not None and random.random() < 0.6:  # 60% chance of stepwise
            interval = random.choice([-2, -1, 1, 2])  # Stepwise motion
            pitch = last_pitch + interval

        # Clamp to valid MIDI range
        return max(48, min(96, pitch))

    def _generate_melodic_contour(self, num_positions: int) -> List[float]:
        """Generate melodic contour curve (arch or wave shape).

        Creates intentional melodic direction: rise, peak, fall.

        Args:
            num_positions: Number of note positions

        Returns:
            List of contour values (-1 to 1, negative=low, positive=high)
        """
        import math

        contour = []
        contour_type = random.choice(['arch', 'inverted_arch', 'wave'])

        for i in range(num_positions):
            t = i / max(1, num_positions - 1)  # 0 to 1

            if contour_type == 'arch':
                # Rise then fall (parabola)
                value = 4 * t * (1 - t)  # Peaks at t=0.5
            elif contour_type == 'inverted_arch':
                # Fall then rise
                value = -4 * t * (1 - t) + 1
            else:  # wave
                # Gentle sine wave
                value = math.sin(t * math.pi * 2)

            contour.append(value)

        return contour

    def _calculate_velocity(self, position: int, total_positions: int, intensity: float) -> int:
        """Calculate velocity with dynamic curve for musical expression.

        Creates crescendo/diminuendo shapes.

        Args:
            position: Current note position
            total_positions: Total number of positions
            intensity: Overall intensity (0-1)

        Returns:
            Velocity (20-100)
        """
        # Base velocity from intensity
        base_velocity = int(20 + (intensity * 60))  # 20-80 range

        # Add dynamic curve (crescendo to middle, diminuendo to end)
        t = position / max(1, total_positions - 1)  # 0 to 1
        dynamic_curve = 4 * t * (1 - t)  # Peaks at middle

        # Apply curve (±20 velocity variation)
        velocity = base_velocity + int(dynamic_curve * 20)

        # Clamp to valid range
        return max(20, min(100, velocity))
