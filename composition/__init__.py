"""Auralis Composition - Generative music algorithms.

This module contains the algorithmic composition logic for generating
chord progressions, melodies, and percussion patterns.
"""

from composition.chord_generator import ChordGenerator, ChordProgression
from composition.melody_generator import MelodyGenerator, MelodyPhrase
from composition.musical_context import MusicalContext
from composition.percussion_generator import PercussionGenerator

__version__ = "2.0.0"

__all__ = [
    "ChordGenerator",
    "ChordProgression",
    "MelodyGenerator",
    "MelodyPhrase",
    "MusicalContext",
    "PercussionGenerator",
]
