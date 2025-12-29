"""Synthesis interface definitions for Auralis audio rendering."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from composition.chord_generator import ChordProgression
    from composition.melody_generator import MelodyPhrase, MusicalContext


class ISynthesisEngine(ABC):
    """Orchestrates music generation and audio synthesis."""

    @abstractmethod
    async def generate_phrase(
        self, context: "MusicalContext", duration_bars: int = 8
    ) -> tuple["ChordProgression", "MelodyPhrase"]:
        """Generate chord progression and melody for a phrase.

        Args:
            context: Current musical parameters (key, mode, BPM, intensity)
            duration_bars: Number of bars to generate (8 or 16)

        Returns:
            Tuple of (ChordProgression, MelodyPhrase)

        Raises:
            GenerationError: If composition fails
        """
        pass

    @abstractmethod
    async def render_phrase(
        self, chords: "ChordProgression", melody: "MelodyPhrase"
    ) -> np.ndarray:
        """Render musical phrase to stereo PCM audio using FluidSynth.

        Args:
            chords: Generated chord progression
            melody: Generated melody phrase

        Returns:
            NumPy array, shape (2, num_samples), dtype float32, range [-1.0, 1.0]

        Raises:
            SynthesisError: If rendering fails or exceeds 100ms latency
        """
        pass

    @abstractmethod
    def get_device(self) -> str:
        """Return current synthesis device.

        Returns:
            Device string: "Metal", "CUDA", or "CPU"
        """
        pass


class IFluidSynthRenderer(ABC):
    """FluidSynth wrapper for sample-based synthesis."""

    @abstractmethod
    def load_soundfont(self, sf2_path: str) -> int:
        """Load SoundFont file into FluidSynth.

        Args:
            sf2_path: Absolute path to .sf2 file

        Returns:
            SoundFont ID (for later preset selection)

        Raises:
            SoundFontLoadError: If file not found or corrupted
        """
        pass

    @abstractmethod
    def select_preset(self, channel: int, sf_id: int, bank: int, preset: int) -> None:
        """Assign SoundFont preset to MIDI channel.

        Args:
            channel: MIDI channel (0-15)
            sf_id: SoundFont ID from load_soundfont()
            bank: Bank number (typically 0)
            preset: Preset number (0-127, General MIDI)

        Raises:
            PresetError: If preset not found in SoundFont
        """
        pass

    @abstractmethod
    def note_on(self, channel: int, pitch: int, velocity: int) -> None:
        """Trigger note on MIDI channel.

        Args:
            channel: MIDI channel (0-15)
            pitch: MIDI note number (0-127)
            velocity: Note velocity (0-127)
        """
        pass

    @abstractmethod
    def note_off(self, channel: int, pitch: int) -> None:
        """Release note on MIDI channel.

        Args:
            channel: MIDI channel (0-15)
            pitch: MIDI note number (0-127)
        """
        pass

    @abstractmethod
    def render(self, num_samples: int) -> np.ndarray:
        """Generate audio samples.

        Args:
            num_samples: Number of samples to render

        Returns:
            NumPy array, shape (2, num_samples), dtype float32

        Raises:
            RenderError: If synthesis fails
        """
        pass

    @abstractmethod
    def configure_reverb(
        self, room_size: float = 0.5, damping: float = 0.5, wet_level: float = 0.2
    ) -> None:
        """Configure FluidSynth reverb settings.

        Args:
            room_size: Reverb room size (0.0-1.0)
            damping: Reverb damping (0.0-1.0)
            wet_level: Wet signal level (0.0-1.0)
        """
        pass
