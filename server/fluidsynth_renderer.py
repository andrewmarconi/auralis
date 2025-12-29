"""FluidSynth renderer for sample-based audio synthesis.

Wraps the pyfluidsynth library to provide high-quality piano and pad sounds
using SoundFont (.sf2) sample libraries.
"""

import logging
from pathlib import Path
from typing import Dict

import fluidsynth
import numpy as np

from server.exceptions import PresetError, RenderError, SoundFontLoadError
from server.interfaces.synthesis import IFluidSynthRenderer

logger = logging.getLogger(__name__)


class FluidSynthRenderer(IFluidSynthRenderer):
    """FluidSynth wrapper for sample-based synthesis."""

    def __init__(self, sample_rate: int = 44100):
        """Initialize FluidSynth synthesizer.

        Args:
            sample_rate: Audio sample rate in Hz (default 44.1kHz)
        """
        self.sample_rate = sample_rate
        self.synth = fluidsynth.Synth(samplerate=float(sample_rate))

        # Configure synthesizer settings for ambient music (BEFORE start())
        # Set polyphony for ambient music (chords + melody with overlaps)
        self.synth.setting('synth.polyphony', 32)  # Max 32 simultaneous voices

        # Use higher gain for better internal resolution (we'll normalize after)
        # Default is 0.2, using 2.0 for 10x higher bit depth
        self.synth.setting('synth.gain', 2.0)

        # Increase audio buffer size for smoother rendering
        self.synth.setting('audio.period-size', 1024)
        self.synth.setting('audio.periods', 8)

        # Disable reverb and chorus for cleaner sound (re-enable later if needed)
        self.synth.setting('synth.reverb.active', 0)
        self.synth.setting('synth.chorus.active', 0)

        self.synth.start()

        # Track loaded SoundFonts
        self.loaded_soundfonts: Dict[int, str] = {}

        logger.info(
            f"FluidSynth initialized at {sample_rate}Hz "
            f"(polyphony=32, gain=2.0, reverb=off, chorus=off)"
        )

    def load_soundfont(self, sf2_path: str) -> int:
        """Load SoundFont file into FluidSynth.

        Args:
            sf2_path: Absolute or relative path to .sf2 file

        Returns:
            SoundFont ID (for later preset selection)

        Raises:
            SoundFontLoadError: If file not found or corrupted
        """
        path = Path(sf2_path)
        if not path.exists():
            raise SoundFontLoadError(f"SoundFont file not found: {sf2_path}")

        if not path.suffix.lower() == ".sf2":
            raise SoundFontLoadError(
                f"Invalid SoundFont file extension: {path.suffix} (expected .sf2)"
            )

        try:
            sf_id = self.synth.sfload(str(path.resolve()))
            if sf_id == -1:
                raise SoundFontLoadError(f"Failed to load SoundFont: {sf2_path}")

            self.loaded_soundfonts[sf_id] = str(path)
            file_size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Loaded SoundFont: {path.name} ({file_size_mb:.1f}MB) → ID {sf_id}"
            )
            return sf_id

        except Exception as e:
            raise SoundFontLoadError(f"Error loading SoundFont {sf2_path}: {e}") from e

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
        if channel < 0 or channel > 15:
            raise PresetError(f"Invalid MIDI channel: {channel} (must be 0-15)")

        if sf_id not in self.loaded_soundfonts:
            raise PresetError(
                f"SoundFont ID {sf_id} not loaded (available: {list(self.loaded_soundfonts.keys())})"
            )

        try:
            result = self.synth.program_select(channel, sf_id, bank, preset)
            if result == -1:
                raise PresetError(
                    f"Failed to select preset {preset} from bank {bank} "
                    f"(SoundFont ID {sf_id})"
                )

            logger.debug(
                f"Selected preset: channel={channel}, sf_id={sf_id}, "
                f"bank={bank}, preset={preset}"
            )

        except Exception as e:
            raise PresetError(
                f"Error selecting preset on channel {channel}: {e}"
            ) from e

    def note_on(self, channel: int, pitch: int, velocity: int) -> None:
        """Trigger note on MIDI channel.

        Args:
            channel: MIDI channel (0-15)
            pitch: MIDI note number (0-127)
            velocity: Note velocity (0-127)
        """
        if not (0 <= channel <= 15):
            logger.warning(f"Invalid MIDI channel: {channel}")
            return

        if not (0 <= pitch <= 127):
            logger.warning(f"Invalid MIDI pitch: {pitch}")
            return

        if not (0 <= velocity <= 127):
            velocity = max(0, min(127, velocity))

        self.synth.noteon(channel, pitch, velocity)

    def note_off(self, channel: int, pitch: int) -> None:
        """Release note on MIDI channel.

        Args:
            channel: MIDI channel (0-15)
            pitch: MIDI note number (0-127)
        """
        if not (0 <= channel <= 15):
            logger.warning(f"Invalid MIDI channel: {channel}")
            return

        if not (0 <= pitch <= 127):
            logger.warning(f"Invalid MIDI pitch: {pitch}")
            return

        self.synth.noteoff(channel, pitch)

    def all_notes_off(self, channel: int = None) -> None:
        """Turn off all notes on a channel or all channels.

        Args:
            channel: MIDI channel (0-15), or None for all channels
        """
        if channel is not None:
            if not (0 <= channel <= 15):
                logger.warning(f"Invalid MIDI channel: {channel}")
                return
            # Send MIDI CC 123 (All Notes Off)
            self.synth.cc(channel, 123, 0)
        else:
            # Turn off all notes on all channels
            for ch in range(16):
                self.synth.cc(ch, 123, 0)

    def render(self, num_samples: int) -> np.ndarray:
        """Generate audio samples.

        Args:
            num_samples: Number of samples to render

        Returns:
            NumPy array, shape (2, num_samples), dtype float32

        Raises:
            RenderError: If synthesis fails
        """
        if num_samples <= 0:
            raise RenderError(f"Invalid sample count: {num_samples}")

        try:
            # FluidSynth generates stereo float samples
            # get_samples() returns array of shape (2, num_samples)
            samples = self.synth.get_samples(num_samples)

            # Convert to NumPy array if not already
            if not isinstance(samples, np.ndarray):
                samples = np.array(samples, dtype=np.float32)

            # FluidSynth returns interleaved stereo: [L, R, L, R, L, R, ...]
            # Need to deinterleave and reshape to (2, num_samples)
            if len(samples.shape) == 1:
                # Interleaved format
                left = samples[0::2]  # Every other sample starting from 0
                right = samples[1::2]  # Every other sample starting from 1
                samples = np.stack([left, right], axis=0)
            elif samples.shape != (2, num_samples):
                # Try reshaping if wrong shape
                samples = samples.reshape(2, num_samples)

            # DEBUG: Check raw sample range
            raw_min = np.min(samples)
            raw_max = np.max(samples)
            raw_peak = np.max(np.abs(samples))
            logger.debug(f"Raw FluidSynth samples - min: {raw_min:.4f}, max: {raw_max:.4f}, peak: {raw_peak:.4f}, dtype: {samples.dtype}")

            # FluidSynth with low gain returns int16 values in a SMALL range (e.g., ±1000)
            # We need to normalize to [-1.0, 1.0] based on actual peak, not assuming ±32768
            samples = samples.astype(np.float32)

            if raw_peak > 0:
                # Normalize to 0.8 peak (leave headroom for safety)
                target_peak = 0.8
                normalization_factor = target_peak / raw_peak
                samples = samples * normalization_factor
                logger.debug(f"Normalized by peak: {raw_peak:.1f} → {target_peak:.1f} (factor: {normalization_factor:.6f})")

            # Clamp to valid range
            samples = np.clip(samples, -1.0, 1.0)

            return samples

        except Exception as e:
            raise RenderError(
                f"Error rendering {num_samples} samples: {e}"
            ) from e

    def configure_reverb(
        self, room_size: float = 0.5, damping: float = 0.5, wet_level: float = 0.2
    ) -> None:
        """Configure FluidSynth reverb settings.

        Args:
            room_size: Reverb room size (0.0-1.0)
            damping: Reverb damping (0.0-1.0)
            wet_level: Wet signal level (0.0-1.0)
        """
        # Clamp values to valid range
        room_size = max(0.0, min(1.0, room_size))
        damping = max(0.0, min(1.0, damping))
        wet_level = max(0.0, min(1.0, wet_level))

        try:
            # FluidSynth reverb parameters (not all exposed in pyfluidsynth)
            # We'll use the simplified reverb settings if available
            self.synth.set_reverb(
                roomsize=room_size, damping=damping, width=1.0, level=wet_level
            )
            logger.info(
                f"Reverb configured: room={room_size:.2f}, "
                f"damping={damping:.2f}, wet={wet_level:.2f}"
            )
        except AttributeError:
            # Fallback for older pyfluidsynth versions
            logger.warning(
                "Reverb configuration not supported in this pyfluidsynth version"
            )
        except Exception as e:
            logger.warning(f"Error configuring reverb: {e}")

    def cleanup(self) -> None:
        """Clean up FluidSynth resources."""
        try:
            self.synth.delete()
            logger.info("FluidSynth cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up FluidSynth: {e}")

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.cleanup()
