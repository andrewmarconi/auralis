"""SoundFont file management and preset mapping.

Manages loading and configuration of SoundFont (.sf2) files for
piano and pad instruments.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from server.config import get_config
from server.exceptions import ConfigurationError, SoundFontLoadError
from server.interfaces.synthesis import IFluidSynthRenderer

logger = logging.getLogger(__name__)


@dataclass
class SoundFontPreset:
    """SoundFont preset configuration."""

    instrument_name: str
    sf2_file_path: Path
    preset_number: int
    bank_number: int = 0
    channel: int = 0

    @property
    def file_size_mb(self) -> float:
        """Get SoundFont file size in MB."""
        if self.sf2_file_path.exists():
            return self.sf2_file_path.stat().st_size / (1024 * 1024)
        return 0.0


class SoundFontManager:
    """Manages SoundFont loading and preset configuration."""

    # Preset mapping for instruments
    PIANO_CHANNEL = 0
    PAD_CHANNEL = 1

    PIANO_PRESET = 0  # Grand Piano preset
    PAD_PRESET = 88  # Warm Pad preset (General MIDI)

    def __init__(self, renderer: IFluidSynthRenderer):
        """Initialize SoundFont manager.

        Args:
            renderer: FluidSynth renderer instance
        """
        self.renderer = renderer
        self.config = get_config()
        self.loaded_presets: Dict[str, SoundFontPreset] = {}
        self.soundfont_ids: Dict[str, int] = {}

    def load_all_soundfonts(self) -> None:
        """Load all configured SoundFonts.

        Loads Piano and Pad SoundFonts from configuration paths.

        Raises:
            SoundFontLoadError: If any SoundFont fails to load
            ConfigurationError: If SoundFont paths not configured
        """
        # Define presets to load
        presets: List[SoundFontPreset] = [
            SoundFontPreset(
                instrument_name="piano",
                sf2_file_path=self.config.soundfont_piano,
                preset_number=self.PIANO_PRESET,
                bank_number=0,
                channel=self.PIANO_CHANNEL,
            ),
            SoundFontPreset(
                instrument_name="pad",
                sf2_file_path=self.config.soundfont_gm,
                preset_number=self.PAD_PRESET,
                bank_number=0,
                channel=self.PAD_CHANNEL,
            ),
        ]

        total_size_mb = 0.0

        for preset in presets:
            # Validate file exists
            if not preset.sf2_file_path.exists():
                error_msg = (
                    f"SoundFont not found: {preset.sf2_file_path}\n"
                    f"Please download SoundFonts as described in soundfonts/.env.example"
                )
                raise SoundFontLoadError(error_msg)

            # Load SoundFont
            try:
                sf_id = self.renderer.load_soundfont(str(preset.sf2_file_path))
                self.soundfont_ids[preset.instrument_name] = sf_id

                # Select preset on channel
                self.renderer.select_preset(
                    channel=preset.channel,
                    sf_id=sf_id,
                    bank=preset.bank_number,
                    preset=preset.preset_number,
                )

                # Track loaded preset
                self.loaded_presets[preset.instrument_name] = preset
                total_size_mb += preset.file_size_mb

                logger.info(
                    f"Loaded {preset.instrument_name}: "
                    f"{preset.sf2_file_path.name} "
                    f"(channel={preset.channel}, "
                    f"preset={preset.preset_number}, "
                    f"{preset.file_size_mb:.1f}MB)"
                )

            except Exception as e:
                raise SoundFontLoadError(
                    f"Failed to load {preset.instrument_name} SoundFont: {e}"
                ) from e

        logger.info(
            f"All SoundFonts loaded successfully "
            f"({len(self.loaded_presets)} instruments, {total_size_mb:.1f}MB total)"
        )

        # Configure minimal reverb
        self.configure_reverb()

    def configure_reverb(self) -> None:
        """Configure reverb settings for ambient music."""
        try:
            self.renderer.configure_reverb(
                room_size=self.config.reverb_room_size,
                damping=self.config.reverb_damping,
                wet_level=self.config.reverb_wet_level,
            )
            logger.info("Reverb configured for ambient music")
        except Exception as e:
            logger.warning(f"Could not configure reverb: {e}")

    def get_piano_channel(self) -> int:
        """Get MIDI channel for piano instrument.

        Returns:
            MIDI channel number (0)
        """
        return self.PIANO_CHANNEL

    def get_pad_channel(self) -> int:
        """Get MIDI channel for pad instrument.

        Returns:
            MIDI channel number (1)
        """
        return self.PAD_CHANNEL

    def get_loaded_instruments(self) -> List[str]:
        """Get list of loaded instrument names.

        Returns:
            List of instrument names (e.g., ["piano", "pad"])
        """
        return list(self.loaded_presets.keys())

    def get_total_memory_mb(self) -> float:
        """Get total SoundFont memory usage.

        Returns:
            Total size in megabytes
        """
        return sum(preset.file_size_mb for preset in self.loaded_presets.values())

    def is_loaded(self, instrument_name: str) -> bool:
        """Check if instrument is loaded.

        Args:
            instrument_name: Instrument name ("piano" or "pad")

        Returns:
            True if loaded, False otherwise
        """
        return instrument_name in self.loaded_presets
