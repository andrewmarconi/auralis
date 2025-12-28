"""
SoundFont Manager for FluidSynth Integration

Handles loading, validation, and management of SoundFont (SF2) files for realistic
instrument synthesis. Implements three-layer validation (filesystem, size check,
FluidSynth load test) with fail-fast behavior.
"""

import os
from pathlib import Path
from typing import Optional
import fluidsynth


class SoundFontValidationError(Exception):
    """Raised when SoundFont validation fails."""
    pass


class SoundFontManager:
    """
    Manages SoundFont file loading and validation for FluidSynth synthesis.

    Implements three-layer validation:
    1. Filesystem check: File exists and is readable
    2. Size check: File is >100MB (reasonable SF2 size threshold)
    3. FluidSynth load test: File can be successfully loaded by FluidSynth

    Attributes:
        soundfont_dir: Path to directory containing SF2 files
        required_soundfonts: List of required SoundFont filenames
    """

    # Minimum file size for valid SoundFont (100MB threshold)
    MIN_SOUNDFONT_SIZE_BYTES = 100 * 1024 * 1024  # 100MB

    def __init__(self, soundfont_dir: Optional[str] = None):
        """
        Initialize SoundFont manager.

        Args:
            soundfont_dir: Optional custom path to soundfonts directory.
                          Defaults to ./soundfonts/ or AURALIS_SOUNDFONT_DIR env var.
        """
        if soundfont_dir:
            self.soundfont_dir = Path(soundfont_dir)
        else:
            # Check environment variable, fallback to ./soundfonts/
            env_dir = os.getenv("AURALIS_SOUNDFONT_DIR")
            self.soundfont_dir = Path(env_dir) if env_dir else Path("soundfonts")

        # Required SoundFont file (FluidR3_GM.sf2 contains all GM presets)
        self.required_soundfonts = ["FluidR3_GM.sf2"]

    def validate_all_soundfonts(self) -> None:
        """
        Validate all required SoundFont files using three-layer validation.

        Raises:
            SoundFontValidationError: If any validation layer fails for any required SoundFont
        """
        for sf_filename in self.required_soundfonts:
            sf_path = self.soundfont_dir / sf_filename
            self._validate_soundfont(sf_path)

    def _validate_soundfont(self, sf_path: Path) -> None:
        """
        Three-layer validation for a single SoundFont file.

        Args:
            sf_path: Path to SoundFont file

        Raises:
            SoundFontValidationError: If any validation layer fails
        """
        # Layer 1: Filesystem check
        if not sf_path.exists():
            raise SoundFontValidationError(
                f"SoundFont file not found: {sf_path}\n"
                f"Expected location: {sf_path.absolute()}\n"
                f"Please download FluidR3_GM.sf2 and place it in the soundfonts/ directory.\n"
                f"Download: https://github.com/urish/cinto/raw/master/media/FluidR3%20GM.sf2"
            )

        if not sf_path.is_file():
            raise SoundFontValidationError(
                f"SoundFont path is not a file: {sf_path}"
            )

        if not os.access(sf_path, os.R_OK):
            raise SoundFontValidationError(
                f"SoundFont file is not readable: {sf_path}\n"
                f"Check file permissions."
            )

        # Layer 2: Size check (>100MB threshold)
        file_size = sf_path.stat().st_size
        if file_size < self.MIN_SOUNDFONT_SIZE_BYTES:
            raise SoundFontValidationError(
                f"SoundFont file too small: {sf_path}\n"
                f"Size: {file_size / (1024*1024):.1f}MB (expected >100MB)\n"
                f"File may be corrupted or incomplete."
            )

        # Layer 3: FluidSynth load test
        try:
            # Create temporary FluidSynth instance to test loading
            test_synth = fluidsynth.Synth(samplerate=44100.0)
            sfid = test_synth.sfload(str(sf_path.absolute()))

            if sfid == -1:
                raise SoundFontValidationError(
                    f"FluidSynth failed to load SoundFont: {sf_path}\n"
                    f"File may be corrupted or not a valid SF2 file."
                )

            # Cleanup test instance
            test_synth.delete()

        except Exception as e:
            raise SoundFontValidationError(
                f"FluidSynth load test failed for: {sf_path}\n"
                f"Error: {str(e)}"
            )

    def get_soundfont_path(self, filename: str) -> Path:
        """
        Get full path to a SoundFont file.

        Args:
            filename: SoundFont filename (e.g., "FluidR3_GM.sf2")

        Returns:
            Full path to SoundFont file
        """
        return self.soundfont_dir / filename
