"""Musical context and parameter definitions."""

from dataclasses import dataclass
from typing import Literal

ModeType = Literal["aeolian", "dorian", "lydian", "phrygian"]


@dataclass
class MusicalContext:
    """Encapsulates current generative parameters.

    Attributes:
        key: Root MIDI pitch (60=C, 62=D, 64=E, 67=G, 69=A)
        mode: Scale mode ("aeolian", "dorian", "lydian", "phrygian")
        bpm: Tempo in beats per minute (40-90)
        intensity: Note density multiplier (0.0-1.0)
        key_signature: Human-readable key (e.g., "C minor")
    """

    key: int
    mode: ModeType
    bpm: float
    intensity: float
    key_signature: str

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Validate key (allow all 12 chromatic keys: C=60 through B=71)
        if not (60 <= self.key <= 71):
            raise ValueError(
                f"Invalid key: {self.key} (must be 60-71, corresponding to C-B)"
            )

        # Validate mode
        valid_modes = {"aeolian", "dorian", "lydian", "phrygian"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid mode: {self.mode} (must be one of {valid_modes})"
            )

        # Validate BPM (expanded range for Phase 6 controls)
        if not (60 <= self.bpm <= 120):
            raise ValueError(f"Invalid BPM: {self.bpm} (must be 60-120)")

        # Validate intensity
        if not (0.0 <= self.intensity <= 1.0):
            raise ValueError(
                f"Invalid intensity: {self.intensity} (must be 0.0-1.0)"
            )

    @classmethod
    def default(cls) -> "MusicalContext":
        """Create default musical context.

        Returns:
            Default context: C minor Aeolian, 70 BPM, 0.5 intensity
        """
        return cls(
            key=60,  # C
            mode="aeolian",
            bpm=70.0,
            intensity=0.5,
            key_signature="C minor",
        )
