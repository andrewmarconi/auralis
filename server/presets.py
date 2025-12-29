"""Musical presets for Auralis.

Predefined parameter combinations for different listening contexts:
Focus, Meditation, Sleep, Bright.
"""

import logging
from typing import Dict

from composition.musical_context import MusicalContext

logger = logging.getLogger(__name__)


# Preset definitions
PRESETS: Dict[str, Dict[str, any]] = {
    "focus": {
        "name": "Focus",
        "description": "Balanced Dorian mode for concentration and productivity",
        "key": 62,  # D
        "mode": "dorian",
        "bpm": 60.0,
        "intensity": 0.5,
        "key_signature": "D Dorian",
    },
    "meditation": {
        "name": "Meditation",
        "description": "Calming Aeolian mode for deep relaxation",
        "key": 60,  # C
        "mode": "aeolian",
        "bpm": 60.0,
        "intensity": 0.3,
        "key_signature": "C Aeolian (Natural Minor)",
    },
    "sleep": {
        "name": "Sleep",
        "description": "Minimal Phrygian mode for rest and sleep",
        "key": 64,  # E
        "mode": "phrygian",
        "bpm": 60.0,
        "intensity": 0.2,
        "key_signature": "E Phrygian",
    },
    "bright": {
        "name": "Bright",
        "description": "Uplifting Lydian mode for energy and positivity",
        "key": 67,  # G
        "mode": "lydian",
        "bpm": 70.0,
        "intensity": 0.6,
        "key_signature": "G Lydian",
    },
}


def get_preset(preset_name: str) -> MusicalContext:
    """Get preset by name.

    Args:
        preset_name: Preset name ("focus", "meditation", "sleep", "bright")

    Returns:
        MusicalContext configured for the preset

    Raises:
        KeyError: If preset name not found
    """
    preset_name_lower = preset_name.lower()

    if preset_name_lower not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(
            f"Unknown preset: {preset_name}. Available presets: {available}"
        )

    preset = PRESETS[preset_name_lower]

    logger.info(f"Loading preset: {preset['name']} - {preset['description']}")

    return MusicalContext(
        key=preset["key"],
        mode=preset["mode"],
        bpm=preset["bpm"],
        intensity=preset["intensity"],
        key_signature=preset["key_signature"],
    )


def list_presets() -> list[Dict[str, str]]:
    """List all available presets.

    Returns:
        List of preset metadata dictionaries
    """
    return [
        {
            "id": preset_id,
            "name": preset["name"],
            "description": preset["description"],
            "key_signature": preset["key_signature"],
            "bpm": str(preset["bpm"]),
            "intensity": str(preset["intensity"]),
        }
        for preset_id, preset in PRESETS.items()
    ]


def get_default_preset() -> MusicalContext:
    """Get the default preset (Focus).

    Returns:
        MusicalContext for Focus preset
    """
    return get_preset("focus")
