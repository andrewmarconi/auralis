"""Configuration management for Auralis server.

Loads and validates environment variables using Pydantic settings.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class AuralisConfig(BaseSettings):
    """Auralis server configuration loaded from environment variables."""

    # Server settings
    env: Literal["development", "production", "test"] = Field(
        default="development", alias="AURALIS_ENV"
    )
    host: str = Field(default="0.0.0.0", alias="AURALIS_HOST")
    port: int = Field(default=8000, alias="AURALIS_PORT", ge=1024, le=65535)

    # SoundFont paths
    soundfont_piano: Path = Field(
        default=Path("soundfonts/soundfonts/Salamander-Grand-Piano.sf2"),
        alias="SOUNDFONT_PIANO",
    )
    soundfont_gm: Path = Field(
        default=Path("soundfonts/soundfonts/FluidR3_GM.sf2"), alias="SOUNDFONT_GM"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="AURALIS_LOG_LEVEL"
    )

    # Performance settings
    synthesis_polyphony: int = Field(default=32, ge=8, le=256)
    reverb_enabled: bool = Field(default=True)
    reverb_room_size: float = Field(default=0.5, ge=0.0, le=1.0)
    reverb_damping: float = Field(default=0.5, ge=0.0, le=1.0)
    reverb_wet_level: float = Field(default=0.2, ge=0.0, le=1.0)
    ring_buffer_capacity: int = Field(default=20, ge=5, le=100)

    @field_validator("soundfont_piano", "soundfont_gm")
    @classmethod
    def validate_soundfont_path(cls, v: Path) -> Path:
        """Validate that SoundFont paths are relative or absolute paths."""
        # Don't check existence during config load - will validate at startup
        return v

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton configuration instance
_config: AuralisConfig | None = None


def get_config() -> AuralisConfig:
    """Get the global configuration instance.

    Returns:
        AuralisConfig: Configuration singleton
    """
    global _config
    if _config is None:
        _config = AuralisConfig()
    return _config
