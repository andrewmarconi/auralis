"""Custom exceptions for Auralis server."""


class AuralisError(Exception):
    """Base exception for all Auralis errors."""

    pass


class GenerationError(AuralisError):
    """Error during musical composition generation."""

    pass


class SynthesisError(AuralisError):
    """Error during audio synthesis/rendering."""

    pass


class SoundFontLoadError(SynthesisError):
    """Error loading SoundFont file."""

    pass


class PresetError(SynthesisError):
    """Error selecting or using SoundFont preset."""

    pass


class RenderError(SynthesisError):
    """Error during audio rendering."""

    pass


class BufferError(AuralisError):
    """Error in ring buffer operations."""

    pass


class ConfigurationError(AuralisError):
    """Error in server configuration."""

    pass
