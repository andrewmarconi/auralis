"""Base exception classes for Auralis error handling."""


class AuralisError(Exception):
    """Base exception for all Auralis errors."""

    pass


class BufferError(AuralisError):
    """Base exception for buffer-related errors."""

    pass


class BufferFullError(BufferError):
    """Raised when attempting to write to full buffer."""

    pass


class BufferUnderrunError(BufferError):
    """Raised when buffer underruns during read."""

    pass


class SynthesisError(AuralisError):
    """Base exception for synthesis-related errors."""

    pass


class DeviceError(SynthesisError):
    """Raised when GPU device initialization fails."""

    pass


class EncodingError(AuralisError):
    """Raised when chunk encoding/decoding fails."""

    pass


class ConnectionError(AuralisError):
    """Raised when WebSocket connection fails."""

    pass


class RateLimitError(AuralisError):
    """Raised when rate limit exceeded."""

    pass
