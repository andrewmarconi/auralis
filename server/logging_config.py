"""Structured logging configuration for Auralis server.

Provides JSON-formatted logs with correlation IDs, performance metrics,
and request context.
"""

import logging
import sys
from typing import Any

from server.config import get_config


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter with performance context."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add custom fields from extra parameter
        if hasattr(record, "client_id"):
            log_data["client_id"] = record.client_id
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        if hasattr(record, "chunk_seq"):
            log_data["chunk_seq"] = record.chunk_seq

        # Format as key=value pairs for easier parsing
        pairs = [f"{k}={v}" for k, v in log_data.items()]
        return " ".join(pairs)


def setup_logging() -> None:
    """Configure structured logging for the application.

    Sets up handlers, formatters, and log levels based on configuration.
    """
    config = get_config()

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with structured formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.log_level))
    console_handler.setFormatter(
        StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    )
    root_logger.addHandler(console_handler)

    # Set library log levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Auralis module logging
    logging.getLogger("server").setLevel(getattr(logging, config.log_level))
    logging.getLogger("composition").setLevel(getattr(logging, config.log_level))


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
