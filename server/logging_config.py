"""Logging configuration for performance monitoring."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging for Auralis performance monitoring."""

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger("auralis")
    logger.setLevel(numeric_level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    file_handler = RotatingFileHandler(
        logs_dir / "auralis.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(f"auralis.{name}")
