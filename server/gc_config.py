"""
Garbage Collection Configuration for Real-Time Audio Performance

Implements GC tuning to minimize collection frequency during audio streaming,
reducing latency spikes and ensuring smooth playback.

Tasks: T080-T083 [US3]
"""

import gc
from dataclasses import dataclass
from loguru import logger


@dataclass
class GCConfig:
    """
    Garbage collection configuration thresholds.

    Attributes:
        threshold0: Gen-0 collection threshold (minor collections)
        threshold1: Gen-1 collection threshold (intermediate collections)
        threshold2: Gen-2 collection threshold (major collections)
    """
    threshold0: int
    threshold1: int
    threshold2: int


class RealTimeGCConfig:
    """
    Garbage collection tuning for real-time audio streaming (T081).

    Increases GC thresholds to reduce collection frequency during audio processing,
    minimizing latency spikes while maintaining acceptable memory overhead.

    Based on research.md recommendations:
    - Gen-0: 50000 (vs default 700) - reduce minor collections
    - Gen-1: 500 (vs default 10) - balanced intermediate collections
    - Gen-2: 1000 (vs default 10) - minimize major collections
    """

    # Real-time optimized thresholds (from research.md)
    REALTIME_THRESHOLDS = GCConfig(
        threshold0=50000,  # Reduce gen-0 frequency significantly
        threshold1=500,    # Moderate gen-1 frequency
        threshold2=1000,   # Minimize expensive gen-2 collections
    )

    # Python default thresholds (for reference)
    DEFAULT_THRESHOLDS = GCConfig(
        threshold0=700,
        threshold1=10,
        threshold2=10,
    )

    @staticmethod
    def apply_realtime_config() -> None:
        """
        Apply GC configuration optimized for real-time audio (T081).

        This increases collection thresholds to reduce GC frequency during
        audio streaming, minimizing latency spikes while accepting slightly
        higher memory overhead.

        Call this during server startup (integrate in main.py - T082).
        """
        config = RealTimeGCConfig.REALTIME_THRESHOLDS

        gc.set_threshold(
            config.threshold0,
            config.threshold1,
            config.threshold2
        )

        logger.info(
            f"✓ Real-time GC configuration applied: "
            f"({config.threshold0}, {config.threshold1}, {config.threshold2})"
        )

    @staticmethod
    def get_current_config() -> GCConfig:
        """
        Get current GC thresholds.

        Returns:
            GCConfig with current threshold values
        """
        thresholds = gc.get_threshold()
        return GCConfig(
            threshold0=thresholds[0],
            threshold1=thresholds[1],
            threshold2=thresholds[2]
        )

    @staticmethod
    def restore_default_config() -> None:
        """
        Restore Python default GC configuration.

        Useful for testing or non-real-time modes.
        """
        config = RealTimeGCConfig.DEFAULT_THRESHOLDS

        gc.set_threshold(
            config.threshold0,
            config.threshold1,
            config.threshold2
        )

        logger.info(
            f"✓ Default GC configuration restored: "
            f"({config.threshold0}, {config.threshold1}, {config.threshold2})"
        )

    @staticmethod
    def get_gc_stats() -> dict:
        """
        Get current GC statistics for monitoring (T083).

        Returns detailed statistics for all 3 generations, useful for
        Prometheus metrics export.

        Returns:
            Dictionary with GC statistics by generation
        """
        stats = gc.get_stats()

        return {
            "gen0": {
                "collections": stats[0].get("collections", 0),
                "collected": stats[0].get("collected", 0),
                "uncollectable": stats[0].get("uncollectable", 0),
            },
            "gen1": {
                "collections": stats[1].get("collections", 0),
                "collected": stats[1].get("collected", 0),
                "uncollectable": stats[1].get("uncollectable", 0),
            },
            "gen2": {
                "collections": stats[2].get("collections", 0),
                "collected": stats[2].get("collected", 0),
                "uncollectable": stats[2].get("uncollectable", 0),
            },
            "current_config": {
                "threshold0": gc.get_threshold()[0],
                "threshold1": gc.get_threshold()[1],
                "threshold2": gc.get_threshold()[2],
            }
        }

    @staticmethod
    def get_collection_counts() -> tuple[int, int, int]:
        """
        Get collection counts for all generations.

        Returns:
            Tuple of (gen0_count, gen1_count, gen2_count)
        """
        counts = gc.get_count()
        return (counts[0], counts[1], counts[2])

    @staticmethod
    def disable_gc() -> None:
        """
        Disable automatic garbage collection.

        Use with extreme caution - only for critical real-time sections.
        Must manually call gc.collect() periodically to prevent memory leaks.
        """
        gc.disable()
        logger.warning("⚠ Automatic garbage collection DISABLED - manual collection required")

    @staticmethod
    def enable_gc() -> None:
        """
        Re-enable automatic garbage collection.
        """
        gc.enable()
        logger.info("✓ Automatic garbage collection enabled")

    @staticmethod
    def is_gc_enabled() -> bool:
        """
        Check if automatic GC is currently enabled.

        Returns:
            True if GC is enabled, False otherwise
        """
        return gc.isenabled()
