"""Garbage collection tuning for real-time audio performance.

Reduces GC pauses by increasing collection thresholds, minimizing
interruptions during audio synthesis and streaming.
"""

import gc
import logging

logger = logging.getLogger(__name__)

# GC threshold configuration
# Default Python: (700, 10, 10) - very aggressive
# Auralis tuned: (10000, 20, 20) - ~14× fewer gen0 collections
GC_THRESHOLD_GEN0 = 10000  # Collect after 10k allocations (vs 700 default)
GC_THRESHOLD_GEN1 = 20  # Collect after 20 gen0 collections (vs 10)
GC_THRESHOLD_GEN2 = 20  # Collect after 20 gen1 collections (vs 10)


def configure_gc() -> None:
    """Configure garbage collection for real-time audio performance.

    Sets GC thresholds to reduce collection frequency:
    - Generation 0: 10,000 allocations (vs 700 default)
    - Generation 1: 20 gen0 collections (vs 10)
    - Generation 2: 20 gen1 collections (vs 10)

    Results in ~14× fewer gen0 collections, reducing pause frequency
    from ~every 50ms to ~every 700ms.

    Call this once at server startup before audio processing begins.
    """
    old_thresholds = gc.get_threshold()
    gc.set_threshold(GC_THRESHOLD_GEN0, GC_THRESHOLD_GEN1, GC_THRESHOLD_GEN2)
    new_thresholds = gc.get_threshold()

    logger.info(
        f"GC thresholds updated: {old_thresholds} → {new_thresholds} "
        f"(~14× reduction in gen0 collections)"
    )


def get_gc_stats() -> dict[str, int]:
    """Get current garbage collection statistics.

    Returns:
        Dictionary with collection counts:
        - gen0: Generation 0 collections (short-lived objects)
        - gen1: Generation 1 collections
        - gen2: Generation 2 collections (expensive, long-lived objects)
    """
    counts = gc.get_count()
    stats = gc.get_stats()

    return {
        "gen0": stats[0]["collections"] if stats else 0,
        "gen1": stats[1]["collections"] if len(stats) > 1 else 0,
        "gen2": stats[2]["collections"] if len(stats) > 2 else 0,
        "current_counts": {
            "gen0": counts[0],
            "gen1": counts[1],
            "gen2": counts[2],
        },
    }


def disable_gc_during_synthesis() -> None:
    """Temporarily disable GC during synthesis (EXPERIMENTAL).

    WARNING: Only use for short synthesis bursts (<1 second).
    Must manually re-enable with gc.enable() afterward.

    Not recommended for MVP - GC tuning via configure_gc() is sufficient.
    """
    gc.disable()
    logger.warning("GC disabled - must manually re-enable after synthesis")


def enable_gc() -> None:
    """Re-enable garbage collection after disable_gc_during_synthesis()."""
    gc.enable()
    logger.info("GC re-enabled")
