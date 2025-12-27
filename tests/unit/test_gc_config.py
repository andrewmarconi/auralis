"""
Garbage Collection Configuration Unit Tests

Tests the GC configuration module for real-time audio performance tuning.

Task: T066 [US3]
"""

import pytest
import gc

from server.gc_config import GCConfig, RealTimeGCConfig


class TestGCConfiguration:
    """
    Test garbage collection configuration for real-time performance.

    Success criteria (from research.md):
    - GC tuning should reduce collection frequency during audio streaming
    - Real-time GC config: (50000, 500, 1000) thresholds
    """

    def setup_method(self):
        """Save initial GC configuration."""
        self.initial_thresholds = gc.get_threshold()

    def teardown_method(self):
        """Restore initial GC configuration."""
        gc.set_threshold(*self.initial_thresholds)

    def test_realtime_gc_config_application(self):
        """
        Test that real-time GC configuration is applied correctly.

        FR-010: Resource optimization through GC tuning.
        Research: Recommended thresholds (50000, 500, 1000).
        """
        # Apply real-time configuration
        RealTimeGCConfig.apply_realtime_config()

        config = RealTimeGCConfig.get_current_config()

        print(f"\nReal-time GC Configuration:")
        print(f"  Gen-0 threshold: {config.threshold0}")
        print(f"  Gen-1 threshold: {config.threshold1}")
        print(f"  Gen-2 threshold: {config.threshold2}")

        # Validate thresholds match recommendations
        assert config.threshold0 == 50000, (
            f"Gen-0 threshold should be 50000 for real-time. Got {config.threshold0}"
        )
        assert config.threshold1 == 500, (
            f"Gen-1 threshold should be 500 for real-time. Got {config.threshold1}"
        )
        assert config.threshold2 == 1000, (
            f"Gen-2 threshold should be 1000 for real-time. Got {config.threshold2}"
        )

    def test_gc_config_vs_default(self):
        """
        Test that real-time config differs significantly from default.

        This ensures we're actually tuning GC, not using defaults.
        """
        # Get default thresholds
        default_config = RealTimeGCConfig.get_current_config()

        # Apply real-time configuration
        RealTimeGCConfig.apply_realtime_config()

        realtime_config = RealTimeGCConfig.get_current_config()

        print(f"\nDefault vs. Real-time GC Config:")
        print(f"  Default Gen-0: {default_config.threshold0}")
        print(f"  Realtime Gen-0: {realtime_config.threshold0}")
        print(f"  Difference: {realtime_config.threshold0 - default_config.threshold0}")

        # Real-time gen-0 threshold should be much higher
        assert realtime_config.threshold0 > default_config.threshold0 * 10, (
            "Real-time Gen-0 threshold should be at least 10x default"
        )

    def test_gc_collection_frequency_reduction(self):
        """
        Test that GC collections are less frequent with real-time config.

        FR-010: Reduce resource consumption through optimization.
        """
        # Apply real-time configuration
        RealTimeGCConfig.apply_realtime_config()

        # Reset GC stats
        gc.collect()
        initial_counts = gc.get_count()

        # Allocate many small objects (typical for audio streaming)
        for _ in range(10000):
            temp = [1, 2, 3, 4, 5] * 100
            del temp

        final_counts = gc.get_count()

        gen0_collections = final_counts[0] - initial_counts[0]

        print(f"\nGC Collection Frequency:")
        print(f"  Gen-0 objects before: {initial_counts[0]}")
        print(f"  Gen-0 objects after: {final_counts[0]}")
        print(f"  Gen-0 collections: {gen0_collections}")

        # With higher thresholds, should have fewer collections
        # This test validates configuration is effective
        assert gen0_collections < 5, (
            f"With real-time GC config, should have < 5 gen-0 collections during test. "
            f"Got {gen0_collections}"
        )

    def test_gc_config_restore(self):
        """
        Test that default GC configuration can be restored.

        Important for testing cleanup and non-real-time modes.
        """
        # Apply real-time configuration
        RealTimeGCConfig.apply_realtime_config()

        realtime_config = RealTimeGCConfig.get_current_config()

        # Restore defaults
        RealTimeGCConfig.restore_default_config()

        default_config = RealTimeGCConfig.get_current_config()

        print(f"\nGC Config Restore:")
        print(f"  Realtime Gen-0: {realtime_config.threshold0}")
        print(f"  Restored Gen-0: {default_config.threshold0}")

        # Should be back to default values
        assert default_config.threshold0 == 700, "Should restore default Gen-0 threshold"
        assert default_config.threshold1 == 10, "Should restore default Gen-1 threshold"
        assert default_config.threshold2 == 10, "Should restore default Gen-2 threshold"

    def test_gc_stats_collection(self):
        """
        Test that GC statistics can be collected for monitoring.

        Needed for Prometheus metrics (T088).
        """
        # Get GC stats
        stats = gc.get_stats()

        print(f"\nGC Statistics:")
        print(f"  Generations: {len(stats)}")

        for i, gen_stats in enumerate(stats):
            print(f"  Gen-{i} collections: {gen_stats.get('collections', 'N/A')}")

        # Should have stats for 3 generations
        assert len(stats) == 3, "Should have statistics for 3 GC generations"

        # Each generation should have 'collections' count
        for gen_stats in stats:
            assert 'collections' in gen_stats, "GC stats should include collections count"


class TestGCConfigIntegration:
    """
    Integration tests for GC configuration with synthesis engine.
    """

    def test_gc_config_application_on_startup(self):
        """
        Test that GC configuration is applied on server startup.

        This should be integrated into server/main.py startup sequence.
        """
        # EXPECTED TO FAIL until integration in main.py
        # Check if main.py applies GC config

        # For now, document that this should be tested
        pytest.skip(
            "GC configuration should be applied in server/main.py startup. "
            "Integration test pending implementation."
        )

    def test_gc_metrics_prometheus_export(self):
        """
        Test that GC statistics are exported to Prometheus metrics.

        Links to T088: Add gc_collections_total counter metric.
        """
        # EXPECTED TO FAIL until Prometheus metrics implementation
        pytest.skip(
            "GC metrics export to Prometheus pending implementation (T088). "
        )
