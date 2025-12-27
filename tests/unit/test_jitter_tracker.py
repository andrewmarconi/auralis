"""Unit test for JitterTracker EMA calculations."""

import pytest
import time
from statistics import mean, stdev

from server.buffer_management import JitterTracker


def test_ema_jitter_tracking():
    """Test EMA-based jitter calculation."""
    tracker = JitterTracker(window_size=50, alpha=0.1)

    # Simulate 100 chunks with varying jitter
    base_time = time.time()
    chunk_interval = 0.1  # 100ms per chunk

    for i in range(100):
        expected_time = base_time + (i * chunk_interval)

        # Add realistic jitter (±5-20ms)
        import random

        jitter_ms = random.uniform(-0.020, 0.020)
        actual_time = expected_time + jitter_ms

        tracker.record_chunk(expected_time, actual_time)

    # Verify jitter statistics
    mean_jitter = tracker.get_current_jitter()
    jitter_std = tracker.get_jitter_std()
    recommended_buffer = tracker.get_recommended_buffer_ms(confidence=0.95)
    tier_buffer = tracker.get_tier_buffer_ms()

    print(f"Mean jitter: {mean_jitter:.2f}ms")
    print(f"Jitter std dev: {jitter_std:.2f}ms")
    print(f"Recommended buffer (95% confidence): {recommended_buffer:.2f}ms")
    print(f"Tier-based buffer: {tier_buffer:.2f}ms")

    # Assertions
    assert 0 <= mean_jitter <= 30, f"Mean jitter out of range: {mean_jitter}ms"
    # Tier-based buffer should be in 200-3000ms range
    assert 200 <= tier_buffer <= 3000, f"Buffer recommendation out of range: {tier_buffer}ms"
    assert tracker.get_underrun_rate() == 0.0, "Should have zero underruns"

    print("✅ EMA jitter tracking working correctly")


def test_jitter_update_dynamics():
    """Test that EMA responds to changing jitter patterns."""
    tracker = JitterTracker(window_size=50, alpha=0.1)

    # Phase 1: Stable jitter (low)
    for i in range(25):
        expected_time = i * 0.1
        actual_time = expected_time + 0.005  # ±5ms jitter
        tracker.record_chunk(expected_time, actual_time)

    stable_jitter = tracker.get_current_jitter()
    assert stable_jitter < 10, f"Stable phase jitter too high: {stable_jitter}ms"

    # Phase 2: High jitter (network degradation)
    for i in range(25, 50):
        expected_time = i * 0.1
        actual_time = expected_time + 0.030  # ±30ms jitter
        tracker.record_chunk(expected_time, actual_time)

    high_jitter = tracker.get_current_jitter()
    assert high_jitter > stable_jitter, (
        f"High jitter phase should show increased jitter: {high_jitter:.2f}ms > {stable_jitter:.2f}ms"
    )

    # Phase 3: Return to stable (network recovery)
    for i in range(50, 75):
        expected_time = i * 0.1
        actual_time = expected_time + 0.005
        tracker.record_chunk(expected_time, actual_time)

    recovered_jitter = tracker.get_current_jitter()
    assert recovered_jitter < high_jitter, (
        f"Recovery phase should show decreased jitter: {recovered_jitter:.2f}ms < {high_jitter:.2f}ms"
    )

    print("✅ Jitter update dynamics working correctly")


def test_confidence_level_buffer_recommendation():
    """Test that higher confidence requires larger buffer."""
    tracker = JitterTracker(window_size=100, alpha=0.1)

    # Simulate jitter with known distribution (std=10ms)
    import random

    for i in range(100):
        expected_time = i * 0.1
        # Normal distribution with 10ms std
        jitter_ms = random.gauss(0, 0.010)
        actual_time = expected_time + jitter_ms
        tracker.record_chunk(expected_time, actual_time)

    # Test different confidence levels
    buffer_95 = tracker.get_recommended_buffer_ms(confidence=0.95)
    buffer_99 = tracker.get_recommended_buffer_ms(confidence=0.99)

    assert buffer_95 < buffer_99, (
        f"95% confidence buffer ({buffer_95:.2f}ms) should be smaller than 99% ({buffer_99:.2f}ms)"
    )

    assert buffer_95 >= tracker.get_current_jitter() + 2 * tracker.get_jitter_std(), (
        "95% buffer should cover mean + 2*std"
    )

    assert buffer_99 >= tracker.get_current_jitter() + 3 * tracker.get_jitter_std(), (
        "99% buffer should cover mean + 3*std"
    )

    print(f"95% buffer: {buffer_95:.2f}ms (mean + 2σ)")
    print(f"99% buffer: {buffer_99:.2f}ms (mean + 3σ)")
    print("✅ Confidence level buffer recommendations working correctly")


if __name__ == "__main__":
    test_ema_jitter_tracking()
    test_jitter_update_dynamics()
    test_confidence_level_buffer_recommendation()
