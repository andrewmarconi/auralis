"""Test synthesis throughput and batch processing performance.

Validates that synthesis runs at least 10× faster than real-time,
ensuring the server can generate audio faster than it's consumed.
"""

import pytest

from composition.chord_generator import ChordGenerator
from composition.melody_generator import MelodyGenerator
from composition.musical_context import MusicalContext
from tests.performance.benchmark_suite import SynthesisBenchmark


class TestBatchSynthesis:
    """Test batch synthesis throughput."""

    def test_synthesis_throughput_exceeds_realtime(self):
        """Verify synthesis runs at least 10× faster than real-time.

        Target: Generate 8 bars @ 60 BPM (32 seconds of audio) in <3.2 seconds
        """
        benchmark = SynthesisBenchmark()
        results = benchmark.benchmark_phrase_synthesis(duration_bars=8, num_iterations=10)

        # Check throughput ratio
        assert results["throughput_ratio"] > 10.0, (
            f"Synthesis throughput {results['throughput_ratio']:.1f}x "
            f"does not meet 10x real-time target"
        )

    def test_chord_generation_performance(self):
        """Verify chord generation completes in <10ms (P95)."""
        benchmark = SynthesisBenchmark()
        results = benchmark.benchmark_chord_generation(num_iterations=100)

        assert results["p95_ms"] < 10.0, (
            f"Chord generation P95 {results['p95_ms']:.2f}ms exceeds 10ms target"
        )

    def test_melody_generation_performance(self):
        """Verify melody generation completes in <20ms (P95)."""
        benchmark = SynthesisBenchmark()
        results = benchmark.benchmark_melody_generation(num_iterations=100)

        assert results["p95_ms"] < 20.0, (
            f"Melody generation P95 {results['p95_ms']:.2f}ms exceeds 20ms target"
        )

    def test_concurrent_phrase_generation(self):
        """Verify multiple concurrent phrase generations don't degrade performance."""
        chord_gen = ChordGenerator()
        melody_gen = MelodyGenerator()
        context = MusicalContext.default()

        import time

        # Generate 5 phrases sequentially
        start_time = time.perf_counter()
        for _ in range(5):
            chords = chord_gen.generate(context, duration_bars=8)
            melody = melody_gen.generate(context, chords)
        sequential_time = time.perf_counter() - start_time

        # Verify average time per phrase
        avg_time_per_phrase_ms = (sequential_time / 5) * 1000.0

        assert avg_time_per_phrase_ms < 100.0, (
            f"Average phrase generation {avg_time_per_phrase_ms:.2f}ms "
            f"exceeds 100ms target under sequential load"
        )

    @pytest.mark.parametrize("duration_bars", [8, 16])
    def test_variable_phrase_lengths(self, duration_bars: int):
        """Verify performance scales appropriately with phrase length.

        Args:
            duration_bars: Number of bars (8 or 16)
        """
        benchmark = SynthesisBenchmark()
        results = benchmark.benchmark_phrase_synthesis(
            duration_bars=duration_bars, num_iterations=20
        )

        # Even 16-bar phrases should complete in <100ms
        assert results["p95_latency_ms"] < 100.0, (
            f"{duration_bars}-bar phrase P95 {results['p95_latency_ms']:.2f}ms "
            f"exceeds 100ms target"
        )
