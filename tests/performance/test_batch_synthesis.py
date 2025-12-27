"""
GPU Batch Synthesis Performance Benchmark

Tests the performance improvements from batch processing of chord rendering
on GPU vs. sequential processing.

Task: T063 [US3]
"""

import pytest
import torch
import numpy as np
import time
from typing import List, Tuple

from server.synthesis_engine import SynthesisEngine


class TestBatchSynthesisPerformance:
    """
    Test GPU batch synthesis performance vs. sequential processing.

    Success criteria (from spec.md SC-007):
    - Batch synthesis should reduce processing time by at least 40% vs. sequential
    - GPU acceleration should show clear performance benefit over CPU
    """

    @pytest.fixture
    def synthesis_engine(self):
        """Create synthesis engine instance for testing."""
        return SynthesisEngine(sample_rate=44100)

    @pytest.fixture
    def test_chords(self) -> List[Tuple[int, int, str]]:
        """Generate test chord progression (8 chords)."""
        # Format: (onset_sample, root_midi, chord_type)
        return [
            (0, 60, "major"),       # C major
            (11025, 62, "minor"),   # D minor
            (22050, 64, "major"),   # E major
            (33075, 65, "major"),   # F major
            (44100, 67, "major"),   # G major
            (55125, 69, "minor"),   # A minor
            (66150, 71, "dim"),     # B diminished
            (77175, 60, "major"),   # C major (resolution)
        ]

    def test_batch_vs_sequential_performance(self, synthesis_engine, test_chords):
        """
        Test that batch synthesis is at least 40% faster than sequential.

        FR-007: Hardware acceleration should reduce synthesis processing time.
        SC-007: At least 40% reduction compared to CPU-only processing.
        """
        duration_sec = 8.0
        sample_rate = 44100

        # Warm-up to ensure GPU is initialized
        _ = synthesis_engine.render_chords(
            test_chords[:2], duration_sec=2.0, bpm=70
        )

        # Measure sequential processing (simulated by rendering chords one-by-one)
        sequential_start = time.perf_counter()
        sequential_results = []
        for chord in test_chords:
            result = synthesis_engine.render_chords(
                [chord], duration_sec=1.0, bpm=70
            )
            sequential_results.append(result)
        sequential_time = time.perf_counter() - sequential_start

        # Measure batch processing (all chords at once)
        batch_start = time.perf_counter()
        batch_result = synthesis_engine.render_chords(
            test_chords, duration_sec=duration_sec, bpm=70
        )
        batch_time = time.perf_counter() - batch_start

        # Calculate performance improvement
        speedup_ratio = sequential_time / batch_time
        improvement_pct = ((sequential_time - batch_time) / sequential_time) * 100

        print(f"\nBatch Synthesis Performance:")
        print(f"  Sequential time: {sequential_time*1000:.2f}ms")
        print(f"  Batch time: {batch_time*1000:.2f}ms")
        print(f"  Speedup: {speedup_ratio:.2f}x")
        print(f"  Improvement: {improvement_pct:.1f}%")

        # EXPECTED TO FAIL until batch optimization is implemented
        assert improvement_pct >= 40.0, (
            f"Batch synthesis should be at least 40% faster than sequential. "
            f"Got {improvement_pct:.1f}% improvement (target: 40%)"
        )

    def test_gpu_acceleration_benefit(self, synthesis_engine, test_chords):
        """
        Test that GPU acceleration provides measurable performance benefit.

        FR-004: Use available hardware acceleration to reduce CPU load.
        """
        device_info = synthesis_engine.device_manager.get_device_info()

        # Skip if no GPU available
        if device_info["type"] == "cpu":
            pytest.skip("No GPU available for testing GPU acceleration")

        duration_sec = 4.0

        # Measure GPU synthesis time
        gpu_start = time.perf_counter()
        gpu_result = synthesis_engine.render_chords(
            test_chords[:4], duration_sec=duration_sec, bpm=70
        )
        gpu_time = time.perf_counter() - gpu_start

        # For comparison, we'd need CPU-only mode
        # This test validates GPU synthesis completes in acceptable time
        print(f"\nGPU Synthesis Performance:")
        print(f"  Device: {device_info['type']} - {device_info['name']}")
        print(f"  Synthesis time: {gpu_time*1000:.2f}ms for {duration_sec}s audio")
        print(f"  Real-time factor: {duration_sec/gpu_time:.2f}x")

        # Should synthesize faster than real-time (at least 2x for 4-chord phrase)
        assert gpu_time < duration_sec / 2, (
            f"GPU synthesis should be at least 2x faster than real-time. "
            f"Got {duration_sec/gpu_time:.2f}x"
        )

    def test_batch_size_scaling(self, synthesis_engine):
        """
        Test that batch processing scales well with different batch sizes.

        FR-010: Resource reduction through optimization.
        """
        batch_sizes = [1, 2, 4, 8, 16]
        timings = []

        for batch_size in batch_sizes:
            # Generate test chords for this batch size
            test_chords = [
                (i * 11025, 60 + (i % 12), "major")
                for i in range(batch_size)
            ]

            # Measure synthesis time
            start = time.perf_counter()
            _ = synthesis_engine.render_chords(
                test_chords,
                duration_sec=batch_size * 1.0,
                bpm=70
            )
            elapsed = time.perf_counter() - start

            timings.append({
                "batch_size": batch_size,
                "time_ms": elapsed * 1000,
                "time_per_chord_ms": (elapsed * 1000) / batch_size
            })

        print(f"\nBatch Size Scaling:")
        for timing in timings:
            print(f"  Batch {timing['batch_size']:2d}: "
                  f"{timing['time_ms']:6.2f}ms total, "
                  f"{timing['time_per_chord_ms']:5.2f}ms per chord")

        # EXPECTED TO FAIL until batch optimization is implemented
        # Time per chord should decrease as batch size increases
        time_per_chord_1 = timings[0]["time_per_chord_ms"]
        time_per_chord_16 = timings[-1]["time_per_chord_ms"]

        improvement = ((time_per_chord_1 - time_per_chord_16) / time_per_chord_1) * 100

        assert improvement >= 30.0, (
            f"Batch processing should improve per-chord time by at least 30% at larger batches. "
            f"Got {improvement:.1f}% improvement (batch-16 vs batch-1)"
        )

    def test_memory_allocation_efficiency(self, synthesis_engine, test_chords):
        """
        Test that batch synthesis uses memory efficiently without excessive allocations.

        FR-005: Maintain stable memory usage.
        """
        device_info = synthesis_engine.device_manager.get_device_info()

        # Skip if not on CUDA (MPS doesn't expose memory stats easily)
        if device_info["type"] != "cuda":
            pytest.skip("Memory tracking only available on CUDA devices")

        # Clear GPU cache before test
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB

        # Render multiple batches
        for _ in range(10):
            _ = synthesis_engine.render_chords(
                test_chords, duration_sec=8.0, bpm=70
            )

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        final_memory = torch.cuda.memory_allocated() / 1024**2  # MB

        memory_growth = final_memory - initial_memory

        print(f"\nMemory Allocation Efficiency:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Peak: {peak_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Growth: {memory_growth:.2f} MB")

        # Memory should not grow significantly across batches
        assert memory_growth < 10.0, (
            f"Memory growth should be minimal (<10MB) across multiple batches. "
            f"Got {memory_growth:.2f}MB growth"
        )


@pytest.mark.benchmark
class TestBatchSynthesisBenchmark:
    """
    Detailed benchmarks for batch synthesis optimization.

    These tests are for profiling and optimization work, not CI.
    """

    def test_large_batch_benchmark(self):
        """Benchmark synthesis of large chord progression (32 chords)."""
        engine = SynthesisEngine(sample_rate=44100)

        # Generate 32-chord progression
        chords = [
            (i * 11025, 60 + (i % 12), "major")
            for i in range(32)
        ]

        start = time.perf_counter()
        result = engine.render_chords(chords, duration_sec=32.0, bpm=70)
        elapsed = time.perf_counter() - start

        print(f"\nLarge Batch Benchmark (32 chords):")
        print(f"  Total time: {elapsed*1000:.2f}ms")
        print(f"  Time per chord: {(elapsed*1000)/32:.2f}ms")
        print(f"  Real-time factor: {32.0/elapsed:.2f}x")

        # Should still be faster than real-time
        assert elapsed < 32.0 / 2, "Large batch should synthesize at least 2x faster than real-time"
