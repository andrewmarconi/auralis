"""
torch.compile() Optimization Performance Benchmark

Tests the performance improvements from using torch.compile() on synthesis methods.
Requires PyTorch 2.0+.

Task: T064 [US3]
"""

import pytest
import torch
import time
import sys

from server.synthesis_engine import SynthesisEngine


# Check if torch.compile is available (PyTorch 2.0+)
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile') and sys.version_info >= (3, 8)


class TestTorchCompileOptimization:
    """
    Test torch.compile() performance optimization for synthesis methods.

    Success criteria (from research.md):
    - torch.compile should provide 20-30% latency reduction for synthesis
    - Compiled methods should maintain audio quality (no numerical degradation)
    """

    @pytest.fixture
    def synthesis_engine(self):
        """Create synthesis engine instance for testing."""
        return SynthesisEngine(sample_rate=44100)

    @pytest.fixture
    def test_chords(self):
        """Generate test chord progression."""
        return [
            (0, 60, "major"),
            (11025, 62, "minor"),
            (22050, 64, "major"),
            (33075, 65, "major"),
        ]

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE,
        reason="torch.compile requires PyTorch 2.0+ and Python 3.8+"
    )
    def test_compile_performance_improvement(self, synthesis_engine, test_chords):
        """
        Test that torch.compile provides at least 20% performance improvement.

        FR-010: Reduce resource consumption by at least 30% vs. baseline.
        Research: torch.compile expected to provide 20-30% latency reduction.
        """
        duration_sec = 4.0
        num_runs = 10

        # Baseline: Non-compiled synthesis
        baseline_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = synthesis_engine.render_chords(
                test_chords, duration_sec=duration_sec, bpm=70
            )
            baseline_times.append(time.perf_counter() - start)

        baseline_avg = sum(baseline_times) / len(baseline_times)

        # EXPECTED TO FAIL: Compiled synthesis not yet implemented
        # This test validates that once implemented, compile provides speedup

        # For now, we simulate what the test should check:
        # 1. Engine should have compiled synthesis methods
        # 2. Compiled methods should be faster than non-compiled

        # Check if engine has compiled synthesis attribute
        has_compiled_synthesis = hasattr(synthesis_engine, '_compiled_render_chords')

        if not has_compiled_synthesis:
            pytest.fail(
                "SynthesisEngine should have compiled synthesis methods. "
                "Expected attribute: _compiled_render_chords"
            )

        # Measure compiled synthesis performance
        compiled_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = synthesis_engine._compiled_render_chords(
                test_chords, duration_sec=duration_sec, bpm=70
            )
            compiled_times.append(time.perf_counter() - start)

        compiled_avg = sum(compiled_times) / len(compiled_times)

        # Calculate improvement
        improvement_pct = ((baseline_avg - compiled_avg) / baseline_avg) * 100

        print(f"\ntorch.compile Performance:")
        print(f"  Baseline avg: {baseline_avg*1000:.2f}ms")
        print(f"  Compiled avg: {compiled_avg*1000:.2f}ms")
        print(f"  Improvement: {improvement_pct:.1f}%")
        print(f"  Speedup: {baseline_avg/compiled_avg:.2f}x")

        assert improvement_pct >= 20.0, (
            f"torch.compile should provide at least 20% performance improvement. "
            f"Got {improvement_pct:.1f}% (target: 20%)"
        )

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE,
        reason="torch.compile requires PyTorch 2.0+ and Python 3.8+"
    )
    def test_compile_audio_quality_preservation(self, synthesis_engine, test_chords):
        """
        Test that torch.compile preserves audio quality (no numerical errors).

        FR-010: Optimizations should not compromise audio quality.
        """
        duration_sec = 2.0

        # Render with non-compiled method
        baseline_audio = synthesis_engine.render_chords(
            test_chords[:2], duration_sec=duration_sec, bpm=70
        )

        # EXPECTED TO FAIL until compiled synthesis is implemented
        if not hasattr(synthesis_engine, '_compiled_render_chords'):
            pytest.fail("Compiled synthesis method not yet implemented")

        # Render with compiled method
        compiled_audio = synthesis_engine._compiled_render_chords(
            test_chords[:2], duration_sec=duration_sec, bpm=70
        )

        # Audio should be numerically very similar (allowing for floating point precision)
        max_diff = abs(baseline_audio - compiled_audio).max()
        mean_diff = abs(baseline_audio - compiled_audio).mean()

        print(f"\nAudio Quality Preservation:")
        print(f"  Max difference: {max_diff}")
        print(f"  Mean difference: {mean_diff}")

        # Differences should be minimal (floating point precision)
        assert max_diff < 100, (
            f"Compiled synthesis should produce nearly identical audio. "
            f"Max difference: {max_diff} (threshold: 100)"
        )
        assert mean_diff < 1.0, (
            f"Compiled synthesis should have minimal mean difference. "
            f"Mean difference: {mean_diff} (threshold: 1.0)"
        )

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE,
        reason="torch.compile requires PyTorch 2.0+ and Python 3.8+"
    )
    def test_compile_warmup_overhead(self, synthesis_engine, test_chords):
        """
        Test that torch.compile warmup overhead is acceptable.

        Note: First compilation is slow, but subsequent calls are fast.
        """
        if not hasattr(synthesis_engine, '_compiled_render_chords'):
            pytest.skip("Compiled synthesis not yet implemented")

        duration_sec = 1.0

        # First call (includes compilation overhead)
        warmup_start = time.perf_counter()
        _ = synthesis_engine._compiled_render_chords(
            test_chords[:1], duration_sec=duration_sec, bpm=70
        )
        warmup_time = time.perf_counter() - warmup_start

        # Subsequent calls (no compilation overhead)
        steady_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = synthesis_engine._compiled_render_chords(
                test_chords[:1], duration_sec=duration_sec, bpm=70
            )
            steady_times.append(time.perf_counter() - start)

        steady_avg = sum(steady_times) / len(steady_times)

        print(f"\nCompilation Warmup:")
        print(f"  First call (with compilation): {warmup_time*1000:.2f}ms")
        print(f"  Steady state avg: {steady_avg*1000:.2f}ms")
        print(f"  Warmup overhead: {(warmup_time - steady_avg)*1000:.2f}ms")

        # Warmup should complete in reasonable time (< 5 seconds)
        assert warmup_time < 5.0, (
            f"Compilation warmup should complete in < 5 seconds. "
            f"Got {warmup_time:.2f}s"
        )

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE,
        reason="torch.compile requires PyTorch 2.0+ and Python 3.8+"
    )
    def test_device_compatibility(self, synthesis_engine):
        """
        Test that torch.compile works correctly on different devices (Metal/CUDA/CPU).
        """
        device_info = synthesis_engine.device_manager.get_device_info()

        print(f"\nDevice Compatibility:")
        print(f"  Device: {device_info['type']} - {device_info['name']}")

        test_chords = [(0, 60, "major")]

        try:
            # Should not raise errors on any device
            _ = synthesis_engine.render_chords(
                test_chords, duration_sec=0.5, bpm=70
            )

            if hasattr(synthesis_engine, '_compiled_render_chords'):
                _ = synthesis_engine._compiled_render_chords(
                    test_chords, duration_sec=0.5, bpm=70
                )

            print(f"  ✓ Synthesis works on {device_info['type']}")

        except Exception as e:
            pytest.fail(f"Synthesis failed on {device_info['type']}: {e}")


@pytest.mark.benchmark
class TestTorchCompileBenchmark:
    """
    Detailed benchmarks for torch.compile optimization.

    These tests are for profiling and optimization work, not CI.
    """

    @pytest.mark.skipif(
        not TORCH_COMPILE_AVAILABLE,
        reason="torch.compile requires PyTorch 2.0+"
    )
    def test_compile_modes_comparison(self):
        """
        Compare different torch.compile modes (default, reduce-overhead, max-autotune).
        """
        engine = SynthesisEngine(sample_rate=44100)
        test_chords = [(i * 11025, 60, "major") for i in range(4)]

        # This test would compare different compile modes
        # For now, just document that this should be tested

        print("\ntorch.compile Modes:")
        print("  default: Balanced compilation")
        print("  reduce-overhead: Minimize compilation overhead")
        print("  max-autotune: Maximum optimization (slower compile)")

        # EXPECTED TO FAIL until implementation
        pytest.skip("torch.compile mode comparison not yet implemented")
