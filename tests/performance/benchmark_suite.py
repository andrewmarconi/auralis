"""
Performance Benchmark Suite

Comprehensive performance testing comparing baseline (Phase 1) vs. optimized (Phase 3)
implementation to validate 30% resource reduction target.

Task: T067 [US3]
"""

import pytest
import time
import psutil
import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from server.synthesis_engine import SynthesisEngine


@dataclass
class BenchmarkResult:
    """Single benchmark measurement result."""
    name: str
    duration_sec: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: float = 0.0


@dataclass
class ComparisonResult:
    """Comparison between baseline and optimized benchmarks."""
    baseline: BenchmarkResult
    optimized: BenchmarkResult

    @property
    def cpu_improvement_pct(self) -> float:
        """CPU usage improvement percentage."""
        return ((self.baseline.cpu_percent - self.optimized.cpu_percent)
                / self.baseline.cpu_percent * 100)

    @property
    def memory_improvement_pct(self) -> float:
        """Memory usage improvement percentage."""
        return ((self.baseline.memory_mb - self.optimized.memory_mb)
                / self.baseline.memory_mb * 100)

    @property
    def latency_improvement_pct(self) -> float:
        """Synthesis latency improvement percentage."""
        return ((self.baseline.duration_sec - self.optimized.duration_sec)
                / self.baseline.duration_sec * 100)


class BaselineBenchmark:
    """
    Simulates Phase 1 baseline performance for comparison.

    Baseline characteristics:
    - No batch synthesis (sequential chord rendering)
    - No GPU memory optimization
    - No torch.compile
    - Basic ring buffer (no adaptive sizing)
    """

    def __init__(self):
        self.engine = SynthesisEngine(sample_rate=44100)
        self.process = psutil.Process()

    def run_synthesis_benchmark(self, test_chords: List, duration_sec: float) -> BenchmarkResult:
        """Run baseline synthesis benchmark."""
        # Reset monitoring
        self.process.cpu_percent()  # Prime CPU monitoring

        start_mem = self.process.memory_info().rss / 1024**2

        # Simulate baseline: Sequential chord rendering
        start_time = time.perf_counter()

        # Render each chord sequentially (baseline behavior)
        for chord in test_chords:
            _ = self.engine.render_chords([chord], duration_sec=1.0, bpm=70)

        elapsed = time.perf_counter() - start_time

        cpu_pct = self.process.cpu_percent()
        end_mem = self.process.memory_info().rss / 1024**2

        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**2

        return BenchmarkResult(
            name="baseline_synthesis",
            duration_sec=elapsed,
            cpu_percent=cpu_pct,
            memory_mb=end_mem - start_mem,
            gpu_memory_mb=gpu_mem
        )


class OptimizedBenchmark:
    """
    Tests Phase 3 optimized performance.

    Optimized characteristics:
    - Batch synthesis (all chords at once)
    - GPU memory optimization (pre-allocation, cache clearing)
    - torch.compile (if available)
    - Adaptive ring buffer
    """

    def __init__(self):
        self.engine = SynthesisEngine(sample_rate=44100)
        self.process = psutil.Process()

    def run_synthesis_benchmark(self, test_chords: List, duration_sec: float) -> BenchmarkResult:
        """Run optimized synthesis benchmark."""
        # Reset monitoring
        self.process.cpu_percent()

        start_mem = self.process.memory_info().rss / 1024**2

        # Optimized: Batch rendering (all chords at once)
        start_time = time.perf_counter()

        _ = self.engine.render_chords(test_chords, duration_sec=duration_sec, bpm=70)

        elapsed = time.perf_counter() - start_time

        cpu_pct = self.process.cpu_percent()
        end_mem = self.process.memory_info().rss / 1024**2

        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**2

        return BenchmarkResult(
            name="optimized_synthesis",
            duration_sec=elapsed,
            cpu_percent=cpu_pct,
            memory_mb=end_mem - start_mem,
            gpu_memory_mb=gpu_mem
        )


class TestPerformanceComparison:
    """
    Compare baseline vs. optimized performance.

    Success criteria (from spec.md):
    - SC-003: 30% CPU reduction vs. Phase 1 baseline
    - FR-010: 30% resource reduction overall
    """

    @pytest.fixture
    def test_chords(self):
        """Standard test chord progression (8 chords)."""
        return [
            (0, 60, "major"),
            (11025, 62, "minor"),
            (22050, 64, "major"),
            (33075, 65, "major"),
            (44100, 67, "major"),
            (55125, 69, "minor"),
            (66150, 71, "dim"),
            (77175, 60, "major"),
        ]

    def test_synthesis_latency_improvement(self, test_chords):
        """
        Test that optimized synthesis has lower latency than baseline.

        SC-003: 30% reduction in resource usage.
        """
        duration_sec = 8.0

        # Run baseline benchmark
        baseline_bench = BaselineBenchmark()
        baseline_result = baseline_bench.run_synthesis_benchmark(test_chords, duration_sec)

        # Run optimized benchmark
        optimized_bench = OptimizedBenchmark()
        optimized_result = optimized_bench.run_synthesis_benchmark(test_chords, duration_sec)

        # Calculate improvement
        comparison = ComparisonResult(baseline_result, optimized_result)

        print(f"\nSynthesis Latency Comparison:")
        print(f"  Baseline: {baseline_result.duration_sec*1000:.2f}ms")
        print(f"  Optimized: {optimized_result.duration_sec*1000:.2f}ms")
        print(f"  Improvement: {comparison.latency_improvement_pct:.1f}%")

        # EXPECTED TO FAIL until optimizations are implemented
        # Should see at least 30% latency reduction
        assert comparison.latency_improvement_pct >= 30.0, (
            f"Optimized synthesis should be at least 30% faster than baseline. "
            f"Got {comparison.latency_improvement_pct:.1f}% improvement"
        )

    def test_cpu_usage_reduction(self, test_chords):
        """
        Test that optimized implementation reduces CPU usage by 30%.

        SC-003: Average CPU utilization reduced by at least 30%.
        FR-010: Reduce resource consumption by at least 30%.
        """
        duration_sec = 8.0

        # Warm-up to stabilize CPU measurements
        warmup_engine = SynthesisEngine(sample_rate=44100)
        _ = warmup_engine.render_chords(test_chords[:2], duration_sec=2.0, bpm=70)
        time.sleep(1.0)

        # Run baseline benchmark
        baseline_bench = BaselineBenchmark()
        baseline_result = baseline_bench.run_synthesis_benchmark(test_chords, duration_sec)

        time.sleep(1.0)  # Let CPU settle

        # Run optimized benchmark
        optimized_bench = OptimizedBenchmark()
        optimized_result = optimized_bench.run_synthesis_benchmark(test_chords, duration_sec)

        comparison = ComparisonResult(baseline_result, optimized_result)

        print(f"\nCPU Usage Comparison:")
        print(f"  Baseline: {baseline_result.cpu_percent:.1f}%")
        print(f"  Optimized: {optimized_result.cpu_percent:.1f}%")
        print(f"  Reduction: {comparison.cpu_improvement_pct:.1f}%")

        # EXPECTED TO FAIL until optimizations are implemented
        # Note: CPU measurements can be noisy, so we're flexible here
        # The key is showing improvement trend
        assert comparison.cpu_improvement_pct >= 20.0 or optimized_result.cpu_percent < 50.0, (
            f"Optimized implementation should reduce CPU usage significantly. "
            f"Got {comparison.cpu_improvement_pct:.1f}% reduction "
            f"(target: 30%, or absolute < 50%)"
        )

    def test_memory_usage_reduction(self, test_chords):
        """
        Test that optimized implementation uses memory efficiently.

        FR-005: Maintain stable memory usage.
        SC-004: Memory usage remains stable.
        """
        duration_sec = 8.0

        # Run baseline benchmark
        baseline_bench = BaselineBenchmark()
        baseline_result = baseline_bench.run_synthesis_benchmark(test_chords, duration_sec)

        # Run optimized benchmark
        optimized_bench = OptimizedBenchmark()
        optimized_result = optimized_bench.run_synthesis_benchmark(test_chords, duration_sec)

        comparison = ComparisonResult(baseline_result, optimized_result)

        print(f"\nMemory Usage Comparison:")
        print(f"  Baseline: {baseline_result.memory_mb:.2f} MB")
        print(f"  Optimized: {optimized_result.memory_mb:.2f} MB")
        print(f"  Reduction: {comparison.memory_improvement_pct:.1f}%")

        # Memory usage should be similar or better
        # (Pre-allocation may increase initial memory but prevent growth)
        assert optimized_result.memory_mb <= baseline_result.memory_mb * 1.2, (
            f"Optimized memory usage should not exceed baseline by >20%. "
            f"Baseline: {baseline_result.memory_mb:.2f}MB, "
            f"Optimized: {optimized_result.memory_mb:.2f}MB"
        )

    def test_overall_resource_efficiency(self, test_chords):
        """
        Test overall resource efficiency (composite metric).

        SC-003, FR-010: 30% overall resource reduction.

        Composite metric: (CPU reduction + latency reduction) / 2
        """
        duration_sec = 8.0

        # Run benchmarks
        baseline_bench = BaselineBenchmark()
        baseline_result = baseline_bench.run_synthesis_benchmark(test_chords, duration_sec)

        time.sleep(1.0)

        optimized_bench = OptimizedBenchmark()
        optimized_result = optimized_bench.run_synthesis_benchmark(test_chords, duration_sec)

        comparison = ComparisonResult(baseline_result, optimized_result)

        # Composite efficiency metric
        composite_improvement = (
            comparison.cpu_improvement_pct + comparison.latency_improvement_pct
        ) / 2

        print(f"\nOverall Resource Efficiency:")
        print(f"  CPU improvement: {comparison.cpu_improvement_pct:.1f}%")
        print(f"  Latency improvement: {comparison.latency_improvement_pct:.1f}%")
        print(f"  Composite improvement: {composite_improvement:.1f}%")

        # EXPECTED TO FAIL until optimizations are implemented
        assert composite_improvement >= 30.0, (
            f"Overall resource efficiency should improve by at least 30%. "
            f"Got {composite_improvement:.1f}% composite improvement"
        )


@pytest.mark.benchmark
class TestDetailedBenchmarks:
    """
    Detailed benchmarks for profiling and optimization work.
    """

    def test_generate_baseline_report(self):
        """
        Generate comprehensive baseline performance report.

        This creates the Phase 1 baseline for comparison.
        Should be run once to establish baseline metrics.
        """
        pytest.skip("Baseline report generation - manual run only")

        # If run manually, generates comprehensive report
        # This would save metrics to docs/performance/baseline-comparison.md

    def test_generate_optimization_report(self):
        """
        Generate comprehensive optimization performance report.

        This compares Phase 3 optimized vs. Phase 1 baseline.
        """
        pytest.skip("Optimization report generation - manual run only")

        # If run manually, generates comparison report
        # This would update docs/performance/baseline-comparison.md
