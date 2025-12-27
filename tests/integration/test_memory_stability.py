"""
Memory Stability Integration Test

Tests memory stability over extended streaming sessions (8+ hours simulation).
Validates memory leak prevention and stable resource usage.

Task: T065 [US3]
"""

import pytest
import time
import psutil
import torch
from typing import List, Dict

from server.synthesis_engine import SynthesisEngine


class TestMemoryStability:
    """
    Test memory stability over extended streaming sessions.

    Success criteria (from spec.md):
    - SC-004: Memory usage remains stable over 8 hours with <10% growth
    - FR-005: Maintain stable memory usage without leaks
    """

    @pytest.fixture
    def synthesis_engine(self):
        """Create synthesis engine instance for testing."""
        return SynthesisEngine(sample_rate=44100)

    @pytest.fixture
    def process(self):
        """Get current process for memory monitoring."""
        return psutil.Process()

    def get_memory_stats(self, process) -> Dict[str, float]:
        """Get current memory statistics in MB."""
        mem_info = process.memory_info()

        stats = {
            "rss_mb": mem_info.rss / 1024**2,  # Resident Set Size
            "vms_mb": mem_info.vms / 1024**2,  # Virtual Memory Size
        }

        # Add GPU memory if available
        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2

        return stats

    def test_short_term_memory_stability(self, synthesis_engine, process):
        """
        Test memory stability over 100 synthesis iterations (simulates ~10 minutes).

        FR-005: Maintain stable memory usage without leaks.
        """
        test_chords = [
            (0, 60, "major"),
            (11025, 62, "minor"),
            (22050, 64, "major"),
            (33075, 65, "major"),
        ]

        num_iterations = 100
        duration_sec = 4.0

        # Initial memory snapshot
        initial_stats = self.get_memory_stats(process)
        memory_samples = [initial_stats["rss_mb"]]

        print(f"\nShort-term Memory Stability Test ({num_iterations} iterations):")
        print(f"  Initial RSS: {initial_stats['rss_mb']:.2f} MB")

        # Run synthesis iterations
        for i in range(num_iterations):
            _ = synthesis_engine.render_chords(
                test_chords, duration_sec=duration_sec, bpm=70
            )

            # Sample memory every 10 iterations
            if i % 10 == 0:
                current_stats = self.get_memory_stats(process)
                memory_samples.append(current_stats["rss_mb"])

        # Final memory snapshot
        final_stats = self.get_memory_stats(process)

        memory_growth_mb = final_stats["rss_mb"] - initial_stats["rss_mb"]
        memory_growth_pct = (memory_growth_mb / initial_stats["rss_mb"]) * 100

        print(f"  Final RSS: {final_stats['rss_mb']:.2f} MB")
        print(f"  Growth: {memory_growth_mb:.2f} MB ({memory_growth_pct:.1f}%)")
        print(f"  Samples: {len(memory_samples)}")

        # EXPECTED TO FAIL until memory leak prevention is implemented
        # Memory growth should be < 10%
        assert memory_growth_pct < 10.0, (
            f"Memory growth should be < 10% over {num_iterations} iterations. "
            f"Got {memory_growth_pct:.1f}% ({memory_growth_mb:.2f} MB)"
        )

    @pytest.mark.slow
    def test_extended_memory_stability_simulation(self, synthesis_engine, process):
        """
        Test memory stability over extended period (simulates 1 hour).

        SC-004: Memory usage remains stable with <10% growth.

        Note: This is a faster simulation, not a real 8-hour test.
        For full 8-hour test, use test_eight_hour_memory_stability.
        """
        test_chords = [
            (0, 60, "major"),
            (11025, 62, "minor"),
            (22050, 64, "major"),
            (33075, 65, "major"),
        ]

        # Simulate 1 hour: 600 iterations of 6-second audio
        num_iterations = 600
        duration_sec = 6.0
        sample_interval = 60  # Sample every 60 iterations (~6 minutes)

        initial_stats = self.get_memory_stats(process)
        memory_samples = [initial_stats["rss_mb"]]

        print(f"\nExtended Memory Stability Simulation ({num_iterations} iterations):")
        print(f"  Initial RSS: {initial_stats['rss_mb']:.2f} MB")

        start_time = time.time()

        for i in range(num_iterations):
            _ = synthesis_engine.render_chords(
                test_chords, duration_sec=duration_sec, bpm=70
            )

            # Sample memory periodically
            if i % sample_interval == 0:
                current_stats = self.get_memory_stats(process)
                memory_samples.append(current_stats["rss_mb"])

                elapsed_min = (time.time() - start_time) / 60
                print(f"  Iteration {i:4d} ({elapsed_min:.1f}min): "
                      f"RSS={current_stats['rss_mb']:.2f} MB")

        final_stats = self.get_memory_stats(process)
        elapsed_time = time.time() - start_time

        memory_growth_mb = final_stats["rss_mb"] - initial_stats["rss_mb"]
        memory_growth_pct = (memory_growth_mb / initial_stats["rss_mb"]) * 100

        print(f"\n  Final RSS: {final_stats['rss_mb']:.2f} MB")
        print(f"  Growth: {memory_growth_mb:.2f} MB ({memory_growth_pct:.1f}%)")
        print(f"  Test duration: {elapsed_time/60:.1f} minutes")
        print(f"  Memory samples: {memory_samples}")

        # Analyze memory trend
        if len(memory_samples) > 2:
            # Calculate linear regression to detect memory leak
            import numpy as np
            x = np.arange(len(memory_samples))
            y = np.array(memory_samples)
            slope = np.polyfit(x, y, 1)[0]

            print(f"  Memory growth rate: {slope:.4f} MB per sample")

            # Project to 8 hours
            samples_per_hour = 3600 / (duration_sec * sample_interval)
            projected_8h_growth = slope * samples_per_hour * 8

            print(f"  Projected 8-hour growth: {projected_8h_growth:.2f} MB")

        # EXPECTED TO FAIL until memory leak prevention is implemented
        assert memory_growth_pct < 10.0, (
            f"Memory growth should be < 10% over extended period. "
            f"Got {memory_growth_pct:.1f}% ({memory_growth_mb:.2f} MB)"
        )

    @pytest.mark.slow
    @pytest.mark.manual
    def test_eight_hour_memory_stability(self, synthesis_engine, process):
        """
        Full 8-hour memory stability test (manual run only).

        SC-004: Memory usage remains stable over 8 hours with <10MB growth.

        This test takes 8+ hours to run and should only be executed manually
        before production deployment.

        Usage: pytest tests/integration/test_memory_stability.py::test_eight_hour_memory_stability -v
        """
        pytest.skip("8-hour test must be run manually with -m manual flag")

        # If run manually:
        test_chords = [
            (0, 60, "major"),
            (11025, 62, "minor"),
            (22050, 64, "major"),
            (33075, 65, "major"),
        ]

        duration_hours = 8
        duration_sec = 4.0
        iterations_per_hour = int(3600 / duration_sec)
        total_iterations = iterations_per_hour * duration_hours

        initial_stats = self.get_memory_stats(process)
        memory_samples = []

        print(f"\n8-Hour Memory Stability Test:")
        print(f"  Duration: {duration_hours} hours")
        print(f"  Total iterations: {total_iterations}")
        print(f"  Initial RSS: {initial_stats['rss_mb']:.2f} MB")

        start_time = time.time()

        for i in range(total_iterations):
            _ = synthesis_engine.render_chords(
                test_chords, duration_sec=duration_sec, bpm=70
            )

            # Sample memory every hour
            if i % iterations_per_hour == 0:
                current_stats = self.get_memory_stats(process)
                memory_samples.append(current_stats["rss_mb"])

                hours_elapsed = (time.time() - start_time) / 3600
                print(f"  Hour {hours_elapsed:.1f}: RSS={current_stats['rss_mb']:.2f} MB")

        final_stats = self.get_memory_stats(process)

        memory_growth_mb = final_stats["rss_mb"] - initial_stats["rss_mb"]
        memory_growth_pct = (memory_growth_mb / initial_stats["rss_mb"]) * 100

        print(f"\n  Final RSS: {final_stats['rss_mb']:.2f} MB")
        print(f"  Growth: {memory_growth_mb:.2f} MB ({memory_growth_pct:.1f}%)")

        # SC-004: <10MB growth over 8 hours
        assert memory_growth_mb < 10.0, (
            f"Memory growth should be < 10 MB over 8 hours. "
            f"Got {memory_growth_mb:.2f} MB"
        )

    def test_gpu_memory_cleanup(self, synthesis_engine, process):
        """
        Test that GPU memory is properly cleaned up after synthesis.

        FR-005: Prevent memory leaks through periodic cleanup.
        """
        device_info = synthesis_engine.device_manager.get_device_info()

        if device_info["type"] not in ["cuda", "mps"]:
            pytest.skip("GPU memory testing only available on CUDA/Metal devices")

        test_chords = [(i * 11025, 60, "major") for i in range(8)]

        # Clear GPU cache before test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            initial_gpu_mb = torch.cuda.memory_allocated() / 1024**2

            # Run multiple synthesis iterations
            for _ in range(100):
                _ = synthesis_engine.render_chords(
                    test_chords, duration_sec=8.0, bpm=70
                )

            final_gpu_mb = torch.cuda.memory_allocated() / 1024**2
            peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024**2

            gpu_growth_mb = final_gpu_mb - initial_gpu_mb

            print(f"\nGPU Memory Cleanup:")
            print(f"  Initial: {initial_gpu_mb:.2f} MB")
            print(f"  Peak: {peak_gpu_mb:.2f} MB")
            print(f"  Final: {final_gpu_mb:.2f} MB")
            print(f"  Growth: {gpu_growth_mb:.2f} MB")

            # EXPECTED TO FAIL until periodic GPU cache clearing is implemented
            # GPU memory should return to near initial after iterations
            assert gpu_growth_mb < 50.0, (
                f"GPU memory growth should be minimal (<50MB) with periodic cleanup. "
                f"Got {gpu_growth_mb:.2f} MB growth"
            )
