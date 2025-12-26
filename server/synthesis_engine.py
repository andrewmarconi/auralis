"""
GPU-Accelerated Audio Synthesis Engine

Real-time ambient music synthesis using torchsynth with Metal/CUDA acceleration.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray


class GPUDeviceManager:
    """
    Manages GPU device selection for audio synthesis.

    Priority order:
    1. Metal (Apple Silicon M1/M2/M4)
    2. CUDA (NVIDIA GPUs)
    3. CPU (fallback)
    """

    @staticmethod
    def get_optimal_device() -> torch.device:
        """
        Detect and return optimal compute device.

        Returns:
            torch.device configured for best available hardware
        """
        # Check for Apple Metal (MPS)
        if torch.backends.mps.is_available():
            try:
                device = torch.device("mps")
                # Test device with small operation
                test_tensor = torch.zeros(1, device=device)
                logger.info("✓ Metal (MPS) device detected and verified")
                return device
            except Exception as e:
                logger.warning(f"Metal available but failed verification: {e}")

        # Check for NVIDIA CUDA
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                # Test device with small operation
                test_tensor = torch.zeros(1, device=device)
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✓ CUDA device detected: {gpu_name}")
                return device
            except Exception as e:
                logger.warning(f"CUDA available but failed verification: {e}")

        # Fallback to CPU
        logger.warning("⚠ No GPU detected - using CPU (may impact real-time performance)")
        return torch.device("cpu")

    @staticmethod
    def get_device_info() -> dict:
        """
        Get detailed information about current device.

        Returns:
            Dictionary with device type, name, and capabilities
        """
        device = GPUDeviceManager.get_optimal_device()

        info = {
            "type": device.type,
            "available_backends": [],
        }

        if device.type == "mps":
            info["name"] = "Apple Metal Performance Shaders"
            info["available_backends"].append("metal")
        elif device.type == "cuda":
            info["name"] = torch.cuda.get_device_name(0)
            info["available_backends"].append("cuda")
            info["cuda_version"] = torch.version.cuda
            info["memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1024**2
        else:
            info["name"] = "CPU"
            info["available_backends"].append("cpu")

        return info


class SynthesisEngine:
    """
    Real-time audio synthesis engine using GPU acceleration.

    Renders ambient music from musical parameters using torchsynth.
    Optimized for <100ms processing latency.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize synthesis engine with GPU acceleration.

        Args:
            sample_rate: Audio sampling rate in Hz
            device: Compute device (None = auto-detect optimal)
        """
        self.sample_rate = sample_rate

        # Auto-detect optimal device if not specified
        if device is None:
            self.device = GPUDeviceManager.get_optimal_device()
        else:
            self.device = device

        logger.info(f"Synthesis engine initialized on {self.device}")

        # Device capabilities
        self.device_info = GPUDeviceManager.get_device_info()
        self.gpu_available = self.device.type in ["mps", "cuda"]

    def render_phrase(
        self,
        chords: list[tuple[int, int, str]],
        melody: list[tuple[int, int, float, float]],
        duration_sec: float,
    ) -> NDArray[np.float32]:
        """
        Render a musical phrase to audio.

        Args:
            chords: List of (onset_sample, root_midi, chord_type)
            melody: List of (onset_sample, pitch_midi, velocity, duration_sec)
            duration_sec: Total phrase duration in seconds

        Returns:
            Stereo audio array, shape (2, num_samples), float32 [-1, 1]
        """
        num_samples = int(duration_sec * self.sample_rate)

        # TODO: Implement full torchsynth rendering in Phase 3
        # For now, generate simple sine waves as placeholder
        audio = self._render_simple_synthesis(melody, num_samples)

        return audio

    def _render_simple_synthesis(
        self,
        melody: list[tuple[int, int, float, float]],
        num_samples: int,
    ) -> NDArray[np.float32]:
        """
        Simple sine wave synthesis (placeholder for torchsynth).

        Args:
            melody: List of (onset_sample, pitch_midi, velocity, duration_sec)
            num_samples: Total output length in samples

        Returns:
            Stereo audio array, shape (2, num_samples), float32
        """
        # Create time array on GPU
        t = torch.linspace(
            0, num_samples / self.sample_rate, num_samples, device=self.device
        )

        # Initialize output
        audio = torch.zeros(num_samples, device=self.device)

        # Render each note as sine wave
        for onset_sample, pitch_midi, velocity, duration_sec in melody:
            # MIDI to frequency
            freq = 440.0 * (2.0 ** ((pitch_midi - 69) / 12.0))

            # Note duration in samples
            note_samples = int(duration_sec * self.sample_rate)
            note_end = min(onset_sample + note_samples, num_samples)

            if onset_sample >= num_samples:
                continue

            # Generate sine wave segment
            note_t = t[onset_sample:note_end] - t[onset_sample]
            sine_wave = torch.sin(2 * np.pi * freq * note_t) * velocity

            # Apply simple envelope (ADSR approximation)
            envelope = self._create_simple_envelope(len(note_t), duration_sec)
            audio[onset_sample:note_end] += sine_wave * envelope

        # Normalize and convert to stereo
        audio = torch.tanh(audio * 0.5)  # Soft clipping
        audio_np = audio.cpu().numpy()

        # Duplicate to stereo
        stereo = np.stack([audio_np, audio_np], axis=0)

        return stereo.astype(np.float32)

    def _create_simple_envelope(
        self, num_samples: int, duration_sec: float
    ) -> torch.Tensor:
        """
        Create simple ADSR envelope.

        Args:
            num_samples: Number of samples in envelope
            duration_sec: Total duration in seconds

        Returns:
            Envelope curve as torch tensor
        """
        envelope = torch.ones(num_samples, device=self.device)

        # Attack (10% of duration)
        attack_samples = int(num_samples * 0.1)
        if attack_samples > 0:
            envelope[:attack_samples] = torch.linspace(
                0, 1, attack_samples, device=self.device
            )

        # Release (20% of duration)
        release_samples = int(num_samples * 0.2)
        if release_samples > 0:
            envelope[-release_samples:] = torch.linspace(
                1, 0, release_samples, device=self.device
            )

        return envelope

    def get_latency_ms(self) -> float:
        """
        Estimate synthesis latency in milliseconds.

        Returns:
            Expected latency for typical phrase rendering
        """
        # TODO: Actual benchmarking in performance tests
        if self.gpu_available:
            return 50.0  # Estimated GPU latency
        else:
            return 150.0  # Estimated CPU latency

    def get_status(self) -> dict:
        """
        Get synthesis engine status.

        Returns:
            Dictionary with device info and performance metrics
        """
        return {
            "device_type": self.device.type,
            "device_info": self.device_info,
            "gpu_available": self.gpu_available,
            "sample_rate": self.sample_rate,
            "estimated_latency_ms": self.get_latency_ms(),
        }
