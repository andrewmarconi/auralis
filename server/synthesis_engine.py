"""
GPU-Accelerated Audio Synthesis Engine

Real-time ambient music synthesis using torchsynth with Metal/CUDA acceleration.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
import math


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


class AmbientPadVoice(torch.nn.Module):
    """
    Ambient pad voice with dual oscillators, ADSR envelope, and LFO modulation.

    Designed for lush, evolving chord pads with rich harmonic content.
    Uses pure PyTorch for GPU acceleration and full control.
    """

    def __init__(self, sample_rate: int = 44100, device: Optional[torch.device] = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device or torch.device("cpu")

    def forward(
        self,
        pitch: torch.Tensor,
        duration_samples: int,
        velocity: float = 0.6,
    ) -> torch.Tensor:
        """
        Generate ambient pad sound.

        Args:
            pitch: MIDI pitch value (tensor)
            duration_samples: Duration in samples
            velocity: Note velocity (0.0-1.0)

        Returns:
            Audio tensor of shape (duration_samples,)
        """
        # Convert MIDI to frequency
        freq = 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))

        # Create time array
        t = torch.arange(duration_samples, device=self.device, dtype=torch.float32) / self.sample_rate

        # OSC1: Pure sine wave (fundamental)
        osc1 = torch.sin(2.0 * math.pi * freq * t)

        # OSC2: Sawtooth wave (harmonics) detuned slightly
        detune_ratio = 2.0 ** (5.0 / 1200.0)  # +5 cents
        freq2 = freq * detune_ratio
        # Sawtooth: 2 * (t * freq % 1) - 1
        saw_phase = (t * freq2) % 1.0
        osc2 = 2.0 * saw_phase - 1.0

        # Mix oscillators (50/50)
        mixed = (osc1 + osc2) * 0.5

        # LFO for subtle movement (0.5 Hz sine wave)
        lfo = torch.sin(2.0 * math.pi * 0.5 * t) * 0.2

        # Apply LFO modulation
        mixed = mixed * (1.0 + lfo)

        # ADSR envelope (very slow ambient evolution)
        # Attack: 2.5s, Decay: 3s, Sustain: 75%, Release: 3s
        envelope = self._create_adsr_envelope(
            duration_samples,
            attack_sec=2.5,
            decay_sec=3.0,
            sustain_level=0.75,
            release_sec=3.0
        )

        # Apply envelope and velocity
        output = mixed * envelope * velocity

        return output

    def _create_adsr_envelope(
        self,
        num_samples: int,
        attack_sec: float,
        decay_sec: float,
        sustain_level: float,
        release_sec: float
    ) -> torch.Tensor:
        """Create exponential ADSR envelope."""
        envelope = torch.ones(num_samples, device=self.device)

        attack_samples = int(attack_sec * self.sample_rate)
        decay_samples = int(decay_sec * self.sample_rate)
        release_samples = int(release_sec * self.sample_rate)

        # Attack (exponential rise)
        if attack_samples > 0 and attack_samples < num_samples:
            attack_curve = 1.0 - torch.exp(
                -5.0 * torch.linspace(0, 1, attack_samples, device=self.device)
            )
            envelope[:attack_samples] = attack_curve

        # Decay (exponential fall to sustain)
        decay_start = attack_samples
        decay_end = min(decay_start + decay_samples, num_samples)
        if decay_end > decay_start:
            decay_len = decay_end - decay_start
            decay_curve = sustain_level + (1.0 - sustain_level) * torch.exp(
                -3.0 * torch.linspace(0, 1, decay_len, device=self.device)
            )
            envelope[decay_start:decay_end] = decay_curve

        # Sustain (constant level)
        sustain_start = decay_end
        sustain_end = max(0, num_samples - release_samples)
        if sustain_end > sustain_start:
            envelope[sustain_start:sustain_end] = sustain_level

        # Release (exponential fall to zero)
        if release_samples > 0 and release_samples < num_samples:
            release_start = num_samples - release_samples
            release_curve = sustain_level * torch.exp(
                -5.0 * torch.linspace(0, 1, release_samples, device=self.device)
            )
            envelope[release_start:] = release_curve

        return envelope


class LeadVoice(torch.nn.Module):
    """
    Lead melody voice with clean tone and articulate envelope.

    Designed for clear, expressive melody lines over ambient pads.
    Uses pure PyTorch for GPU acceleration.
    """

    def __init__(self, sample_rate: int = 44100, device: Optional[torch.device] = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device or torch.device("cpu")

    def forward(
        self,
        pitch: torch.Tensor,
        duration_samples: int,
        velocity: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate lead melody sound.

        Args:
            pitch: MIDI pitch value (tensor)
            duration_samples: Duration in samples
            velocity: Note velocity (0.0-1.0)

        Returns:
            Audio tensor of shape (duration_samples,)
        """
        # Convert MIDI to frequency
        freq = 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))

        # Create time array
        t = torch.arange(duration_samples, device=self.device, dtype=torch.float32) / self.sample_rate

        # Generate pure sine wave for clean tone
        signal = torch.sin(2.0 * math.pi * freq * t)

        # ADSR envelope (smooth ambient response)
        # Attack: 300ms, Decay: 500ms, Sustain: 80%, Release: 800ms
        envelope = self._create_adsr_envelope(
            duration_samples,
            attack_sec=0.3,
            decay_sec=0.5,
            sustain_level=0.8,
            release_sec=0.8
        )

        # Apply envelope and velocity
        output = signal * envelope * velocity

        return output

    def _create_adsr_envelope(
        self,
        num_samples: int,
        attack_sec: float,
        decay_sec: float,
        sustain_level: float,
        release_sec: float
    ) -> torch.Tensor:
        """Create exponential ADSR envelope."""
        envelope = torch.ones(num_samples, device=self.device)

        attack_samples = int(attack_sec * self.sample_rate)
        decay_samples = int(decay_sec * self.sample_rate)
        release_samples = int(release_sec * self.sample_rate)

        # Attack (exponential rise)
        if attack_samples > 0 and attack_samples < num_samples:
            attack_curve = 1.0 - torch.exp(
                -5.0 * torch.linspace(0, 1, attack_samples, device=self.device)
            )
            envelope[:attack_samples] = attack_curve

        # Decay (exponential fall to sustain)
        decay_start = attack_samples
        decay_end = min(decay_start + decay_samples, num_samples)
        if decay_end > decay_start:
            decay_len = decay_end - decay_start
            decay_curve = sustain_level + (1.0 - sustain_level) * torch.exp(
                -3.0 * torch.linspace(0, 1, decay_len, device=self.device)
            )
            envelope[decay_start:decay_end] = decay_curve

        # Sustain (constant level)
        sustain_start = decay_end
        sustain_end = max(0, num_samples - release_samples)
        if sustain_end > sustain_start:
            envelope[sustain_start:sustain_end] = sustain_level

        # Release (exponential fall to zero)
        if release_samples > 0 and release_samples < num_samples:
            release_start = num_samples - release_samples
            release_curve = sustain_level * torch.exp(
                -5.0 * torch.linspace(0, 1, release_samples, device=self.device)
            )
            envelope[release_start:] = release_curve

        return envelope


class KickVoice(torch.nn.Module):
    """
    Deep kick drum voice using frequency sweep.

    Generates low-frequency pulses for subtle rhythmic anchoring.
    Uses pure PyTorch for GPU acceleration.
    """

    def __init__(self, sample_rate: int = 44100, device: Optional[torch.device] = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device or torch.device("cpu")

    def forward(self, duration_samples: int, velocity: float = 0.7) -> torch.Tensor:
        """
        Generate kick drum sound.

        Args:
            duration_samples: Duration in samples (~100ms typical)
            velocity: Kick velocity (0.0-1.0)

        Returns:
            Audio tensor of shape (duration_samples,)
        """
        # Frequency sweep: 150Hz → 40Hz (sub-bass)
        start_freq = 150.0
        end_freq = 40.0

        # Create exponential frequency sweep (using exp instead of logspace for MPS compatibility)
        log_start = math.log(start_freq)
        log_end = math.log(end_freq)
        log_curve = torch.linspace(log_start, log_end, duration_samples, device=self.device)
        freq_curve = torch.exp(log_curve)

        # Create time array and integrate frequency for phase
        t = torch.arange(duration_samples, device=self.device, dtype=torch.float32) / self.sample_rate
        # Phase = cumulative sum of frequency * dt
        phase = torch.cumsum(freq_curve / self.sample_rate, dim=0)

        # Generate sine wave with frequency sweep
        signal = torch.sin(2.0 * math.pi * phase)

        # Tight exponential envelope for punchy kick
        # Attack: 1ms, Decay: 150ms
        envelope = torch.exp(-10.0 * t / 0.15)  # Exponential decay with 150ms time constant

        # Apply envelope and velocity
        return signal * envelope * velocity


class SwellVoice(torch.nn.Module):
    """
    Granular swell voice using filtered noise.

    Creates textured ambient swells for atmospheric depth.
    Uses pure PyTorch for GPU acceleration.
    """

    def __init__(self, sample_rate: int = 44100, device: Optional[torch.device] = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device or torch.device("cpu")

    def forward(self, duration_samples: int, velocity: float = 0.5) -> torch.Tensor:
        """
        Generate granular swell sound.

        Args:
            duration_samples: Duration in samples (2-4 seconds typical)
            velocity: Swell velocity (0.0-1.0)

        Returns:
            Audio tensor of shape (duration_samples,)
        """
        # Generate white noise
        noise_signal = torch.randn(duration_samples, device=self.device)

        # Create time array for LFO
        t = torch.arange(duration_samples, device=self.device, dtype=torch.float32) / self.sample_rate

        # Create slow LFO for amplitude modulation (0.2 Hz)
        lfo_signal = torch.sin(2.0 * math.pi * 0.2 * t)

        # Apply LFO modulation to noise (creates granular texture)
        modulated = noise_signal * (0.5 + lfo_signal * 0.5)

        # Create swell envelope (fade in → sustain → fade out)
        attack_samples = duration_samples // 4
        sustain_samples = duration_samples // 2
        release_samples = duration_samples - attack_samples - sustain_samples

        envelope = torch.ones(duration_samples, device=self.device)

        # Attack (smooth fade in)
        if attack_samples > 0:
            attack_curve = 1.0 - torch.exp(
                -3.0 * torch.linspace(0, 1, attack_samples, device=self.device)
            )
            envelope[:attack_samples] = attack_curve

        # Release (smooth fade out)
        if release_samples > 0:
            release_curve = torch.exp(
                -3.0 * torch.linspace(0, 1, release_samples, device=self.device)
            )
            envelope[-release_samples:] = release_curve

        # Apply envelope and velocity
        output = modulated * envelope * velocity

        # Low-pass filter effect (simple moving average)
        # This softens harsh noise for ambient aesthetic
        kernel_size = 100  # ~2.3ms at 44.1kHz
        if duration_samples > kernel_size:
            kernel = torch.ones(kernel_size, device=self.device) / kernel_size
            output = torch.nn.functional.conv1d(
                output.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=kernel_size // 2
            ).squeeze()
            # Trim to original length
            output = output[:duration_samples]

        return output


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

        # Initialize voice modules
        self.pad_voice = AmbientPadVoice(sample_rate=sample_rate, device=self.device)
        self.lead_voice = LeadVoice(sample_rate=sample_rate, device=self.device)
        self.kick_voice = KickVoice(sample_rate=sample_rate, device=self.device)
        self.swell_voice = SwellVoice(sample_rate=sample_rate, device=self.device)

        logger.info("✓ Ambient pad, lead, and percussion voices initialized")

    def render_phrase(
        self,
        chords: list[tuple[int, int, str]],
        melody: list[tuple[int, int, float, float]],
        duration_sec: float,
        kicks: list[tuple[int, float]] | None = None,
        swells: list[tuple[int, float, float]] | None = None,
    ) -> NDArray[np.float32]:
        """
        Render a musical phrase to audio using torchsynth voices.

        Args:
            chords: List of (onset_sample, root_midi, chord_type)
            melody: List of (onset_sample, pitch_midi, velocity, duration_sec)
            duration_sec: Total phrase duration in seconds
            kicks: Optional list of (onset_sample, velocity)
            swells: Optional list of (onset_sample, duration_sec, velocity)

        Returns:
            Stereo audio array, shape (2, num_samples), float32 [-1, 1]
        """
        num_samples = int(duration_sec * self.sample_rate)

        # Initialize output buffer on GPU
        audio = torch.zeros(num_samples, device=self.device)

        # Render chord pads
        audio = self._render_chord_pads(audio, chords, num_samples)

        # Render melody lead
        audio = self._render_melody_lead(audio, melody, num_samples)

        # Render percussion (if provided)
        if kicks:
            audio = self._render_kicks(audio, kicks, num_samples)
        if swells:
            audio = self._render_swells(audio, swells, num_samples)

        # Normalize and convert to stereo
        audio = torch.tanh(audio * 0.7)  # Soft clipping for dynamics
        audio_np = audio.cpu().numpy()

        # Create stereo image
        stereo = np.stack([audio_np, audio_np], axis=0)

        return stereo.astype(np.float32)

    def _render_chord_pads(
        self,
        audio: torch.Tensor,
        chords: list[tuple[int, int, str]],
        num_samples: int,
    ) -> torch.Tensor:
        """
        Render chord progressions as ambient pads with improved voicing.

        Args:
            audio: Existing audio buffer to add to
            chords: List of (onset_sample, root_midi, chord_type)
            num_samples: Total buffer length

        Returns:
            Audio buffer with chord pads added
        """
        from composition.melody_generator import CHORD_INTERVALS

        for onset_sample, root_midi, chord_type in chords:
            if onset_sample >= num_samples:
                continue

            # Get chord intervals
            intervals = CHORD_INTERVALS.get(chord_type, [0, 3, 7])

            # Calculate chord duration (until next chord or end of phrase)
            next_onset = num_samples
            for next_chord in chords:
                if next_chord[0] > onset_sample:
                    next_onset = next_chord[0]
                    break

            duration_samples = min(next_onset - onset_sample, num_samples - onset_sample)

            # Improved voicing: spread across multiple octaves for clarity
            # Use bass note one octave below, and spread upper voices
            bass_root = root_midi - 12  # One octave lower for bass

            # Render bass note (root) for harmonic foundation
            bass_pitch = torch.tensor(bass_root, dtype=torch.float32, device=self.device)
            bass_signal = self.pad_voice(
                pitch=bass_pitch,
                duration_samples=duration_samples,
                velocity=0.5,  # Increased from 0.4 for better presence
            )
            end_sample = min(onset_sample + duration_samples, num_samples)
            audio[onset_sample:end_sample] += bass_signal[:end_sample - onset_sample] * 0.8

            # Render chord tones in mid-range with varied velocities for depth
            for idx, interval in enumerate(intervals):
                pitch_midi = root_midi + interval
                pitch_tensor = torch.tensor(pitch_midi, dtype=torch.float32, device=self.device)

                # Vary velocity slightly for each voice (creates depth)
                voice_velocity = 0.45 + (idx * 0.05)  # 0.45, 0.50, 0.55

                # Generate pad voice
                pad_signal = self.pad_voice(
                    pitch=pitch_tensor,
                    duration_samples=duration_samples,
                    velocity=voice_velocity,
                )

                # Add to buffer
                audio[onset_sample:end_sample] += pad_signal[:end_sample - onset_sample]

        return audio

    def _render_melody_lead(
        self,
        audio: torch.Tensor,
        melody: list[tuple[int, int, float, float]],
        num_samples: int,
    ) -> torch.Tensor:
        """
        Render melody as lead voice.

        Args:
            audio: Existing audio buffer to add to
            melody: List of (onset_sample, pitch_midi, velocity, duration_sec)
            num_samples: Total buffer length

        Returns:
            Audio buffer with melody lead added
        """
        for onset_sample, pitch_midi, velocity, duration_sec in melody:
            if onset_sample >= num_samples:
                continue

            # Calculate note duration in samples
            duration_samples = int(duration_sec * self.sample_rate)
            end_sample = min(onset_sample + duration_samples, num_samples)
            actual_duration = end_sample - onset_sample

            if actual_duration <= 0:
                continue

            # Generate lead voice
            pitch_tensor = torch.tensor(pitch_midi, dtype=torch.float32, device=self.device)
            lead_signal = self.lead_voice(
                pitch=pitch_tensor,
                duration_samples=actual_duration,
                velocity=velocity,
            )

            # Add to buffer
            audio[onset_sample:end_sample] += lead_signal

        return audio

    def _render_kicks(
        self,
        audio: torch.Tensor,
        kicks: list[tuple[int, float]],
        num_samples: int,
    ) -> torch.Tensor:
        """
        Render kick drum events.

        Args:
            audio: Existing audio buffer to add to
            kicks: List of (onset_sample, velocity)
            num_samples: Total buffer length

        Returns:
            Audio buffer with kicks added
        """
        # Kick duration: ~100ms
        kick_duration_samples = int(0.1 * self.sample_rate)

        for onset_sample, velocity in kicks:
            # Skip if onset is beyond buffer
            if onset_sample >= num_samples:
                continue

            # Ensure onset is not negative (shouldn't happen but be safe)
            onset_sample = max(0, onset_sample)

            # Calculate actual duration (clip at end if necessary)
            end_sample = min(onset_sample + kick_duration_samples, num_samples)
            actual_duration = end_sample - onset_sample

            if actual_duration <= 0:
                continue

            # Generate kick with actual duration
            kick_signal = self.kick_voice(
                duration_samples=actual_duration,
                velocity=velocity,
            )

            # Add to buffer
            audio[onset_sample:end_sample] += kick_signal

        return audio

    def _render_swells(
        self,
        audio: torch.Tensor,
        swells: list[tuple[int, float, float]],
        num_samples: int,
    ) -> torch.Tensor:
        """
        Render granular swell events.

        Args:
            audio: Existing audio buffer to add to
            swells: List of (onset_sample, duration_sec, velocity)
            num_samples: Total buffer length

        Returns:
            Audio buffer with swells added
        """
        for onset_sample, duration_sec, velocity in swells:
            if onset_sample >= num_samples:
                continue

            # Calculate swell duration in samples
            duration_samples = int(duration_sec * self.sample_rate)
            end_sample = min(onset_sample + duration_samples, num_samples)
            actual_duration = end_sample - onset_sample

            if actual_duration <= 0:
                continue

            # Generate swell
            swell_signal = self.swell_voice(
                duration_samples=actual_duration,
                velocity=velocity,
            )

            # Add to buffer
            audio[onset_sample:end_sample] += swell_signal

        return audio

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
