# Torchsynth Integration Guide

## Overview

This document details how Auralis integrates with torchsynth for GPU-accelerated audio synthesis, including fallback strategies and cross-platform compatibility.

## Important Notes

**Torchsynth Status**: As of December 2025, torchsynth (v1.0.2) is considered an inactive project with no updates since 2021. However, it remains functional and compatible with modern PyTorch versions (requires PyTorch 1.8+).

**Sources**:
- [torchsynth on PyPI](https://pypi.org/project/torchsynth/)
- [torchsynth GitHub Repository](https://github.com/torchsynth/torchsynth)

---

## 1. Torchsynth Architecture Mapping

### 1.1 Available Modules

Torchsynth provides these core synthesis modules:

| Torchsynth Module | Purpose | Auralis Usage |
|------------------|---------|---------------|
| `VCO` (Voltage Controlled Oscillator) | Waveform generation | Lead melody, chord tones |
| `ADSR` | Envelope shaping | Note dynamics (attack, decay, sustain, release) |
| `LFO` (Low Frequency Oscillator) | Modulation source | Filter sweeps, vibrato |
| `Noise` | Noise generation | Percussion textures, ambient background |
| `SineVCO` | Pure sine waves | Sub-bass, smooth pads |
| `SquareSawVCO` | Square/saw waves | Harmonic-rich tones |
| `MonophonicKeyboard` | MIDI note handling | Melody voice management |

### 1.2 Custom Synthesizer Using Torchsynth

```python
# auralis/synthesis/torchsynth_engine.py
import torch
import numpy as np
from torchsynth.synth import AbstractSynth
from torchsynth.config import SynthConfig
from torchsynth.module import (
    SineVCO,
    SquareSawVCO,
    ADSR,
    MonophonicKeyboard,
    AudioMixer,
    Noise,
    LFO,
    ControlRateUpsample,
)
from typing import List, Tuple, Dict
from loguru import logger


class AmbientSynth(AbstractSynth):
    """
    Custom ambient synthesizer using torchsynth modules.

    Signal chain:
    - 2× oscillators (sine + square/saw) for pad
    - 1× sine oscillator for lead melody
    - ADSR envelope per voice
    - LFO for filter modulation
    - Noise generator for texture
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        buffer_size: int = 4410,  # 100ms at 44.1kHz
        device: str = "cpu",
    ):
        synthconfig = SynthConfig(
            batch_size=1,
            reproducible=False,
            sample_rate=sample_rate,
            buffer_size=buffer_size,
        )
        super().__init__(synthconfig)

        self.device = device

        # Keyboard for MIDI note control
        self.keyboard = MonophonicKeyboard(
            synthconfig,
            keyboard_octave=4,
            keyboard_note_number=69,  # A4 = 440Hz
        )

        # Oscillators for pad layer
        self.pad_osc1 = SineVCO(synthconfig)
        self.pad_osc2 = SquareSawVCO(synthconfig, tuning_max=2.0)

        # Lead oscillator
        self.lead_osc = SineVCO(synthconfig)

        # Envelopes
        self.pad_envelope = ADSR(
            synthconfig,
            attack=0.5,  # 500ms attack for pads
            decay=2.0,   # 2s decay
            sustain=0.7, # 70% sustain
            release=1.0, # 1s release
        )

        self.lead_envelope = ADSR(
            synthconfig,
            attack=0.05,  # 50ms attack for lead
            decay=0.1,    # 100ms decay
            sustain=0.8,  # 80% sustain
            release=0.2,  # 200ms release
        )

        # LFO for modulation
        self.lfo = LFO(synthconfig, mod_depth=0.3, frequency=0.5)  # 0.5 Hz

        # Noise generator for texture
        self.noise = Noise(synthconfig)

        # Audio mixer
        self.mixer = AudioMixer(synthconfig, n_input=4)  # 4 sources

        # Upsample control signals to audio rate
        self.upsample = ControlRateUpsample(synthconfig)

        # Move to device
        self.to(device)

    def forward(
        self,
        midi_note: torch.Tensor,  # Shape: (batch, 1) - MIDI note number
        note_on: torch.Tensor,    # Shape: (batch, 1) - 1.0 = note on, 0.0 = note off
        voice_type: str = "pad",  # "pad" or "lead"
    ) -> torch.Tensor:
        """
        Render audio for a single note.

        Args:
            midi_note: MIDI note number (e.g., 60 = C4)
            note_on: Note gate (1.0 = on, 0.0 = off)
            voice_type: Type of voice ("pad" or "lead")

        Returns:
            Audio tensor shape (batch, buffer_size)
        """
        # Set keyboard MIDI note
        self.keyboard.midi_f0 = midi_note
        pitch = self.keyboard()

        if voice_type == "pad":
            # Pad voice: two oscillators with LFO modulation
            osc1_out = self.pad_osc1(pitch)
            osc2_out = self.pad_osc2(pitch, mod_signal=self.lfo())

            # Apply envelope
            env = self.pad_envelope(note_on)
            env_upsampled = self.upsample(env)

            # Mix oscillators
            mixed = (osc1_out * 0.5 + osc2_out * 0.5) * env_upsampled

        elif voice_type == "lead":
            # Lead voice: single sine with sharp envelope
            lead_out = self.lead_osc(pitch)

            # Apply envelope
            env = self.lead_envelope(note_on)
            env_upsampled = self.upsample(env)

            mixed = lead_out * env_upsampled

        else:
            raise ValueError(f"Unknown voice_type: {voice_type}")

        return mixed


class TorchsynthAmbientEngine:
    """
    High-level synthesis engine using torchsynth.
    Handles phrase rendering with chords and melody.
    """

    def __init__(self, sample_rate: int = 44100, device: str = "cpu"):
        self.sr = sample_rate
        self.device = device
        self.synth = AmbientSynth(
            sample_rate=sample_rate,
            buffer_size=4410,  # 100ms chunks
            device=device,
        )
        logger.info(f"Initialized torchsynth engine on device: {device}")

    def render_phrase(
        self,
        chords: List[Tuple[int, int, str]],  # (onset_samples, root_midi, chord_type)
        melody: List[Tuple[int, int, float, float]],  # (onset, pitch, velocity, duration)
        percussion: List[Dict],
        duration_sec: float,
    ) -> np.ndarray:
        """
        Render a complete musical phrase.

        Returns:
            Stereo audio as numpy array, shape (2, num_samples)
        """
        num_samples = int(duration_sec * self.sr)
        audio = torch.zeros((2, num_samples), device=self.device)

        try:
            # Render chord layer
            chord_audio = self._render_chords(chords, num_samples)
            audio += chord_audio * 0.5  # -6dB for pads

            # Render melody layer
            melody_audio = self._render_melody(melody, num_samples)
            audio += melody_audio * 0.7  # -3dB for lead

            # Render percussion (uses noise generator)
            perc_audio = self._render_percussion(percussion, num_samples)
            audio += perc_audio * 0.3  # -10dB for percussion

            # Soft limiter
            audio = torch.tanh(audio * 0.95)

            # Convert to numpy
            return audio.cpu().numpy().astype(np.float32)

        except Exception as e:
            logger.error(f"Torchsynth rendering failed: {e}", exc_info=True)
            # Return silence on error
            return np.zeros((2, num_samples), dtype=np.float32)

    def _render_chords(
        self, chords: List[Tuple[int, int, str]], num_samples: int
    ) -> torch.Tensor:
        """Render chord progression using pad voice."""
        audio = torch.zeros((2, num_samples), device=self.device)

        for chord_idx, (onset, root_midi, chord_type) in enumerate(chords):
            # Get chord notes
            chord_notes = self._get_chord_notes(root_midi, chord_type)

            # Determine duration
            if chord_idx < len(chords) - 1:
                duration_samples = chords[chord_idx + 1][0] - onset
            else:
                duration_samples = num_samples - onset

            # Render each note in chord
            for note_midi in chord_notes:
                note_audio = self._render_note(
                    midi_note=note_midi,
                    onset_samples=onset,
                    duration_samples=duration_samples,
                    voice_type="pad",
                )

                # Add to audio (mono -> stereo duplicate for now)
                if onset + len(note_audio) <= num_samples:
                    audio[0, onset : onset + len(note_audio)] += note_audio
                    audio[1, onset : onset + len(note_audio)] += note_audio

        return audio

    def _render_melody(
        self, melody: List[Tuple[int, int, float, float]], num_samples: int
    ) -> torch.Tensor:
        """Render melody using lead voice."""
        audio = torch.zeros((2, num_samples), device=self.device)

        for onset, pitch_midi, velocity, duration_sec in melody:
            duration_samples = int(duration_sec * self.sr)

            note_audio = self._render_note(
                midi_note=pitch_midi,
                onset_samples=onset,
                duration_samples=duration_samples,
                voice_type="lead",
                velocity=velocity,
            )

            # Add to audio (centered pan)
            if onset + len(note_audio) <= num_samples:
                audio[0, onset : onset + len(note_audio)] += note_audio * velocity
                audio[1, onset : onset + len(note_audio)] += note_audio * velocity

        return audio

    def _render_percussion(
        self, percussion: List[Dict], num_samples: int
    ) -> torch.Tensor:
        """Render percussion using noise bursts."""
        audio = torch.zeros((2, num_samples), device=self.device)

        # Use synth's noise generator
        for perc_event in percussion:
            perc_type = perc_event.get("type", "kick")
            onset = perc_event.get("onset_sample", 0)
            velocity = perc_event.get("velocity", 1.0)

            if perc_type == "kick":
                # Generate kick using sine sweep (handled separately)
                kick_duration = int(0.1 * self.sr)
                # TODO: Implement kick synthesis
                pass

            elif perc_type == "swell":
                # Noise swell
                swell_duration = int(perc_event.get("duration_sec", 2.0) * self.sr)
                # Generate noise burst
                noise_burst = torch.randn(swell_duration, device=self.device) * velocity * 0.3

                # Apply envelope
                env = torch.linspace(0, 1, swell_duration // 4, device=self.device)
                env = torch.cat(
                    [
                        env,
                        torch.ones(swell_duration // 2, device=self.device),
                        torch.linspace(1, 0, swell_duration // 4, device=self.device),
                    ]
                )

                noise_burst = noise_burst[: len(env)] * env

                # Add to audio
                if onset + len(noise_burst) <= num_samples:
                    audio[0, onset : onset + len(noise_burst)] += noise_burst
                    audio[1, onset : onset + len(noise_burst)] += noise_burst

        return audio

    def _render_note(
        self,
        midi_note: int,
        onset_samples: int,
        duration_samples: int,
        voice_type: str = "pad",
        velocity: float = 1.0,
    ) -> torch.Tensor:
        """
        Render a single note using torchsynth.

        Returns:
            Audio tensor shape (duration_samples,)
        """
        # Convert to tensors
        midi_tensor = torch.tensor([[float(midi_note)]], device=self.device)

        # Generate note-on gate signal
        note_on = torch.ones((1, 1), device=self.device)

        # Render note
        # Note: torchsynth renders fixed buffer_size chunks
        # We need to concatenate multiple chunks to reach duration_samples
        num_chunks = int(np.ceil(duration_samples / self.synth.buffer_size))
        chunks = []

        for i in range(num_chunks):
            # Last chunk may need note-off
            if i == num_chunks - 1:
                note_on = torch.zeros((1, 1), device=self.device)

            chunk = self.synth(
                midi_note=midi_tensor, note_on=note_on, voice_type=voice_type
            )
            chunks.append(chunk.squeeze(0))

        # Concatenate and trim to exact duration
        full_note = torch.cat(chunks)[:duration_samples]

        return full_note * velocity

    def _get_chord_notes(self, root_midi: int, chord_type: str) -> List[int]:
        """Get MIDI notes for a chord."""
        from auralis.music.theory import CHORD_INTERVALS

        intervals = CHORD_INTERVALS.get(chord_type, [0, 4, 7])  # Default major
        notes = [root_midi + interval for interval in intervals]

        # Add octave below for richness
        notes.append(root_midi - 12)

        return notes[:4]  # Max 4 voices
```

---

## 2. Backend Detection and Fallback Strategy

### 2.1 Detection Logic

```python
# auralis/core/device_manager.py
import torch
import platform
from typing import Tuple, Literal
from loguru import logger

DeviceType = Literal["mps", "cuda", "cpu"]


def detect_device() -> Tuple[DeviceType, str]:
    """
    Detect optimal device with comprehensive checks.

    Returns:
        (device_type, device_string)
    """
    # 1. Check Apple Metal (MPS)
    if platform.system() == "Darwin":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                # Test MPS with a small operation
                test_tensor = torch.zeros(1, device="mps")
                logger.info("✓ Apple Metal (MPS) available and functional")
                return "mps", "mps"
            except Exception as e:
                logger.warning(f"MPS available but not functional: {e}")

    # 2. Check NVIDIA CUDA
    if torch.cuda.is_available():
        try:
            # Test CUDA
            test_tensor = torch.zeros(1, device="cuda:0")
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ CUDA available: {device_name}")
            return "cuda", "cuda:0"
        except Exception as e:
            logger.warning(f"CUDA available but not functional: {e}")

    # 3. CPU fallback
    logger.warning("⚠ No GPU detected. Using CPU (performance degraded)")
    return "cpu", "cpu"
```

### 2.2 Fallback Chain

```
┌─────────────────────────────────────┐
│  1. Torchsynth on MPS/CUDA (GPU)   │
│     Target: >50× real-time          │
└──────────┬──────────────────────────┘
           │ Fails
           ▼
┌─────────────────────────────────────┐
│  2. Torchsynth on CPU               │
│     Target: >5× real-time           │
└──────────┬──────────────────────────┘
           │ Fails or too slow
           ▼
┌─────────────────────────────────────┐
│  3. Numpy-based Simple Synth        │
│     Pure Python, guaranteed to work │
│     Target: >2× real-time           │
└──────────┬──────────────────────────┘
           │ Fails
           ▼
┌─────────────────────────────────────┐
│  4. Stream Silence + Log Error      │
│     Graceful degradation            │
└─────────────────────────────────────┘
```

### 2.3 Fallback Implementation

```python
# auralis/synthesis/engine_factory.py
from typing import Protocol
import numpy as np
from loguru import logger


class SynthesisEngine(Protocol):
    """Protocol that all synthesis engines must implement."""

    def render_phrase(
        self,
        chords: list,
        melody: list,
        percussion: list,
        duration_sec: float,
    ) -> np.ndarray:
        """Render phrase to stereo audio."""
        ...


class SynthesisEngineFactory:
    """Factory for creating synthesis engines with fallbacks."""

    @staticmethod
    def create_engine(sample_rate: int = 44100) -> SynthesisEngine:
        """
        Create best available synthesis engine.

        Tries in order:
        1. Torchsynth on GPU
        2. Torchsynth on CPU
        3. Numpy fallback synth
        """
        from auralis.core.device_manager import detect_device

        device_type, device_str = detect_device()

        # Try torchsynth
        try:
            from auralis.synthesis.torchsynth_engine import TorchsynthAmbientEngine

            engine = TorchsynthAmbientEngine(sample_rate=sample_rate, device=device_str)
            logger.info(f"✓ Using torchsynth engine on {device_str}")
            return engine

        except ImportError as e:
            logger.warning(f"Torchsynth not available: {e}")

        except Exception as e:
            logger.error(f"Torchsynth initialization failed: {e}")

        # Fallback to numpy-based synth
        try:
            from auralis.synthesis.numpy_synth import NumpySimpleSynth

            engine = NumpySimpleSynth(sample_rate=sample_rate)
            logger.warning("⚠ Using numpy fallback synth (limited quality)")
            return engine

        except Exception as e:
            logger.critical(f"All synthesis engines failed: {e}")
            # Return silence engine as last resort
            from auralis.synthesis.silence_engine import SilenceEngine

            return SilenceEngine(sample_rate=sample_rate)
```

---

## 3. Cross-Platform Compatibility

### 3.1 Platform-Specific Configuration

| Platform | Optimal Device | Fallback | Notes |
|----------|---------------|----------|-------|
| **macOS (Apple Silicon)** | `mps` | `cpu` | M1/M2/M3/M4 chips, requires macOS 12.3+ |
| **Linux (NVIDIA GPU)** | `cuda:0` | `cpu` | Requires CUDA 11.8+ drivers |
| **Linux (CPU only)** | `cpu` | N/A | Works but slow (2-5× real-time) |
| **Windows (NVIDIA GPU)** | `cuda:0` | `cpu` | CUDA support, MPS not available |
| **Windows (CPU only)** | `cpu` | N/A | Works but slow |

### 3.2 Installation Instructions by Platform

**macOS (Apple Silicon)**
```bash
# Install PyTorch with MPS support
uv pip install torch>=2.5.0 torchaudio>=2.5.0

# Verify MPS
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

**Linux (NVIDIA GPU)**
```bash
# Install PyTorch with CUDA 12.1
uv pip install torch>=2.5.0+cu121 torchaudio>=2.5.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**CPU-Only (Any Platform)**
```bash
# Standard PyTorch (CPU)
uv pip install torch>=2.5.0 torchaudio>=2.5.0
```

---

## 4. Performance Benchmarks

### 4.1 Expected Performance

Based on torchsynth documentation (16,200× real-time on reference hardware):

| Hardware | Expected RTF | 8-bar Phrase (22s) | Notes |
|----------|-------------|-------------------|-------|
| **M4 Mac (MPS)** | 50-100× | 220-440ms | Excellent for production |
| **M1 Mac (MPS)** | 30-50× | 440-733ms | Good for production |
| **RTX 3080 (CUDA)** | 100-200× | 110-220ms | Excellent for production |
| **RTX 2060 (CUDA)** | 50-80× | 275-440ms | Good for production |
| **8-core CPU** | 3-8× | 2.75-7.3s | Development only |
| **4-core CPU** | 1-3× | 7.3-22s | Barely real-time |

**RTF** = Real-Time Factor (how many times faster than real-time playback)

### 4.2 Profiling Code

```python
# auralis/profiling/benchmark.py
import time
import torch
from loguru import logger


def benchmark_synthesis(engine, test_phrase: dict, num_runs: int = 10):
    """
    Benchmark synthesis performance.

    Args:
        engine: Synthesis engine instance
        test_phrase: Dict with chords, melody, percussion
        num_runs: Number of benchmark runs
    """
    times = []
    duration_sec = 22.0  # 8 bars at 70 BPM

    logger.info(f"Running {num_runs} benchmark iterations...")

    for i in range(num_runs):
        start = time.perf_counter()

        audio = engine.render_phrase(
            chords=test_phrase["chords"],
            melody=test_phrase["melody"],
            percussion=test_phrase["percussion"],
            duration_sec=duration_sec,
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        logger.debug(f"Run {i+1}: {elapsed:.3f}s")

    mean_time = np.mean(times)
    std_time = np.std(times)
    rtf = duration_sec / mean_time

    logger.info("=== Benchmark Results ===")
    logger.info(f"Mean render time: {mean_time:.3f}s ± {std_time:.3f}s")
    logger.info(f"Real-time factor: {rtf:.1f}×")
    logger.info(f"For 1 minute of audio: {60 / rtf:.2f}s render time")

    if rtf < 2:
        logger.error("❌ Performance CRITICAL: Cannot maintain real-time streaming")
    elif rtf < 5:
        logger.warning("⚠ Performance WARNING: Limited headroom for real-time")
    else:
        logger.success(f"✓ Performance GOOD: {rtf:.1f}× real-time factor")

    return {"mean": mean_time, "std": std_time, "rtf": rtf}
```

---

## 5. Known Limitations

### 5.1 Torchsynth Limitations

1. **Fixed Buffer Size**: Torchsynth renders in fixed-size chunks (buffer_size). Variable-length notes require concatenation.

2. **Limited Polyphony**: Each synth instance is typically monophonic. For chords, need to render each note separately and mix.

3. **No Built-in Effects**: Reverb, delay, etc. must be implemented separately (use pedalboard or custom).

4. **MPS Limitations** (Apple Silicon):
   - Some operations may fall back to CPU
   - Not all PyTorch ops are MPS-optimized
   - Memory transfers between MPS and CPU can be slow

5. **Project Maintenance**: Torchsynth is inactive (last update 2021). May have compatibility issues with future PyTorch versions.

### 5.2 Mitigation Strategies

- **Fixed buffers**: Pre-calculate total duration, concatenate chunks
- **Polyphony**: Render voices in parallel, mix outputs
- **Effects**: Use pedalboard for reverb/delay (Phase 3)
- **MPS issues**: Comprehensive testing, CPU fallback
- **Maintenance**: Pin torchsynth version, maintain fork if needed

---

## 6. References

- [PyTorch MPS Backend Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [torchsynth on PyPI](https://pypi.org/project/torchsynth/)
- [torchsynth GitHub](https://github.com/torchsynth/torchsynth)
- [Accelerated PyTorch on Mac](https://developer.apple.com/metal/pytorch/)
