# FluidSynth Integration Specification
## Sample-Based Synthesis for Realistic Instrument Voices

**Version**: 1.0
**Status**: Planning
**Created**: 2025-12-28
**Target**: Auralis v0.3.0+

---

## Executive Summary

This specification outlines the integration of **FluidSynth** sample-based synthesis to replace basic oscillator-based synthesis with realistic instrument voices (piano, harpsichord, synth pads).

**Key Goals:**
- Replace basic oscillators with realistic instrument timbres using SoundFonts
- Maintain <100ms audio processing latency constraint
- Preserve real-time streaming performance (44.1kHz/16-bit PCM)
- Provide immediate, production-ready realistic instrument sounds

**Approach:**
1. **Phase 1 (MVP)**: FluidSynth integration for piano, harpsichord, pads (1-2 weeks)
2. **Phase 2 (Optimization)**: CPU performance tuning and multi-client testing (1 week)

---

## Current State Analysis

### Existing Synthesis Architecture

**Location**: `server/synthesis_engine.py`

**Current Voice Modules** (PyTorch-based):
```python
AmbientPadVoice    # Dual oscillators (sine + sawtooth) + LFO
LeadVoice          # Pure sine with ADSR
KickVoice          # Frequency sweep (150Hz → 40Hz)
SwellVoice         # Filtered noise with LFO
```

**Performance Characteristics**:
- Apple Silicon M4: 50-100× real-time
- NVIDIA RTX 3080: 100-200× real-time
- GPU-accelerated via Metal/CUDA
- JIT-compiled kernels with batch processing

**Limitations**:
- Basic timbres lack realism (sine/sawtooth oscillators)
- No acoustic instrument modeling
- Limited expressiveness for melodic content

---

## Phase 1: FluidSynth Integration (MVP)

### 1.1 Technical Architecture

**Component Overview**:
```
┌─────────────────────────────────────────────────────────────┐
│                    Composition Layer                        │
│  (chord_generator, melody_generator - UNCHANGED)            │
└────────────────────┬────────────────────────────────────────┘
                     │ Musical events (MIDI-like)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               Synthesis Engine (MODIFIED)                   │
│  ┌──────────────────────┐  ┌──────────────────────────┐    │
│  │  PyTorch Voices      │  │  FluidSynth Voices       │    │
│  │  - KickVoice         │  │  - PianoVoice            │    │
│  │  - SwellVoice        │  │  - HarpsichordVoice      │    │
│  │  (percussion/FX)     │  │  - AmbientPadVoice       │    │
│  └──────────────────────┘  └──────────────────────────┘    │
│           │                           │                     │
│           └───────────┬───────────────┘                     │
│                       ▼                                     │
│              Audio Mixer (Stereo Mix)                       │
└───────────────────────┬─────────────────────────────────────┘
                        │ PCM audio (44.1kHz/16-bit)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          Streaming Layer (UNCHANGED)                        │
│  (ring_buffer, WebSocket streaming)                         │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decisions**:
- **Hybrid approach**: Keep PyTorch voices for percussion/FX, use FluidSynth for melodic instruments
- **CPU-based synthesis**: FluidSynth runs on CPU (no GPU acceleration)
- **Per-phrase rendering**: FluidSynth generates complete phrase audio upfront
- **No streaming mode**: Render entire phrase to NumPy array, then stream via existing pipeline

### 1.2 FluidSynth Voice Interface

**New Abstract Base Class**:
```python
# server/interfaces/synthesis.py (EXTEND)

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class SampleBasedVoice(ABC):
    """Base class for sample-based synthesis (FluidSynth, etc.)"""

    @abstractmethod
    def load_soundfont(self, sf2_path: str, preset: int) -> None:
        """Load SoundFont and select preset."""
        pass

    @abstractmethod
    def render_notes(
        self,
        notes: List[Tuple[int, int, float, float]],  # (onset_sample, pitch, velocity, duration)
        duration_samples: int,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Render MIDI-like note events to audio.

        Returns:
            Audio array, shape (num_samples,) mono or (2, num_samples) stereo
        """
        pass

    @abstractmethod
    def note_on(self, channel: int, pitch: int, velocity: int) -> None:
        """Trigger note."""
        pass

    @abstractmethod
    def note_off(self, channel: int, pitch: int) -> None:
        """Release note."""
        pass
```

**FluidSynth Implementation**:
```python
# server/fluidsynth_voice.py (NEW FILE)

import fluidsynth
import numpy as np
from typing import List, Tuple
from server.interfaces.synthesis import SampleBasedVoice

class FluidSynthVoice(SampleBasedVoice):
    """FluidSynth-based voice for realistic instrument synthesis."""

    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.fs = fluidsynth.Synth(samplerate=sample_rate)
        self.fs.start(driver="file")  # Render to memory, not audio device
        self.sfid = None

    def load_soundfont(self, sf2_path: str, preset: int = 0) -> None:
        """Load SoundFont file and select instrument preset."""
        self.sfid = self.fs.sfload(sf2_path)
        self.fs.program_select(0, self.sfid, 0, preset)

    def render_notes(
        self,
        notes: List[Tuple[int, int, float, float]],
        duration_samples: int,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        Render MIDI-like note events to audio.

        Args:
            notes: List of (onset_sample, pitch_midi, velocity_0_1, duration_sec)
            duration_samples: Total phrase duration in samples
            sample_rate: Target sample rate (must match self.sample_rate)

        Returns:
            Stereo audio, shape (2, duration_samples), dtype float32
        """
        if sample_rate != self.sample_rate:
            raise ValueError(f"Sample rate mismatch: {sample_rate} != {self.sample_rate}")

        # Pre-allocate output buffer
        audio_buffer = np.zeros((duration_samples, 2), dtype=np.float32)

        # Sort notes by onset time
        notes_sorted = sorted(notes, key=lambda n: n[0])

        # Schedule all note events
        events = []
        for onset_sample, pitch, velocity, duration in notes_sorted:
            velocity_midi = int(velocity * 127)
            events.append(('noteon', onset_sample, pitch, velocity_midi))
            events.append(('noteoff', onset_sample + int(duration * sample_rate), pitch))

        events.sort(key=lambda e: e[1])  # Sort by time

        # Render audio with scheduled events
        current_sample = 0
        for event_type, event_sample, pitch, *args in events:
            # Render silence up to event
            if event_sample > current_sample:
                chunk_size = event_sample - current_sample
                chunk = self.fs.get_samples(chunk_size)
                audio_buffer[current_sample:event_sample] = chunk
                current_sample = event_sample

            # Trigger event
            if event_type == 'noteon':
                velocity = args[0]
                self.fs.noteon(0, pitch, velocity)
            elif event_type == 'noteoff':
                self.fs.noteoff(0, pitch)

        # Render remaining audio (release tails)
        if current_sample < duration_samples:
            chunk = self.fs.get_samples(duration_samples - current_sample)
            audio_buffer[current_sample:] = chunk

        # Convert to (2, num_samples) format
        return audio_buffer.T

    def note_on(self, channel: int, pitch: int, velocity: int) -> None:
        """Trigger note (for interactive use)."""
        self.fs.noteon(channel, pitch, velocity)

    def note_off(self, channel: int, pitch: int) -> None:
        """Release note (for interactive use)."""
        self.fs.noteoff(channel, pitch)

    def __del__(self):
        """Cleanup FluidSynth instance."""
        if hasattr(self, 'fs'):
            self.fs.delete()
```

### 1.3 Integration with SynthesisEngine

**Modify**: `server/synthesis_engine.py`

**Changes Required**:
1. Import FluidSynthVoice
2. Initialize FluidSynth voices in `__init__`
3. Load SoundFonts from configurable paths
4. Modify rendering methods to use FluidSynth for melodic voices
5. Mix PyTorch and FluidSynth outputs

**Updated `__init__` method**:
```python
def __init__(self, device: torch.device, sample_rate: int = 44100):
    self.device = device
    self.sample_rate = sample_rate

    # Existing PyTorch voices (keep for percussion/FX)
    self.kick_voice = KickVoice(sample_rate, device)
    self.swell_voice = SwellVoice(sample_rate, device)

    # NEW: FluidSynth voices for melodic instruments
    self.piano_voice = FluidSynthVoice(sample_rate, channels=2)
    self.harpsichord_voice = FluidSynthVoice(sample_rate, channels=2)
    self.pad_voice = FluidSynthVoice(sample_rate, channels=2)

    # Load SoundFonts (paths from config/environment)
    self._load_soundfonts()

    # Pre-allocated buffers (existing)
    self.max_duration_sec = 32.0
    self.max_samples = int(self.max_duration_sec * sample_rate)
    self.buffer = torch.zeros(2, self.max_samples, device=device, dtype=torch.float32)

def _load_soundfonts(self):
    """Load SoundFont files for each voice."""
    import os
    sf_dir = os.getenv("AURALIS_SOUNDFONT_DIR", "./soundfonts")

    # Load piano (preset 0 = Acoustic Grand Piano)
    piano_path = os.path.join(sf_dir, "piano.sf2")
    self.piano_voice.load_soundfont(piano_path, preset=0)

    # Load harpsichord (preset 6 = Harpsichord)
    harpsi_path = os.path.join(sf_dir, "harpsichord.sf2")
    self.harpsichord_voice.load_soundfont(harpsi_path, preset=6)

    # Load pad (preset 88 = Warm Pad or custom SF2)
    pad_path = os.path.join(sf_dir, "pads.sf2")
    self.pad_voice.load_soundfont(pad_path, preset=88)
```

**Updated rendering method**:
```python
def render_phrase(
    self,
    chords: List[Tuple[int, int, str]],
    melody: List[Tuple[int, int, float, float]],
    percussion: List[Tuple[int, str, float]],
    duration_sec: float
) -> np.ndarray:
    """
    Render complete musical phrase mixing PyTorch and FluidSynth voices.

    Returns:
        Stereo audio, shape (2, num_samples), dtype float32
    """
    duration_samples = int(duration_sec * self.sample_rate)

    # 1. Render chord pads (FluidSynth)
    chord_notes = self._chords_to_notes(chords, duration_sec)
    pad_audio_np = self.pad_voice.render_notes(chord_notes, duration_samples)

    # 2. Render melody (FluidSynth - piano or harpsichord based on config)
    melody_audio_np = self.piano_voice.render_notes(melody, duration_samples)

    # 3. Render percussion (PyTorch - keep existing)
    kick_audio = self._render_kicks(percussion, duration_samples)
    swell_audio = self._render_swells(percussion, duration_samples)

    # Convert PyTorch to NumPy
    kick_audio_np = kick_audio.cpu().numpy()
    swell_audio_np = swell_audio.cpu().numpy()

    # 4. Mix all layers
    mixed = (
        pad_audio_np * 0.4 +       # Pad layer (40%)
        melody_audio_np * 0.5 +    # Melody layer (50%)
        kick_audio_np * 0.3 +      # Kick layer (30%)
        swell_audio_np * 0.2       # Swell layer (20%)
    )

    # 5. Soft clipping
    mixed = np.tanh(mixed * 0.7)

    return mixed.astype(np.float32)

def _chords_to_notes(
    self,
    chords: List[Tuple[int, int, str]],
    duration_sec: float
) -> List[Tuple[int, int, float, float]]:
    """
    Convert chord events to individual note events.

    Args:
        chords: List of (onset_sample, root_midi, chord_type)
        duration_sec: Total phrase duration

    Returns:
        List of (onset_sample, pitch, velocity, duration)
    """
    notes = []
    chord_duration = 2.0  # Seconds per chord (adjust based on BPM)

    for onset, root, chord_type in chords:
        # Get chord intervals
        intervals = CHORD_INTERVALS.get(chord_type, [0, 4, 7])  # Default to major

        # Bass note
        notes.append((onset, root - 12, 0.5, chord_duration))

        # Chord tones
        for i, interval in enumerate(intervals):
            pitch = root + interval
            velocity = 0.45 + i * 0.025  # Slightly increase velocity for upper voices
            notes.append((onset, pitch, velocity, chord_duration))

    return notes
```

### 1.4 SoundFont Acquisition

**Recommended Free SoundFonts**:

1. **Piano**:
   - [Salamander Grand Piano](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html) (48kHz, 24-bit, ~200MB)
   - [FluidR3_GM.sf2](http://www.musescore.org/download/fluid-soundfont.tar.gz) (General MIDI set with piano)

2. **Harpsichord**:
   - FluidR3_GM.sf2 (includes harpsichord preset)
   - [Versilian Community Sample Library](https://github.com/sgossner/VCSL) (comprehensive, 1.5GB)

3. **Ambient Pads**:
   - [Arachno SoundFont](https://www.arachnosoft.com/main/soundfont.php) (high-quality pads, ~150MB)
   - Custom pads from [Creative Commons SF2 collections](https://musical-artifacts.com/artifacts?tags=soundfont)

**Directory Structure**:
```
soundfonts/
├── piano.sf2           # Salamander or FluidR3
├── harpsichord.sf2     # FluidR3 or VCSL
└── pads.sf2            # Arachno or custom
```

**Environment Configuration**:
```bash
# .env
AURALIS_SOUNDFONT_DIR=/path/to/soundfonts
AURALIS_PIANO_PRESET=0          # Acoustic Grand Piano
AURALIS_HARPSICHORD_PRESET=6    # Harpsichord
AURALIS_PAD_PRESET=88           # Warm Pad
```

### 1.5 Dependencies

**Add to `pyproject.toml`**:
```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "pyfluidsynth>=1.3.2",  # FluidSynth Python bindings
]
```

**System Dependencies** (required for FluidSynth):
```bash
# macOS
brew install fluidsynth

# Ubuntu/Debian
sudo apt-get install fluidsynth libfluidsynth-dev

# Windows (via vcpkg or pre-built binaries)
vcpkg install fluidsynth
```

**Installation command**:
```bash
uv add pyfluidsynth
```

---

## Phase 2: Performance Optimization

### 2.1 Latency Benchmarking

**Test Scenarios**:
1. Single phrase rendering (8 bars, 70 BPM)
2. Concurrent rendering (4 phrases in parallel)
3. Multi-client streaming (10+ concurrent WebSocket clients)

**Benchmark Script**: `tests/performance/test_fluidsynth_latency.py`

```python
import pytest
import time
from server.synthesis_engine import SynthesisEngine
from composition.chord_generator import ChordGenerator
from composition.melody_generator import MelodyGenerator

def test_fluidsynth_phrase_latency():
    """Measure end-to-end phrase rendering latency."""
    engine = SynthesisEngine(device=torch.device("cpu"), sample_rate=44100)
    chord_gen = ChordGenerator(key="C", mode="dorian")
    melody_gen = MelodyGenerator()

    # Generate musical events
    chords = chord_gen.generate_progression(num_chords=8)
    melody = melody_gen.generate_melody(chords)
    percussion = []

    # Measure rendering time
    start = time.perf_counter()
    audio = engine.render_phrase(chords, melody, percussion, duration_sec=16.0)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Verify latency constraint
    assert elapsed_ms < 100, f"Latency {elapsed_ms:.2f}ms exceeds 100ms target"

    # Calculate real-time factor
    rtf = 16000.0 / elapsed_ms  # 16 seconds / elapsed time
    print(f"Real-time factor: {rtf:.1f}x")
    assert rtf > 1.0, "Not achieving real-time synthesis"

def test_concurrent_phrase_rendering():
    """Test parallel phrase rendering for multiple clients."""
    import concurrent.futures

    engine = SynthesisEngine(device=torch.device("cpu"))

    def render_phrase_task():
        # ... generate and render phrase
        pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(render_phrase_task) for _ in range(10)]
        results = [f.result() for f in futures]

    # Verify all completed successfully
    assert all(r is not None for r in results)
```

**Performance Targets**:
- Single phrase: <100ms rendering time (requirement: <100ms)
- Real-time factor: >10× (16s audio in <1.6s)
- 10 concurrent clients: <500ms total latency per client

### 2.2 CPU Optimization Strategies

**If latency exceeds targets**:

1. **Reduce SoundFont size**:
   - Use smaller SF2 files with fewer velocity layers
   - Downsample to 22.05kHz (if 44.1kHz too heavy)

2. **Optimize FluidSynth settings**:
   ```python
   fs = fluidsynth.Synth(samplerate=44100)
   fs.setting('synth.reverb.active', 'no')   # Disable reverb
   fs.setting('synth.chorus.active', 'no')   # Disable chorus
   fs.setting('synth.polyphony', '32')       # Limit polyphony
   fs.setting('audio.period-size', '64')     # Smaller buffer
   ```

3. **Pre-render common patterns**:
   - Cache frequently used chord voicings
   - Store rendered phrases in LRU cache

4. **Process pooling**:
   ```python
   # Use multiprocessing to distribute FluidSynth across CPU cores
   from multiprocessing import Pool

   def render_worker(voice_type, notes, duration):
       # Each worker has own FluidSynth instance
       pass

   with Pool(processes=4) as pool:
       results = pool.starmap(render_worker, tasks)
   ```

### 2.3 Memory Management

**Monitor**:
- SoundFont memory footprint (SF2 files loaded into RAM)
- Audio buffer allocations
- FluidSynth internal buffers

**Optimization**:
```python
# Lazy loading of SoundFonts
class LazyFluidSynthVoice(FluidSynthVoice):
    def __init__(self, sf2_path: str, preset: int):
        self._sf2_path = sf2_path
        self._preset = preset
        self._loaded = False
        super().__init__()

    def render_notes(self, *args, **kwargs):
        if not self._loaded:
            self.load_soundfont(self._sf2_path, self._preset)
            self._loaded = True
        return super().render_notes(*args, **kwargs)
```

---

## Future Enhancements

### Potential Future Directions

Once FluidSynth integration is stable and validated, potential enhancements include:

1. **Additional Effects Processing**:
   - Integrate pedalboard for reverb, chorus, delay
   - Per-instrument effect chains
   - Dynamic effect automation based on musical context

2. **Advanced SoundFont Management**:
   - Multiple SoundFont layers per instrument
   - Dynamic SoundFont switching based on musical mode/key
   - Velocity layer optimization for more expressive dynamics

3. **Custom Sample Libraries**:
   - Curated sample sets for specific ambient music aesthetics
   - Multi-sampled recordings of unique instruments
   - Custom articulations for evolving textures

4. **Alternative Synthesis Technologies**:
   - Neural audio synthesis (RAVE, DDSP) for GPU acceleration
   - Physical modeling for expressive instrument behavior
   - Wavetable synthesis for evolving pad textures

These enhancements can be evaluated based on user feedback and performance requirements after Phase 2 completion.

---

## Testing Strategy

### Unit Tests

**File**: `tests/unit/test_fluidsynth_voice.py`

```python
import pytest
import numpy as np
from server.fluidsynth_voice import FluidSynthVoice

def test_soundfont_loading():
    """Test SoundFont file loading."""
    voice = FluidSynthVoice(sample_rate=44100)
    voice.load_soundfont("soundfonts/piano.sf2", preset=0)
    assert voice.sfid is not None

def test_note_rendering():
    """Test basic note rendering."""
    voice = FluidSynthVoice(sample_rate=44100)
    voice.load_soundfont("soundfonts/piano.sf2", preset=0)

    # Single note: C4, velocity 0.8, duration 1.0s
    notes = [(0, 60, 0.8, 1.0)]
    audio = voice.render_notes(notes, duration_samples=44100)

    assert audio.shape == (2, 44100)  # Stereo
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) > 0.01  # Audio generated

def test_polyphonic_rendering():
    """Test polyphonic (chord) rendering."""
    voice = FluidSynthVoice()
    voice.load_soundfont("soundfonts/piano.sf2", preset=0)

    # C major chord: C4, E4, G4
    notes = [
        (0, 60, 0.7, 2.0),
        (0, 64, 0.7, 2.0),
        (0, 67, 0.7, 2.0),
    ]
    audio = voice.render_notes(notes, duration_samples=88200)  # 2 seconds

    assert audio.shape == (2, 88200)
    assert np.max(np.abs(audio)) > 0.05
```

### Integration Tests

**File**: `tests/integration/test_hybrid_synthesis.py`

```python
def test_hybrid_synthesis_pipeline():
    """Test complete synthesis pipeline with FluidSynth + PyTorch."""
    from server.synthesis_engine import SynthesisEngine
    from composition.chord_generator import ChordGenerator

    engine = SynthesisEngine(device=torch.device("cpu"))
    chord_gen = ChordGenerator(key="C", mode="aeolian")

    chords = chord_gen.generate_progression(num_chords=4)
    melody = [(0, 60, 0.8, 0.5), (22050, 64, 0.7, 0.5)]
    percussion = []

    audio = engine.render_phrase(chords, melody, percussion, duration_sec=8.0)

    assert audio.shape == (2, 352800)  # 8 seconds stereo
    assert -1.0 <= np.min(audio) <= 0.0  # Valid range
    assert 0.0 <= np.max(audio) <= 1.0

def test_audio_quality_metrics():
    """Test audio output quality (no clipping, appropriate RMS)."""
    # ... render phrase ...

    # Check for hard clipping
    assert np.sum(np.abs(audio) >= 0.99) == 0

    # Check RMS level (should be in reasonable range)
    rms = np.sqrt(np.mean(audio ** 2))
    assert 0.1 <= rms <= 0.5  # Not silent, not distorted
```

### Performance Tests

**File**: `tests/performance/test_fluidsynth_latency.py` (as detailed in Phase 2)

---

## Risk Assessment

### High-Risk Items

1. **CPU Latency Bottleneck**
   - **Risk**: FluidSynth may not meet <100ms latency on CPU
   - **Mitigation**: Benchmark early, optimize settings, consider RAVE migration
   - **Fallback**: Keep PyTorch oscillators, use FluidSynth for preview/offline only

2. **SoundFont Quality**
   - **Risk**: Free SF2 files may sound poor or have artifacts
   - **Mitigation**: Test multiple SoundFont sources, A/B comparison
   - **Fallback**: Purchase professional SoundFonts ($50-200)

3. **Cross-Platform Compatibility**
   - **Risk**: FluidSynth native library may have install issues on Windows
   - **Mitigation**: Provide detailed install docs, Docker image, pre-built wheels
   - **Fallback**: Platform-specific builds, cloud rendering

### Medium-Risk Items

4. **Memory Footprint**
   - **Risk**: Large SF2 files consume excessive RAM (500MB-1GB)
   - **Mitigation**: Use compressed SF2s, lazy loading, unload unused fonts
   - **Impact**: May limit concurrent clients on memory-constrained servers

5. **RAVE Training Complexity**
   - **Risk**: Training RAVE models requires ML expertise, quality data
   - **Mitigation**: Use pre-trained models, collaborate with audio ML community
   - **Impact**: Phase 3 may take longer than 4-6 weeks

### Low-Risk Items

6. **Integration Complexity**
   - **Risk**: Mixing PyTorch and FluidSynth outputs may cause sync issues
   - **Mitigation**: Well-defined interfaces, extensive integration tests
   - **Impact**: Minor debugging time

---

## Success Metrics

### Phase 1 (MVP)
- ✅ FluidSynth voices integrated for piano, harpsichord, pads
- ✅ Latency <100ms for single phrase rendering
- ✅ Audio quality subjectively better than oscillator-based synthesis
- ✅ No regressions in WebSocket streaming performance
- ✅ Unit + integration test coverage >80%

### Phase 2 (Optimization)
- ✅ Real-time factor >10× (can render 16s audio in <1.6s)
- ✅ Support 10+ concurrent WebSocket clients with <500ms latency
- ✅ Memory usage <500MB per server instance
- ✅ Cross-platform builds (macOS, Linux, Windows)
- ✅ Documentation complete with deployment guides

---

## Timeline

### Phase 1: FluidSynth MVP (1-2 weeks)
- **Day 1-2**: Install dependencies, acquire SoundFonts
- **Day 3-5**: Implement `FluidSynthVoice` class
- **Day 6-8**: Integrate with `SynthesisEngine`
- **Day 9-10**: Unit tests, integration tests
- **Day 11-14**: Bug fixes, documentation, code review

### Phase 2: Optimization (1 week)
- **Day 1-2**: Latency benchmarking
- **Day 3-4**: CPU optimization (settings, caching)
- **Day 5-6**: Multi-client stress testing
- **Day 7**: Performance documentation, deployment guide

**Total Estimated Time**: 2-3 weeks for production-ready FluidSynth implementation

---

## Open Questions

1. **SoundFont Selection**: Which specific SF2 files provide best quality/performance trade-off?
   - **Action**: Benchmark top 3 piano SF2s (Salamander, FluidR3, Maestro)

2. **Voice Allocation**: Should we use FluidSynth for all melodic content or just specific layers?
   - **Action**: Experiment with different configurations, gather user feedback

3. **Reverb/Effects**: Should we add reverb in FluidSynth or as separate pedalboard processing?
   - **Action**: Compare CPU overhead, audio quality of both approaches

4. **Streaming vs Batch**: Should we render complete phrases or stream in chunks?
   - **Action**: Latency analysis, determine if chunk-based rendering is feasible

5. **SoundFont Licensing**: Ensure all SoundFonts used are properly licensed for production use
   - **Action**: Review licenses for Salamander, FluidR3, and other selected SoundFonts

---

## References

### FluidSynth Documentation
- [FluidSynth official site](https://www.fluidsynth.org/)
- [pyFluidSynth GitHub](https://github.com/nwhitehead/pyfluidsynth)
- [sf2_loader documentation](https://pypi.org/project/sf2-loader/)

### SoundFont Resources
- [FreePatterns SoundFonts](https://freepats.zenvoid.org/)
- [Salamander Grand Piano](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html)
- [Musical Artifacts (CC-licensed)](https://musical-artifacts.com/artifacts?tags=soundfont)

### Auralis Internal Documentation
- [CLAUDE.md](../CLAUDE.md) - Project guidelines
- [system_architecture.md](./system_architecture.md) - Current architecture
- [Constitution](../.specify/memory/constitution.md) - Project principles

---

## Appendix A: FluidSynth Configuration Reference

```python
# Optimized settings for low-latency real-time synthesis
FLUIDSYNTH_SETTINGS = {
    'synth.reverb.active': 'no',         # Disable reverb (CPU intensive)
    'synth.chorus.active': 'no',         # Disable chorus
    'synth.polyphony': '32',             # Max simultaneous notes
    'synth.midi-channels': '16',         # Standard MIDI channels
    'synth.gain': '0.8',                 # Output gain (prevent clipping)
    'audio.period-size': '64',           # Buffer size (lower = lower latency)
    'audio.periods': '4',                # Number of buffers
    'synth.sample-rate': '44100',        # Sample rate
    'synth.overflow.important': 'yes',   # Don't drop notes if polyphony exceeded
    'synth.overflow.percussion': 'yes',  # Allow percussion overflow
}
```

## Appendix B: MIDI Preset Reference (General MIDI)

```python
# General MIDI Preset Numbers (for FluidR3_GM.sf2)
GM_PRESETS = {
    # Piano Family (0-7)
    'acoustic_grand_piano': 0,
    'bright_acoustic_piano': 1,
    'electric_grand_piano': 2,
    'honky_tonk_piano': 3,
    'electric_piano_1': 4,
    'electric_piano_2': 5,
    'harpsichord': 6,
    'clavinet': 7,

    # Chromatic Percussion (8-15)
    'celesta': 8,
    'glockenspiel': 9,
    'music_box': 10,
    # ...

    # Synth Pad (88-95)
    'pad_new_age': 88,
    'pad_warm': 89,
    'pad_polysynth': 90,
    'pad_choir': 91,
    'pad_bowed': 92,
    'pad_metallic': 93,
    'pad_halo': 94,
    'pad_sweep': 95,
}
```

---

**Document Version**: 1.0
**Last Updated**: 2025-12-28
**Author**: Claude Code (Auralis Development Team)
**Status**: Ready for Implementation
