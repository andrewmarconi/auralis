# Internal Interfaces Contract: Auralis Python ABCs

**Feature**: Internal Python Interfaces
**Branch**: `001-mvp-v2`
**Version**: 1.0.0
**Date**: 2025-12-28

## Overview

This document defines Abstract Base Classes (ABCs) for internal Python interfaces in the Auralis codebase. These interfaces enable:

1. **Dependency Injection**: Testable, swappable implementations
2. **Clear Contracts**: Well-defined boundaries between modules
3. **Type Safety**: mypy validation of interface compliance
4. **Modularity**: Independent testing of each layer

All interfaces are located in `server/interfaces/` and use Python's `abc` module.

---

## Synthesis Interfaces

### ISynthesisEngine

**Location**: `server/interfaces/synthesis.py`

**Purpose**: Orchestrates composition generation and FluidSynth rendering.

**Interface**:
```python
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from composition.chord_generator import ChordProgression
from composition.melody_generator import MelodyPhrase

class ISynthesisEngine(ABC):
    """Orchestrates music generation and audio synthesis."""

    @abstractmethod
    async def generate_phrase(
        self,
        context: MusicalContext,
        duration_bars: int = 8
    ) -> Tuple[ChordProgression, MelodyPhrase]:
        """
        Generate chord progression and melody for a phrase.

        Args:
            context: Current musical parameters (key, mode, BPM, intensity)
            duration_bars: Number of bars to generate (8 or 16)

        Returns:
            Tuple of (ChordProgression, MelodyPhrase)

        Raises:
            GenerationError: If composition fails
        """
        pass

    @abstractmethod
    async def render_phrase(
        self,
        chords: ChordProgression,
        melody: MelodyPhrase
    ) -> np.ndarray:
        """
        Render musical phrase to stereo PCM audio using FluidSynth.

        Args:
            chords: Generated chord progression
            melody: Generated melody phrase

        Returns:
            NumPy array, shape (2, num_samples), dtype float32, range [-1.0, 1.0]

        Raises:
            SynthesisError: If rendering fails or exceeds 100ms latency
        """
        pass

    @abstractmethod
    def get_device(self) -> str:
        """Return current synthesis device ("Metal", "CUDA", "CPU")."""
        pass
```

**Implementation**: `server/synthesis_engine.py`

**Example Usage**:
```python
engine: ISynthesisEngine = SynthesisEngine(
    chord_generator=chord_gen,
    melody_generator=melody_gen,
    fluidsynth_renderer=synth_renderer
)

context = MusicalContext(key=60, mode="aeolian", bpm=60, intensity=0.5)
chords, melody = await engine.generate_phrase(context, duration_bars=8)
audio = await engine.render_phrase(chords, melody)

assert audio.shape[0] == 2  # Stereo
assert audio.dtype == np.float32
assert np.all(np.abs(audio) <= 1.0)  # No clipping
```

---

### IFluidSynthRenderer

**Location**: `server/interfaces/synthesis.py`

**Purpose**: Wraps FluidSynth library for sample-based audio rendering.

**Interface**:
```python
from abc import ABC, abstractmethod
import numpy as np

class IFluidSynthRenderer(ABC):
    """FluidSynth wrapper for sample-based synthesis."""

    @abstractmethod
    def load_soundfont(self, sf2_path: str) -> int:
        """
        Load SoundFont file into FluidSynth.

        Args:
            sf2_path: Absolute path to .sf2 file

        Returns:
            SoundFont ID (for later preset selection)

        Raises:
            SoundFontLoadError: If file not found or corrupted
        """
        pass

    @abstractmethod
    def select_preset(self, channel: int, sf_id: int, bank: int, preset: int) -> None:
        """
        Assign SoundFont preset to MIDI channel.

        Args:
            channel: MIDI channel (0-15)
            sf_id: SoundFont ID from load_soundfont()
            bank: Bank number (typically 0)
            preset: Preset number (0-127, General MIDI)

        Raises:
            PresetError: If preset not found in SoundFont
        """
        pass

    @abstractmethod
    def note_on(self, channel: int, pitch: int, velocity: int) -> None:
        """
        Trigger note on MIDI channel.

        Args:
            channel: MIDI channel
            pitch: MIDI note number (0-127)
            velocity: Note velocity (0-127)
        """
        pass

    @abstractmethod
    def note_off(self, channel: int, pitch: int) -> None:
        """Release note on MIDI channel."""
        pass

    @abstractmethod
    def render(self, num_samples: int) -> np.ndarray:
        """
        Generate audio samples.

        Args:
            num_samples: Number of samples to render

        Returns:
            NumPy array, shape (2, num_samples), dtype float32

        Raises:
            RenderError: If synthesis fails
        """
        pass

    @abstractmethod
    def configure_reverb(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.2
    ) -> None:
        """
        Configure FluidSynth reverb settings.

        Args:
            room_size: Reverb room size (0.0-1.0)
            damping: Reverb damping (0.0-1.0)
            wet_level: Wet signal level (0.0-1.0)
        """
        pass
```

**Implementation**: `server/fluidsynth_renderer.py`

---

## Buffer Interfaces

### IRingBuffer

**Location**: `server/interfaces/buffer.py`

**Purpose**: Thread-safe circular buffer for audio chunks.

**Interface**:
```python
from abc import ABC, abstractmethod
from typing import Optional
from server.audio_chunk import AudioChunk

class IRingBuffer(ABC):
    """Thread-safe ring buffer for audio chunks."""

    @abstractmethod
    def write(self, chunk: AudioChunk) -> bool:
        """
        Add chunk to buffer.

        Args:
            chunk: Audio chunk to write

        Returns:
            True if written, False if buffer full

        Thread-safe: Uses internal lock
        """
        pass

    @abstractmethod
    def read(self) -> Optional[AudioChunk]:
        """
        Remove chunk from buffer.

        Returns:
            AudioChunk if available, None if empty

        Thread-safe: Uses internal lock
        """
        pass

    @abstractmethod
    def get_depth(self) -> int:
        """
        Return current buffer depth (number of buffered chunks).

        Returns:
            Integer in range [0, capacity]
        """
        pass

    @abstractmethod
    def is_full(self) -> bool:
        """Check if buffer is full (write would fail)."""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if buffer is empty (read would return None)."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all chunks from buffer."""
        pass
```

**Implementation**: `server/ring_buffer.py`

**Example Usage**:
```python
buffer: IRingBuffer = RingBuffer(capacity=20)

# Producer thread
chunk = AudioChunk(data=pcm_data, seq=1, timestamp=time.time())
if buffer.write(chunk):
    print(f"Buffered chunk {chunk.seq}, depth: {buffer.get_depth()}")
else:
    print("Buffer full, applying back-pressure")
    time.sleep(0.01)

# Consumer thread
chunk = buffer.read()
if chunk:
    await websocket.send_bytes(chunk.to_base64())
else:
    print("Buffer empty, underrun detected")
```

---

### IBufferManager

**Location**: `server/interfaces/buffer.py`

**Purpose**: Manages back-pressure and buffer health monitoring.

**Interface**:
```python
from abc import ABC, abstractmethod

class IBufferManager(ABC):
    """Manages buffer back-pressure and health monitoring."""

    @abstractmethod
    async def wait_for_space(self, timeout_sec: float = 1.0) -> bool:
        """
        Wait until buffer has space for writing.

        Args:
            timeout_sec: Maximum wait time

        Returns:
            True if space available, False if timeout

        Implements back-pressure: Sleeps if depth < 2 chunks
        """
        pass

    @abstractmethod
    def get_health_status(self) -> str:
        """
        Return buffer health status.

        Returns:
            One of: "healthy" (3-4 chunks), "low" (1-2), "emergency" (<1), "full" (5+)
        """
        pass

    @abstractmethod
    def record_underrun(self) -> None:
        """Record buffer underrun event (metrics tracking)."""
        pass

    @abstractmethod
    def record_overflow(self) -> None:
        """Record buffer overflow event (metrics tracking)."""
        pass
```

**Implementation**: `server/buffer_management.py`

---

## Metrics Interfaces

### IMetricsCollector

**Location**: `server/interfaces/metrics.py`

**Purpose**: Collects and aggregates performance metrics.

**Interface**:
```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class IMetricsCollector(ABC):
    """Collects performance metrics (latency, buffer health, memory)."""

    @abstractmethod
    def record_synthesis_latency(self, latency_ms: float) -> None:
        """Record synthesis timing sample."""
        pass

    @abstractmethod
    def record_network_latency(self, client_id: str, latency_ms: float) -> None:
        """Record network delivery timing for specific client."""
        pass

    @abstractmethod
    def record_end_to_end_latency(self, latency_ms: float) -> None:
        """Record total generation → playback latency."""
        pass

    @abstractmethod
    def record_buffer_underrun(self) -> None:
        """Increment underrun counter."""
        pass

    @abstractmethod
    def record_buffer_overflow(self) -> None:
        """Increment overflow counter."""
        pass

    @abstractmethod
    def record_disconnect(self) -> None:
        """Increment disconnect counter."""
        pass

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        """
        Return current metrics as JSON-serializable dict.

        Returns:
            Dict matching PerformanceMetrics schema (see data-model.md)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all counters and histograms (typically not used)."""
        pass
```

**Implementation**: `server/metrics.py`

---

### IMemoryMonitor

**Location**: `server/interfaces/metrics.py`

**Purpose**: Tracks memory usage and detects leaks.

**Interface**:
```python
from abc import ABC, abstractmethod

class IMemoryMonitor(ABC):
    """Monitors process memory usage and detects leaks."""

    @abstractmethod
    def get_current_usage_mb(self) -> float:
        """Return current memory usage in MB."""
        pass

    @abstractmethod
    def record_snapshot(self) -> None:
        """Take memory snapshot for leak detection."""
        pass

    @abstractmethod
    def detect_leak(self, threshold_mb: float = 10.0) -> bool:
        """
        Check if memory has grown beyond threshold since last snapshot.

        Args:
            threshold_mb: Growth threshold in MB

        Returns:
            True if leak suspected
        """
        pass

    @abstractmethod
    def get_gc_stats(self) -> Dict[str, int]:
        """
        Return garbage collection statistics.

        Returns:
            Dict with keys: gen0, gen1, gen2 (collection counts)
        """
        pass
```

**Implementation**: `server/memory_monitor.py`

---

## Client-Side Interfaces (JavaScript)

### IJitterTracker

**Location**: `server/interfaces/jitter.py` (TypeScript definitions for client)

**Purpose**: Tracks network jitter using exponential moving average.

**Interface** (TypeScript):
```typescript
interface IJitterTracker {
  /**
   * Record arrival time variance for a chunk.
   * @param expectedTime Expected arrival time (ms)
   * @param actualTime Actual arrival time (ms)
   */
  recordArrival(expectedTime: number, actualTime: number): void;

  /**
   * Get current jitter estimate (EMA).
   * @returns Jitter in milliseconds
   */
  getCurrentJitter(): number;

  /**
   * Get variance of arrival times.
   * @returns Variance in milliseconds squared
   */
  getVariance(): number;

  /**
   * Reset jitter tracking (e.g., after reconnect).
   */
  reset(): void;
}
```

**Implementation**: Client-side in `client/audio_client_worklet.js`

**Example**:
```javascript
class JitterTracker {
  constructor(alpha = 0.1) {
    this.alpha = alpha;  // EMA smoothing factor
    this.jitter = 0;
    this.variance = 0;
  }

  recordArrival(expectedTime, actualTime) {
    const error = Math.abs(actualTime - expectedTime);
    this.jitter = this.alpha * error + (1 - this.alpha) * this.jitter;
    this.variance = this.alpha * (error ** 2) + (1 - this.alpha) * this.variance;
  }

  getCurrentJitter() {
    return this.jitter;
  }

  getVariance() {
    return this.variance;
  }

  reset() {
    this.jitter = 0;
    this.variance = 0;
  }
}
```

---

## Testing Interfaces

All interfaces should be tested with:

1. **Type Checking**: `mypy server/ --strict` validates interface compliance
2. **Mock Implementations**: Test doubles for unit tests
3. **Integration Tests**: Real implementations in integration tests

### Example Mock

```python
class MockFluidSynthRenderer(IFluidSynthRenderer):
    """Mock FluidSynth for testing without actual synthesis."""

    def __init__(self):
        self.loaded_soundfonts = {}
        self.preset_channels = {}

    def load_soundfont(self, sf2_path: str) -> int:
        sf_id = len(self.loaded_soundfonts)
        self.loaded_soundfonts[sf_id] = sf2_path
        return sf_id

    def select_preset(self, channel: int, sf_id: int, bank: int, preset: int) -> None:
        self.preset_channels[channel] = (sf_id, bank, preset)

    def note_on(self, channel: int, pitch: int, velocity: int) -> None:
        pass  # No-op in mock

    def note_off(self, channel: int, pitch: int) -> None:
        pass  # No-op in mock

    def render(self, num_samples: int) -> np.ndarray:
        # Return silence
        return np.zeros((2, num_samples), dtype=np.float32)

    def configure_reverb(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.2
    ) -> None:
        pass  # No-op in mock


# Usage in tests
def test_synthesis_engine():
    mock_renderer = MockFluidSynthRenderer()
    engine = SynthesisEngine(
        chord_generator=chord_gen,
        melody_generator=melody_gen,
        fluidsynth_renderer=mock_renderer  # Inject mock
    )

    audio = await engine.render_phrase(chords, melody)
    assert audio.shape == (2, expected_samples)
```

---

## Dependency Injection Container

### Location: `server/di_container.py`

**Purpose**: Centralized dependency injection for all interfaces.

**Implementation**:
```python
from dependency_injector import containers, providers
from server.interfaces.synthesis import ISynthesisEngine, IFluidSynthRenderer
from server.interfaces.buffer import IRingBuffer, IBufferManager
from server.interfaces.metrics import IMetricsCollector, IMemoryMonitor

class DIContainer(containers.DeclarativeContainer):
    """Dependency injection container for Auralis."""

    # Configuration
    config = providers.Configuration()

    # Synthesis
    fluidsynth_renderer = providers.Singleton(
        FluidSynthRenderer,
        sample_rate=config.audio.sample_rate,
        soundfont_path=config.audio.soundfont_path
    )

    synthesis_engine = providers.Singleton(
        SynthesisEngine,
        chord_generator=providers.Factory(ChordGenerator),
        melody_generator=providers.Factory(MelodyGenerator),
        fluidsynth_renderer=fluidsynth_renderer
    )

    # Buffers
    ring_buffer = providers.Singleton(
        RingBuffer,
        capacity=config.buffer.capacity
    )

    buffer_manager = providers.Singleton(
        BufferManager,
        ring_buffer=ring_buffer
    )

    # Metrics
    metrics_collector = providers.Singleton(MetricsCollector)
    memory_monitor = providers.Singleton(MemoryMonitor)

# Usage
container = DIContainer()
container.config.from_yaml('config.yaml')

engine: ISynthesisEngine = container.synthesis_engine()
buffer: IRingBuffer = container.ring_buffer()
metrics: IMetricsCollector = container.metrics_collector()
```

---

## Interface Documentation Standards

All interfaces must include:

1. **Docstrings**: Full documentation for all methods
2. **Type Hints**: Complete type annotations (args, returns)
3. **Raises**: Document all exceptions
4. **Thread Safety**: Note if methods are thread-safe
5. **Performance**: Document any latency/memory requirements

**Example**:
```python
@abstractmethod
async def render_phrase(
    self,
    chords: ChordProgression,
    melody: MelodyPhrase
) -> np.ndarray:
    """
    Render musical phrase to stereo PCM audio using FluidSynth.

    This method MUST complete within 100ms (real-time audio constraint).
    FluidSynth synthesis is CPU-bound; GPU acceleration not available.

    Args:
        chords: Generated chord progression with onset times
        melody: Generated melody phrase with note events

    Returns:
        NumPy array, shape (2, num_samples), dtype float32
        Range: [-1.0, 1.0] (soft clipping applied)
        Duration: Calculated from BPM and bar count

    Raises:
        SynthesisError: If rendering exceeds 100ms latency
        SoundFontError: If SoundFonts not loaded

    Thread-safe: No (use one instance per synthesis thread)
    Performance: <100ms target, typically 30-50ms on M4/M2
    """
    pass
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-28 | Initial internal interfaces specification |

---

**Next Steps**:
- Implement ABC classes in `server/interfaces/`
- Create concrete implementations in `server/`
- Write mock implementations for testing
- Set up dependency injection container
- Run `mypy server/ --strict` to validate type compliance

**Related Contracts**:
- [websocket-api.md](websocket-api.md) - WebSocket streaming protocol
- [http-api.md](http-api.md) - REST endpoints
- [data-model.md](../data-model.md) - Entity definitions
