# Comprehensive Testing Strategy

## Overview

This document defines the testing approach for Auralis, covering unit tests, integration tests, performance benchmarks, and end-to-end testing.

---

## 1. Test Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── unit/                    # Fast, isolated tests
│   ├── test_chord_generator.py
│   ├── test_melody_generator.py
│   ├── test_percussion_generator.py
│   ├── test_ring_buffer.py
│   ├── test_config.py
│   ├── test_music_theory.py
│   └── test_device_manager.py
│
├── integration/             # Component interaction tests
│   ├── test_composition_engine.py
│   ├── test_synthesis_pipeline.py
│   ├── test_websocket_streaming.py
│   └── test_api_endpoints.py
│
├── performance/             # Benchmarks
│   ├── test_synthesis_benchmark.py
│   └── test_streaming_performance.py
│
└── e2e/                     # End-to-end browser tests
    └── test_client_playback.py
```

---

## 2. Unit Tests

### 2.1 Chord Generator Tests

```python
# tests/unit/test_chord_generator.py
import pytest
import numpy as np
from auralis.composition.chord_generator import ChordProgressionGenerator


def test_chord_generator_determinism():
    """Test that seeded generator produces reproducible output."""
    gen1 = ChordProgressionGenerator(seed=42)
    gen2 = ChordProgressionGenerator(seed=42)

    progression1 = gen1.generate(length_bars=8)
    progression2 = gen2.generate(length_bars=8)

    assert progression1 == progression2


def test_chord_generator_length():
    """Test that generator returns correct number of chords."""
    gen = ChordProgressionGenerator()

    for length in [4, 8, 16, 32]:
        progression = gen.generate(length_bars=length)
        assert len(progression) == length


def test_chord_generator_valid_chords():
    """Test that all generated chords are valid."""
    gen = ChordProgressionGenerator()
    valid_chords = {"i", "ii", "III", "iv", "v", "VI", "VII"}

    progression = gen.generate(length_bars=100)  # Large sample

    for chord in progression:
        assert chord in valid_chords


def test_transition_matrix_properties():
    """Test that transition matrix is valid stochastic matrix."""
    gen = ChordProgressionGenerator()

    # Each row should sum to 1.0
    row_sums = gen.transition_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0)

    # All probabilities should be non-negative
    assert (gen.transition_matrix >= 0).all()
```

### 2.2 Ring Buffer Tests

```python
# tests/unit/test_ring_buffer.py
import pytest
import numpy as np
from auralis.streaming.ring_buffer import RingBuffer


def test_ring_buffer_write_read():
    """Test basic write and read operations."""
    buffer = RingBuffer(sample_rate=44100, capacity_sec=1.0)

    # Write 100ms of audio
    audio = np.random.randn(2, 4410).astype(np.float32)
    buffer.write(audio)

    # Read back
    read_audio = buffer.read(4410)

    assert read_audio.shape == (2, 4410)
    assert buffer.buffer_depth_samples() == 0  # Buffer should be empty


def test_ring_buffer_wraparound():
    """Test that buffer correctly wraps around."""
    buffer = RingBuffer(sample_rate=44100, capacity_sec=0.5)  # Small buffer

    # Write more than capacity
    for _ in range(10):
        audio = np.random.randn(2, 4410).astype(np.float32)
        buffer.write(audio)

    # Should not overflow
    depth = buffer.buffer_depth_samples()
    capacity = int(0.5 * 44100)
    assert depth <= capacity


def test_ring_buffer_thread_safety():
    """Test concurrent writes and reads (simplified)."""
    import threading

    buffer = RingBuffer(sample_rate=44100, capacity_sec=2.0)
    errors = []

    def writer():
        try:
            for _ in range(100):
                audio = np.random.randn(2, 441).astype(np.float32)
                buffer.write(audio)
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(100):
                _ = buffer.read(441)
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
```

### 2.3 Configuration Tests

```python
# tests/unit/test_config.py
import pytest
from pydantic import ValidationError
from auralis.core.config import AuralisConfig


def test_config_defaults():
    """Test that config loads with sensible defaults."""
    config = AuralisConfig()

    assert config.port == 8000
    assert config.sample_rate == 44100
    assert config.default_bpm == 70


def test_config_validation_bpm():
    """Test BPM validation."""
    # Valid BPM
    config = AuralisConfig(default_bpm=70)
    assert config.default_bpm == 70

    # Invalid BPM
    with pytest.raises(ValidationError):
        AuralisConfig(default_bpm=300)  # Too high

    with pytest.raises(ValidationError):
        AuralisConfig(default_bpm=10)  # Too low


def test_config_validation_intensity():
    """Test intensity validation."""
    # Valid intensity
    config = AuralisConfig(default_intensity=0.5)
    assert config.default_intensity == 0.5

    # Invalid intensity
    with pytest.raises(ValidationError):
        AuralisConfig(default_intensity=1.5)  # Out of range
```

---

## 3. Integration Tests

### 3.1 Composition Engine Tests

```python
# tests/integration/test_composition_engine.py
import pytest
import asyncio
from auralis.composition.engine import CompositionEngine, GenerationParameters


@pytest.mark.asyncio
async def test_engine_generates_phrases():
    """Test that engine continuously generates phrases."""
    engine = CompositionEngine(sample_rate=44100, phrase_queue_size=2)
    await engine.start()

    # Get 3 phrases
    phrases = []
    for _ in range(3):
        phrase = await asyncio.wait_for(engine.get_next_phrase(), timeout=10.0)
        phrases.append(phrase)

    await engine.stop()

    # Verify all phrases are valid
    for phrase in phrases:
        assert phrase.phrase_id is not None
        assert len(phrase.chords) == 8  # Default bars_per_phrase
        assert phrase.duration_sec > 0


@pytest.mark.asyncio
async def test_engine_parameter_updates():
    """Test parameter updates propagate correctly."""
    engine = CompositionEngine(sample_rate=44100)
    await engine.start()

    # Update params
    await engine.update_params({"key": "D minor", "bpm": 90, "intensity": 0.8})

    # Get phrase with new params
    phrase = await asyncio.wait_for(engine.get_next_phrase(), timeout=10.0)

    assert phrase.key == "D minor"
    assert phrase.bpm == 90
    assert phrase.intensity == 0.8

    await engine.stop()


@pytest.mark.asyncio
async def test_engine_backpressure():
    """Test that engine respects queue size limits."""
    engine = CompositionEngine(sample_rate=44100, phrase_queue_size=2)
    await engine.start()

    # Let queue fill up
    await asyncio.sleep(2.0)

    # Queue should be at or near capacity
    queue_size = engine.phrase_queue.qsize()
    assert queue_size <= 2  # Should not exceed max size

    await engine.stop()
```

### 3.2 WebSocket Streaming Tests

```python
# tests/integration/test_websocket_streaming.py
import pytest
from fastapi.testclient import TestClient
from server.main import app


def test_websocket_connection():
    """Test WebSocket connection and basic streaming."""
    client = TestClient(app)

    with client.websocket_connect("/ws/stream") as websocket:
        # Receive first audio chunk
        data = websocket.receive_json()

        assert data["type"] == "audio"
        assert "data" in data  # base64 encoded audio
        assert data["sample_rate"] == 44100


def test_websocket_control_messages():
    """Test sending control messages."""
    client = TestClient(app)

    with client.websocket_connect("/ws/stream") as websocket:
        # Send control update
        websocket.send_json({
            "type": "control",
            "key": "C minor",
            "bpm": 80,
            "intensity": 0.7,
        })

        # Should still receive audio
        data = websocket.receive_json()
        assert data["type"] == "audio"


@pytest.mark.asyncio
async def test_multiple_clients():
    """Test that server handles multiple concurrent clients."""
    client = TestClient(app)

    # Connect multiple clients
    with client.websocket_connect("/ws/stream") as ws1:
        with client.websocket_connect("/ws/stream") as ws2:
            # Both should receive audio
            data1 = ws1.receive_json()
            data2 = ws2.receive_json()

            assert data1["type"] == "audio"
            assert data2["type"] == "audio"
```

---

## 4. Performance Benchmarks

### 4.1 Synthesis Benchmark

```python
# tests/performance/test_synthesis_benchmark.py
import pytest
import time
import numpy as np
from auralis.synthesis.engine_factory import SynthesisEngineFactory


def test_synthesis_performance(benchmark):
    """Benchmark synthesis render time."""
    engine = SynthesisEngineFactory.create_engine(sample_rate=44100)

    # Test phrase
    test_phrase = {
        "chords": [(0, 57, "i"), (22050, 57, "VI")],
        "melody": [(0, 60, 0.7, 1.0), (11025, 62, 0.7, 1.5)],
        "percussion": [],
    }

    # Benchmark
    result = benchmark(
        engine.render_phrase,
        **test_phrase,
        duration_sec=22.0,
    )

    # Verify output
    assert result.shape[0] == 2  # Stereo
    assert result.shape[1] == int(22.0 * 44100)


def test_real_time_factor():
    """Test that synthesis achieves acceptable real-time factor."""
    engine = SynthesisEngineFactory.create_engine(sample_rate=44100)

    duration_sec = 22.0  # 8 bars at 70 BPM
    test_phrase = {
        "chords": [(0, 57, "i")] * 8,
        "melody": [],
        "percussion": [],
    }

    start = time.perf_counter()
    audio = engine.render_phrase(**test_phrase, duration_sec=duration_sec)
    elapsed = time.perf_counter() - start

    rtf = duration_sec / elapsed

    print(f"\nReal-time factor: {rtf:.1f}×")
    print(f"Render time: {elapsed:.3f}s for {duration_sec:.1f}s of audio")

    # Minimum acceptable RTF (must be faster than real-time)
    assert rtf > 1.0, f"Synthesis too slow: {rtf:.1f}× real-time"

    # Warn if below optimal threshold
    if rtf < 5.0:
        pytest.warn(f"Low real-time factor: {rtf:.1f}× (target: >5×)")
```

### 4.2 Streaming Performance

```python
# tests/performance/test_streaming_performance.py
import pytest
import asyncio
import time


@pytest.mark.asyncio
async def test_streaming_latency():
    """Measure end-to-end streaming latency."""
    from server.main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    latencies = []

    with client.websocket_connect("/ws/stream") as websocket:
        for _ in range(10):
            send_time = time.time()

            # Receive chunk
            data = websocket.receive_json()

            receive_time = time.time()
            latency_ms = (receive_time - send_time) * 1000

            latencies.append(latency_ms)

    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)

    print(f"\nAvg latency: {avg_latency:.1f}ms")
    print(f"Max latency: {max_latency:.1f}ms")

    # Target: < 200ms average latency
    assert avg_latency < 200, f"High average latency: {avg_latency:.1f}ms"
```

---

## 5. Test Fixtures

```python
# tests/conftest.py
import pytest
import asyncio
import numpy as np
from auralis.composition.engine import CompositionEngine
from auralis.synthesis.engine_factory import SynthesisEngineFactory


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 44100


@pytest.fixture
def test_audio(sample_rate):
    """Generate test audio signal."""
    duration_sec = 1.0
    num_samples = int(duration_sec * sample_rate)
    return np.random.randn(2, num_samples).astype(np.float32) * 0.1


@pytest.fixture
def test_phrase():
    """Standard test phrase for synthesis."""
    return {
        "chords": [(0, 57, "i"), (22050, 60, "iv"), (44100, 62, "V")],
        "melody": [
            (0, 60, 0.7, 1.0),
            (11025, 62, 0.75, 1.5),
            (33075, 64, 0.7, 2.0),
        ],
        "percussion": [
            {"type": "kick", "onset_sample": 0, "velocity": 0.8},
            {"type": "swell", "onset_sample": 22050, "duration_sec": 2.0, "velocity": 0.6},
        ],
    }


@pytest.fixture
async def composition_engine(sample_rate):
    """Create and start a composition engine."""
    engine = CompositionEngine(sample_rate=sample_rate, phrase_queue_size=2)
    await engine.start()
    yield engine
    await engine.stop()


@pytest.fixture
def synthesis_engine(sample_rate):
    """Create synthesis engine."""
    return SynthesisEngineFactory.create_engine(sample_rate=sample_rate)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

---

## 6. Testing Commands

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=auralis --cov=server --cov-report=html --cov-report=term
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance benchmarks
pytest tests/performance/ -v

# Specific test file
pytest tests/unit/test_chord_generator.py

# Specific test function
pytest tests/unit/test_chord_generator.py::test_chord_generator_determinism
```

### Run with Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Run only async tests
pytest -m asyncio
```

---

## 7. Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: ["3.12"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Lint with ruff
        run: ruff check auralis/ tests/

      - name: Type check with mypy
        run: mypy auralis/ --strict

      - name: Run tests
        run: pytest --cov=auralis --cov=server --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

---

## 8. Test Coverage Goals

| Component | Target Coverage | Current |
|-----------|----------------|---------|
| **Core** | 90%+ | TBD |
| **Composition** | 85%+ | TBD |
| **Synthesis** | 80%+ | TBD |
| **Streaming** | 85%+ | TBD |
| **API** | 90%+ | TBD |
| **Overall** | 85%+ | TBD |

---

## 9. Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Synthesis RTF** (GPU) | >50× | Benchmark test |
| **Synthesis RTF** (CPU) | >5× | Benchmark test |
| **Streaming Latency** | <200ms | Integration test |
| **Memory Usage** | <500MB | Profiling |
| **Client Connect Time** | <1s | E2E test |

---

This comprehensive testing strategy ensures reliability, performance, and maintainability throughout development.
