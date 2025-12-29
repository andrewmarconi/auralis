# Project Structure & Module Organization

## Complete Directory Layout

```
auralis/
├── auralis/                      # Main Python package
│   ├── __init__.py
│   │
│   ├── core/                     # Core infrastructure
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration management (Pydantic settings)
│   │   └── device_manager.py    # Hardware detection (MPS/CUDA/CPU)
│   │
│   ├── music/                    # Music theory utilities
│   │   ├── __init__.py
│   │   └── theory.py             # Keys, scales, chords, intervals
│   │
│   ├── composition/              # Musical generation
│   │   ├── __init__.py
│   │   ├── engine.py             # CompositionEngine (main orchestrator)
│   │   ├── chord_generator.py   # Markov chord progressions
│   │   ├── melody_generator.py  # Constraint-based melody
│   │   ├── percussion_generator.py  # Sparse percussion events
│   │   └── transformer_melody.py    # (Phase 2) GPT-based melody
│   │
│   ├── synthesis/                # Audio rendering
│   │   ├── __init__.py
│   │   ├── engine_factory.py    # Factory with fallbacks
│   │   ├── torchsynth_engine.py # Torchsynth implementation
│   │   ├── numpy_synth.py       # CPU fallback (simple synth)
│   │   ├── silence_engine.py    # Last-resort fallback
│   │   └── effects.py           # (Phase 3) Reverb, delay
│   │
│   ├── streaming/                # Audio buffering & encoding
│   │   ├── __init__.py
│   │   ├── ring_buffer.py       # Thread-safe circular buffer
│   │   ├── client_queue.py      # Per-client async queues
│   │   └── opus_codec.py        # (Phase 3) Opus compression
│   │
│   ├── api/                      # API schemas & validation
│   │   ├── __init__.py
│   │   ├── validation.py        # Pydantic models (ControlParameters, etc.)
│   │   └── routes.py            # REST API endpoints (/api/v1/...)
│   │
│   └── monitoring/               # Metrics & logging
│       ├── __init__.py
│       ├── metrics.py           # Performance metrics collector
│       └── logger.py            # Structured logging setup
│
├── server/                       # FastAPI server
│   ├── __init__.py
│   ├── main.py                  # App entrypoint, startup/shutdown
│   ├── websocket.py             # WebSocket streaming endpoint
│   ├── client_manager.py        # Client connection management
│   └── synthesis_worker.py      # Background synthesis loop
│
├── client/                       # Web frontend
│   ├── index.html               # Main UI
│   ├── audio_client.js          # AudioWorklet-based playback
│   ├── audio_worklet.js         # Web Audio worklet processor
│   ├── controls.js              # UI controls (key, BPM, intensity)
│   └── styles.css               # Basic styling
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── unit/                    # Unit tests
│   │   ├── test_chord_generator.py
│   │   ├── test_melody_generator.py
│   │   ├── test_ring_buffer.py
│   │   └── test_config.py
│   ├── integration/             # Integration tests
│   │   ├── test_composition_engine.py
│   │   ├── test_synthesis_pipeline.py
│   │   └── test_websocket_streaming.py
│   └── performance/             # Performance benchmarks
│       └── test_synthesis_benchmark.py
│
├── docs/                         # Documentation
│   ├── implementation_plan.md
│   ├── implementation_strategies.md
│   ├── system_architecture.md
│   ├── technical_specifications.md
│   ├── torchsynth_integration.md
│   ├── composition_engine.md
│   ├── project_structure.md     # This file
│   ├── testing_strategy.md
│   ├── deployment_guide.md
│   └── api_reference.md
│
├── scripts/                      # Utility scripts
│   ├── benchmark.py             # Run performance benchmarks
│   ├── generate_sample.py       # Generate offline audio sample
│   └── setup_dev_env.sh         # Dev environment setup
│
├── deployment/                   # Deployment configs
│   ├── Dockerfile               # Production container
│   ├── Dockerfile.dev           # Development container
│   ├── docker-compose.yml       # Multi-service orchestration
│   ├── nginx.conf               # Reverse proxy config
│   └── systemd/                 # Systemd service files
│       └── auralis.service
│
├── .github/                      # GitHub configuration
│   └── workflows/
│       ├── ci.yml               # CI/CD pipeline
│       └── release.yml          # Release automation
│
├── pyproject.toml                # Project metadata & dependencies
├── .env.example                  # Example environment variables
├── .gitignore
├── LICENSE                       # MIT License
├── README.md                     # Project overview
└── main.py                       # Quick-start entrypoint (development)
```

---

## Module Responsibilities

### Core Package (`auralis/`)

**Purpose**: Reusable library for generative ambient music

#### `core/`
- **config.py**: Load/validate environment variables, configuration management
- **device_manager.py**: Detect hardware (MPS/CUDA/CPU), select optimal device

#### `music/`
- **theory.py**: Music theory constants (keys, scales, chords, intervals)

#### `composition/`
- **engine.py**: Main orchestrator, manages async generation loop
- **chord_generator.py**: Markov chain chord progressions
- **melody_generator.py**: Constraint-based melody generation
- **percussion_generator.py**: Sparse percussion event generation
- **transformer_melody.py** (Phase 2): ML-based melody generation

#### `synthesis/`
- **engine_factory.py**: Create synthesis engine with fallbacks
- **torchsynth_engine.py**: Torchsynth GPU-accelerated synthesis
- **numpy_synth.py**: Simple CPU fallback
- **effects.py** (Phase 3): Audio effects (reverb, delay)

#### `streaming/`
- **ring_buffer.py**: Thread-safe circular audio buffer
- **client_queue.py**: Per-client asyncio queues
- **opus_codec.py** (Phase 3): Opus compression

#### `api/`
- **validation.py**: Pydantic schemas for API
- **routes.py**: FastAPI route definitions

#### `monitoring/`
- **metrics.py**: Performance metrics collection
- **logger.py**: Structured logging configuration

---

### Server Package (`server/`)

**Purpose**: FastAPI application for streaming audio

- **main.py**: FastAPI app, startup/shutdown hooks, state management
- **websocket.py**: WebSocket `/ws/stream` endpoint
- **client_manager.py**: Track connected clients, enforce limits
- **synthesis_worker.py**: Background task that renders phrases

---

### Client (`client/`)

**Purpose**: Browser-based audio playback

- **index.html**: Main UI with controls
- **audio_client.js**: AudioContext + AudioWorklet setup
- **audio_worklet.js**: AudioWorklet processor (runs in audio thread)
- **controls.js**: UI event handlers
- **styles.css**: Minimal styling

---

### Tests (`tests/`)

**Purpose**: Comprehensive test coverage

- **unit/**: Fast, isolated tests for individual components
- **integration/**: Tests for component interaction
- **performance/**: Benchmarks and profiling

---

### Scripts (`scripts/`)

**Purpose**: Development and utility scripts

- **benchmark.py**: Measure synthesis performance
- **generate_sample.py**: Offline audio generation for testing
- **setup_dev_env.sh**: Automated dev environment setup

---

### Deployment (`deployment/`)

**Purpose**: Production deployment configurations

- **Dockerfile**: Production container image
- **docker-compose.yml**: Multi-container setup (server + nginx)
- **nginx.conf**: Reverse proxy for WebSocket + static files
- **systemd/**: Linux service management

---

## Import Conventions

### Absolute Imports

All imports use absolute package names:

```python
# Good ✓
from auralis.core.config import config
from auralis.composition.engine import CompositionEngine
from auralis.music.theory import KEY_TO_ROOT_MIDI

# Bad ✗
from ..core.config import config  # Relative imports
from core.config import config     # Missing package prefix
```

### Package-level Exports

Each package's `__init__.py` exports its public API:

```python
# auralis/composition/__init__.py
from auralis.composition.engine import CompositionEngine
from auralis.composition.chord_generator import ChordProgressionGenerator
from auralis.composition.melody_generator import ConstrainedMelodyGenerator

__all__ = [
    "CompositionEngine",
    "ChordProgressionGenerator",
    "ConstrainedMelodyGenerator",
]
```

Usage:

```python
# Clean imports
from auralis.composition import CompositionEngine

# Instead of
from auralis.composition.engine import CompositionEngine
```

---

## Naming Conventions

### Files & Modules

- **snake_case**: `chord_generator.py`, `ring_buffer.py`
- One primary class per file (where appropriate)
- Filename matches primary class: `CompositionEngine` → `engine.py`

### Classes

- **PascalCase**: `CompositionEngine`, `RingBuffer`, `AuralisConfig`
- Descriptive names: `TorchsynthAmbientEngine` (not `TEngine`)

### Functions & Variables

- **snake_case**: `generate_phrase()`, `buffer_depth_ms`
- Verbs for functions: `render_phrase()`, `update_params()`
- Nouns for variables: `sample_rate`, `chunk_duration_ms`

### Constants

- **UPPER_SNAKE_CASE**: `KEY_TO_ROOT_MIDI`, `CHORD_INTERVALS`
- Defined in `music/theory.py` or at module top-level

### Private Members

- **Leading underscore**: `_generation_loop()`, `_params_lock`
- Indicates internal implementation detail

---

## Type Annotations

**All public functions must have type hints:**

```python
from typing import List, Dict, Optional, Tuple
import numpy as np

def render_phrase(
    chords: List[Tuple[int, int, str]],
    melody: List[Tuple[int, int, float, float]],
    percussion: List[Dict],
    duration_sec: float,
) -> np.ndarray:
    """
    Render a musical phrase to audio.

    Args:
        chords: List of (onset_sample, root_midi, chord_type)
        melody: List of (onset_sample, pitch, velocity, duration)
        percussion: Percussion event dictionaries
        duration_sec: Total phrase duration in seconds

    Returns:
        Stereo audio array, shape (2, num_samples), float32
    """
    ...
```

**Run type checking:**

```bash
mypy auralis/ --strict
```

---

## Development Workflow

### 1. Setup Environment

```bash
# Clone repo
git clone https://github.com/yourusername/auralis.git
cd auralis

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Copy example env
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 2. Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=auralis --cov-report=html

# Specific test file
pytest tests/unit/test_chord_generator.py

# Run benchmarks
pytest tests/performance/ -v
```

### 3. Code Quality

```bash
# Format code
black auralis/ tests/

# Lint
ruff check auralis/ tests/

# Type check
mypy auralis/ --strict
```

### 4. Run Development Server

```bash
# Using main.py
python main.py

# Or using uvicorn directly
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000

# Open client
open client/index.html
```

---

## Package Distribution

### Build Package

```bash
# Build wheel
python -m build

# Output: dist/auralis-0.1.0-py3-none-any.whl
```

### Install from Wheel

```bash
pip install dist/auralis-0.1.0-py3-none-any.whl
```

### Publish to PyPI (Future)

```bash
# Test PyPI
python -m twine upload --repository testpypi dist/*

# Production PyPI
python -m twine upload dist/*
```

---

## Configuration Management

### Environment Variables

Stored in `.env` (git-ignored):

```bash
AURALIS_ENV=development
AURALIS_HOST=0.0.0.0
AURALIS_PORT=8000
AURALIS_DEVICE=auto
AURALIS_LOG_LEVEL=INFO
```

### Loading Configuration

```python
from auralis.core.config import config

# Access settings
print(config.port)  # 8000
print(config.device)  # "auto"
```

### Multiple Environments

```
.env                 # Default (development)
.env.production      # Production overrides
.env.test            # Testing overrides
```

Load specific env:

```bash
export AURALIS_ENV_FILE=.env.production
python -m uvicorn server.main:app
```

---

This structure follows Python best practices and scales from MVP to production deployment.
