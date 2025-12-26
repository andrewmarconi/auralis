# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Auralis is a real-time generative ambient music engine that composes and streams evolving soundscapes using Python 3.12+, FastAPI, PyTorch, and torchsynth. The system generates continuous, non-repeating ambient music through Markov chord progressions, constraint-based melodies, and GPU-accelerated synthesis, streaming audio at 44.1kHz/16-bit PCM over WebSockets.

## Core Architecture

The system follows a strict modular pipeline:

1. **Composition Layer** ([composition/](composition/)) - Generates musical phrases using:
   - Markov chain chord progressions (bigram, order 2)
   - Constraint-based melody generation (70% chord tones, 25% scale, 5% chromatic)
   - Sparse ambient percussion events

2. **Synthesis Layer** ([server/](server/)) - Renders audio using:
   - GPU-accelerated torchsynth (Metal on Apple Silicon, CUDA on NVIDIA)
   - Real-time polyphonic voice management (8-16 voices)
   - Wavetable oscillators with ADSR envelopes and pitch glide

3. **Streaming Layer** ([server/](server/)) - Manages real-time delivery via:
   - Thread-safe ring buffer (10-20 chunks, 1-2 second capacity)
   - WebSocket endpoint streaming base64-encoded PCM chunks (100ms/~17.6kB each)
   - Back-pressure flow control with adaptive buffering

4. **Client Layer** ([client/](client/)) - Browser playback using:
   - Web Audio API with AudioWorklet processing
   - Adaptive ring buffer (300-500ms target latency)
   - Real-time controls for key, BPM, and intensity

**Critical Performance Constraint**: All code changes MUST preserve <100ms audio processing latency. No blocking operations in the audio pipeline.

## Development Commands

### Environment Setup
```bash
# Install uv package manager (required for all Python operations)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment (ONLY via uv)
uv venv

# Activate environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies in development mode
uv pip install -e ".[dev]"
```

### Running the Server
```bash
# Development server with auto-reload
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000

# Quick-start alternative
python main.py

# Production server (multi-worker)
uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Testing
```bash
# Run all tests with coverage
pytest

# Coverage report (HTML + terminal)
pytest --cov=auralis --cov-report=html --cov-report=term

# Specific test file
pytest tests/unit/test_chord_generator.py

# Integration tests only (audio performance tests)
pytest tests/integration/ -v

# Performance benchmarks
pytest tests/performance/ -v
```

### Code Quality
```bash
# Format code (Black)
black auralis/ tests/

# Lint (Ruff)
ruff check auralis/ tests/

# Type checking (mypy)
mypy auralis/ --strict

# Run all quality checks
black auralis/ tests/ && ruff check auralis/ tests/ && mypy auralis/ --strict
```

### Dependency Management
```bash
# Add new dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Remove dependency
uv remove package-name

# Update dependencies
uv pip compile --upgrade pyproject.toml
```

## Constitution (Non-Negotiable Principles)

### I. UV-First Development
**ALL Python operations MUST use uv exclusively**. Never use `python -m venv`, `pip install`, or manual virtual environment creation. This ensures reproducible builds and consistent dependency management.

- ✅ Correct: `uv add fastapi`, `uv run pytest`
- ❌ Wrong: `pip install fastapi`, `python -m pytest`

### II. Real-Time Audio Performance
**ALL code changes MUST preserve real-time audio performance**. Audio streaming at 44.1kHz, 16-bit PCM in 100ms chunks cannot be interrupted.

- Prohibited: Blocking I/O in audio pipeline, inefficient algorithms causing glitches, memory leaks
- Required: <100ms processing latency, GPU acceleration for synthesis, asyncio for concurrent operations

### III. Modular Architecture
**ALL components MUST follow strict separation**:
- `server/` - FastAPI endpoints, WebSocket streaming, synthesis engine
- `composition/` - Generative algorithms (chords, melody, percussion)
- `client/` - Web Audio API playback, browser UI
- `tests/` - Unit, integration, and performance tests

Circular dependencies are forbidden. Each module must be independently testable.

### IV. GPU-Accelerated Synthesis
**ALL audio synthesis MUST prioritize GPU acceleration**:
- Metal on Apple Silicon (M1/M2/M4)
- CUDA on NVIDIA GPUs
- CPU-only fallback allowed but NOT default

PyTorch operations should automatically use the appropriate device.

### V. WebSocket Streaming Protocol
**ALL client-server audio communication MUST use WebSockets**. No REST for audio data.

- WebSocket frames: base64-encoded PCM chunks with timing metadata
- Adaptive client buffering required for seamless playback
- REST API only for control messages (`/api/control`, `/api/status`, `/api/metrics`)

## Project Structure

```
auralis/
├── server/                    # FastAPI server & synthesis
│   ├── main.py               # App entrypoint, startup/shutdown
│   ├── synthesis_engine.py   # torchsynth GPU rendering
│   ├── ring_buffer.py        # Thread-safe audio buffer
│   └── streaming_server.py   # WebSocket streaming logic
│
├── composition/              # Musical generation algorithms
│   ├── chord_generator.py   # Markov chord progressions
│   ├── melody_generator.py  # Constraint-based melody
│   └── percussion_generator.py  # Sparse percussion events
│
├── client/                   # Web Audio API client
│   ├── index.html           # UI with controls
│   ├── audio_client_worklet.js      # AudioContext setup
│   └── audio_worklet_processor.js   # Audio thread processor
│
├── tests/
│   ├── integration/         # WebSocket, latency, GPU tests
│   └── performance/         # Real-time constraint benchmarks
│
├── docs/                    # Comprehensive documentation
│   ├── system_architecture.md     # Detailed architecture diagrams
│   ├── project_structure.md       # Module organization guide
│   ├── implementation_plan.md     # Development roadmap
│   └── technical_specifications.md
│
├── specs/001-phase1-mvp/    # Current feature specification
│   ├── spec.md              # User stories, requirements
│   ├── plan.md              # Implementation plan
│   ├── tasks.md             # Task breakdown
│   ├── data-model.md        # Data structures
│   ├── quickstart.md        # Testing scenarios
│   └── contracts/           # API contracts
│
├── .specify/                # SpecKit workflow templates
│   ├── memory/constitution.md    # Project constitution
│   └── templates/           # Spec/plan/task templates
│
├── pyproject.toml           # Project metadata, dependencies
├── main.py                  # Quick-start development entrypoint
└── README.md               # Project overview
```

## Key Implementation Details

### Audio Format Specifications
- **Sample Rate**: 44.1kHz (44,100 samples/second)
- **Bit Depth**: 16-bit PCM
- **Channels**: Stereo (2 channels)
- **Chunk Size**: 100ms (4,410 samples per channel)
- **Chunk Bytes**: 4,410 samples × 2 channels × 2 bytes = ~17.6kB
- **Encoding**: base64 for WebSocket transport (~23.5kB per chunk)

### Ring Buffer Design
- **Capacity**: 10-20 chunks (1-2 seconds)
- **Pre-allocation**: numpy array (prevents GC pauses)
- **Thread-Safe**: Atomic cursors for read/write operations
- **Back-Pressure**: Sleep 10ms if buffer depth < 2 chunks

### Composition Parameters
- **BPM Range**: 60-120 (default: 70)
- **Phrase Duration**: 8 bars typical
- **Markov Order**: 2 (bigram - considers 1 previous chord)
- **Melody Constraints**:
  - 70% chord tones
  - 25% scale notes
  - 5% chromatic passing tones

### GPU Device Selection
```python
# Automatic device detection priority:
# 1. Metal (Apple Silicon M1/M2/M4)
# 2. CUDA (NVIDIA GPUs)
# 3. CPU fallback (not recommended for real-time)
import torch
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
```

## Import Conventions

**Use absolute imports only**:
```python
# ✅ Correct
from auralis.core.config import config
from auralis.composition.engine import CompositionEngine
from server.main import app

# ❌ Wrong
from ..core.config import config  # No relative imports
from core.config import config     # Missing package prefix
```

**Package-level exports** (defined in `__init__.py`):
```python
# Clean imports from public API
from auralis.composition import CompositionEngine
```

## Type Annotations

**ALL public functions MUST have type hints**:
```python
from typing import List, Tuple
import numpy as np

def render_phrase(
    chords: List[Tuple[int, int, str]],
    melody: List[Tuple[int, int, float, float]],
    duration_sec: float,
) -> np.ndarray:
    """
    Render a musical phrase to audio.

    Args:
        chords: List of (onset_sample, root_midi, chord_type)
        melody: List of (onset_sample, pitch, velocity, duration)
        duration_sec: Total phrase duration in seconds

    Returns:
        Stereo audio array, shape (2, num_samples), float32
    """
    ...
```

## Naming Conventions

- **Files/Modules**: `snake_case` (e.g., `chord_generator.py`)
- **Classes**: `PascalCase` (e.g., `CompositionEngine`, `RingBuffer`)
- **Functions/Variables**: `snake_case` (e.g., `generate_phrase()`, `buffer_depth_ms`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `KEY_TO_ROOT_MIDI`, `CHORD_INTERVALS`)
- **Private Members**: `_leading_underscore` (e.g., `_generation_loop()`)

## SpecKit Workflow

This project uses the SpecKit workflow system (`.specify/` directory) for feature development:

### Commands
- `/speckit.specify` - Create feature specification from natural language
- `/speckit.plan` - Generate implementation plan with research and design artifacts
- `/speckit.tasks` - Generate task breakdown from specification and plan
- `/speckit.implement` - Execute tasks from tasks.md
- `/speckit.clarify` - Identify underspecified areas in specifications
- `/speckit.analyze` - Validate consistency across spec/plan/tasks artifacts
- `/speckit.constitution` - Update project constitution and principles

### Feature Branch Structure
```
specs/[###-feature-name]/
├── spec.md           # User stories, requirements, success criteria
├── plan.md           # Technical implementation plan
├── tasks.md          # Task breakdown (generated by /speckit.tasks)
├── research.md       # Phase 0 technical research
├── data-model.md     # Data structures and entities
├── quickstart.md     # Testing scenarios
├── contracts/        # API contracts (WebSocket, REST)
└── checklists/       # Quality validation checklists
```

## Common Development Tasks

### Adding a New Composition Algorithm
1. Create module in `composition/` (e.g., `new_generator.py`)
2. Follow naming convention: `[Purpose]Generator` class
3. Implement with type hints and docstrings
4. Add unit tests in `tests/unit/test_new_generator.py`
5. Add integration test in `tests/integration/`
6. Update `composition/__init__.py` exports

### Adding a WebSocket Endpoint
1. Define endpoint in `server/websocket.py`
2. Use asyncio for non-blocking operations
3. Document contract in `specs/[feature]/contracts/websocket-api.md`
4. Add integration test verifying latency constraints
5. Update client handler in `client/audio_client_worklet.js`

### GPU Synthesis Optimization
1. Profile with `torch.profiler` or `nvidia-smi`
2. Ensure operations use GPU device (check `.to(device)`)
3. Batch operations where possible (voice management)
4. Benchmark against `<100ms` latency requirement in `tests/performance/`

## Environment Variables

Create `.env` from `.env.example`:
```bash
AURALIS_ENV=development
AURALIS_HOST=0.0.0.0
AURALIS_PORT=8000
AURALIS_DEVICE=auto          # auto | mps | cuda | cpu
AURALIS_LOG_LEVEL=INFO
```

## Testing Strategy

### Unit Tests
- Fast, isolated tests for individual components
- Mock external dependencies (GPU, network)
- Located in `tests/unit/`

### Integration Tests
- Test component interactions (composition → synthesis → streaming)
- Verify WebSocket streaming under load
- Measure audio latency and quality metrics
- Located in `tests/integration/`

### Performance Tests
- Benchmark synthesis latency (<100ms requirement)
- Measure GPU utilization effectiveness
- Test concurrent client capacity
- Located in `tests/performance/`

## Documentation

Extensive documentation in [docs/](docs/):
- [system_architecture.md](docs/system_architecture.md) - Detailed layer-by-layer architecture
- [project_structure.md](docs/project_structure.md) - Module organization and import conventions
- [implementation_plan.md](docs/implementation_plan.md) - Development phases and roadmap
- [technical_specifications.md](docs/technical_specifications.md) - Technical deep-dive
- [torchsynth_integration.md](docs/torchsynth_integration.md) - GPU synthesis details

## Key Dependencies

### Core Framework
- **FastAPI** (0.127.0) - Async web framework with WebSocket support
- **uvicorn** (0.40.0) - ASGI server with WebSocket improvements
- **Pydantic** (2.12.0) - Data validation and settings management

### Audio Processing
- **PyTorch** (2.5.0) - GPU acceleration (Metal/CUDA)
- **torchaudio** (2.5.0) - Audio I/O and processing
- **torchsynth** (1.0.2) - Differentiable synthesizer
- **numpy** (1.26.0+) - NumPy 2.x compatible
- **pedalboard** (0.9.19) - Spotify's audio effects library

### Development Tools
- **pytest** - Test framework with asyncio support
- **pytest-cov** - Coverage reporting
- **ruff** - Fast Python linter
- **black** - Code formatter
- **mypy** - Static type checker

## References

- **Project Constitution**: [.specify/memory/constitution.md](.specify/memory/constitution.md)
- **Current Feature Spec**: [specs/001-phase1-mvp/spec.md](specs/001-phase1-mvp/spec.md)
- **Agent Guidelines**: [AGENTS.md](AGENTS.md)
- **Main README**: [README.md](README.md)
