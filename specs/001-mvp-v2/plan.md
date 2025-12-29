# Implementation Plan: Auralis MVP v2.0 - Real-Time Generative Ambient Music Streaming Engine

**Branch**: `001-mvp-v2` | **Date**: 2025-12-28 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-mvp-v2/spec.md`

## Summary

Auralis MVP v2.0 delivers a real-time generative ambient music streaming engine that combines:

1. **Generative Composition**: Markov chain chord progressions and constraint-based melodies following ambient music principles
2. **Sample-Based Synthesis**: FluidSynth rendering with SoundFont libraries (piano + pads) for high-quality timbres
3. **Real-Time Streaming**: WebSocket delivery of 44.1kHz/16-bit PCM audio in 100ms chunks with <800ms end-to-end latency
4. **Browser Controls**: Web interface with parameter adjustments (key, mode, intensity, BPM) and presets
5. **Performance Monitoring**: REST endpoints for system status and metrics

**Primary Technical Approach**:
- FluidSynth sample-based synthesis replaces oscillator-based synthesis for realistic piano/pad sounds
- Thread-safe ring buffer (2-second capacity) enables continuous streaming without dropouts
- Client-side adaptive buffering (300-500ms) handles network jitter
- GPU acceleration via PyTorch (Metal/CUDA) for audio effects, CPU fallback for FluidSynth synthesis

## Technical Context

**Language/Version**: Python 3.12+ via uv (required for async features, PyTorch compatibility)

**Primary Dependencies**:
- **Server Framework**: FastAPI 0.127+ (async WebSocket), uvicorn 0.40+ (ASGI server)
- **Audio Synthesis**: pyfluidsynth 1.3.2+ (FluidSynth Python bindings), FluidSynth 2.x native library
- **Audio Processing**: numpy 1.26+ (NumPy 2.x compatible), scipy 1.11+
- **GPU Acceleration**: PyTorch 2.5+ (Metal/CUDA support), pedalboard 0.9.19 (effects, post-MVP)
- **Networking**: WebSocket (native to FastAPI/uvicorn), base64 encoding for PCM transport

**Storage**:
- **Audio Buffers**: Thread-safe ring buffer (NumPy pre-allocated arrays, 1-2 second capacity)
- **SoundFonts**: Local filesystem (`./soundfonts/` directory, 200-300MB total)
- **Client Settings**: Browser localStorage (key, mode, intensity, BPM persistence)
- **Metrics**: In-memory circular buffers (<10MB RAM)

**Testing**:
- **Integration Tests**: WebSocket streaming under load, audio latency measurement, GPU acceleration verification, buffer underrun detection
- **Performance Tests**: Synthesis latency benchmarks (<100ms target), end-to-end latency tests (<800ms target), memory leak detection (8+ hour sessions)
- **Quality Tests**: Harmonic analysis (95%+ note constraint adherence), listening tests (user surveys)

**Target Platform**:
- **Server**: macOS 12+ (Metal), Ubuntu 20.04+ (CUDA), Windows 10+ (CPU fallback)
- **Client**: Chrome 90+, Edge 90+, Safari 14+ (Web Audio API + WebSocket support)
- **Hardware**: 2+ CPU cores, 1GB+ RAM, optional GPU (Metal M1/M2/M4, NVIDIA CUDA)

**Project Type**: Real-time audio streaming server with generative music engine

**Performance Goals**:
- **Synthesis Latency**: <100ms per phrase (8 bars @ 60 BPM)
- **End-to-End Latency**: <800ms (target 500ms) from generation → synthesis → network → playback
- **Streaming**: Continuous 44.1kHz/16-bit PCM, 100ms chunks, >98% on-time delivery
- **Concurrency**: 10+ simultaneous WebSocket clients without audio degradation
- **Memory**: <500MB total footprint including SoundFonts

**Constraints**:
- **No Blocking Operations**: Audio pipeline must use asyncio, no synchronous I/O in synthesis/streaming path
- **Real-Time Requirement**: <100ms audio processing latency is non-negotiable (gates all code changes)
- **GPU Acceleration**: FluidSynth is CPU-only (sample-based), PyTorch effects use GPU when available
- **Base64 Overhead**: ~33% bandwidth penalty for WebSocket PCM transport (acceptable for MVP, Opus compression post-MVP)
- **Browser Auto-Play**: Relies on user navigation to satisfy Web Audio API policies
- **Single-Server**: No multi-server orchestration or session persistence in MVP

**Scale/Scope**:
- **Concurrent Users**: 10+ WebSocket connections per server instance (MVP acceptable)
- **Session Duration**: Support 8+ hour continuous sessions without memory leaks
- **Musical Complexity**: 8-16 voices polyphony, 32 simultaneous notes maximum
- **Phrase Length**: 8-16 bars per generated phrase, rendered upfront

**Known Technical Risks**:
1. **FluidSynth Latency**: CPU synthesis may approach 100ms limit with large SoundFonts (mitigation: benchmark early, optimize settings)
2. **SoundFont Quality**: Free SoundFonts may require licensing review or quality compromise (mitigation: test top 3 options)
3. **Browser Compatibility**: Web Audio API inconsistencies across browsers (mitigation: test Chrome/Edge/Safari, document minimums)
4. **Network Unreliability**: Adaptive buffering may not handle sustained poor connections (mitigation: exponential backoff, user warnings)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Initial Check (Pre-Research)

- ✅ **UV-First**: All Python dependencies managed via `uv add`, no manual venv. Development commands use `uv run`.
- ✅ **Real-Time Performance**: FluidSynth synthesis benchmarked at <100ms. Ring buffer design prevents blocking. Asyncio for concurrency.
- ✅ **Modular Architecture**: Clear separation: `server/` (FastAPI, streaming), `composition/` (generation), `client/` (Web Audio), `tests/` (integration).
- ⚠️ **GPU Acceleration**: FluidSynth is CPU-only (sample-based synthesis limitation). PyTorch GPU used for effects (post-MVP pedalboard integration). **Justification**: Sample-based synthesis quality justifies CPU-only approach; GPU still used for effects pipeline.
- ✅ **WebSocket Protocol**: All audio data via WebSocket (base64 PCM chunks). REST only for `/api/status`, `/api/metrics` control endpoints.
- ✅ **Developer Experience**: Function complexity kept ≤10 (McCabe), clear naming, minimal nesting. FluidSynth integration encapsulated in single `FluidSynthVoice` class.

### Post-Design Check (After Phase 1)

✅ **Design Phase Complete** (December 28, 2024)

- ✅ **UV-First**: All contracts and data model assume uv-managed dependencies (pyfluidsynth, FastAPI, PyTorch via `uv add`). No manual pip or conda references.
- ✅ **Real-Time Performance**: Research confirms FluidSynth <100ms synthesis target achievable (40ms on M4 Mac). Ring buffer design with back-pressure prevents blocking. All WebSocket operations use asyncio.
- ✅ **Modular Architecture**: Data model and contracts enforce clear boundaries: `composition/` → `server/synthesis` → `server/streaming` → `client/`. No circular dependencies. All internal interfaces defined as ABCs in `server/interfaces/`.
- ⚠️ **GPU Acceleration**: Confirmed FluidSynth is CPU-only (sample-based synthesis limitation). PyTorch GPU reserved for effects pipeline (post-MVP pedalboard integration). Design preserves GPU capability for future phases.
- ✅ **WebSocket Protocol**: Contracts specify WebSocket-only streaming (`/ws/stream`). REST endpoints (`/api/status`, `/api/metrics`) strictly for control plane, never audio data. Base64 PCM format documented in [websocket-api.md](contracts/websocket-api.md).
- ✅ **Developer Experience**: Function complexity bounded by interface design (single-responsibility methods). FluidSynth integration encapsulated in `FluidSynthRenderer` class. Adaptive buffering abstracted to `AdaptiveAudioBuffer` class. GC tuning isolated to `gc_config.py` module. All complexity justified in research (see [docs/technical-research.md](../../docs/technical-research.md)).

## Complexity Tracking

### Justified Violations

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| FluidSynth CPU-only synthesis | Sample-based synthesis provides realistic piano/pad timbres essential for ambient music quality. FluidSynth is mature, well-tested, cross-platform. | PyTorch oscillators lack timbral richness; ML-based synthesis requires training data and GPU resources; custom wavetable synthesis adds development complexity without quality guarantee. |
| Base64 PCM encoding | WebSocket transport requires text-based encoding for binary audio data. Base64 is standard, widely supported, simple to implement. | Opus compression requires codec integration, adds latency, complicates client decoding; custom binary protocol increases complexity without MVP validation gain. |

## Project Structure

### Documentation (this feature)

```text
specs/001-mvp-v2/
├── spec.md              # Feature specification (input)
├── plan.md              # This file (implementation plan)
├── research.md          # Phase 0: Technical research & decisions
├── data-model.md        # Phase 1: Entity definitions & relationships
├── quickstart.md        # Phase 1: Testing scenarios & manual QA
├── contracts/           # Phase 1: API contracts
│   ├── websocket-api.md # WebSocket streaming protocol
│   ├── http-api.md      # REST endpoints (/api/status, /api/metrics)
│   └── internal-interfaces.md # FluidSynth, RingBuffer, AudioBuffer interfaces
└── checklists/
    └── requirements.md  # Specification quality checklist (completed)
```

### Source Code (repository root)

```text
auralis/
├── server/                      # Server-side Python code
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entrypoint, startup/shutdown
│   ├── config.py               # Configuration (env vars, SoundFont paths)
│   ├── di_container.py         # Dependency injection container
│   │
│   ├── synthesis_engine.py     # FluidSynth rendering orchestration
│   ├── fluidsynth_renderer.py  # FluidSynth wrapper (load SF2, render notes)
│   ├── soundfont_manager.py    # SoundFont loading, preset mapping
│   ├── presets.py              # Musical presets (Focus, Meditation, Sleep, Bright)
│   │
│   ├── streaming_server.py     # WebSocket endpoint, client connection management
│   ├── ring_buffer.py          # Thread-safe circular buffer
│   ├── buffer_management.py    # Back-pressure logic, chunk delivery
│   │
│   ├── metrics.py              # Performance metrics (latency histograms, counters)
│   ├── memory_monitor.py       # Memory leak detection
│   ├── logging_config.py       # Structured logging setup
│   │
│   ├── device_selector.py      # GPU/CPU device selection (Metal/CUDA/CPU)
│   ├── gc_config.py            # Garbage collection tuning (reduce pauses)
│   │
│   └── interfaces/             # Interface definitions (ABC classes)
│       ├── __init__.py
│       ├── synthesis.py        # ISynthesisEngine, IFluidSynthRenderer
│       ├── buffer.py           # IRingBuffer, IBufferManager
│       ├── metrics.py          # IMetricsCollector, IMemoryMonitor
│       └── jitter.py           # IJitterTracker (client-side buffering)
│
├── composition/                 # Generative music algorithms
│   ├── __init__.py
│   ├── chord_generator.py      # Markov chain chord progressions
│   ├── melody_generator.py     # Constraint-based melody generation
│   └── percussion_generator.py # Ambient percussion (post-MVP, placeholder)
│
├── client/                      # Browser-based client
│   ├── index.html              # Main UI (controls, presets, status indicator)
│   ├── audio_client_worklet.js # AudioContext setup, WebSocket connection
│   ├── audio_worklet_processor.js # AudioWorklet thread (ring buffer, playback)
│   └── debug.html              # Debug interface (metrics visualization)
│
├── soundfonts/                  # SoundFont files (not in git)
│   ├── .gitkeep
│   ├── .env.example            # SoundFont download links
│   └── soundfonts/             # SF2 files (Salamander, Arachno, or FluidR3)
│       └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_smooth_streaming.py      # End-to-end latency, no dropouts
│   │   ├── test_fluidsynth_rendering.py  # Synthesis latency <100ms
│   │   ├── test_buffer_underruns.py      # Buffer health >98%
│   │   └── test_memory_stability.py      # 8+ hour session without leaks
│   │
│   ├── performance/
│   │   ├── __init__.py
│   │   ├── test_chunk_delivery.py        # Network timing variance
│   │   ├── test_batch_synthesis.py       # Synthesis throughput
│   │   └── benchmark_suite.py            # Full performance benchmark
│   │
│   └── unit/                    # Unit tests (created as needed)
│       ├── test_adaptive_buffer.py       # Client buffering logic
│       ├── test_jitter_tracker.py        # Jitter tracking
│       └── test_gc_config.py             # GC tuning validation
│
├── docs/
│   ├── system_architecture.md   # High-level architecture overview
│   ├── project_structure.md     # Module organization guide
│   ├── implementation_plan.md   # Original development roadmap
│   ├── technical_specifications.md
│   └── ambient-music-context.md # Ambient music theory reference
│
├── .specify/                    # SpecKit workflow templates
│   ├── memory/
│   │   └── constitution.md      # Project constitution v1.1.0
│   └── templates/
│       └── [spec/plan/task templates]
│
├── pyproject.toml              # uv project metadata, dependencies
├── .env.example                # Environment variables template
├── .gitignore
├── CLAUDE.md                   # Project guidance for Claude Code
├── README.md                   # Project overview, setup instructions
└── PRD.md                      # Product Requirements Document v2.0
```

**Structure Decision**: Modular architecture with strict separation of concerns:
- **`server/`**: All Python backend code (FastAPI, FluidSynth, WebSocket, buffers)
- **`composition/`**: Pure generation algorithms (no I/O, no network, no synthesis)
- **`client/`**: Pure browser code (Web Audio API, no server logic)
- **`tests/`**: Integration and performance tests (no unit tests in MVP unless needed)
- **`soundfonts/`**: External SoundFont files (not in git, downloaded separately)

This structure enables:
- Independent testing of each layer
- Clear dependency flow (composition → synthesis → streaming → client)
- No circular dependencies
- Easy future extraction (e.g., composition library, synthesis microservice)

## Phase 0: Research & Decisions

See [research.md](research.md) for detailed findings.

### Research Areas

1. **FluidSynth Python Integration**
   - Best practices for pyfluidsynth usage
   - SoundFont loading and memory management
   - Preset selection and voice allocation
   - Reverb/effects configuration (minimal in MVP)

2. **SoundFont Selection**
   - Quality comparison: Salamander Grand Piano, FluidR3_GM, Arachno
   - Licensing verification (CC0, Public Domain, commercial use)
   - File size vs quality trade-offs
   - Loading performance benchmarks

3. **Thread-Safe Ring Buffer**
   - NumPy pre-allocation patterns
   - Atomic cursor management (read/write positions)
   - Back-pressure mechanisms (sleep vs event-driven)
   - GIL considerations for concurrent access

4. **WebSocket Audio Streaming**
   - Chunking strategies (100ms = 4,410 samples/channel)
   - Base64 encoding/decoding performance
   - Timing metadata format (chunk sequence, timestamps)
   - Reconnection and error recovery patterns

5. **Web Audio API Implementation**
   - AudioWorklet vs ScriptProcessorNode (deprecated)
   - Adaptive buffering strategies (300-500ms target)
   - Jitter tracking with EMA (exponential moving average)
   - Browser compatibility (Chrome, Edge, Safari)

6. **Adaptive Buffering**
   - Client-side buffer tier strategy (4 tiers: emergency/low/healthy/full)
   - Jitter measurement and variance tracking
   - Dynamic buffer size adjustment algorithms
   - Underrun recovery without audio glitches

7. **Performance Monitoring**
   - Latency histogram implementation (avg, p50, p95, p99)
   - Circular buffer for metrics (limit memory overhead)
   - Prometheus/Grafana integration patterns (post-MVP)

8. **Memory Management**
   - GC tuning for real-time audio (gc.disable() in audio thread?)
   - SoundFont memory footprint optimization
   - NumPy array reuse patterns
   - Memory leak detection strategies

### Key Decisions

**Research Complete** - See detailed documentation in:
- **Full Research**: [/docs/technical-research.md](../../docs/technical-research.md) (72KB comprehensive guide)
- **Quick Reference**: [/docs/technical-decisions-summary.md](../../docs/technical-decisions-summary.md) (15KB summary)

**Critical Decisions Summary**:

1. **FluidSynth Integration** (pyfluidsynth 1.3.2+)
   - 32-voice polyphony (ambient music is sparse)
   - 3s reverb decay, 20% wet mix (minimal spaciousness)
   - 2 CPU cores, 44.1kHz sample rate
   - Target: <100ms synthesis latency (40ms observed on M4 Mac)

2. **SoundFont Selection** (Salamander Grand Piano + FluidR3_GM)
   - Salamander: 200MB, 16-velocity layers, CC-BY 3.0 license
   - FluidR3_GM: 140MB, excellent pads (presets 88-90), public domain
   - Total: 340MB (within 500MB budget)
   - Preset mapping: Piano on channel 0, Pad on channel 1

3. **Thread-Safe Ring Buffer** (NumPy pre-allocated circular buffer)
   - Capacity: 20 chunks (2 seconds @ 100ms chunks)
   - Data type: int16 stereo PCM, 4,410 samples/chunk
   - threading.Lock for cursor synchronization (Python GIL makes explicit locks practical)
   - Back-pressure: Sleep 10ms when buffer depth <2 chunks

4. **WebSocket Streaming** (Base64-encoded PCM)
   - Chunk size: 100ms (17.6kB raw, ~23.5kB base64)
   - Bandwidth: ~250 kbps per client
   - Metadata: JSON with seq, timestamp, duration_ms, sample_rate, channels
   - Reconnection: Exponential backoff (1s, 2s, 4s, ... up to 60s) with ±20% jitter

5. **Web Audio API** (AudioWorklet, NOT deprecated ScriptProcessorNode)
   - Runs in audio rendering thread (lower latency, GC-isolated)
   - 4-tier adaptive buffering: 300ms, 500ms, 700ms, 1000ms based on jitter
   - Jitter tracking: EMA with α=0.1 (smooth adjustment)
   - Browser support: Chrome 66+, Edge 79+, Safari 14.1+ (with auto-play workaround)

6. **Memory Management**
   - GC tuning: `gc.set_threshold(10000, 20, 20)` - 14× fewer collections
   - Buffer pooling: 20 pre-allocated NumPy buffers (~345KB)
   - Memory budget: 450MB total (340MB SoundFonts + 50MB FluidSynth + 60MB overhead)
   - Leak detection: tracemalloc monitoring, <10MB/hour growth target

## Phase 1: Data Model & Contracts

See [data-model.md](data-model.md) for entity definitions.
See [contracts/](contracts/) for API specifications.

### Data Entities

*To be filled after Phase 0 research*

Key entities from spec:
- ChordProgression (onset_time, root_pitch, chord_type)
- MelodyPhrase (onset_time, pitch, velocity, duration)
- MusicalContext (key, mode, BPM, intensity)
- FluidSynthVoice (SoundFont wrapper)
- SoundFontPreset (file path, preset number)
- AudioBuffer (NumPy stereo array)
- RingBuffer (circular buffer, cursors)
- WebSocketConnection (client state)
- AudioChunk (PCM data, metadata)
- PerformanceMetrics (histograms, counters)
- SystemStatus (uptime, connections, device)

### API Contracts

*To be filled after Phase 0 research*

Key contracts:
- WebSocket streaming protocol (`/ws/stream`)
- REST endpoints (`/api/status`, `/api/metrics`)
- Internal interfaces (ISynthesisEngine, IRingBuffer, etc.)

### Testing Scenarios

See [quickstart.md](quickstart.md) for manual testing procedures.

## Phase 2: Task Breakdown

**NOTE**: Phase 2 (task breakdown) is generated by the `/speckit.tasks` command, NOT by `/speckit.plan`.

The implementation plan concludes here. To generate tasks, run:

```bash
/speckit.tasks
```

This will create `tasks.md` with dependency-ordered implementation tasks.

## Appendix: Architecture Diagrams

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Browser Client                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Web Audio API (AudioContext + AudioWorklet)     │   │
│  │ • Adaptive ring buffer (300-500ms)              │   │
│  │ • Jitter tracking (EMA)                         │   │
│  │ • Base64 PCM decoding                           │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ UI Controls (index.html)                        │   │
│  │ • Key, mode, intensity, BPM sliders             │   │
│  │ • Presets (Focus, Meditation, Sleep, Bright)    │   │
│  │ • Connection status indicator                   │   │
│  │ • localStorage persistence                      │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ WebSocket (bidirectional)
                   │ Audio: base64 PCM chunks (100ms)
                   │ Control: JSON messages (key, BPM, etc.)
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Server (Python 3.12+)              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ WebSocket Endpoint (/ws/stream)                 │   │
│  │ • Client connection management                  │   │
│  │ • Chunk delivery from ring buffer               │   │
│  │ • Control message handling (asyncio)            │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ REST API (/api/status, /api/metrics)            │   │
│  │ • System status: uptime, connections, device    │   │
│  │ • Metrics: latency histograms, buffer health    │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Async generation loop
                   ▼
┌─────────────────────────────────────────────────────────┐
│           Composition + Synthesis Engine                │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ChordGenerator (Markov chain, order 2)          │   │
│  │ • Modal contexts (Aeolian, Dorian, Lydian, etc.)│   │
│  │ • 8-16 bar progressions                         │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ MelodyGenerator (constraint-based)              │   │
│  │ • 70% chord tones, 25% scale, 5% chromatic      │   │
│  │ • Velocity randomization (20-100)               │   │
│  │ • Note probability (50-80%)                     │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ FluidSynth Synthesis (sample-based)             │   │
│  │ • Piano voice (SoundFont preset 0)              │   │
│  │ • Pad voice (SoundFont preset 88-90)            │   │
│  │ • Minimal reverb (3s decay, 20% wet)            │   │
│  │ • Renders to 44.1kHz stereo PCM                 │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Thread-Safe Ring Buffer                         │   │
│  │ • 2-second capacity (10-20 chunks)              │   │
│  │ • Pre-allocated NumPy arrays                    │   │
│  │ • Atomic read/write cursors                     │   │
│  │ • Back-pressure (sleep if depth <2)             │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
┌──────────────┐
│ MusicalContext│ (key, mode, BPM, intensity)
│ (from client)│
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│ ChordGenerator      │──→ ChordProgression
│ (Markov chain)      │    [(onset, root, type), ...]
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│ MelodyGenerator     │──→ MelodyPhrase
│ (constraint-based)  │    [(onset, pitch, velocity, duration), ...]
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│ FluidSynthRenderer  │──→ AudioBuffer
│ (sample synthesis)  │    NumPy array (2, num_samples) float32
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│ RingBuffer          │──→ AudioChunk (100ms each)
│ (thread-safe)       │    base64-encoded PCM + metadata
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│ WebSocketConnection │──→ Client (Browser)
│ (streaming)         │    AudioWorklet playback
└─────────────────────┘
```

### Client-Side Adaptive Buffering

```
┌─────────────────────────────────────────────────────────┐
│              Client Ring Buffer (300-500ms)             │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  │ Chunk 1  │  │ Chunk 2  │  │ Chunk 3  │  │ Chunk 4  │
│  │ 100ms    │  │ 100ms    │  │ 100ms    │  │ 100ms    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘
│       ▲                                          ▲       │
│       │                                          │       │
│   Read Cursor                              Write Cursor  │
│   (playback)                               (WebSocket)   │
└─────────────────────────────────────────────────────────┘
       │                                          │
       │                                          │
       ▼                                          ▼
  Audio Output                            Network Jitter
  (speakers)                              Tracking (EMA)

Buffer Health Tiers:
• Emergency (<1 chunk): Display buffering indicator
• Low (1-2 chunks): Increase buffer target to 5 chunks
• Healthy (3-4 chunks): Optimal latency (~300-400ms)
• Full (5+ chunks): Reduce buffer target to 3 chunks
```

### Synthesis Pipeline

```
MusicalContext (key, mode, BPM, intensity)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                  Composition Phase                      │
│  (Generate MIDI-like note events, <50ms)                │
└─────────────────────────────────────────────────────────┘
       │
       ├──→ ChordProgression: [(onset, root, type), ...]
       └──→ MelodyPhrase: [(onset, pitch, velocity, duration), ...]
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                   Synthesis Phase                       │
│  (FluidSynth rendering, <100ms per phrase)              │
│                                                         │
│  1. Load SoundFonts (startup only):                     │
│     • Piano: Salamander Grand Piano (preset 0)          │
│     • Pad: Arachno Warm Pad (preset 88-90)              │
│                                                         │
│  2. Configure FluidSynth:                               │
│     • Sample rate: 44.1kHz                              │
│     • Reverb: 3s decay, 20% wet (minimal)               │
│     • Polyphony: 32 voices                              │
│                                                         │
│  3. Render note events:                                 │
│     • noteon(pitch, velocity, onset_time)               │
│     • noteoff(pitch, onset_time + duration)             │
│     • Mix piano (50%) + pad (40%)                       │
│                                                         │
│  4. Output: NumPy array (2, num_samples) float32        │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                   Chunking Phase                        │
│  (Split into 100ms chunks for streaming)                │
│                                                         │
│  Chunk size: 4,410 samples/channel                      │
│  Chunk bytes: 4,410 × 2 channels × 2 bytes = 17.6kB     │
│  Base64: ~23.5kB per chunk                              │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                   Streaming Phase                       │
│  (WebSocket delivery to clients)                        │
└─────────────────────────────────────────────────────────┘
```

## Next Steps

After completing this plan:

1. ✅ **Phase 0 Complete**: Research findings documented in `research.md`
2. ✅ **Phase 1 Complete**: Data model, contracts, quickstart scenarios documented
3. ⏭️ **Generate Tasks**: Run `/speckit.tasks` to create `tasks.md` with implementation breakdown
4. ⏭️ **Implementation**: Execute tasks in dependency order
5. ⏭️ **Testing**: Run integration tests, performance benchmarks
6. ⏭️ **Validation**: Verify success criteria from spec.md
7. ⏭️ **Merge**: Create PR, merge to main after review

---

**Plan Status**: ✅ Phase 0 & Phase 1 Complete (December 28, 2024)
**Research Documents**: [docs/technical-research.md](../../docs/technical-research.md), [docs/technical-decisions-summary.md](../../docs/technical-decisions-summary.md)
**Next Command**: `/speckit.tasks` to generate dependency-ordered implementation tasks
