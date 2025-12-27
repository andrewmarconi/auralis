# Implementation Plan: Performance Optimizations

**Branch**: `003-performance-optimizations` | **Date**: 2025-12-26 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-performance-optimizations/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Optimize Auralis for production-grade performance by ensuring smooth audio streaming (99% chunk delivery within 50ms), supporting 10+ concurrent users without degradation, and reducing resource consumption by 30% compared to Phase 1 baseline. Focuses on audio buffer management, GPU acceleration, concurrent WebSocket handling, and memory leak prevention to deliver uninterrupted ambient music streaming for extended sessions (8+ hours).

## Technical Context

**Language/Version**: Python 3.12+ via uv
**Primary Dependencies**: FastAPI, uvicorn, PyTorch, torchsynth, asyncio, WebSockets, numpy
**Storage**: Thread-safe ring buffer (2-second capacity), in-memory audio chunks
**Testing**: Integration tests for latency/jitter, performance benchmarks, load testing
**Target Platform**: macOS (Metal), Linux (CUDA/CPU), Web clients (Chrome/Firefox/Safari)
**Project Type**: Real-time audio streaming server with GPU-accelerated synthesis
**Performance Goals**:
- 99% of chunks delivered within 50ms of schedule
- <100ms total audio processing latency (synthesis + buffering + transmission)
- 30% reduction in resource usage vs. Phase 1 baseline
- Support 10+ concurrent users without degradation
- Stable memory usage over 8+ hour sessions

**Current Baseline** (Phase 1 MVP):
- Single-threaded synthesis with basic ring buffer
- Sequential WebSocket handling (one client at a time)
- No adaptive buffering or jitter management
- Basic GPU device selection, no memory optimization
- Minimal performance monitoring

**Optimization Targets**:
1. **Buffer Management**: NEEDS RESEARCH - adaptive buffering, jitter reduction, back-pressure control
2. **Concurrency**: NEEDS RESEARCH - FastAPI/asyncio patterns for 10+ concurrent WebSocket streams
3. **GPU Optimization**: NEEDS RESEARCH - batch synthesis, memory management, Metal/CUDA-specific tuning
4. **Memory Leaks**: NEEDS RESEARCH - prevention techniques, profiling tools, cleanup patterns
5. **Monitoring**: NEEDS RESEARCH - performance metrics, observability, alerting strategies

**Constraints**:
- No blocking operations in audio pipeline (preserves <100ms latency)
- GPU acceleration required for synthesis
- WebSocket-only for audio data (constitution requirement)
- Must maintain backward compatibility with Phase 2 controls API

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Initial Check (Before Phase 0 Research)

✅ **UV-First**: All Python dependencies managed via uv, no manual venv creation
- Performance optimizations will not introduce new package managers
- Profiling/monitoring tools added via `uv add` only

✅ **Real-Time Performance**: Audio streaming latency <100ms preserved in all changes
- All optimizations measured against <100ms total latency requirement
- No blocking operations introduced in audio pipeline
- Buffer optimizations specifically target latency reduction

✅ **Modular Architecture**: Clear separation of server/, composition/, client/ modules
- Performance improvements contained within existing module boundaries
- New monitoring code isolated in dedicated module if needed
- No circular dependencies introduced

✅ **GPU Acceleration**: Metal/CUDA utilization prioritized for synthesis
- GPU optimization is a primary focus area
- Memory management improvements maintain GPU-first approach
- Fallback to CPU remains available but not optimized

✅ **WebSocket Protocol**: Audio data exclusively via WebSocket streaming
- Concurrency improvements apply to WebSocket connections only
- Performance metrics may use REST API (non-audio data)
- No change to audio streaming protocol

**Initial Gate Status**: PASSED - All constitution principles satisfied.

---

### Re-Evaluation (After Phase 1 Design - 2025-12-26)

✅ **UV-First Development** - STILL PASSES
- **Verified in research.md**: All dependencies added via `uv add prometheus-client psutil`
- **Verified in quickstart.md**: All test commands use `uv run` prefix
- **No violations**: Zero manual venv creation or alternative package managers

✅ **Real-Time Audio Performance (<100ms latency)** - STILL PASSES
- **Verified in research.md**: Adaptive buffer tiers, EMA jitter tracking, and token bucket all designed to maintain <100ms latency
- **Verified in data-model.md**: All buffer models (AdaptiveRingBuffer, JitterTracker) track timing constraints
- **Verified in contracts/websocket-api.md**: Explicit timing SLA "99% of chunks delivered within 50ms"
- **Verified in contracts/metrics-api.md**: Alert rule triggers if p99 synthesis latency >100ms
- **Verified in quickstart.md**: All test scenarios verify <100ms latency targets
- **No violations**: Zero blocking operations introduced in audio pipeline

✅ **Modular Architecture (server/, composition/, client/)** - STILL PASSES
- **Verified in research.md**: Optimizations clearly separated by module (buffer in server/ring_buffer.py, synthesis in server/synthesis_engine.py)
- **Verified in data-model.md**: Models organized by architectural concern (Buffer Management, WebSocket Concurrency, GPU Optimization)
- **Verified in contracts/internal-interfaces.md**: Abstract interfaces (IRingBuffer, ISynthesisEngine, IStreamingServer) with dependency injection prevent circular dependencies
- **No violations**: All changes respect existing module boundaries

✅ **GPU-Accelerated Synthesis (Metal/CUDA priority)** - STILL PASSES
- **Verified in research.md**: Comprehensive GPU optimization section (batch processing, torch.compile, kernel fusion, device-specific tuning)
- **Verified in data-model.md**: DeviceInfo and DeviceSelector models with Metal/CUDA priority over CPU
- **Verified in contracts/internal-interfaces.md**: IDeviceSelector interface enforces priority: Metal > CUDA > CPU fallback
- **Verified in quickstart.md**: Multiple GPU tests (device selection, batch synthesis, torch.compile benchmarks)
- **No violations**: GPU optimization is a primary focus, CPU fallback exists but not optimized

✅ **WebSocket Streaming Protocol (audio only via WebSocket)** - STILL PASSES
- **Verified in research.md**: All audio streaming optimizations target WebSocket protocol (broadcast architecture, per-client cursors, token bucket)
- **Verified in data-model.md**: BroadcastRingBuffer and WebSocketClientState models exclusively for WebSocket streaming
- **Verified in contracts/websocket-api.md**: Detailed WebSocket protocol specification for audio chunks (base64-encoded PCM)
- **Verified in contracts/metrics-api.md**: REST API only for Prometheus metrics (non-audio data), separated from audio streaming
- **No violations**: Audio data remains exclusively WebSocket, no REST alternative introduced

**Final Gate Status**: ✅ PASSED - All 5 constitution principles verified against Phase 1 design artifacts (research.md, data-model.md, contracts/websocket-api.md, contracts/metrics-api.md, contracts/internal-interfaces.md, quickstart.md). Zero violations detected. Ready to proceed to Phase 2 (task generation via /speckit.tasks).

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
# Auralis Modular Architecture - Performance Optimization Targets

server/
├── main.py                 # FastAPI entrypoint + metrics endpoints
│                           # [OPTIMIZE: Add Prometheus/metrics integration]
├── synthesis_engine.py     # torchsynth GPU rendering
│                           # [OPTIMIZE: Batch synthesis, GPU memory management]
├── ring_buffer.py          # Thread-safe audio buffer
│                           # [OPTIMIZE: Adaptive sizing, jitter reduction]
├── streaming_server.py     # WebSocket streaming logic
│                           # [OPTIMIZE: Concurrent connection handling]
└── presets.py              # Generation presets (Phase 2)
                            # [NO CHANGES]

composition/
├── chord_generator.py      # Markov chord progressions
├── melody_generator.py     # Constraint-based melody
└── percussion_generator.py # Sparse percussion events
                            # [MINIMAL CHANGES: Performance only if profiling shows bottlenecks]

client/
├── index.html              # UI with controls
├── audio_client_worklet.js # AudioContext setup
│                           # [OPTIMIZE: Adaptive client buffering]
├── audio_worklet_processor.js # Audio thread processor
└── debug.html              # Debug interface

tests/
├── integration/            # WebSocket, latency, GPU tests
│                           # [ADD: Concurrent user load tests]
└── performance/            # Real-time constraint benchmarks
                            # [ADD: Memory leak tests, resource profiling]
```

**Structure Decision**: Performance optimizations will primarily target `server/` modules:
- `ring_buffer.py`: Adaptive buffer management
- `synthesis_engine.py`: GPU optimization and memory management
- `streaming_server.py`: Concurrent WebSocket handling
- `main.py`: Performance monitoring and metrics
- Client-side: Adaptive buffering improvements in `audio_client_worklet.js`

Composition modules remain unchanged unless profiling identifies bottlenecks. New test files added to `tests/performance/` and `tests/integration/` for load testing and memory profiling.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
