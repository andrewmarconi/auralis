# Implementation Plan: Phase 1 MVP - Real-time Ambient Music Streaming

**Branch**: `001-phase1-mvp` | **Date**: 2024-12-26 | **Spec**: [specs/001-phase1-mvp/spec.md](spec.md)
**Input**: Feature specification from `/specs/001-phase1-mvp/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Phase 1 MVP delivers end-to-end ambient music streaming with real-time generation using Markov chord progressions, constraint-based melodies, and torchsynth synthesis. Research confirms bigram Markov chains with ambient-biased transitions provide sufficient musical coherence for MVP. WebSocket streaming with base64-encoded 16-bit PCM chunks at 100ms intervals achieves target <800ms latency. GPU acceleration via Metal/CUDA ensures real-time synthesis performance.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.12+ via uv  
**Primary Dependencies**: FastAPI, PyTorch, torchsynth, asyncio, WebSockets  
**Storage**: Audio ring buffers, temporary in-memory storage  
**Testing**: Integration tests for real-time audio performance  
**Target Platform**: Linux server with GPU support (Metal/CUDA), Web clients  
**Project Type**: Real-time audio streaming server  
**Performance Goals**: <100ms audio processing latency, continuous 44.1kHz streaming  
**Constraints**: No blocking operations in audio pipeline, GPU acceleration required  
**Scale/Scope**: Concurrent WebSocket connections, real-time synthesis engine

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **UV-First**: All Python dependencies managed via uv, no manual venv creation
- **Real-Time Performance**: Audio streaming latency <100ms preserved in all changes  
- **Modular Architecture**: Clear separation of server/, composition/, client/ modules
- **GPU Acceleration**: Metal/CUDA utilization prioritized for synthesis
- **WebSocket Protocol**: Audio data exclusively via WebSocket streaming

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
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
# Auralis Modular Architecture
server/
├── main.py                 # FastAPI entrypoint
├── synthesis_engine.py    # torchsynth synthesis core
├── ring_buffer.py         # Audio buffering
└── streaming_server.py    # WebSocket streaming logic

composition/
├── chord_generator.py     # Markov chord engine
├── melody_generator.py    # Constraint-based melody
└── percussion_generator.py# Ambient percussion

client/
├── audio_client.js        # Web Audio API playback
└── index.html             # Control interface

tests/
├── integration/
│   ├── test_websocket_streaming.py
│   ├── test_audio_latency.py
│   └── test_gpu_acceleration.py
└── performance/
    └── test_real_time_constraints.py
```

**Structure Decision**: Modular architecture with server/ (FastAPI + synthesis), composition/ (generative algorithms), client/ (Web Audio API), and tests/ directories as specified in plan structure

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
