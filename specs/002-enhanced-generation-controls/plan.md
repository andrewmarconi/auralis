# Implementation Plan: Enhanced Generation & Controls

**Branch**: `002-enhanced-generation-controls` | **Date**: 2025-12-26 | **Spec**: specs/002-enhanced-generation-controls/spec.md
**Input**: Feature specification from `/specs/002-enhanced-generation-controls/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement enhanced music generation algorithms and additional control parameters beyond Phase 1's basic key, BPM, and intensity controls. Focus on sophisticated Markov chains, constraint-based melody generation, and real-time parameter validation with GPU acceleration.

## Technical Context

**Language/Version**: Python 3.12+ via uv  
**Primary Dependencies**: FastAPI, PyTorch, torchsynth, asyncio, WebSockets  
**Storage**: Audio ring buffers, temporary in-memory storage  
**Testing**: Integration tests for real-time audio performance  
**Target Platform**: Linux server with GPU support (Metal/CUDA), Web clients  
**Project Type**: Real-time audio streaming server  
**Performance Goals**: <100ms audio processing latency, continuous 44.1kHz streaming  
**Constraints**: No blocking operations in audio pipeline, GPU acceleration required  
**Scale/Scope**: Concurrent WebSocket connections, real-time synthesis engine

**Unknowns/Clarifications**:
- Specific enhanced Markov chain algorithms for chord progressions (NEEDS CLARIFICATION: higher-order chains vs. weighted probabilities)
- Entropy calculation method for chord progression variety (NEEDS CLARIFICATION: Shannon entropy on chord transitions)
- Parameter validation logic for conflicts (NEEDS CLARIFICATION: rule-based vs. ML-based detection)
- Automatic parameter adjustment for computational overload (NEEDS CLARIFICATION: scaling algorithms)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **UV-First**: All Python dependencies managed via uv, no manual venv creation ✓
- **Real-Time Performance**: Audio streaming latency <100ms preserved in all changes ✓  
- **Modular Architecture**: Clear separation of server/, composition/, client/ modules ✓
- **GPU Acceleration**: Metal/CUDA utilization prioritized for synthesis ✓
- **WebSocket Protocol**: Audio data exclusively via WebSocket streaming ✓

## Project Structure

### Documentation (this feature)

```text
specs/002-enhanced-generation-controls/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
# Auralis Modular Architecture
server/
├── main.py                 # FastAPI entrypoint
├── synthesis_engine.py    # torchsynth synthesis core
├── ring_buffer.py         # Audio buffering
└── streaming_server.py    # WebSocket streaming logic

composition/
├── chord_generator.py     # Markov chord engine (enhanced)
├── melody_generator.py    # Constraint-based melody (enhanced)
└── percussion_generator.py# Ambient percussion

client/
├── audio_client.js        # Web Audio API playback
└── index.html             # Control interface (extended parameters)

tests/
├── integration/
│   ├── test_websocket_streaming.py
│   ├── test_audio_latency.py
│   └── test_gpu_acceleration.py
└── performance/
    └── test_real_time_constraints.py
```

**Structure Decision**: Maintains existing modular architecture with enhancements to composition/ algorithms and client/ controls. No new modules required.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations detected - all constitution principles maintained.</content>
<parameter name="filePath">/Users/andrew/Develop/auralis/specs/001-enhanced-generation-controls/plan.md