# Implementation Plan: FluidSynth Sample-Based Instrument Synthesis

**Branch**: `004-fluidsynth-integration` | **Date**: 2025-12-28 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-fluidsynth-integration/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Replace oscillator-based synthesis with realistic sampled instruments (Acoustic Grand Piano, Polysynth Pads, Choir voices) using FluidSynth and SoundFont libraries. Maintain existing PyTorch synthesis for percussion. Preserve <100ms latency and 44.1kHz streaming quality while introducing sample-based rendering for melodic, pad, and swell voices. Hybrid synthesis system mixing FluidSynth-rendered instruments with PyTorch-rendered percussion.

## Technical Context

**Language/Version**: Python 3.12+ via uv
**Primary Dependencies**: FastAPI, PyTorch, asyncio, WebSockets, **pyfluidsynth** (official Python bindings)
**New Dependencies**: FluidSynth library (via brew/apt), SoundFont (SF2) file management, numpy for audio mixing
**Storage**: Audio ring buffers, SoundFont files (SF2) in `./soundfonts/` directory, loaded at startup
**Testing**: Integration tests for real-time audio performance, sample playback latency, polyphony limits, voice stealing behavior
**Target Platform**: Linux/macOS server with CPU for sample playback, GPU for PyTorch percussion (Metal/CUDA), Web clients
**Project Type**: Real-time audio streaming server with hybrid synthesis (sample-based + synthesized)
**Performance Goals**: <100ms total audio processing latency, continuous 44.1kHz/16-bit stereo streaming
**Constraints**: No blocking operations in audio pipeline, sample playback must be CPU-efficient, FluidSynth latency budget 30-50ms
**Scale/Scope**: Concurrent WebSocket connections, polyphonic sample playback (15+ notes), voice stealing algorithm
**Integration Points**: Existing composition layer (chord_generator, melody_generator), existing synthesis_engine (keep PyTorch kicks), existing streaming infrastructure
**Key Technical Decisions**:
- **Python Binding**: pyfluidsynth (ctypes-based, real-time capable, <10ms latency)
- **SoundFont Library**: FluidR3_GM.sf2 (142MB, MIT license, all 128 GM presets)
- **Storage Location**: `./soundfonts/` directory (gitignored, startup validation, fail-fast if missing)
- **GM Preset Mapping**: Acoustic Grand Piano (0), Pad Polysynth (90), Choir Aahs (52), Voice Oohs (53)
- **Hybrid Architecture**: FluidSynth for melody/pads/swells, PyTorch for kicks
- **Polyphony Management**: FluidSynth built-in voice stealing (oldest-first with release-phase preference, configured via `synth.polyphony=20`)
- **Audio Mixing**: Weighted sum (40% pads, 50% melody, 30% kicks, 20% swells) with auto-gain scaling and soft knee limiting (threshold=0.8, knee=0.1)
- **Startup Validation**: Three-layer validation (filesystem, size check >100MB, FluidSynth load test), fail fast if any validation fails
- **Sample Rate Handling**: FluidSynth automatic resampling to 44.1kHz via 4th-order polynomial interpolation (transparent, no manual preprocessing)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Initial Check (Pre-Research)

- **UV-First**: ✅ PASS - FluidSynth Python bindings will be added via `uv add`, all dependency management through uv
- **Real-Time Performance**: ⚠️ CONDITIONAL PASS - Must validate FluidSynth can render samples within <100ms latency budget. Research required to confirm CPU sample playback meets real-time constraints
- **Modular Architecture**: ✅ PASS - FluidSynth integration contained within synthesis layer (server/synthesis_engine.py or new server/fluidsynth_renderer.py), composition layer unchanged, client unchanged
- **GPU Acceleration**: ⚠️ JUSTIFIED VIOLATION - FluidSynth is CPU-based sample playback, NOT GPU-accelerated. Violation justified because:
  - Sample-based synthesis fundamentally requires CPU (playing pre-recorded audio samples)
  - GPU retained for existing PyTorch percussion synthesis
  - User explicitly requested realistic instrument timbres, requiring sample playback
  - Alternative GPU approaches (neural synthesis) rejected in specification clarification
  - Hybrid approach minimizes performance impact by limiting CPU synthesis to melodic/pad/swell voices only
- **WebSocket Protocol**: ✅ PASS - No changes to streaming protocol, FluidSynth output mixed into existing PCM chunk pipeline
- **Developer Experience**: ✅ PASS - FluidSynth has well-documented Python API, integration should maintain low complexity. Voice stealing and mixing logic must be kept under complexity threshold (≤10 McCabe)

### Final Check (Post-Research & Design)

- **UV-First**: ✅ PASS - Confirmed: `uv add pyfluidsynth`, system dependencies via brew/apt, all Python management through uv
- **Real-Time Performance**: ✅ PASS - Research confirms FluidSynth latency 5-20ms typical, 4th-order resampling adds 2-3ms, total budget 47-80ms well within <100ms requirement (see [research.md](research.md) Performance Budget Validation)
- **Modular Architecture**: ✅ PASS - Design creates 3 new modules (fluidsynth_renderer.py, soundfont_manager.py, voice_manager.py) maintaining single-responsibility, existing modules minimally modified
- **GPU Acceleration**: ⚠️ JUSTIFIED VIOLATION - Confirmed: FluidSynth is CPU-only, violation justified (see Complexity Tracking table). Hybrid approach preserves GPU for PyTorch kicks.
- **WebSocket Protocol**: ✅ PASS - Confirmed: No API changes required (see [contracts/README.md](contracts/README.md)), 100% backward compatible
- **Developer Experience**: ✅ PASS - Design uses FluidSynth built-in voice stealing (no complex manual implementation), mixing logic is simple weighted sum, estimated complexity ≤8 McCabe

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
# Auralis Modular Architecture (with FluidSynth Integration)
server/
├── main.py                      # FastAPI entrypoint (unchanged)
├── synthesis_engine.py          # MODIFIED: Hybrid synthesis coordinator
├── fluidsynth_renderer.py       # NEW: FluidSynth sample-based rendering
├── soundfont_manager.py         # NEW: SoundFont loading and validation
├── voice_manager.py             # NEW: Polyphony and voice stealing logic
├── ring_buffer.py               # Audio buffering (unchanged)
└── streaming_server.py          # WebSocket streaming logic (unchanged)

composition/
├── chord_generator.py           # Markov chord engine (unchanged)
├── melody_generator.py          # Constraint-based melody (unchanged)
└── percussion_generator.py      # Ambient percussion (unchanged)

client/
├── audio_client_worklet.js      # Web Audio API playback (unchanged)
├── audio_worklet_processor.js   # Audio thread processor (unchanged)
└── index.html                   # Control interface (unchanged)

soundfonts/                      # NEW: SoundFont file storage
├── acoustic_grand_piano.sf2     # Acoustic Grand Piano (GM preset 0)
├── pad_polysynth.sf2            # Pad Polysynth (GM preset 90)
└── choir_voices.sf2             # Choir Aahs (52) & Voice Oohs (53)

tests/
├── integration/
│   ├── test_websocket_streaming.py       # Existing (unchanged)
│   ├── test_audio_latency.py             # Existing (unchanged)
│   ├── test_fluidsynth_rendering.py      # NEW: FluidSynth integration tests
│   ├── test_hybrid_synthesis.py          # NEW: Mixed FluidSynth + PyTorch
│   └── test_polyphony_voice_stealing.py  # NEW: Voice management tests
└── performance/
    └── test_real_time_constraints.py     # MODIFIED: Add FluidSynth latency tests
```

**Structure Decision**:
- **New Modules**: Three new modules for FluidSynth integration (fluidsynth_renderer.py, soundfont_manager.py, voice_manager.py) maintain separation of concerns
- **Modified Existing**: synthesis_engine.py becomes hybrid coordinator, delegating to FluidSynth renderer for melody/pads/swells, PyTorch for kicks
- **SoundFont Storage**: New soundfonts/ directory at repo root for SF2 files (excluded from git via .gitignore, documented in README for manual download)
- **Testing**: New integration tests for FluidSynth-specific functionality, modified performance tests to include sample playback latency
- **No Client Changes**: Streaming protocol unchanged, client unaware of synthesis method

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| GPU Acceleration principle (CPU-based FluidSynth) | Realistic instrument timbres require pre-recorded audio samples, which fundamentally need CPU-based sample playback | GPU synthesis approaches (torchsynth oscillators, neural synthesis like RAVE) cannot produce authentic acoustic instrument sounds. User explicitly rejected neural synthesis in spec clarification. PyTorch oscillators produce "synthetic" sound unsuitable for piano/choir realism. Sample-based synthesis is the only viable approach for authentic instrument timbres. |
| Increased architectural complexity (3 new modules) | FluidSynth integration requires dedicated modules for: (1) soundfont_manager.py for SF2 loading/validation, (2) fluidsynth_renderer.py for sample playback, (3) voice_manager.py for polyphony limits | Integrating FluidSynth directly into synthesis_engine.py would violate single-responsibility principle and create >10 McCabe complexity. Monolithic approach rejected because it couples SoundFont management, voice stealing, and audio rendering in one module, making testing and maintenance difficult. |
