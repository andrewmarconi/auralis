# Tasks: Auralis MVP v2.0 - Real-Time Generative Ambient Music Streaming Engine

**Input**: Design documents from `/specs/001-mvp-v2/`
**Prerequisites**: plan.md, spec.md (5 user stories), data-model.md, contracts/ (websocket-api.md, http-api.md, internal-interfaces.md), quickstart.md

**Tests**: This project includes integration and performance tests to validate real-time audio performance requirements.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4, US5)
- Include exact file paths in descriptions

## Path Conventions

Based on plan.md structure:
- **Server**: `server/` at repository root
- **Composition**: `composition/` at repository root
- **Client**: `client/` at repository root
- **Tests**: `tests/` at repository root
- **Docs**: `docs/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure per plan.md

- [X] T001 Create .gitignore with Python patterns (__pycache__/, *.pyc, .venv/, venv/, dist/, *.egg-info/, .DS_Store, Thumbs.db, *.tmp, *.swp, .idea/)
- [X] T002 [P] Create server/ directory structure (server/__init__.py, server/interfaces/)
- [X] T003 [P] Create composition/ directory structure (composition/__init__.py)
- [X] T004 [P] Create client/ directory structure (client/index.html placeholder)
- [X] T005 [P] Create tests/ directory structure (tests/__init__.py, tests/integration/, tests/performance/, tests/unit/)
- [X] T006 [P] Create soundfonts/ directory structure (soundfonts/.gitkeep, soundfonts/.env.example with SoundFont download links)
- [X] T007 Initialize pyproject.toml with project metadata and Python 3.12+ requirement
- [X] T008 Add core server dependencies via uv: fastapi==0.127.0, uvicorn==0.40.0, pydantic==2.12.0
- [X] T009 [P] Add audio processing dependencies via uv: numpy==1.26.0, scipy==1.11.0
- [X] T010 [P] Add FluidSynth dependency via uv: pyfluidsynth==1.3.2
- [X] T011 [P] Add dev dependencies via uv: pytest, pytest-cov, pytest-asyncio, black, ruff, mypy
- [X] T012 Create .env.example with AURALIS_ENV, AURALIS_HOST, AURALIS_PORT, SOUNDFONT_PIANO, SOUNDFONT_GM, AURALIS_LOG_LEVEL
- [X] T013 [P] Create server/config.py for environment variable configuration and validation
- [X] T014 [P] Create server/logging_config.py for structured logging setup

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Core Synthesis Infrastructure

- [X] T015 Create server/interfaces/__init__.py with ISynthesisEngine, IFluidSynthRenderer ABCs per internal-interfaces.md
- [X] T016 [P] Create server/interfaces/buffer.py with IRingBuffer, IBufferManager ABCs per internal-interfaces.md
- [X] T017 [P] Create server/interfaces/metrics.py with IMetricsCollector, IMemoryMonitor ABCs per internal-interfaces.md
- [X] T018 Create server/device_selector.py for GPU/CPU device detection (Metal/CUDA/CPU priority)
- [X] T019 Create server/gc_config.py with gc.set_threshold(10000, 20, 20) for real-time audio GC tuning
- [X] T020 Create server/fluidsynth_renderer.py implementing IFluidSynthRenderer (FluidSynth wrapper, SoundFont loading, note rendering)
- [X] T021 Create server/soundfont_manager.py for SoundFont file management (Piano channel 0, Pad channel 1, preset mapping)

### Streaming Infrastructure

- [X] T022 Create server/ring_buffer.py implementing IRingBuffer (NumPy pre-allocated circular buffer, threading.Lock, 20 chunk capacity)
- [X] T023 [P] Create server/buffer_management.py with back-pressure logic (sleep 10ms if depth <2 chunks)
- [X] T024 [P] Create server/metrics.py implementing IMetricsCollector (latency histograms, event counters, memory tracking)
- [X] T025 [P] Create server/memory_monitor.py implementing IMemoryMonitor (tracemalloc integration, <500MB budget tracking)
- [X] T026 Create server/streaming_server.py with WebSocket endpoint skeleton per websocket-api.md (/ws/stream)

### Composition Infrastructure

- [X] T027 Create composition/__init__.py with ChordGenerator, MelodyGenerator exports
- [X] T028 Create composition/chord_generator.py with Markov chain (bigram order 2) chord progression generation
- [X] T029 [P] Create composition/melody_generator.py with constraint-based melody generation (70% chord tones, 25% scale, 5% chromatic)
- [X] T030 Create composition/percussion_generator.py as placeholder (post-MVP, raises NotImplementedError)

### Server Entrypoint

- [X] T031 Create server/__init__.py with package exports
- [X] T032 Create server/main.py with FastAPI app initialization, startup/shutdown lifecycle, CORS middleware
- [X] T033 [P] Add server/presets.py with musical presets (Focus: Dorian/60 BPM/0.5, Meditation: Aeolian/50/0.3, Sleep: Phrygian/40/0.2, Bright: Lydian/70/0.6)
- [X] T034 [P] Create server/di_container.py for dependency injection container pattern

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 4 - Low-Latency Streaming Performance (Priority: P1) 🎯 MVP Foundation

**Goal**: Achieve <800ms end-to-end latency with <100ms synthesis, enabling real-time parameter changes and smooth streaming

**Independent Test**: Measure end-to-end latency from generation to playback (target <800ms 95th percentile), synthesis performance (<100ms per phrase), buffer health (>98% chunks on time)

### Performance Tests for US4

- [ ] T035 [P] [US4] Create tests/performance/__init__.py
- [ ] T036 [P] [US4] Create tests/performance/benchmark_suite.py with synthesis latency benchmarking (<100ms target for 8 bars @ 60 BPM)
- [ ] T037 [P] [US4] Create tests/performance/test_batch_synthesis.py to measure synthesis throughput (10× faster than real-time minimum)
- [ ] T038 [P] [US4] Create tests/performance/test_chunk_delivery.py to validate >98% on-time delivery with <50ms timing variance

### Implementation for US4

- [ ] T039 [US4] Implement server/synthesis_engine.py orchestrating ChordGenerator → MelodyGenerator → FluidSynthRenderer pipeline
- [ ] T040 [US4] Add FluidSynth reverb configuration (3s decay, 20% wet, 0.6 roomsize, 0.5 damping) to fluidsynth_renderer.py
- [ ] T041 [US4] Implement AudioBuffer chunking logic in synthesis_engine.py (split stereo PCM into 100ms chunks of 4,410 samples)
- [ ] T042 [US4] Add base64 PCM encoding to AudioChunk serialization in streaming_server.py per websocket-api.md
- [ ] T043 [US4] Implement continuous generation loop in synthesis_engine.py with back-pressure (check ring buffer depth <2, sleep 10ms)
- [ ] T044 [US4] Add GPU acceleration detection and device selection in server/main.py startup (Metal/CUDA/CPU priority)
- [ ] T045 [US4] Implement synthesis latency tracking in metrics.py (record phrase generation time, calculate avg/p50/p95/p99)
- [ ] T046 [US4] Add end-to-end latency measurement (server timestamp in AudioChunk, client reports playback time)

### Integration Tests for US4

- [ ] T047 [P] [US4] Create tests/integration/__init__.py
- [ ] T048 [US4] Create tests/integration/test_fluidsynth_rendering.py to verify <100ms synthesis latency on sample 8-bar phrase
- [ ] T049 [US4] Create tests/integration/test_buffer_underruns.py to validate buffer health >98% over 5-minute streaming session

**Checkpoint**: Performance infrastructure complete - can now build playback features with confidence in latency targets

---

## Phase 4: User Story 1 - Instant Ambient Playback (Priority: P1) 🎯 MVP Core

**Goal**: User opens browser and immediately hears evolving ambient music within 2 seconds without configuration

**Independent Test**: Navigate to application URL, verify audio plays within 2 seconds, streams continuously for 30+ minutes without glitches, exhibits gradual musical evolution

### Client Implementation for US1

- [ ] T050 [P] [US1] Create client/index.html with minimal UI (connection status indicator, title, auto-play setup)
- [ ] T051 [P] [US1] Create client/audio_worklet_processor.js implementing AudioWorklet thread (decode base64 PCM, queue to buffer, output to speakers)
- [ ] T052 [US1] Create client/audio_client_worklet.js implementing AudioContext setup, WebSocket connection to /ws/stream, adaptive buffering (300-500ms target)
- [ ] T053 [US1] Implement 4-tier adaptive buffering in audio_client_worklet.js (Emergency <1 chunk, Low 1-2, Healthy 3-4, Full 5+)
- [ ] T054 [US1] Add EMA jitter tracking (α=0.1) to audio_client_worklet.js for buffer size adjustment
- [ ] T055 [US1] Implement base64 PCM decoding (Int16Array → Float32Array normalization) in audio_worklet_processor.js

### Server Integration for US1

- [ ] T056 [US1] Complete WebSocket endpoint implementation in streaming_server.py (accept connections, stream AudioChunks with seq/timestamp/data)
- [ ] T057 [US1] Implement WebSocket message handler for audio chunk delivery (read from RingBuffer, serialize to JSON per websocket-api.md)
- [ ] T058 [US1] Add connection state management in streaming_server.py (track client_id, connected_at, last_chunk_seq, buffer_health)
- [ ] T059 [US1] Integrate synthesis_engine continuous loop with RingBuffer (generate phrases, chunk audio, write to buffer)
- [ ] T060 [US1] Add SoundFont loading validation in server/main.py startup (verify Piano and Pad SF2 files exist, load presets)

### REST API for US1

- [ ] T061 [P] [US1] Implement GET /api/status endpoint in server/main.py per http-api.md (uptime, connections, buffer_depth, device, soundfont_loaded, synthesis_active)
- [ ] T062 [P] [US1] Implement GET /api/metrics endpoint in server/main.py per http-api.md (synthesis_latency, network_latency, buffer_underruns, memory_usage_mb)
- [ ] T063 [P] [US1] Implement GET / root endpoint in server/main.py to serve client/index.html
- [ ] T064 [P] [US1] Mount static file serving for /client/* in server/main.py (serve .js files)

### Integration Tests for US1

- [ ] T065 [US1] Create tests/integration/test_smooth_streaming.py to verify 30-minute continuous playback without dropouts on stable network
- [ ] T066 [US1] Add time-to-first-audio measurement test (verify <2 seconds from page load to audible output)

**Checkpoint**: Core MVP complete - user can load page and hear ambient music immediately

---

## Phase 5: User Story 2 - Musical Coherence and Quality (Priority: P1) 🎯 MVP Quality

**Goal**: Harmonically pleasing music with 95%+ note constraint adherence, no dissonant surprises, ambient-appropriate progressions

**Independent Test**: Analyze generated MIDI/audio for harmonic consistency (95%+ notes within chord/scale constraints), listening tests with target users (75%+ "harmonically pleasing"), average session duration 45+ minutes

### Composition Enhancement for US2

- [X] T067 [US2] Enhance chord_generator.py to enforce modal constraints (Aeolian, Dorian, Lydian, Phrygian scale degrees)
- [X] T068 [US2] Add chord transition smoothing in chord_generator.py (avoid V-I cadences, prefer circular/modal progressions)
- [X] T069 [US2] Implement harmonic rhythm control in chord_generator.py (transitions at 30-90 second intervals, configurable via intensity)
- [X] T070 [US2] Enhance melody_generator.py note selection to enforce 70% chord tones, 25% scale notes, 5% chromatic passing tones
- [X] T071 [US2] Add velocity humanization in melody_generator.py (randomize within 20-100 range, weighted by intensity)
- [X] T072 [US2] Implement note probability (50-80%) in melody_generator.py to create sparse ambient texture
- [X] T073 [US2] Add phrase boundary detection in synthesis_engine.py (apply parameter changes only at phrase boundaries, not mid-phrase)

### Quality Validation for US2

- [ ] T074 [P] [US2] Create tests/unit/test_adaptive_buffer.py to validate client-side buffering logic (jitter tracking, tier transitions)
- [ ] T075 [P] [US2] Create tests/unit/test_jitter_tracker.py to verify EMA calculations (α=0.1, convergence behavior)
- [ ] T076 [US2] Add harmonic analysis test in tests/integration/ to verify 95%+ note constraint adherence over 100 generated phrases
- [ ] T077 [US2] Add phrase transition smoothness test to validate zero hard cuts or abrupt key changes in 30-minute sessions

**Checkpoint**: Musical quality validated - output matches ambient music principles

---

## Phase 6: User Story 3 - Personalized Control (Priority: P2) ✨ Enhancement

**Goal**: User adjusts key, mode, intensity, tempo or selects presets, changes apply smoothly within 5 seconds at phrase boundaries

**Independent Test**: Adjust each control parameter, verify changes apply within 5 seconds at phrase boundaries, confirm settings persist across browser refresh

### UI Controls for US3

- [X] T078 [US3] Enhance client/index.html with control UI (key selector dropdown: C/D/E/G/A major/minor, mode selector: Aeolian/Dorian/Lydian/Phrygian, intensity slider: 0.0-1.0, BPM slider: 60-120)
- [X] T079 [US3] Add preset buttons to client/index.html (Focus, Meditation, Sleep, Bright with visual styling)
- [X] T080 [US3] Implement localStorage persistence in audio_client_worklet.js (save key/mode/intensity/BPM on change, restore on page load)
- [X] T081 [US3] Add control change handlers in audio_client_worklet.js (send WebSocket JSON messages on slider/dropdown changes)
- [X] T082 [US3] Implement preset click handlers in audio_client_worklet.js (load preset values from presets.py mapping, update UI, send to server)

### Server Control Handling for US3

- [X] T083 [US3] Add WebSocket control message handler in streaming_server.py (parse JSON messages for key/mode/intensity/BPM updates)
- [X] T084 [US3] Implement MusicalContext updates in synthesis_engine.py (apply changes at next phrase boundary, not mid-phrase)
- [X] T085 [US3] Add parameter validation in streaming_server.py (BPM [60, 120], intensity [0.0, 1.0], supported keys/modes only)
- [X] T086 [US3] Add parameter change logging in synthesis_engine.py (INFO level, track user adjustments for debugging)

### Integration Tests for US3

- [X] T087 [US3] Create integration test for parameter changes (simulate WebSocket control messages, verify changes applied within 5 seconds)
- [X] T088 [US3] Add preset activation test (verify preset values match specification, changes applied correctly)

**Checkpoint**: User controls complete - personalization enhances core playback experience

---

## Phase 7: User Story 5 - Graceful Error Recovery (Priority: P2) 🛡️ Reliability

**Goal**: Network disconnects auto-reconnect, GPU fallback to CPU, buffer underflow displays indicator and resumes smoothly

**Independent Test**: Simulate network disconnects (verify auto-reconnect), disable GPU (verify CPU fallback), induce buffer underflow (verify buffering indicator and smooth resume)

### Error Handling for US5

- [ ] T089 [US5] Implement exponential backoff reconnection in audio_client_worklet.js (1s, 2s, 4s, 8s intervals up to 60s max, ±20% jitter)
- [ ] T090 [US5] Add WebSocket onclose handler in audio_client_worklet.js (trigger reconnection, display "Reconnecting..." status)
- [ ] T091 [US5] Add WebSocket onerror handler in audio_client_worklet.js (log error, update connection status indicator)
- [ ] T092 [US5] Implement buffer underflow detection in audio_worklet_processor.js (display "Buffering..." indicator when buffer empty)
- [ ] T093 [US5] Add GPU unavailable fallback in server/device_selector.py (log warning, notify user via /api/status endpoint)
- [ ] T094 [US5] Implement browser compatibility check in client/index.html (detect Chrome 90+, Edge 90+, Safari 14+, show error if unsupported)
- [ ] T095 [US5] Add SoundFont loading error handling in server/main.py (fail gracefully with clear error message if SF2 missing/corrupted)

### Error Recovery Tests for US5

- [ ] T096 [P] [US5] Create integration test for WebSocket reconnection (simulate disconnect, verify auto-reconnect within 10 seconds)
- [ ] T097 [P] [US5] Create test for CPU fallback (mock GPU unavailable, verify synthesis continues on CPU)
- [ ] T098 [US5] Add buffer underflow recovery test (pause chunk delivery, verify buffering indicator, resume delivery, verify smooth playback)

**Checkpoint**: Error recovery complete - system handles failures gracefully

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories, final validation, documentation

### Testing & Validation

- [ ] T099 [P] Create tests/unit/test_gc_config.py to validate GC tuning (verify thresholds set to 10000/20/20)
- [ ] T100 [P] Create tests/integration/test_memory_stability.py to verify <500MB memory over 8+ hour session with <10MB growth
- [ ] T101 Run all quickstart.md scenarios manually (Scenarios 1-15) and document any discrepancies
- [ ] T102 Run pytest with coverage report: `uv run pytest --cov=server --cov=composition --cov-report=html --cov-report=term`
- [ ] T103 Validate all integration test targets met (synthesis <100ms, end-to-end <800ms, buffer health >98%, memory <500MB)

### Code Quality

- [ ] T104 [P] Run black code formatter: `black server/ composition/ tests/`
- [ ] T105 [P] Run ruff linter: `ruff check server/ composition/ tests/`
- [ ] T106 [P] Run mypy type checker: `mypy server/ composition/ --strict`
- [ ] T107 Fix any linting, formatting, or type errors reported by T104-T106

### Documentation

- [ ] T108 [P] Create README.md with project overview, setup instructions (FluidSynth installation, SoundFont download, uv setup)
- [ ] T109 [P] Update CLAUDE.md if needed (ensure it reflects final structure and any new conventions)
- [ ] T110 [P] Create client/debug.html with real-time metrics visualization (synthesis latency chart, buffer health, memory usage)
- [ ] T111 Add inline documentation to complex functions (FluidSynth rendering, Markov chain logic, adaptive buffering)

### Final Validation

- [ ] T112 Verify time-to-first-audio <2 seconds on 10 Mbps connection (load page, measure to audible output)
- [ ] T113 Conduct 30-minute listening test (verify no dropouts, smooth transitions, musical coherence)
- [ ] T114 Test concurrent client support (10+ simultaneous WebSocket connections, verify no audio degradation)
- [ ] T115 Validate all success criteria from spec.md (SC-001 through SC-030, document pass/fail for each)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion (T001-T014) - BLOCKS all user stories
- **User Story 4 (Phase 3)**: Depends on Foundational completion (T015-T034) - Performance foundation
- **User Story 1 (Phase 4)**: Depends on Foundational completion (T015-T034) and US4 (T035-T049) - Core MVP
- **User Story 2 (Phase 5)**: Depends on Foundational completion and US1 (T050-T066) - Enhances musical quality
- **User Story 3 (Phase 6)**: Depends on Foundational completion and US1 - Adds personalization
- **User Story 5 (Phase 7)**: Depends on Foundational completion and US1 - Adds error recovery
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

- **US4 (Low-Latency Performance)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **US1 (Instant Playback)**: Can start after US4 complete - Builds on performance foundation
- **US2 (Musical Quality)**: Can start after US1 complete - Enhances playback musical coherence
- **US3 (Controls)**: Can start after US1 complete - Adds customization to working playback
- **US5 (Error Recovery)**: Can start after US1 complete - Adds robustness to working playback

### Within Each User Story

- Tests SHOULD be written and FAIL before implementation (TDD approach recommended but not required)
- Interfaces before implementations
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

**Setup Phase (T001-T014)**:
- T002, T003, T004, T005, T006 (directory creation) can run in parallel
- T008, T009, T010, T011 (dependency additions) can run in parallel
- T013, T014 (config/logging files) can run in parallel

**Foundational Phase (T015-T034)**:
- T016, T017 (interface files) can run in parallel after T015
- T023, T024, T025 (buffer management, metrics, memory monitor) can run in parallel after T022
- T033, T034 (presets, DI container) can run in parallel

**US4 Phase (T035-T049)**:
- T035, T036, T037, T038 (test files) can run in parallel

**US1 Phase (T050-T066)**:
- T050, T051 (client HTML and AudioWorklet) can run in parallel
- T061, T062, T063, T064 (REST endpoints and static serving) can run in parallel

**US2 Phase (T067-T077)**:
- T074, T075 (unit tests) can run in parallel

**Polish Phase (T099-T115)**:
- T099, T100 (unit tests) can run in parallel
- T104, T105, T106 (code quality checks) can run in parallel
- T108, T109, T110 (documentation) can run in parallel

---

## Parallel Example: User Story 4 (Performance)

```bash
# Launch all test file creation tasks together:
Task T035: "Create tests/performance/__init__.py"
Task T036: "Create tests/performance/benchmark_suite.py"
Task T037: "Create tests/performance/test_batch_synthesis.py"
Task T038: "Create tests/performance/test_chunk_delivery.py"
```

---

## Parallel Example: User Story 1 (Core Playback)

```bash
# Launch all REST endpoint implementations together:
Task T061: "Implement GET /api/status in server/main.py"
Task T062: "Implement GET /api/metrics in server/main.py"
Task T063: "Implement GET / root endpoint in server/main.py"
Task T064: "Mount static file serving for /client/*"
```

---

## Implementation Strategy

### MVP First (US4 + US1 Only)

1. Complete Phase 1: Setup (T001-T014)
2. Complete Phase 2: Foundational (T015-T034) - CRITICAL foundation
3. Complete Phase 3: User Story 4 - Performance (T035-T049)
4. Complete Phase 4: User Story 1 - Core Playback (T050-T066)
5. **STOP and VALIDATE**: Run quickstart.md Scenarios 1-4, verify time-to-first-audio <2s, 30-min streaming
6. Deploy/demo if ready - This is minimal viable product!

### Incremental Delivery (Add Quality and Features)

1. MVP validated (Setup + Foundational + US4 + US1) → Foundation ready
2. Add Phase 5: User Story 2 - Musical Quality (T067-T077) → Test independently → Deploy/Demo
3. Add Phase 6: User Story 3 - Controls (T078-T088) → Test independently → Deploy/Demo
4. Add Phase 7: User Story 5 - Error Recovery (T089-T098) → Test independently → Deploy/Demo
5. Complete Phase 8: Polish (T099-T115) → Final validation → Production release

### Parallel Team Strategy

With multiple developers:

1. **Week 1**: Team completes Setup + Foundational together (T001-T034)
2. **Week 2**: Once Foundational is done:
   - Developer A: User Story 4 - Performance (T035-T049)
   - Developer B: Begin User Story 1 client-side (T050-T055)
   - Developer C: Begin User Story 1 server-side (T056-T060)
3. **Week 3**: MVP integration:
   - Developer A: Complete US1 REST API (T061-T064)
   - Developer B: US1 integration tests (T065-T066)
   - Developer C: Begin User Story 2 (T067-T073)
4. **Week 4**: Quality and enhancements:
   - Developer A: User Story 3 - Controls (T078-T088)
   - Developer B: User Story 5 - Error Recovery (T089-T098)
   - Developer C: Complete User Story 2 (T074-T077)
5. **Week 5**: Polish and release (T099-T115)

---

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- [Story] label maps task to specific user story (US1-US5) for traceability
- Each user story should be independently completable and testable
- Tests should fail before implementing (TDD recommended)
- Commit after each task or logical group for incremental progress
- Stop at any checkpoint to validate story independently
- Prioritize US4 + US1 for MVP (performance + playback), then add US2 (quality), then US3/US5 (enhancements)
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- All file paths are relative to repository root (`/Users/andrew/Develop/auralis/`)
- GPU acceleration is optional (CPU fallback must work) per constitutional check

---

**Task Generation Complete**: 115 tasks organized into 8 phases, covering 5 user stories (3 P1, 2 P2)
**MVP Scope**: Phases 1-4 (Setup + Foundational + US4 + US1) = 66 tasks
**Full Feature Scope**: All 8 phases = 115 tasks
**Estimated Parallel Opportunities**: 31 tasks marked [P] for parallel execution

**Next Steps**:
1. Review task breakdown with team
2. Begin implementation with Phase 1: Setup (T001-T014)
3. Track progress by marking tasks [X] as completed in this file
