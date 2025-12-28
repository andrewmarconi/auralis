# Tasks: FluidSynth Sample-Based Instrument Synthesis

**Input**: Design documents from `/specs/004-fluidsynth-integration/`
**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md)

**Tests**: This feature includes performance and integration tests for real-time audio quality validation (SC-002, SC-003, SC-004)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

---

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- Server modules: `server/`
- Composition: `composition/` (unchanged)
- Client: `client/` (unchanged)
- Tests: `tests/integration/`, `tests/performance/`
- SoundFonts: `soundfonts/` (new directory, gitignored)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependencies, and SoundFont acquisition

- [X] T001 Install system FluidSynth library via brew (macOS) or apt (Linux)
- [X] T002 Add pyfluidsynth Python bindings via `uv add pyfluidsynth`
- [X] T003 [P] Create soundfonts/ directory for SoundFont file storage
- [X] T004 [P] Download FluidR3_GM.sf2 SoundFont (142MB) from MuseScore and place in soundfonts/
- [X] T005 [P] Update .gitignore to exclude soundfonts/*.sf2 files
- [X] T006 [P] Add soundfonts/.gitkeep to preserve directory in git
- [X] T007 [P] Update README.md with SoundFont download instructions

**Checkpoint**: ✅ Dependencies installed, SoundFont downloaded, gitignore configured

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core FluidSynth infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T008 Create server/soundfont_manager.py with SoundFontManager class for SF2 loading and validation
- [X] T009 Implement startup validation in soundfont_manager.py (three-layer: filesystem, size check >100MB, FluidSynth load test)
- [X] T010 Implement fail-fast behavior in soundfont_manager.py if SoundFont missing/corrupted (FR-016)
- [X] T011 Create server/fluidsynth_renderer.py with FluidSynthRenderer base class
- [X] T012 Implement FluidSynth initialization in fluidsynth_renderer.py (sample_rate=44100, 4th-order interpolation)
- [X] T013 Configure FluidSynth polyphony settings in fluidsynth_renderer.py (synth.polyphony=20, overflow.released=yes)
- [X] T014 Implement SoundFont loading in fluidsynth_renderer.py via soundfont_manager integration
- [X] T015 [P] Add optional environment variable AURALIS_SOUNDFONT_DIR to .env.example
- [X] T016 Update server/main.py startup sequence to validate SoundFonts via soundfont_manager before initializing synthesis_engine
- [X] T017 Add startup validation error handling in server/main.py (exit with status 1 if validation fails)

**Checkpoint**: ✅ Foundation ready - FluidSynth infrastructure operational, user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Realistic Acoustic Grand Piano Timbres (Priority: P1) 🎯 MVP

**Goal**: Replace oscillator-based piano with realistic sampled Acoustic Grand Piano (GM preset 0) for melodic phrases

**Independent Test**: Stream ambient music with melodic content and verify listeners hear realistic piano tones (not synthetic oscillators). Subjective listening test confirms 80%+ preference for sample-based piano over oscillators (SC-001).

### Implementation for User Story 1

- [X] T018 [P] [US1] Implement piano voice rendering method in server/fluidsynth_renderer.py (GM preset 0, channel 0)
- [X] T019 [P] [US1] Add note event handling (note_on/note_off) for piano in server/fluidsynth_renderer.py
- [X] T020 [P] [US1] Implement render_notes() method for piano in server/fluidsynth_renderer.py (converts MIDI note list to audio)
- [X] T021 [US1] Modify server/synthesis_engine.py to integrate FluidSynthRenderer for melody rendering
- [X] T022 [US1] Update server/synthesis_engine.py render_phrase() to call fluidsynth_renderer.render_notes() for melody
- [X] T023 [US1] Add audio format conversion in server/synthesis_engine.py (FluidSynth int16 → float32 normalization)

### Tests for User Story 1

- [X] T024 [P] [US1] Create tests/integration/test_fluidsynth_rendering.py with piano rendering test (FR-001)
- [X] T025 [P] [US1] Add polyphonic piano chord test (3-5 simultaneous notes) in tests/integration/test_fluidsynth_rendering.py (SC-010)
- [X] T026 [P] [US1] Add velocity range test (0.0-1.0) for piano in tests/integration/test_fluidsynth_rendering.py (FR-005, SC-011)
- [ ] T027 [US1] Run manual listening test per quickstart.md Test Scenario 2.1 (piano timbre validation)

**Checkpoint**: ✅ Piano melody implementation complete - ready for manual listening validation

---

## Phase 4: User Story 2 - Rich Polysynth Pad Textures (Priority: P2)

**Goal**: Add sampled polysynth pad voices (GM preset 90) for chord progressions, providing warm atmospheric depth

**Independent Test**: Stream music with chord progressions and verify pads sound warm, spacious, and enveloping (not thin or electronic). Listeners describe textures as atmospheric.

### Implementation for User Story 2

- [ ] T028 [P] [US2] Implement pad voice rendering method in server/fluidsynth_renderer.py (GM preset 90, channel 1)
- [ ] T029 [P] [US2] Add chord rendering support in server/fluidsynth_renderer.py for polyphonic pads
- [ ] T030 [US2] Modify server/synthesis_engine.py to integrate FluidSynthRenderer for pad chord rendering
- [ ] T031 [US2] Update server/synthesis_engine.py render_phrase() to call fluidsynth_renderer for chord pads
- [ ] T032 [US2] Ensure smooth chord transitions in server/fluidsynth_renderer.py (no clicks/pops, FR-011)

### Tests for User Story 2

- [ ] T033 [P] [US2] Add pad rendering test in tests/integration/test_fluidsynth_rendering.py (FR-002)
- [ ] T034 [P] [US2] Add sustained pad test (5+ seconds) in tests/integration/test_fluidsynth_rendering.py (verify character maintained)
- [ ] T035 [US2] Run manual listening test per quickstart.md Test Scenario 2.2 (pad texture validation)

**Checkpoint**: At this point, both piano (US1) AND pads (US2) should work independently with realistic timbres

---

## Phase 5: User Story 3 - Uninterrupted Real-Time Streaming (Priority: P1)

**Goal**: Ensure FluidSynth integration maintains <100ms latency and smooth audio streaming for 10+ concurrent listeners

**Independent Test**: Monitor audio stream latency and verify no dropouts during sustained listening sessions. Latency <100ms (SC-002), 10× real-time rendering (SC-003), 10+ concurrent streams (SC-004).

### Implementation for User Story 3

- [ ] T036 [P] [US3] Implement weighted sum mixing in server/synthesis_engine.py (40% pads, 50% melody, 30% kicks, 20% swells)
- [ ] T037 [P] [US3] Implement auto-gain scaling in server/synthesis_engine.py (scale weights >1.0 to preserve headroom)
- [ ] T038 [P] [US3] Implement soft knee limiting in server/synthesis_engine.py (threshold=0.8, knee=0.1, FR-015)
- [ ] T039 [US3] Integrate FluidSynth stereo output with PyTorch mono percussion in server/synthesis_engine.py
- [ ] T040 [US3] Convert PyTorch mono kicks to stereo (duplicate channels) in server/synthesis_engine.py
- [ ] T041 [US3] Ensure final mix output format matches WebSocket requirements (44.1kHz, 16-bit stereo PCM, FR-008)

### Tests for User Story 3

- [ ] T042 [P] [US3] Add latency measurement test in tests/performance/test_real_time_constraints.py (verify <100ms total, SC-002)
- [ ] T043 [P] [US3] Add real-time factor benchmark in tests/performance/test_fluidsynth_performance.py (16s phrase in <1.6s, SC-003)
- [ ] T044 [P] [US3] Add concurrent stream load test in tests/integration/test_websocket_streaming.py (10+ streams, SC-004)
- [ ] T045 [P] [US3] Add clipping prevention test in tests/integration/test_hybrid_synthesis.py (verify max sample <0.99, SC-006)
- [ ] T046 [US3] Run performance benchmarks per quickstart.md Test Scenario 3 (latency, real-time factor, concurrent streams)

**Checkpoint**: At this point, piano, pads, and kicks should all mix correctly with <100ms latency and support 10+ concurrent streams

---

## Phase 6: User Story 4 - Sampled Choir Swells & Hybrid Mixing (Priority: P2)

**Goal**: Add sampled choir voices (Choir Aahs GM 52, Voice Oohs GM 53) for swell effects and complete hybrid mixing (FluidSynth + PyTorch)

**Independent Test**: Generate music with percussion, melody, pads, and swell events, verifying all elements present in final audio with balanced mix and choir-like swell textures.

### Implementation for User Story 4

- [ ] T047 [P] [US4] Implement choir voice rendering methods in server/fluidsynth_renderer.py (GM presets 52 and 53)
- [ ] T048 [P] [US4] Add swell event handling in server/fluidsynth_renderer.py for choir voices
- [ ] T049 [US4] Modify server/synthesis_engine.py to integrate FluidSynthRenderer for swell rendering
- [ ] T050 [US4] Update server/synthesis_engine.py render_phrase() to include choir swells in final mix
- [ ] T051 [US4] Implement complete 4-layer mixing in server/synthesis_engine.py (pads + melody + kicks + swells)
- [ ] T052 [US4] Verify mix weight balance in server/synthesis_engine.py (FR-007: 40% pads, 50% melody, 30% kicks, 20% swells)

### Tests for User Story 4

- [ ] T053 [P] [US4] Add choir swell rendering test in tests/integration/test_fluidsynth_rendering.py (FR-003)
- [ ] T054 [P] [US4] Create tests/integration/test_hybrid_synthesis.py with 4-layer mix verification test
- [ ] T055 [P] [US4] Add mixing weights validation test in tests/integration/test_hybrid_synthesis.py (verify relative loudness)
- [ ] T056 [US4] Run manual listening test per quickstart.md Test Scenario 5 (hybrid mix, choir swells)

**Checkpoint**: All user stories complete - full hybrid synthesis operational (piano, pads, kicks, choir swells)

---

## Phase 7: Polyphony & Voice Stealing (Cross-Cutting)

**Goal**: Validate polyphonic playback (15+ notes) and graceful voice stealing behavior (>20 notes)

**Independent Test**: Render phrases with 15-25 simultaneous notes and verify graceful handling without crashes, clicks, or pops.

### Implementation for Polyphony

- [ ] T057 [P] Verify FluidSynth polyphony configuration in server/fluidsynth_renderer.py (synth.polyphony=20 already set in T013)
- [ ] T058 [P] Verify FluidSynth voice stealing settings in server/fluidsynth_renderer.py (overflow.released=yes already set in T013)
- [ ] T059 Document voice stealing behavior in server/fluidsynth_renderer.py comments (FluidSynth built-in, oldest-first algorithm)

### Tests for Polyphony

- [ ] T060 [P] Create tests/integration/test_polyphony_voice_stealing.py with 15-note simultaneous rendering test (SC-007)
- [ ] T061 [P] Add 25-note voice stealing test in tests/integration/test_polyphony_voice_stealing.py (verify graceful degradation, FR-017)
- [ ] T062 [P] Add click detection test in tests/integration/test_polyphony_voice_stealing.py (verify no clicks during voice stealing, SC-008)
- [ ] T063 Run polyphony tests per quickstart.md Test Scenario 4 (15+ notes, voice stealing, click-free)

**Checkpoint**: Polyphony and voice stealing validated (15+ notes clean, >20 notes gracefully steal oldest)

---

## Phase 8: Sample Rate Resampling Validation (Cross-Cutting)

**Goal**: Verify FluidSynth automatic resampling handles non-44.1kHz SoundFonts transparently

**Independent Test**: Load SoundFont with 48kHz samples and verify output is always 44.1kHz (FR-018)

### Implementation for Resampling

- [ ] T064 Verify FluidSynth resampling configuration in server/fluidsynth_renderer.py (4th-order interpolation already set in T012)
- [ ] T065 Document automatic resampling behavior in server/fluidsynth_renderer.py comments (FluidSynth built-in, transparent)

### Tests for Resampling

- [ ] T066 [P] Add sample rate resampling test in tests/integration/test_fluidsynth_rendering.py (verify 48kHz SF2 → 44.1kHz output, FR-018)
- [ ] T067 Run resampling test per quickstart.md Test Scenario 6 (48kHz→44.1kHz automatic resampling)

**Checkpoint**: Sample rate resampling validated (FluidSynth automatically handles all sample rates)

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, cleanup, and final validation

- [ ] T068 [P] Update README.md with FluidSynth integration overview and SoundFont setup instructions
- [ ] T069 [P] Update docs/system_architecture.md to document hybrid synthesis architecture
- [ ] T070 [P] Create docs/FLUIDSYNTH_INTEGRATION.md with technical details (optional)
- [ ] T071 [P] Add inline documentation to server/fluidsynth_renderer.py (docstrings for all public methods)
- [ ] T072 [P] Add inline documentation to server/soundfont_manager.py (docstrings for all public methods)
- [ ] T073 [P] Code cleanup: Remove unused imports, fix linting issues (run ruff check)
- [ ] T074 [P] Code formatting: Apply Black formatter to all modified files
- [ ] T075 [P] Type hint validation: Run mypy on server/fluidsynth_renderer.py, server/soundfont_manager.py
- [ ] T076 Run complete quickstart.md validation checklist (all test scenarios 1-7)
- [ ] T077 Perform manual listening quality validation (subjective assessment per spec.md qualitative outcomes)
- [ ] T078 Verify constitution compliance (UV-First, Real-Time Performance, Modular Architecture, Developer Experience)
- [ ] T079 [P] Update CHANGELOG.md or release notes with FluidSynth integration summary

**Checkpoint**: Feature complete, documented, tested, and validated

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup (Phase 1) completion - BLOCKS all user stories
- **User Stories (Phases 3-6)**: All depend on Foundational (Phase 2) completion
  - US1 Piano (P1): Can start after Foundational - **MVP scope** 🎯
  - US2 Pads (P2): Can start after Foundational - independent of US1
  - US3 Performance (P1): Requires US1 piano + US2 pads for mixing (sequential after US1/US2)
  - US4 Choir Swells (P2): Requires US3 mixing infrastructure (sequential after US3)
- **Polyphony (Phase 7)**: Can start after Foundational - validates all stories
- **Resampling (Phase 8)**: Can start after Foundational - validates all stories
- **Polish (Phase 9)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1 - Piano)**: Can start after Foundational (Phase 2) - No dependencies on other stories ✅ **MVP**
- **User Story 2 (P2 - Pads)**: Can start after Foundational (Phase 2) - No dependencies on other stories ✅ **Can run parallel with US1**
- **User Story 3 (P1 - Performance/Mixing)**: Requires US1 piano + US2 pads complete (needs sources to mix) ⚠️ **Sequential after US1/US2**
- **User Story 4 (P2 - Choir Swells)**: Requires US3 mixing infrastructure complete ⚠️ **Sequential after US3**

### Within Each User Story

- Tests (if applicable) can be written before or during implementation
- Models/infrastructure before services
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

**Phase 1 (Setup)**: Tasks T003-T007 can run in parallel
**Phase 2 (Foundational)**: Tasks T008-T010 (soundfont_manager.py) parallel with T011-T014 (fluidsynth_renderer.py)
**Phase 3 (US1 Piano)**: Tasks T018-T020 parallel (different methods), T024-T026 tests parallel
**Phase 4 (US2 Pads)**: Tasks T028-T029 parallel, T033-T034 tests parallel
**Phase 5 (US3 Performance)**: Tasks T036-T038 parallel, T042-T045 tests parallel
**Phase 6 (US4 Choir)**: Tasks T047-T048 parallel, T053-T055 tests parallel
**Phase 7 (Polyphony)**: Tasks T057-T059 parallel, T060-T062 tests parallel
**Phase 9 (Polish)**: Tasks T068-T072 documentation parallel, T073-T075 cleanup parallel

**Cross-Story Parallelism**:
- US1 (Piano) and US2 (Pads) can be developed in parallel by different developers after Foundational phase
- Polyphony testing (Phase 7) and Resampling testing (Phase 8) can run in parallel with US3/US4 implementation

---

## Parallel Example: User Story 1 (Piano)

```bash
# Launch all rendering method implementations in parallel:
Task T018: "Implement piano voice rendering method in server/fluidsynth_renderer.py"
Task T019: "Add note event handling for piano in server/fluidsynth_renderer.py"
Task T020: "Implement render_notes() method for piano in server/fluidsynth_renderer.py"

# Launch all tests in parallel (after implementation):
Task T024: "Piano rendering test in tests/integration/test_fluidsynth_rendering.py"
Task T025: "Polyphonic piano chord test in tests/integration/test_fluidsynth_rendering.py"
Task T026: "Velocity range test for piano in tests/integration/test_fluidsynth_rendering.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only) 🎯

**Recommended approach for initial delivery:**

1. ✅ Complete Phase 1: Setup (T001-T007) - ~1-2 hours
2. ✅ Complete Phase 2: Foundational (T008-T017) - ~4-6 hours
3. ✅ Complete Phase 3: User Story 1 - Piano (T018-T027) - ~6-8 hours
4. **STOP and VALIDATE**: Test User Story 1 independently via manual listening
5. Deploy/demo realistic piano synthesis MVP

**MVP Deliverable**: Ambient music with realistic Acoustic Grand Piano melody (replacing oscillator-based synthesis) while maintaining <100ms latency and real-time streaming.

### Incremental Delivery

**Full feature rollout:**

1. ✅ MVP: Setup + Foundational + US1 Piano → **Deploy v1 (realistic piano)** 🎯
2. ✅ Add US2 Pads → Test independently → **Deploy v2 (piano + pads)**
3. ✅ Add US3 Performance/Mixing → Test independently → **Deploy v3 (optimized mix)**
4. ✅ Add US4 Choir Swells → Test independently → **Deploy v4 (complete hybrid synthesis)**
5. ✅ Add Polyphony validation → **Deploy v5 (polyphony tested)**
6. ✅ Polish & Documentation → **Deploy v6 (production-ready)**

Each version adds value without breaking previous functionality.

### Parallel Team Strategy

**With 2-3 developers:**

1. **Week 1**: Team completes Setup (Phase 1) + Foundational (Phase 2) together
2. **Week 2**: Once Foundational done:
   - Developer A: User Story 1 (Piano)
   - Developer B: User Story 2 (Pads)
3. **Week 3**:
   - Developer A: User Story 3 (Performance/Mixing) - depends on US1/US2
   - Developer B: User Story 4 (Choir Swells) - can start after US3 begins
4. **Week 4**: Both developers work on Polyphony (Phase 7), Resampling (Phase 8), Polish (Phase 9) in parallel

---

## Task Statistics

- **Total Tasks**: 79
- **Phase 1 (Setup)**: 7 tasks
- **Phase 2 (Foundational)**: 10 tasks
- **Phase 3 (US1 - Piano)**: 10 tasks (6 implementation + 4 tests)
- **Phase 4 (US2 - Pads)**: 8 tasks (5 implementation + 3 tests)
- **Phase 5 (US3 - Performance)**: 11 tasks (6 implementation + 5 tests)
- **Phase 6 (US4 - Choir Swells)**: 10 tasks (6 implementation + 4 tests)
- **Phase 7 (Polyphony)**: 7 tasks (3 implementation + 4 tests)
- **Phase 8 (Resampling)**: 4 tasks (2 implementation + 2 tests)
- **Phase 9 (Polish)**: 12 tasks

**Parallel Opportunities**: 45+ tasks marked [P] can run in parallel (57% parallelizable)

**MVP Scope**: 27 tasks (Setup + Foundational + US1) - approximately 12-16 hours for experienced developer

**Independent Test Criteria**:
- ✅ US1 (Piano): Manual listening confirms realistic piano timbre (not synthetic)
- ✅ US2 (Pads): Manual listening confirms warm, atmospheric pad textures
- ✅ US3 (Performance): Automated tests confirm <100ms latency, 10× real-time, 10+ streams
- ✅ US4 (Choir Swells): Manual listening confirms choir-like swells, balanced 4-layer mix

---

## Format Validation

✅ **All tasks follow strict checklist format**:
- [x] Checkbox prefix (`- [ ]`)
- [x] Sequential Task IDs (T001-T079)
- [x] [P] markers for parallelizable tasks (45 tasks)
- [x] [Story] labels for user story tasks (US1, US2, US3, US4)
- [x] File paths included in descriptions
- [x] Clear action verbs (Create, Implement, Add, Update, Verify, etc.)

---

## Notes

- **[P] tasks**: Different files, no dependencies, can run simultaneously
- **[Story] labels**: Maps task to specific user story for traceability and independent testing
- **Each user story is independently completable**: Can stop after any story and have working feature subset
- **MVP = US1 only**: Realistic piano is the core value proposition (P1 priority)
- **Tests included**: Performance tests (SC-002, SC-003, SC-004) and integration tests (FR-001, FR-002, FR-003)
- **No API changes**: All tasks are internal synthesis modifications (100% backward compatible)
- **Commit frequently**: After each task or logical group
- **Validate at checkpoints**: Test stories independently before proceeding
