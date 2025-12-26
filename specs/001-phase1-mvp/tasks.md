---

description: "Task list for Phase 1 MVP - Real-time Ambient Music Streaming implementation"
---

# Tasks: Phase 1 MVP - Real-time Ambient Music Streaming

**Input**: Design documents from `/specs/001-phase1-mvp/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/
**Tests**: Integration tests for real-time audio performance per feature specification

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: Repository root with server/, composition/, client/, tests/
- **Paths**: Use plan.md structure with absolute paths from repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create modular project structure per implementation plan
- [X] T002 Initialize project with uv and core dependencies via `uv add fastapi torch torchsynth numpy`
- [X] T003 [P] Create pyproject.toml with Python 3.12+ specification and metadata
- [X] T004 [P] Create server/ directory with module structure
- [X] T005 [P] Create composition/ directory with module structure  
- [X] T006 [P] Create client/ directory with static assets structure
- [X] T007 [P] Create tests/ directory with integration/ and performance/ subdirectories

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T008 Implement real-time audio processing pipeline with ring buffer in server/ring_buffer.py
- [X] T009 [P] Implement WebSocket streaming server with audio chunk delivery in server/streaming_server.py
- [ ] T010 [P] Setup GPU acceleration detection and configuration in server/synthesis_engine.py
- [X] T011 [P] Create base audio synthesis modules with torchsynth integration in server/synthesis_engine.py
- [X] T012 [P] Configure adaptive client buffering for seamless playback in client/audio_client.js
- [X] T013 [P] Setup performance monitoring for audio latency metrics in server/main.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Basic Streaming Experience (Priority: P1) 🎯 MVP

**Goal**: User wants to listen to continuously generated ambient music through their web browser with minimal setup and immediate playback

**Independent Test**: Can be tested by navigating to web client and verifying that ambient music starts playing automatically within 2 seconds, with continuous streaming for at least 5 minutes without interruptions

### Implementation for User Story 1

- [X] T014 [US1] Create ChordProgression model in composition/chord_generator.py with ambient-optimized Markov transitions
- [X] T015 [US1] Create MelodyPhrase model in composition/melody_generator.py with harmonic constraints
- [X] T016 [US1] Create AudioChunk model in server/streaming_server.py for WebSocket message structure
- [X] T017 [US1] Create RingBuffer entity in server/ring_buffer.py with thread-safe operations
- [X] T018 [US1] Create SynthesisParameters model in server/main.py for key/BPM/intensity controls
- [X] T019 [US1] Implement Markov chord progression generator in composition/chord_generator.py with 8-bar phrases
- [X] T020 [US1] Implement constraint-based melody generator in composition/melody_generator.py with chord fitting
- [X] T021 [US1] Implement ambient synthesis engine in server/synthesis_engine.py with torchsynth voices
- [X] T022 [US1] Implement WebSocket streaming endpoint in server/streaming_server.py with 100ms audio chunks
- [X] T023 [US1] Create web client HTML interface in client/index.html with auto-playback
- [X] T024 [US1] Implement Web Audio API client in client/audio_client.js with base64 decoding
- [X] T025 [US1] Add graceful error handling and user notifications in server/main.py
- [X] T026 [US1] Implement control parameters API endpoint in server/main.py for key/BPM/intensity

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Generation Quality Validation (Priority: P2)

**Goal**: System must generate musically coherent ambient progressions that sound pleasant and maintain harmonic consistency throughout streaming session

**Independent Test**: Can be tested by listening to 30 minutes of generated content and verifying harmonic consistency, appropriate ambient pacing, and musical coherence without dissonant transitions

### Implementation for User Story 2

- [X] T027 [US2] Add harmonic validation to chord progression generator in composition/chord_generator.py
- [X] T028 [US2] Add dissonance detection to melody generator in composition/melody_generator.py
- [X] T029 [US2] Enhance synthesis engine with pad voice for ambient texture in server/synthesis_engine.py
- [X] T030 [US2] Add phrase transition smoothing in server/synthesis_engine.py for seamless flow
- [X] T031 [US2] Implement audio quality monitoring in server/main.py for harmonic consistency
- [X] T032 [US2] Add browser compatibility checks in client/audio_client.js for Chrome/Edge support

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Performance Requirements (Priority: P2)

**Goal**: System must maintain responsive performance under typical usage conditions with acceptable latency and resource usage

**Independent Test**: Can be measured by monitoring system metrics during a 10-minute streaming session, verifying latency stays below 800ms and CPU usage remains reasonable

### Implementation for User Story 3

- [X] T033 [US3] Add GPU acceleration fallback logic in server/synthesis_engine.py with user notifications
- [X] T034 [US3] Implement ring buffer underflow/overflow detection in server/ring_buffer.py
- [X] T035 [US3] Add performance metrics collection in server/main.py for latency monitoring
- [X] T036 [US3] Optimize audio processing for <100ms chunk generation in server/synthesis_engine.py
- [X] T037 [US3] Add connection recovery with exponential backoff in client/audio_client.js
- [X] T038 [US3] Implement health check endpoint in server/main.py for system status

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T039 [P] Create comprehensive documentation in docs/ with API and setup guides
- [X] T040 [P] Add integration tests for WebSocket streaming in tests/integration/test_websocket_streaming.py
- [X] T041 [P] Add performance tests for audio latency in tests/performance/test_audio_latency.py
- [X] T042 [P] Add GPU acceleration tests in tests/integration/test_gpu_acceleration.py
- [X] T043 [P] Add real-time constraints tests in tests/performance/test_real_time_constraints.py
- [X] T044 [P] Update quickstart.md with complete user guide and troubleshooting
- [X] T045 Code cleanup and performance optimization across all modules
- [X] T046 Security hardening for WebSocket connections and input validation
- [X] T047 Error handling refinement with user-friendly messages

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Models before services before endpoints before integration
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all foundational tasks together:
Task: "T008 Implement real-time audio processing pipeline with ring buffer in server/ring_buffer.py"
Task: "T009 [P] Implement WebSocket streaming server with audio chunk delivery in server/streaming_server.py"
Task: "T010 [P] Setup GPU acceleration detection and configuration in server/synthesis_engine.py"

# Launch all User Story 1 models together:
Task: "T014 [US1] Create ChordProgression model in composition/chord_generator.py"
Task: "T015 [US1] Create MelodyPhrase model in composition/melody_generator.py"
Task: "T016 [US1] Create AudioChunk model in server/streaming_server.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify constitution compliance: UV-first, real-time performance, modular architecture, GPU acceleration, WebSocket protocol
- Monitor audio latency throughout development to ensure <800ms end-to-end target
- All file paths follow plan.md modular structure (server/, composition/, client/, tests/)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence