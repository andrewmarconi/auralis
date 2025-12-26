# Tasks: Enhanced Generation & Controls

**Input**: Design documents from `/specs/002-enhanced-generation-controls/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Not requested in feature specification - no test tasks included.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Project structure**: server/, composition/, client/ at repository root
- **Paths**: Use absolute paths for clarity

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Feature-specific project setup and dependencies

- [x] T001 Add new dependencies for enhanced algorithms: numpy for entropy calculations
- [x] T002 [P] Create in-memory data structures for presets and parameter history

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core enhancements that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T003 Extend SynthesisParameters model in server/main.py with melody_complexity, chord_progression_variety, harmonic_density
- [x] T004 [P] Implement parameter validation logic in server/main.py for conflict detection
- [x] T005 [P] Implement automatic parameter adjustment in server/main.py for overload prevention
- [x] T006 Update /api/control endpoint in server/main.py to handle extended parameters

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Advanced Music Parameter Control (Priority: P1) 🎯 MVP

**Goal**: Add new control parameters beyond basic key, BPM, and intensity for fine-tuned music generation

**Independent Test**: Adjust new parameters via web interface and verify they affect generated music output without breaking existing functionality

### Implementation for User Story 1

- [x] T007 [P] [US1] Update client/index.html to add sliders for melody_complexity, chord_progression_variety, harmonic_density
- [x] T008 [US1] Implement parameter validation feedback in client/index.html
- [x] T009 [US1] Update client audio_client.js to send extended parameters to server
- [x] T010 [US1] Add parameter conflict warnings display in client/index.html

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Enhanced Generation Algorithms (Priority: P2)

**Goal**: Implement sophisticated Markov chains and entropy-based metrics for higher-quality music generation

**Independent Test**: Compare generated music quality metrics against baseline, verify increased harmonic variety and complexity

### Implementation for User Story 2

- [x] T011 [P] [US2] Implement second-order Markov chains in composition/chord_generator.py
- [x] T012 [P] [US2] Add entropy calculation for chord progression variety in composition/chord_generator.py
- [x] T013 [US2] Enhance melody generation with adjustable complexity in composition/melody_generator.py
- [x] T014 [US2] Integrate enhanced algorithms with existing synthesis engine in server/synthesis_engine.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Real-time Parameter Feedback (Priority: P3)

**Goal**: Provide immediate visual and audio feedback when adjusting parameters for better user experience

**Independent Test**: Monitor UI responsiveness and audio continuity during parameter changes, verify <50ms visual feedback

### Implementation for User Story 3

- [x] T015 [P] [US3] Add visual parameter effect indicators in client/index.html
- [x] T016 [US3] Implement smooth audio transitions in server/streaming_server.py for parameter changes
- [x] T017 [US3] Update WebSocket broadcasting in server/streaming_server.py for extended parameters
- [x] T018 [US3] Add parameter change history display in client/index.html

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Preset management, documentation, and final integration

- [x] T019 [P] Implement preset management endpoints in server/main.py (/api/presets)
- [x] T020 [P] Add parameter history tracking in server/main.py
- [x] T021 Update quickstart.md with new parameter usage examples
- [x] T022 Update server/main.py status endpoint to include new parameters
- [x] T023 Final integration testing across all user stories

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Independent of other stories
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Independent of other stories

### Within Each User Story

- Client updates before server integration
- Core implementation before feedback/polish
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel
- Client and server tasks within a story can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch client and server updates together:
Task: "Update client/index.html to add sliders for melody_complexity, chord_progression_variety, harmonic_density"
Task: "Update client audio_client.js to send extended parameters to server"

# Launch validation tasks together:
Task: "Implement parameter validation feedback in client/index.html"
Task: "Add parameter conflict warnings display in client/index.html"
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
5. Complete Phase 6 → Final integration

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (client controls)
   - Developer B: User Story 2 (algorithms)
   - Developer C: User Story 3 (feedback) + Phase 6
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence</content>
<parameter name="filePath">/Users/andrew/Develop/auralis/specs/002-enhanced-generation-controls/tasks.md