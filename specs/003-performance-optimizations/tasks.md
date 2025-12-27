# Tasks: Performance Optimizations

**Input**: Design documents from `/specs/003-performance-optimizations/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Performance and integration tests are included as per feature requirements for production-grade optimization validation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Server**: `server/` at repository root
- **Client**: `client/` at repository root
- **Tests**: `tests/integration/`, `tests/performance/`, `tests/unit/`
- **Composition**: `composition/` at repository root (minimal changes expected)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency setup for performance optimization implementation

- [X] T001 Install performance monitoring dependencies via `uv add prometheus-client psutil`
- [X] T002 Create performance testing directory structure: tests/performance/, tests/load/
- [X] T003 [P] Create base exception classes in server/exceptions.py (BufferError, SynthesisError, etc.)
- [X] T004 [P] Create configuration module in server/config.py for performance settings
- [X] T005 Setup logging configuration for performance monitoring in server/logging_config.py

**Checkpoint**: Environment ready with all dependencies and base infrastructure

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core performance infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Create abstract buffer interface IRingBuffer in server/interfaces/buffer.py
- [X] T007 [P] Create abstract jitter tracker interface IJitterTracker in server/interfaces/jitter.py
- [X] T008 [P] Create abstract synthesis interface ISynthesisEngine in server/interfaces/synthesis.py
- [X] T009 [P] Create abstract metrics interface IMetricsCollector in server/interfaces/metrics.py
- [X] T010 Create dependency injection container in server/di_container.py
- [X] T011 Add Prometheus metrics endpoint to server/main.py at /metrics
- [X] T012 Update existing RingBuffer in server/ring_buffer.py to implement IRingBuffer interface

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Smooth Audio Streaming (Priority: P1) 🎯 MVP

**Goal**: Deliver uninterrupted ambient music streaming with <100ms latency, 99% chunk delivery within 50ms, and adaptive buffer management that prevents underruns while minimizing latency.

**Independent Test**: Stream audio for 30+ minutes while monitoring buffer health, jitter metrics, and delivery timing - verify zero audible glitches and 99%+ on-time delivery.

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T013 [P] [US1] Create integration test for smooth audio streaming in tests/integration/test_smooth_streaming.py
- [X] T014 [P] [US1] Create performance test for chunk delivery timing in tests/performance/test_chunk_delivery.py
- [X] T015 [P] [US1] Create unit test for JitterTracker EMA calculations in tests/unit/test_jitter_tracker.py
- [X] T016 [P] [US1] Create unit test for AdaptiveRingBuffer tier transitions in tests/unit/test_adaptive_buffer.py
- [X] T017 [P] [US1] Create integration test for buffer underrun prevention in tests/integration/test_buffer_underruns.py

### Implementation for User Story 1

#### Buffer Management Components

- [X] T018 [P] [US1] Implement ChunkTimestamp dataclass in server/buffer_management.py
- [X] T019 [P] [US1] Implement JitterTracker with EMA in server/buffer_management.py
- [X] T020 [P] [US1] Implement TokenBucket rate limiter in server/buffer_management.py
- [X] T021 [US1] Implement AdaptiveRingBuffer with 4-tier system in server/ring_buffer.py (depends on T019, T020)
- [X] T022 [US1] Add tier adjustment logic to AdaptiveRingBuffer based on jitter and underrun rate
- [X] T023 [US1] Implement get_buffer_health() method returning tier, depth, jitter metrics

#### Synthesis Optimization

- [X] T024 [P] [US1] Create DeviceSelector for GPU device detection in server/device_selector.py
- [X] T025 [US1] Update SynthesisEngine in server/synthesis_engine.py to use DeviceSelector
- [X] T026 [US1] Add memory pre-allocation for audio buffers in server/synthesis_engine.py
- [X] T027 [US1] Implement torch.no_grad() context in synthesis loops to prevent gradient leaks
- [X] T028 [US1] Add GPU cache clearing every 100 renders in server/synthesis_engine.py

#### Client-Side Adaptive Buffering

- [X] T029 [P] [US1] Implement adaptive buffer sizing in client/audio_client_worklet.js
- [X] T030 [US1] Add jitter tracking to AudioWorklet processor in client/audio_worklet_processor.js
- [X] T031 [US1] Implement buffer health status display in client UI (client/index.html)

#### Integration & Monitoring

- [ ] T032 [US1] Integrate AdaptiveRingBuffer into streaming_server.py WebSocket handler
- [ ] T033 [US1] Add buffer tier and jitter metrics to WebSocket status messages
- [ ] T034 [US1] Add Prometheus histogram metrics for chunk delivery jitter
- [ ] T035 [US1] Add Prometheus gauge metrics for buffer depth per client
- [ ] T036 [US1] Create Grafana dashboard JSON for US1 metrics in docs/grafana/smooth-streaming.json

**Checkpoint**: User Story 1 complete - smooth audio streaming with adaptive buffering functional and independently testable

---

## Phase 4: User Story 2 - Concurrent User Support (Priority: P2)

**Goal**: Support 10+ concurrent listeners simultaneously without audio degradation, resource contention, or performance impact on any individual stream.

**Independent Test**: Simulate 10+ simultaneous connections and verify each receives consistent audio quality with stable latency and zero cross-client interference.

### Tests for User Story 2

- [ ] T037 [P] [US2] Create load test for 10 concurrent clients in tests/load/test_concurrent_10_clients.py
- [ ] T038 [P] [US2] Create load test for 20 concurrent clients (stress) in tests/load/test_concurrent_20_clients.py
- [ ] T039 [P] [US2] Create integration test for client isolation in tests/integration/test_client_isolation.py
- [ ] T040 [P] [US2] Create performance test for broadcast encoding efficiency in tests/performance/test_broadcast_encoding.py

### Implementation for User Story 2

#### Broadcast WebSocket Architecture

- [ ] T041 [P] [US2] Implement ClientCursor dataclass in server/websocket_manager.py
- [ ] T042 [P] [US2] Implement WebSocketClientState dataclass in server/websocket_manager.py
- [ ] T043 [US2] Implement BroadcastRingBuffer with per-client cursors in server/ring_buffer.py
- [ ] T044 [US2] Add lock-free concurrent read support to BroadcastRingBuffer
- [ ] T045 [US2] Implement client connection tracking in server/websocket_manager.py

#### Audio Encoding Optimization

- [ ] T046 [P] [US2] Create AudioChunkPool for zero-allocation encoding in server/chunk_encoder.py
- [ ] T047 [US2] Implement base64 encoding pool reuse to reduce allocations
- [ ] T048 [US2] Add broadcast encoding (1× encode, N× send) to streaming_server.py
- [ ] T049 [US2] Add chunk encoding duration metrics to Prometheus

#### Concurrency Management

- [ ] T050 [US2] Update StreamingServer to use BroadcastRingBuffer in server/streaming_server.py
- [ ] T051 [US2] Implement per-client rate limiting with TokenBucket
- [ ] T052 [US2] Add graceful client disconnect handling with cursor cleanup
- [ ] T053 [US2] Implement connection drain period on shutdown (5 seconds)

#### Async Synthesis Engine

- [ ] T054 [US2] Create AsyncSynthesisEngine wrapper in server/async_synthesis.py
- [ ] T055 [US2] Implement thread pool offloading for CPU-bound synthesis operations
- [ ] T056 [US2] Add synthesis request queue with asyncio.Queue
- [ ] T057 [US2] Integrate AsyncSynthesisEngine into streaming_server.py

#### Monitoring & Metrics

- [ ] T058 [US2] Add Prometheus gauge for active WebSocket connections
- [ ] T059 [US2] Add Prometheus counter for WebSocket send errors by type
- [ ] T060 [US2] Add Prometheus histogram for per-client latency distribution
- [ ] T061 [US2] Create Grafana dashboard for concurrent user metrics in docs/grafana/concurrent-users.json
- [ ] T062 [US2] Configure alert rules for connection limits in docs/prometheus/alerts.yml

**Checkpoint**: User Story 2 complete - 10+ concurrent users supported with broadcast architecture and independent testability

---

## Phase 5: User Story 3 - Efficient Resource Utilization (Priority: P3)

**Goal**: Reduce CPU/GPU/memory usage by 30% compared to Phase 1 baseline while maintaining audio quality and achieving stable memory over 8+ hours.

**Independent Test**: Measure resource usage during continuous streaming and compare to baseline benchmarks - verify 30% reduction and <10MB memory growth over 8 hours.

### Tests for User Story 3

- [X] T063 [P] [US3] Create GPU batch synthesis benchmark in tests/performance/test_batch_synthesis.py
- [X] T064 [P] [US3] Create torch.compile optimization benchmark in tests/performance/test_torch_compile.py
- [X] T065 [P] [US3] Create memory stability test (8+ hours) in tests/integration/test_memory_stability.py
- [X] T066 [P] [US3] Create GC configuration test in tests/unit/test_gc_config.py
- [X] T067 [P] [US3] Create resource usage comparison test in tests/performance/benchmark_suite.py

### Implementation for User Story 3

#### GPU Optimization

- [X] T068 [P] [US3] Implement GPU batch processing for chord rendering in server/synthesis_engine.py
- [ ] T069 [US3] Add torch.compile decorator to synthesis methods (requires PyTorch 2.0+)
- [ ] T070 [US3] Implement kernel fusion for voice generation operations
- [X] T071 [US3] Add device-specific tuning for Metal (MPS) in server/device_selector.py
- [X] T072 [US3] Add device-specific tuning for CUDA in server/device_selector.py
- [X] T073 [US3] Implement batch size auto-tuning based on GPU memory

#### Memory Leak Prevention

- [X] T074 [P] [US3] Implement MemorySnapshot dataclass in server/memory_monitor.py
- [X] T075 [P] [US3] Implement MemoryGrowthTracker with linear regression in server/memory_monitor.py
- [X] T076 [US3] Implement MemoryMonitor with periodic sampling in server/memory_monitor.py
- [X] T077 [US3] Add tracemalloc profiling for Python memory tracking
- [X] T078 [US3] Implement memory leak detection with 20MB/hour threshold
- [X] T079 [US3] Add GPU memory monitoring for Metal/CUDA allocation

#### GC Tuning

- [X] T080 [P] [US3] Create GCConfig dataclass in server/gc_config.py
- [X] T081 [US3] Implement RealTimeGCConfig with tuned thresholds (50000, 500, 1000)
- [X] T082 [US3] Apply GC configuration on server startup in server/main.py
- [X] T083 [US3] Add GC statistics collection to Prometheus metrics

#### Performance Monitoring

- [X] T084 [P] [US3] Implement PrometheusMetrics class in server/metrics.py
- [X] T085 [US3] Add synthesis_latency_seconds histogram metric
- [X] T086 [US3] Add memory_usage_mb gauge metric with RSS tracking
- [X] T087 [US3] Add gpu_memory_allocated_mb gauge metric
- [X] T088 [US3] Add gc_collections_total counter metric by generation
- [X] T089 [US3] Add phrase_generation_rate_hz gauge metric
- [X] T090 [US3] Implement async metrics collection every 5 seconds

#### Resource Optimization

- [ ] T091 [US3] Implement object pooling for frequently allocated objects
- [X] T092 [US3] Add memory pre-allocation for synthesis buffers
- [ ] T093 [US3] Optimize composition algorithms if profiling shows bottlenecks (composition/)
- [ ] T094 [US3] Add CPU affinity settings for synthesis thread pool

#### Monitoring Infrastructure

- [X] T095 [US3] Create Prometheus alert rules in docs/prometheus/alerts.yml
- [X] T096 [US3] Create Grafana dashboard for resource utilization in docs/grafana/resource-usage.json
- [X] T097 [US3] Document baseline vs. optimized metrics in docs/performance/baseline-comparison.md
- [X] T098 [US3] Create resource usage report script in scripts/performance/resource_report.py

**Checkpoint**: User Story 3 complete - 30% resource reduction achieved with stable memory and comprehensive monitoring

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and final validation

- [ ] T099 [P] Update AGENTS.md with performance patterns and new dependencies
- [ ] T100 [P] Update README.md with performance optimization details
- [ ] T101 [P] Create performance tuning guide in docs/performance/tuning-guide.md
- [ ] T102 [P] Create troubleshooting guide for common performance issues in docs/performance/troubleshooting.md
- [ ] T103 Run full test suite: pytest tests/ -v --cov=server --cov=client
- [ ] T104 Run quickstart.md validation scenarios for all 3 user stories
- [ ] T105 Validate all Prometheus metrics are correctly exposed at /metrics
- [ ] T106 Validate Grafana dashboards load and display metrics correctly
- [ ] T107 Run 24-hour stability test with all optimizations enabled
- [ ] T108 Create performance optimization summary report in docs/performance/optimization-summary.md
- [ ] T109 Code cleanup: Remove debug logging, unused imports, dead code
- [ ] T110 Security review: Validate rate limiting, input validation, resource limits
- [ ] T111 [P] Update type hints across all modified files
- [ ] T112 [P] Add docstrings to all public methods
- [ ] T113 Create migration guide from Phase 1/2 to Phase 3 in docs/migration/phase3-migration.md

**Checkpoint**: Performance optimizations complete, tested, documented, and ready for production

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - ✅ No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - ⚠️ May integrate with US1 (BroadcastRingBuffer extends AdaptiveRingBuffer) but independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - ✅ Independently testable (monitors US1/US2 but doesn't block them)

**Recommended Order**:
1. Complete US1 first (MVP - smooth streaming is core value)
2. US2 and US3 can proceed in parallel after US1 (if team capacity allows)
3. Otherwise: US1 → US2 → US3 (priority order)

### Within Each User Story

**User Story 1** (Smooth Streaming):
1. Tests first (T013-T017) - all parallelizable
2. Buffer models (T018-T020) - parallelizable
3. AdaptiveRingBuffer (T021-T023) - sequential, depends on T018-T020
4. Synthesis optimization (T024-T028) - T024 first, then T025-T028 parallelizable
5. Client buffering (T029-T031) - T029 first, then T030-T031 sequential
6. Integration (T032-T036) - sequential

**User Story 2** (Concurrent Users):
1. Tests first (T037-T040) - all parallelizable
2. Data models (T041-T042) - parallelizable
3. BroadcastRingBuffer (T043-T045) - depends on T041-T042
4. Encoding pool (T046-T049) - T046-T047 parallelizable, then T048-T049
5. Concurrency (T050-T053) - depends on T043-T049
6. Async synthesis (T054-T057) - sequential
7. Monitoring (T058-T062) - all parallelizable

**User Story 3** (Resource Efficiency):
1. Tests first (T063-T067) - all parallelizable
2. GPU optimization (T068-T073) - T068 first, then others parallelizable
3. Memory models (T074-T075) - parallelizable
4. MemoryMonitor (T076-T079) - depends on T074-T075
5. GC tuning (T080-T083) - T080 first, then T081-T083 sequential
6. Metrics (T084-T090) - T084 first, then T085-T090 parallelizable
7. Optimizations (T091-T094) - all parallelizable
8. Infrastructure (T095-T098) - all parallelizable

### Parallel Opportunities

- **Setup (Phase 1)**: T003, T004 can run in parallel
- **Foundational (Phase 2)**: T007, T008, T009 can run in parallel
- **US1 Tests**: T013-T017 (5 tasks) can all run in parallel
- **US1 Models**: T018-T020 (3 tasks) can run in parallel
- **US1 Synthesis**: T025-T028 (4 tasks) can run in parallel after T024
- **US2 Tests**: T037-T040 (4 tasks) can all run in parallel
- **US2 Models**: T041-T042 (2 tasks) can run in parallel
- **US2 Encoding**: T046-T047 (2 tasks) can run in parallel
- **US2 Monitoring**: T058-T062 (5 tasks) can all run in parallel
- **US3 Tests**: T063-T067 (5 tasks) can all run in parallel
- **US3 GPU**: T069-T073 (5 tasks) can run in parallel after T068
- **US3 Memory Models**: T074-T075 (2 tasks) can run in parallel
- **US3 Metrics**: T085-T090 (6 tasks) can run in parallel after T084
- **US3 Optimizations**: T091-T094 (4 tasks) can all run in parallel
- **US3 Infrastructure**: T095-T098 (4 tasks) can all run in parallel
- **Polish**: T099-T102, T111-T112 can run in parallel
- **Different User Stories**: Once Foundational complete, US1/US2/US3 can all proceed in parallel (if team capacity allows)

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task T013: "Create integration test for smooth audio streaming in tests/integration/test_smooth_streaming.py"
Task T014: "Create performance test for chunk delivery timing in tests/performance/test_chunk_delivery.py"
Task T015: "Create unit test for JitterTracker EMA calculations in tests/unit/test_jitter_tracker.py"
Task T016: "Create unit test for AdaptiveRingBuffer tier transitions in tests/unit/test_adaptive_buffer.py"
Task T017: "Create integration test for buffer underrun prevention in tests/integration/test_buffer_underruns.py"

# Launch all buffer models for User Story 1 together:
Task T018: "Implement ChunkTimestamp dataclass in server/buffer_management.py"
Task T019: "Implement JitterTracker with EMA in server/buffer_management.py"
Task T020: "Implement TokenBucket rate limiter in server/buffer_management.py"

# After T024 completes, launch synthesis optimizations together:
Task T025: "Update SynthesisEngine in server/synthesis_engine.py to use DeviceSelector"
Task T026: "Add memory pre-allocation for audio buffers in server/synthesis_engine.py"
Task T027: "Implement torch.no_grad() context in synthesis loops"
Task T028: "Add GPU cache clearing every 100 renders in server/synthesis_engine.py"
```

---

## Parallel Example: User Story 2

```bash
# Launch all tests for User Story 2 together:
Task T037: "Create load test for 10 concurrent clients in tests/load/test_concurrent_10_clients.py"
Task T038: "Create load test for 20 concurrent clients in tests/load/test_concurrent_20_clients.py"
Task T039: "Create integration test for client isolation in tests/integration/test_client_isolation.py"
Task T040: "Create performance test for broadcast encoding in tests/performance/test_broadcast_encoding.py"

# Launch client state models together:
Task T041: "Implement ClientCursor dataclass in server/websocket_manager.py"
Task T042: "Implement WebSocketClientState dataclass in server/websocket_manager.py"

# Launch monitoring metrics together (after implementation):
Task T058: "Add Prometheus gauge for active WebSocket connections"
Task T059: "Add Prometheus counter for WebSocket send errors by type"
Task T060: "Add Prometheus histogram for per-client latency distribution"
Task T061: "Create Grafana dashboard for concurrent users"
Task T062: "Configure alert rules for connection limits"
```

---

## Parallel Example: User Story 3

```bash
# Launch all tests for User Story 3 together:
Task T063: "Create GPU batch synthesis benchmark in tests/performance/test_batch_synthesis.py"
Task T064: "Create torch.compile optimization benchmark in tests/performance/test_torch_compile.py"
Task T065: "Create memory stability test in tests/integration/test_memory_stability.py"
Task T066: "Create GC configuration test in tests/unit/test_gc_config.py"
Task T067: "Create resource usage comparison in tests/performance/benchmark_suite.py"

# After T068, launch GPU optimizations together:
Task T069: "Add torch.compile decorator to synthesis methods"
Task T070: "Implement kernel fusion for voice generation"
Task T071: "Add Metal (MPS) device-specific tuning"
Task T072: "Add CUDA device-specific tuning"
Task T073: "Implement batch size auto-tuning"

# Launch memory models together:
Task T074: "Implement MemorySnapshot dataclass in server/memory_monitor.py"
Task T075: "Implement MemoryGrowthTracker with linear regression in server/memory_monitor.py"

# After T084, launch all metrics together:
Task T085: "Add synthesis_latency_seconds histogram metric"
Task T086: "Add memory_usage_mb gauge metric"
Task T087: "Add gpu_memory_allocated_mb gauge metric"
Task T088: "Add gc_collections_total counter metric"
Task T089: "Add phrase_generation_rate_hz gauge metric"
Task T090: "Implement async metrics collection every 5 seconds"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

**Goal**: Deliver smooth audio streaming with adaptive buffering as minimum viable product

1. Complete Phase 1: Setup (T001-T005)
2. Complete Phase 2: Foundational (T006-T012) - **CRITICAL BLOCKER**
3. Complete Phase 3: User Story 1 (T013-T036)
4. **STOP and VALIDATE**:
   - Run US1 tests (T013-T017)
   - Stream for 30+ minutes
   - Verify 99% chunk delivery within 50ms
   - Verify buffer tier adjustments work
   - Verify zero audible glitches
5. Deploy/demo if ready

**Estimated Effort**: 8-10 days (1 developer)
**Value**: Core user experience - smooth, uninterrupted audio streaming

---

### Incremental Delivery (All User Stories)

**Goal**: Deliver each user story as an independent, testable increment

1. **Foundation** → Complete Setup + Foundational (T001-T012) → 2-3 days
2. **US1 (MVP)** → Smooth streaming (T013-T036) → Test independently → Deploy/Demo → 6-8 days
3. **US2** → Concurrent users (T037-T062) → Test independently → Deploy/Demo → 5-7 days
4. **US3** → Resource efficiency (T063-T098) → Test independently → Deploy/Demo → 7-9 days
5. **Polish** → Final validation (T099-T113) → 2-3 days

**Total Estimated Effort**: 22-30 days (1 developer, sequential)
**Value**: Each story adds production capability without breaking previous stories

---

### Parallel Team Strategy

**Goal**: Maximize throughput with multiple developers

With 3 developers:

1. **Week 1**: Team completes Setup + Foundational together (T001-T012)
2. **Week 2-3**: Once Foundational done, parallel work:
   - **Developer A**: User Story 1 (T013-T036) - Smooth streaming
   - **Developer B**: User Story 2 (T037-T062) - Concurrent users
   - **Developer C**: User Story 3 (T063-T098) - Resource efficiency
3. **Week 4**: Team collaborates on Polish (T099-T113) and integration testing

**Total Estimated Effort**: 3-4 weeks (3 developers, parallel)
**Value**: All stories delivered simultaneously, faster time to production

**Key Coordination Points**:
- BroadcastRingBuffer (US2) may need to coordinate with AdaptiveRingBuffer (US1)
- Metrics collection (US3) can observe US1/US2 but doesn't block them
- Integration testing in Week 4 ensures all stories work together

---

## Notes

- **[P]** tasks = different files, no dependencies, safe to parallelize
- **[Story]** label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Tests should be written FIRST and FAIL before implementing (TDD approach)
- Commit after each task or logical group for clean git history
- Stop at any checkpoint to validate story independently
- **Constitution compliance**: All tasks respect UV-first, real-time performance, modular architecture, GPU-first, WebSocket-only principles
- **Performance targets**: Every task designed to maintain <100ms latency, support 10+ users, achieve 30% resource reduction

---

## Task Count Summary

- **Phase 1 (Setup)**: 5 tasks
- **Phase 2 (Foundational)**: 7 tasks
- **Phase 3 (US1)**: 24 tasks (5 tests + 19 implementation)
- **Phase 4 (US2)**: 26 tasks (4 tests + 22 implementation)
- **Phase 5 (US3)**: 36 tasks (5 tests + 31 implementation)
- **Phase 6 (Polish)**: 15 tasks

**Total**: 113 tasks

**Parallel Opportunities**: 52 tasks marked [P] (46% parallelizable within constraints)

**Test Coverage**: 14 test tasks (12% of total) ensuring production-grade validation
