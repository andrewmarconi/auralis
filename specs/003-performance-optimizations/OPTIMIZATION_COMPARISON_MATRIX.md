# GPU Optimization Techniques - Comparison Matrix

**Date**: 2025-12-26
**Context**: Real-time audio synthesis with PyTorch (Metal/CUDA)
**Target**: <100ms synthesis latency, 10+ concurrent users, 30% resource reduction

---

## Quick Decision Matrix

| Technique | Impact | Effort | Risk | Metal | CUDA | Priority | Phase |
|-----------|--------|--------|------|-------|------|----------|-------|
| **Memory Pre-Allocation** | 🔴 High | 🟢 Low | 🟢 Low | ✅ | ✅ | P0 | 1 |
| **Batch Voice Rendering** | 🔴 High | 🟡 Med | 🟢 Low | ✅ | ✅ | P0 | 1 |
| **torch.no_grad() Context** | 🟡 Med | 🟢 Low | 🟢 Low | ✅ | ✅ | P0 | 1 |
| **torch.compile (JIT)** | 🟡 Med | 🟡 Med | 🟡 Med | ⚠️ | ✅ | P1 | 2 |
| **Kernel Fusion (Manual)** | 🟡 Med | 🟡 Med | 🟢 Low | ✅ | ✅ | P1 | 2 |
| **CUDA Streams (Async)** | 🟡 Med | 🔴 High | 🟡 Med | ❌ | ✅ | P2 | 3 |
| **Profiling Integration** | 🟡 Med | 🟢 Low | 🟢 Low | ✅ | ✅ | P1 | 1 |
| **Mixed Precision (FP16)** | 🟢 Low | 🟢 Low | 🔴 **Critical** | ❌ | ⚠️ | **AVOID** | N/A |
| **CUDA Graphs** | 🔴 High | 🔴 High | 🔴 High | ❌ | ✅ | P3 | Future |
| **Custom CUDA Kernels** | 🔴 High | 🔴 Very High | 🔴 Very High | ❌ | ✅ | P4 | Research |

**Legend**:
- Impact: 🔴 High (>30% improvement) | 🟡 Medium (10-30%) | 🟢 Low (<10%)
- Effort: 🟢 Low (1-2 days) | 🟡 Medium (3-5 days) | 🔴 High (1+ weeks)
- Risk: 🟢 Low (safe, reversible) | 🟡 Medium (requires testing) | 🔴 High (quality/stability concerns)
- Platform: ✅ Supported | ⚠️ Limited | ❌ Not Available
- Priority: P0 (Critical) → P4 (Future Research)

---

## Detailed Comparison

### 1. Memory Pre-Allocation

| Aspect | Details |
|--------|---------|
| **Latency Reduction** | N/A (reduces variance, not average) |
| **Primary Benefit** | Eliminates GC pauses (audio glitch prevention) |
| **Implementation** | Pre-allocate buffers in `__init__`, reuse via slicing |
| **Memory Overhead** | +5-10 MB baseline (negligible) |
| **Metal Performance** | ⭐⭐⭐⭐⭐ (unified memory benefits) |
| **CUDA Performance** | ⭐⭐⭐⭐⭐ (prevents fragmentation) |
| **Code Complexity** | +5% (simple, one-time change) |
| **Regression Risk** | Very Low (deterministic buffers) |
| **Recommendation** | **IMPLEMENT FIRST** - highest reliability impact |

**Code Snippet**:
```python
# In SynthesisEngine.__init__()
self.max_phrase_samples = int(30.0 * self.sample_rate)
self.audio_buffer_pool = torch.zeros(
    self.max_phrase_samples,
    device=self.device,
    dtype=torch.float32
)
```

---

### 2. Batch Voice Rendering

| Aspect | Details |
|--------|---------|
| **Latency Reduction** | 40-60% for chord synthesis |
| **Primary Benefit** | Maximize GPU parallelism (SIMD operations) |
| **Implementation** | Vectorize multi-voice synthesis (chords, overlapping notes) |
| **Memory Overhead** | Temporary: batch_size × max_duration samples |
| **Metal Performance** | ⭐⭐⭐⭐ (excellent for small batches <16) |
| **CUDA Performance** | ⭐⭐⭐⭐⭐ (scales to large batches 32+) |
| **Code Complexity** | +20% (requires refactoring render loops) |
| **Regression Risk** | Low (isolated to chord rendering initially) |
| **Recommendation** | **IMPLEMENT EARLY** - highest performance gain |

**Code Snippet**:
```python
# Batch all chord voices
pitches = torch.tensor([root-12] + [root+i for i in intervals], device=self.device)
freqs = 440.0 * (2.0 ** ((pitches - 69.0) / 12.0))
# Vectorized oscillator generation...
```

---

### 3. torch.no_grad() Context

| Aspect | Details |
|--------|---------|
| **Latency Reduction** | 5-10% (prevents gradient overhead) |
| **Primary Benefit** | Memory leak prevention, reduce computation graph building |
| **Implementation** | Wrap all inference calls with `with torch.no_grad():` |
| **Memory Overhead** | -10-15% (reduction from disabling gradients) |
| **Metal Performance** | ⭐⭐⭐⭐ (consistent benefit) |
| **CUDA Performance** | ⭐⭐⭐⭐ (consistent benefit) |
| **Code Complexity** | +1% (one-line context wrapper) |
| **Regression Risk** | None (inference-only code) |
| **Recommendation** | **IMPLEMENT IMMEDIATELY** - trivial, zero risk |

**Code Snippet**:
```python
def render_phrase(self, ...):
    with torch.no_grad():  # Single line addition
        # All synthesis operations...
```

---

### 4. torch.compile (JIT Compilation)

| Aspect | Details |
|--------|---------|
| **Latency Reduction** | 20-40% (CUDA), 10-20% (Metal) |
| **Primary Benefit** | Automatic kernel fusion, optimized dispatch |
| **Implementation** | `torch.compile(model.forward, mode="reduce-overhead")` |
| **Memory Overhead** | +10-20 MB (compiled kernel cache) |
| **Metal Performance** | ⭐⭐⭐ (newer backend, less optimized) |
| **CUDA Performance** | ⭐⭐⭐⭐⭐ (mature TorchInductor backend) |
| **Code Complexity** | +5% (warmup logic required) |
| **Regression Risk** | Medium (JIT compilation latency on first call) |
| **Recommendation** | Phase 2 - test warmup overhead carefully |

**Critical Caveat**:
```python
# MUST warmup during init (avoid first-call JIT penalty)
def __init__(self, ...):
    self.forward = torch.compile(self.forward, mode="reduce-overhead")
    # Warmup
    dummy_pitch = torch.tensor(60.0, device=self.device)
    _ = self(dummy_pitch, duration_samples=1000, velocity=0.5)
```

---

### 5. Manual Kernel Fusion

| Aspect | Details |
|--------|---------|
| **Latency Reduction** | 10-20% (reduces kernel launch overhead) |
| **Primary Benefit** | Combine sequential ops into single expression |
| **Implementation** | Merge separate tensor operations into compound expressions |
| **Memory Overhead** | Neutral (eliminates intermediate tensors) |
| **Metal Performance** | ⭐⭐⭐⭐ (benefits from reduced kernel launches) |
| **CUDA Performance** | ⭐⭐⭐⭐ (benefits from reduced kernel launches) |
| **Code Complexity** | +10% (less readable expressions) |
| **Regression Risk** | Low (deterministic transformations) |
| **Recommendation** | Phase 2 - apply to hot paths after profiling |

**Example**:
```python
# Before (3 kernels)
osc = torch.sin(2.0 * math.pi * freq * t)
env = torch.exp(-5.0 * t / 0.15)
output = osc * env * velocity

# After (1 kernel)
output = torch.sin(2.0 * math.pi * freq * t) * torch.exp(-5.0 * t / 0.15) * velocity
```

---

### 6. CUDA Streams (Asynchronous Operations)

| Aspect | Details |
|--------|---------|
| **Latency Reduction** | 15-25% (total delivery time, not synthesis alone) |
| **Primary Benefit** | Overlap GPU compute with CPU-GPU transfers |
| **Implementation** | Use `torch.cuda.Stream()` for concurrent operations |
| **Memory Overhead** | Minimal (stream context) |
| **Metal Performance** | N/A (automatic pipelining by MPS backend) |
| **CUDA Performance** | ⭐⭐⭐⭐ (explicit stream control) |
| **Code Complexity** | +30% (async logic, synchronization) |
| **Regression Risk** | Medium (concurrency bugs, race conditions) |
| **Recommendation** | Phase 3 - CUDA-specific optimization |

**Platform Note**: Metal handles pipelining automatically; no manual stream management needed.

---

### 7. Profiling Integration

| Aspect | Details |
|--------|---------|
| **Latency Reduction** | N/A (measurement, not optimization) |
| **Primary Benefit** | Identify actual bottlenecks, prevent regressions |
| **Implementation** | `torch.profiler` + CI benchmarks |
| **Memory Overhead** | Negligible (profiling overhead ~5% during capture) |
| **Metal Performance** | ✅ Metal System Trace (Instruments) |
| **CUDA Performance** | ✅ NSight Systems, nvidia-smi |
| **Code Complexity** | +10% (profiling infrastructure) |
| **Regression Risk** | None (observability only) |
| **Recommendation** | **IMPLEMENT IN PHASE 1** - foundation for all optimizations |

**Essential Tools**:
- `torch.profiler` → Chrome tracing visualization
- CUDA: `nvidia-smi`, NSight Systems
- Metal: Xcode Instruments (Metal System Trace)

---

### 8. Mixed Precision (FP16/BF16)

| Aspect | Details |
|--------|---------|
| **Latency Reduction** | 10-15% (theoretical, for neural networks) |
| **Primary Benefit** | ❌ **NONE for audio synthesis** |
| **Implementation** | **DO NOT IMPLEMENT** |
| **Memory Overhead** | -50% (reduced precision) |
| **Metal Performance** | ❌ Limited FP16 support, no BF16 |
| **CUDA Performance** | ⚠️ Available but **DEGRADES AUDIO QUALITY** |
| **Code Complexity** | +15% (precision management) |
| **Regression Risk** | 🔴 **CRITICAL** - audible quantization noise |
| **Recommendation** | **AVOID ENTIRELY** - unacceptable quality trade-off |

**Why Rejected**:
- Audio requires 5-6 digits precision; FP16 provides ~3 digits
- Voice accumulation in FP16 compounds rounding errors → clicks, pops
- Minimal performance gain (10-15%) not worth quality degradation
- **Verdict**: Use FP32 for ALL audio operations

---

### 9. CUDA Graphs

| Aspect | Details |
|--------|---------|
| **Latency Reduction** | 30-50% (for repetitive, static patterns) |
| **Primary Benefit** | Pre-capture execution graph, eliminate overhead |
| **Implementation** | `torch.cuda.CUDAGraph()` with graph replay |
| **Memory Overhead** | +20-50 MB (captured graph state) |
| **Metal Performance** | ❌ Not available |
| **CUDA Performance** | ⭐⭐⭐⭐⭐ (when applicable) |
| **Code Complexity** | +40% (graph capture, replay logic) |
| **Regression Risk** | High (requires static input shapes, limited flexibility) |
| **Recommendation** | Phase 3 - research if phrase patterns are repetitive |

**Constraint**: Requires identical tensor shapes across invocations. Ambient music's variability may limit applicability.

---

### 10. Custom CUDA Kernels

| Aspect | Details |
|--------|---------|
| **Latency Reduction** | 50-100% (for specific operations) |
| **Primary Benefit** | Hand-optimized assembly for critical paths |
| **Implementation** | C++/CUDA kernel development, PyTorch extension |
| **Memory Overhead** | Neutral (replaces PyTorch ops) |
| **Metal Performance** | ❌ Not applicable (Metal Shading Language alternative) |
| **CUDA Performance** | ⭐⭐⭐⭐⭐ (maximum possible performance) |
| **Code Complexity** | +200% (separate C++ codebase, build complexity) |
| **Regression Risk** | Very High (numerical precision, edge cases) |
| **Recommendation** | Phase 4+ - only if PyTorch ops insufficient |

**Effort Estimate**: 1-2 weeks per kernel, requires CUDA expertise.

---

## Platform-Specific Recommendations

### Apple Silicon (Metal/MPS)

**Implement**:
1. ✅ Memory pre-allocation (highest impact for unified memory)
2. ✅ Batch voice rendering (good performance, <16 voices)
3. ✅ torch.no_grad() context
4. ⚠️ torch.compile (test carefully, newer backend)
5. ✅ Manual kernel fusion
6. ✅ Metal System Trace profiling

**Avoid**:
- ❌ CUDA streams (not applicable, automatic pipelining)
- ❌ Mixed precision (limited support, quality concerns)
- ❌ CUDA graphs/kernels (CUDA-only)

**Expected Baseline**: ~50-80ms synthesis latency after Phase 1-2 optimizations.

---

### NVIDIA GPU (CUDA)

**Implement**:
1. ✅ Memory pre-allocation (prevent fragmentation)
2. ✅ Batch voice rendering (excellent scaling >32 voices)
3. ✅ torch.no_grad() context
4. ✅ torch.compile (mature backend, high gains)
5. ✅ Manual kernel fusion
6. ✅ CUDA streams (async overlap)
7. ✅ NSight profiling

**Advanced (Phase 3+)**:
- ⚠️ CUDA graphs (if phrase patterns repeat)
- ⚠️ Custom kernels (last resort optimization)

**Avoid**:
- ❌ Mixed precision for audio (quality degradation)

**Expected Baseline**: ~30-50ms synthesis latency after Phase 1-2 optimizations.

---

## Implementation Sequence

### Phase 1: Foundation (Week 1) - Priority: P0

| Day | Task | Expected Outcome |
|-----|------|------------------|
| 1-2 | Memory pre-allocation | Eliminate GC glitches, stable latency |
| 2 | torch.no_grad() wrapper | -10% memory usage |
| 3 | Profiling baseline | Documented bottlenecks, regression thresholds |
| 4-5 | Batch chord rendering | -40% chord synthesis time |

**Cumulative Gain**: 30-40% overall latency reduction, zero audio glitches.

---

### Phase 2: Performance (Week 2-3) - Priority: P1

| Day | Task | Expected Outcome |
|-----|------|------------------|
| 1-3 | torch.compile integration | -20% synthesis time (CUDA), -10% (Metal) |
| 4-5 | Manual kernel fusion (hot paths) | -10% additional reduction |
| 6-7 | Concurrent user testing (10 streams) | Validate scalability claims |
| 8 | CI performance benchmarks | Prevent future regressions |

**Cumulative Gain**: 50-70% total latency reduction vs Phase 1 baseline.

---

### Phase 3: Advanced (Future) - Priority: P2-P3

| Weeks | Task | Expected Outcome |
|-------|------|------------------|
| 1 | CUDA streams (async synthesis) | -15% delivery time (CUDA-only) |
| 2 | CUDA graphs (if applicable) | -30% for repetitive patterns |
| 3+ | Custom kernels (research) | -50%+ for specific ops |

**Cumulative Gain**: Additional 20-30% for advanced CUDA optimizations.

---

## Success Criteria by Phase

### Phase 1 Targets

| Metric | Baseline (est.) | Phase 1 Target | Measurement |
|--------|-----------------|----------------|-------------|
| Synthesis Latency (avg) | 150ms (CPU) / 80ms (GPU) | <50ms | torch.profiler |
| Latency Variance (p95) | Unknown | <80ms | Percentile tracking |
| Audio Glitches | Occasional GC pauses | Zero in 30min test | Manual testing |
| Memory Growth (8hr) | Unknown | <10 MB | tracemalloc |
| GPU Utilization | Unknown | >60% during synthesis | nvidia-smi / Metal HUD |

---

### Phase 2 Targets

| Metric | Phase 1 | Phase 2 Target | Measurement |
|--------|---------|----------------|-------------|
| Synthesis Latency (avg) | <50ms | <30ms | torch.profiler |
| Concurrent Users | 1-2 | 10 with <5% degradation | Load testing |
| Resource Usage | Baseline | -30% CPU vs original | System monitoring |
| Compilation Warmup | N/A | <500ms one-time | Startup profiling |

---

### Phase 3 Targets (CUDA-Specific)

| Metric | Phase 2 | Phase 3 Target | Measurement |
|--------|---------|----------------|-------------|
| Async Overlap | Sequential | 15-25% delivery reduction | NSight timeline |
| Graph Acceleration | N/A | 30-50% (if applicable) | CUDA graph benchmarks |

---

## Risk Assessment Matrix

| Optimization | Quality Risk | Stability Risk | Maintenance Risk | Mitigation |
|--------------|--------------|----------------|------------------|------------|
| Memory Pre-Allocation | 🟢 None | 🟢 Low | 🟢 Low | Deterministic buffers |
| Batch Rendering | 🟢 None | 🟡 Medium | 🟡 Medium | Extensive testing |
| torch.no_grad() | 🟢 None | 🟢 None | 🟢 None | Standard practice |
| torch.compile | 🟢 None | 🟡 Medium | 🟡 Medium | Warmup testing |
| CUDA Streams | 🟢 None | 🟡 Medium | 🔴 High | CUDA-only, async complexity |
| Mixed Precision | 🔴 **Critical** | 🟢 Low | 🟡 Medium | **DO NOT USE** |
| CUDA Graphs | 🟢 None | 🔴 High | 🔴 High | Requires static shapes |
| Custom Kernels | 🟡 Medium | 🔴 High | 🔴 Very High | Expert review required |

**Legend**:
- 🟢 Low: Minimal risk, standard practice
- 🟡 Medium: Requires testing, reversible
- 🔴 High: Significant complexity or quality concerns

---

## Final Recommendations

### Immediate Actions (This Week)

1. ✅ **Memory Pre-Allocation** - Implement first (glitch prevention)
2. ✅ **torch.no_grad()** - Trivial, zero risk
3. ✅ **Profiling Baseline** - Foundation for all optimizations

### High-Priority (Next 2 Weeks)

4. ✅ **Batch Chord Rendering** - Highest performance gain
5. ⚠️ **torch.compile** - Test warmup carefully
6. ✅ **Concurrent User Testing** - Validate scalability

### Future Research

7. ⚠️ **CUDA Streams** - CUDA-specific, async complexity
8. ⚠️ **CUDA Graphs** - High effort, static pattern requirement
9. ❌ **Mixed Precision** - **REJECTED** for audio quality

### Never Implement

- ❌ **FP16/BF16 for Audio** - Unacceptable quality degradation
- ❌ **Blocking I/O in Audio Pipeline** - Violates real-time constraint

---

## Monitoring and Validation

### Continuous Metrics (CI Pipeline)

```python
# Automated performance regression tests
@pytest.mark.benchmark
def test_synthesis_latency_regression():
    engine = SynthesisEngine()
    latency_ms = benchmark_render_phrase(engine, iterations=100)

    assert latency_ms < 50, f"Latency {latency_ms:.2f}ms exceeds 50ms target"

@pytest.mark.benchmark
def test_memory_stability():
    monitor = MemoryMonitor()
    for i in range(1000):
        _ = engine.render_phrase(...)

    assert monitor.growth_mb < 10, "Memory leak detected"
```

### Manual Quality Validation

- **A/B Testing**: Compare optimized vs. baseline audio (spectral analysis)
- **Long-Duration Testing**: 8-hour continuous playback (glitch detection)
- **Concurrent Load Testing**: 10 simultaneous WebSocket connections
- **Platform Parity**: Metal vs. CUDA performance within 20%

---

## References

- **Detailed Research**: [research.md](research.md)
- **Quick Reference**: [GPU_OPTIMIZATION_QUICK_REFERENCE.md](GPU_OPTIMIZATION_QUICK_REFERENCE.md)
- **Feature Specification**: [spec.md](spec.md)
- **Implementation Plan**: [plan.md](plan.md)

**Last Updated**: 2025-12-26
