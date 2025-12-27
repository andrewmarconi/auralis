# GPU Optimization Quick Reference

**Last Updated**: 2025-12-26
**Full Research**: [research.md](research.md)

## Top 6 Optimization Techniques (Ranked by Impact)

### 1. Batch Voice Rendering (40-60% latency reduction)

**Problem**: Sequential per-note synthesis underutilizes GPU parallelism.

**Solution**: Process multiple voices/notes simultaneously using vectorized operations.

```python
# Before (sequential)
for pitch in chord_pitches:
    signal = self.synthesize_voice(pitch)
    audio += signal

# After (batched)
pitches_batch = torch.tensor(chord_pitches, device=self.device)
signals_batch = self.synthesize_batch(pitches_batch)  # All voices in parallel
audio += signals_batch.sum(dim=0)
```

**Effort**: Medium (2-3 days) | **Risk**: Low

---

### 2. Memory Pre-Allocation (Eliminates GC pauses)

**Problem**: Frequent tensor allocations cause memory fragmentation and garbage collection pauses (audio glitches).

**Solution**: Pre-allocate buffers during initialization, reuse for all synthesis.

```python
class SynthesisEngine:
    def __init__(self, ...):
        # Pre-allocate max-size buffers
        self.max_phrase_samples = int(30.0 * self.sample_rate)
        self.audio_buffer_pool = torch.zeros(
            self.max_phrase_samples,
            device=self.device,
            dtype=torch.float32
        )

    def render_phrase(self, ...):
        # Reuse pre-allocated buffer (slice to needed size)
        audio = self.audio_buffer_pool[:num_samples]
        audio.zero_()  # Clear previous data
```

**Effort**: Low (1-2 days) | **Risk**: Very Low

---

### 3. Kernel Fusion via torch.compile (20-30% speedup)

**Problem**: Multiple sequential operations launch separate GPU kernels (overhead).

**Solution**: Use PyTorch 2.x `torch.compile()` to fuse operations automatically.

```python
class AmbientPadVoice(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Enable JIT compilation
        self.forward = torch.compile(
            self.forward,
            mode="reduce-overhead"  # Minimize kernel launch overhead
        )

    def forward(self, pitch, duration_samples, velocity):
        # Operations will be fused into optimized kernels
        ...
```

**Warmup Required**: First call triggers JIT compilation (50-500ms). Run dummy synthesis during initialization.

**Effort**: Low (1-2 days) | **Risk**: Medium (test warmup overhead)

---

### 4. Asynchronous GPU Streams (15-25% total delivery time reduction)

**Problem**: GPU synthesis and CPU-GPU transfers happen sequentially (idle time).

**Solution**: Overlap GPU compute with memory transfers using CUDA streams.

```python
class SynthesisEngine:
    def __init__(self, ...):
        if self.device.type == "cuda":
            self.synthesis_stream = torch.cuda.Stream()
            self.transfer_stream = torch.cuda.Stream()

    async def render_phrase_async(self, ...):
        if self.device.type == "cuda":
            # Synthesis on dedicated stream
            with torch.cuda.stream(self.synthesis_stream):
                audio_tensor = self._synthesize(...)

            # Overlap transfer with next synthesis
            with torch.cuda.stream(self.transfer_stream):
                audio_np = audio_tensor.cpu().numpy()

            torch.cuda.synchronize()
        return audio_np
```

**Platform**: CUDA only (Metal uses automatic pipelining)

**Effort**: Medium (3-4 days) | **Risk**: Medium (async complexity)

---

### 5. Profiling-Driven Optimization (torch.profiler)

**Problem**: Optimizing without measurement leads to wasted effort.

**Solution**: Use `torch.profiler` to identify actual bottlenecks.

```python
from torch.profiler import profile, record_function, ProfilerActivity

def benchmark_synthesis(self):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        with record_function("chord_rendering"):
            self._render_chord_pads(...)

    # Export for Chrome tracing
    prof.export_chrome_trace("synthesis_trace.json")

    # Print summary
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))
```

**View Results**: Open `synthesis_trace.json` in Chrome at `chrome://tracing`

**Effort**: Low (1-2 days) | **Risk**: None (observability only)

---

### 6. Mixed Precision Considerations (DO NOT USE for audio)

**Problem**: FP16/BF16 can speed up neural networks significantly.

**Auralis Verdict**: **Use FP32 for all audio synthesis operations.**

**Rationale**:
- Audio requires 5-6 digits precision; FP16 provides only ~3 (audible quantization noise)
- Accumulating multiple voices in FP16 compounds rounding errors
- Performance gain (~10-15%) not worth quality degradation

**Exception**: Non-audio metadata/control parameters can use FP16 safely.

**Effort**: N/A | **Risk**: Critical quality degradation if misapplied

---

## Platform-Specific Notes

### Metal (Apple Silicon M1/M2/M4)

**Characteristics**:
- Unified CPU/GPU memory (no explicit transfers)
- Newer backend (less mature than CUDA)
- Automatic command queue pipelining

**Optimizations**:
1. Avoid unsupported ops (e.g., `torch.logspace` - use `torch.exp(torch.linspace())`)
2. Use Metal System Trace for profiling (Xcode Instruments)
3. No manual stream management needed (automatic)

**Expected Performance**: 10-20% speedup with `torch.compile` (vs 20-40% on CUDA)

---

### CUDA (NVIDIA GPUs)

**Characteristics**:
- Dedicated GPU memory (PCIe transfer overhead)
- Mature optimization ecosystem (15+ years)
- Explicit stream management required

**Optimizations**:
1. Minimize CPU-GPU transfers (batch data movement)
2. Use CUDA streams for async operations
3. Enable TF32 for matmul: `torch.backends.cuda.matmul.allow_tf32 = True`
4. Profile with `nvidia-smi` and NSight Systems

**Expected Performance**: 20-40% speedup with `torch.compile`

---

## Quick Profiling Commands

### Real-Time GPU Monitoring

```bash
# CUDA
watch -n 0.5 nvidia-smi

# Metal (enable HUD in code)
export METAL_DEVICE_WRAPPER_TYPE=1
python server/main.py
```

### Detailed Profiling

```bash
# CUDA - NSight Systems
nsys profile --trace=cuda,nvtx --output=profile.qdrep python server/main.py

# Metal - Instruments
xcrun xctrace record --template 'Metal System Trace' --launch python server/main.py
```

### Memory Leak Detection

```python
import torch
import tracemalloc

tracemalloc.start()

# Run synthesis loop
for i in range(1000):
    _ = engine.render_phrase(...)

    if i % 100 == 0:
        current, peak = tracemalloc.get_traced_memory()
        gpu_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        print(f"[{i}] Python: {current/1024**2:.1f} MB | GPU: {gpu_mb:.1f} MB")
```

---

## Critical Anti-Patterns to Avoid

### 1. Frequent CPU-GPU Transfers

```python
# BAD: Transfer every note
for note in melody:
    pitch_tensor = torch.tensor(note.pitch, device="cuda")  # Transfer
    signal = synthesize(pitch_tensor)
    signal_cpu = signal.cpu()  # Transfer back

# GOOD: Batch transfers
pitches = torch.tensor([n.pitch for n in melody], device="cuda")
signals = synthesize_batch(pitches)
signals_cpu = signals.cpu()
```

### 2. Gradient Tracking in Inference

```python
# BAD: Gradients accumulated unnecessarily
output = model(input_tensor)  # Builds computation graph

# GOOD: Disable gradients for inference
with torch.no_grad():
    output = model(input_tensor)
```

### 3. Tensor Reference Leaks

```python
# BAD: Unbounded accumulation
self.debug_signals.append(signal)  # Grows forever

# GOOD: Bounded buffer or clear after use
if len(self.debug_signals) > 100:
    self.debug_signals.pop(0)
```

### 4. Blocking Operations in Audio Pipeline

```python
# BAD: Synchronous I/O in render loop
def render_phrase(self, ...):
    audio = self._synthesize(...)
    with open("debug.wav", "wb") as f:  # BLOCKS AUDIO THREAD
        f.write(audio)

# GOOD: Async offload to background thread
async def render_phrase(self, ...):
    audio = self._synthesize(...)
    asyncio.create_task(self._save_debug_async(audio))
```

---

## Implementation Roadmap

### Phase 1: High-Impact, Low-Risk (Week 1)

1. Memory pre-allocation (1-2 days)
2. `torch.no_grad()` wrapper (1 day)
3. Profiling baseline benchmarks (1 day)

**Expected Gain**: 30-40% latency reduction, eliminate GC glitches

---

### Phase 2: Medium-Impact (Week 2-3)

4. Batch chord rendering (2-3 days)
5. `torch.compile` optimization (2-3 days)
6. Integration testing for 10 concurrent users (2 days)

**Expected Gain**: 50-70% total latency reduction vs baseline

---

### Phase 3: Advanced (Future)

7. CUDA streams for async operations (3-4 days)
8. CUDA graphs for repetitive patterns (4-5 days)
9. Custom CUDA kernels (1-2 weeks)

**Expected Gain**: Additional 20-30% for advanced optimizations

---

## Success Metrics

| Metric | Target | Current Estimate |
|--------|--------|------------------|
| Synthesis Latency | <50ms | ~50ms GPU, 150ms CPU |
| Latency Variance (p95) | <80ms | Unknown (needs profiling) |
| Memory Leak (8hr session) | <10 MB growth | Unknown (needs testing) |
| GPU Utilization | >60% during synthesis | Unknown (needs profiling) |
| Concurrent Users | 10 streams, <5% degradation | Untested |
| Resource Reduction | 30% vs baseline | Target for Phase 1-2 |

---

## Next Steps

1. **Baseline Profiling** (Today)
   - Run `torch.profiler` on current implementation
   - Document actual latency, memory usage, GPU utilization
   - Establish regression test thresholds

2. **Implement Phase 1** (This Week)
   - Memory pre-allocation
   - Add `torch.no_grad()` context
   - Verify glitch elimination

3. **Batch Processing** (Next Week)
   - Start with chord rendering (highest impact)
   - Test across different voicings and phrase complexities

4. **Continuous Monitoring** (Ongoing)
   - Add performance benchmarks to CI
   - Alert on >10% regression

---

## Additional Resources

- **Full Research Document**: [research.md](research.md) (comprehensive analysis, code examples, platform comparisons)
- **PyTorch Performance Tuning**: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- **torch.profiler Tutorial**: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- **CUDA Best Practices**: https://pytorch.org/docs/stable/notes/cuda.html
- **MPS Backend Notes**: https://pytorch.org/docs/stable/notes/mps.html
