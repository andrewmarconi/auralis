# Auralis MVP v2.0 Technical Decisions Summary

**Quick Reference Guide for Implementation**
**Date**: December 28, 2024

---

## Critical Decisions at a Glance

### 1. FluidSynth Python Integration (pyfluidsynth)

**Decision**: Use FluidSynth 2.x with pyfluidsynth 1.3.2+ for sample-based synthesis

**Key Parameters**:
- Polyphony: 32 voices (ambient music is sparse)
- Reverb: Enabled (3s decay, 20% wet, 0.6 roomsize, 0.5 damping)
- Chorus: Disabled (saves CPU, not needed)
- CPU cores: 2
- Sample rate: 44,100 Hz

**Performance Target**: <100ms to render 8-bar phrase (32 seconds audio)

**Why Chosen**:
- Realistic piano/pad timbres essential for ambient music
- CPU-based (no GPU dependency issues)
- Mature, cross-platform (20+ years development)
- Minimal Python overhead via C bindings

**Rejected Alternatives**:
- PyTorch torchsynth: Synthetic timbres lack realism
- Custom SF2 renderer: Reinventing wheel, latency uncertainty

---

### 2. SoundFont Selection

**Decision**: Salamander Grand Piano (200MB) + FluidR3_GM (140MB)

**Rationale**:
- **Salamander**: Best free piano SoundFont, 16 velocity layers, CC-BY 3.0 license
- **FluidR3_GM**: Excellent pads (presets 88-90), public domain
- Total: 340MB (within 500MB memory budget)
- Audio quality critical for ambient music

**Preset Mapping**:
- Channel 0: Piano (Salamander, preset 0 - Acoustic Grand Piano)
- Channel 1: Pad (FluidR3_GM, preset 88 - Warm Pad)

**Licensing**:
- Salamander: CC-BY 3.0 (commercial use allowed with attribution)
- FluidR3_GM: Public domain (no attribution required)

**Rejected Alternatives**:
- Arachno SoundFont: Weaker piano quality
- MuseScore General: Too basic for ambient music
- Timbres of Heaven: 350MB too large

---

### 3. Thread-Safe Ring Buffer

**Decision**: NumPy pre-allocated circular buffer with threading.Lock

**Configuration**:
- Capacity: 20 chunks (2 seconds @ 100ms chunks)
- Chunk size: 4,410 samples per channel (100ms @ 44.1kHz)
- Data type: int16 stereo PCM
- Back-pressure threshold: <2 chunks remaining

**Why This Pattern**:
- NumPy pre-allocation avoids GC pauses
- Python GIL makes explicit locks more practical than lockfree atomics
- Circular buffer ensures FIFO ordering
- Back-pressure prevents overflow (sleep 10ms when buffer low)

**Rejected Alternatives**:
- multiprocessing.Queue: Pickling overhead adds latency
- asyncio.Queue: Incompatible with sync synthesis calls
- Lockfree atomics: Complex, GIL makes unnecessary

---

### 4. WebSocket Audio Streaming

**Decision**: Base64-encoded PCM with 100ms chunks

**Format**:
- Chunk size: 100ms (4,410 samples × 2 channels × 2 bytes = 17.6 kB raw)
- Encoding: Base64 (~23.5 kB per chunk)
- Bandwidth: ~250 kbps per client
- Metadata: JSON with seq, timestamp, duration_ms, sample_rate, channels

**Timing Metadata**:
```json
{
  "type": "audio_chunk",
  "seq": 12345,
  "timestamp": 1672531200.123,
  "duration_ms": 100,
  "sample_rate": 44100,
  "channels": 2,
  "audio_data": "BASE64_PCM_HERE..."
}
```

**Reconnection**: Exponential backoff (1s, 2s, 4s, 8s, ... up to 60s max) with ±20% jitter

**Why This Works**:
- Base64 universally supported (binary frames less compatible)
- 100ms chunks balance latency and efficiency
- Sequence numbers detect packet loss
- Timestamps enable latency measurement

**Rejected Alternatives**:
- Binary WebSocket frames: Browser compatibility concerns
- 50ms chunks: Excessive network overhead
- Opus compression: Deferred to Phase 3 (adds encoding latency)

---

### 5. Web Audio API Implementation

**Decision**: AudioWorklet (NOT ScriptProcessorNode)

**Why AudioWorklet**:
- Runs in audio rendering thread (lower latency)
- Isolated from main thread GC pauses
- ScriptProcessorNode is deprecated (will be removed)
- Modern standard (Chrome 66+, Edge 79+, Safari 14.1+)

**Adaptive Buffering (4-Tier)**:
| Jitter | Target Buffer | Latency |
|--------|---------------|---------|
| <10ms  | 300ms (3 chunks) | Low jitter |
| 10-25ms | 500ms (5 chunks) | Moderate |
| 25-50ms | 700ms (7 chunks) | High jitter |
| >50ms  | 1000ms (10 chunks) | Extreme |

**Jitter Tracking**: Exponential Moving Average (EMA) with α = 0.1
```
jitter_ema[t] = 0.1 × jitter[t] + 0.9 × jitter_ema[t-1]
```

**Browser Compatibility**:
- Chrome/Edge: Primary targets
- Safari: Auto-play workaround required (resume AudioContext on user click)
- Firefox: Optional (low priority for MVP)

---

### 6. Memory Management

**GC Tuning**:
```python
gc.set_threshold(10000, 20, 20)  # Increase from default (700, 10, 10)
```
- **Effect**: 14× fewer GC collections
- **Trade-off**: +5-10MB memory for smoother performance
- **Result**: Lower latency variance, fewer audio glitches

**Buffer Pre-allocation**:
- Pool of 20 pre-allocated NumPy buffers (int16, stereo)
- Reuse buffers instead of allocating new (avoids GC)
- Total pool memory: ~345 kB (negligible)

**Memory Budget**:
- SoundFonts: 340MB (Salamander + FluidR3)
- FluidSynth runtime: ~50MB
- Ring buffers: ~2MB
- Python overhead: ~50MB
- **Total**: ~450MB (within 500MB limit)

**Leak Detection**:
- Use `tracemalloc` for monitoring
- Success criterion: <10MB growth per hour

---

## Performance Targets Summary

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Synthesis latency** | <100ms per phrase | Time to render 8-bar phrase |
| **End-to-end latency** | <800ms (target 500ms) | Server generation → client playback |
| **Chunk delivery** | >98% on-time | Within 50ms variance of 100ms target |
| **Buffer underruns** | <1 per 30 min | On stable network |
| **Memory usage** | <500MB | Over 8+ hour sessions |
| **Concurrent clients** | 10+ | Without audio degradation |
| **Time-to-first-audio** | <2 seconds | Page load to audible playback |

---

## Implementation Checklist

**Server Setup**:
- [x] Research FluidSynth integration patterns
- [x] Select SoundFonts (Salamander + FluidR3_GM)
- [ ] Install FluidSynth native library
- [ ] Download SoundFonts to `./soundfonts/`
- [ ] Implement FluidSynthEngine class
- [ ] Implement AudioRingBuffer class
- [ ] Configure GC tuning

**Streaming Infrastructure**:
- [ ] Implement FastAPI WebSocket endpoint
- [ ] Implement base64 encoding for PCM chunks
- [ ] Add timing metadata to messages
- [ ] Implement back-pressure mechanism
- [ ] Add exponential backoff reconnection

**Client Implementation**:
- [ ] Create AudioWorklet processor
- [ ] Implement adaptive buffering algorithm
- [ ] Add EMA jitter tracking
- [ ] Handle Safari auto-play quirks
- [ ] Implement reconnection logic

**Testing**:
- [ ] Benchmark synthesis latency (<100ms target)
- [ ] Measure end-to-end latency (<800ms target)
- [ ] 30-minute continuous streaming test (0 dropouts)
- [ ] 8-hour memory stability test (<500MB limit)
- [ ] Multi-client test (10+ concurrent)

---

## Key Code Snippets

### FluidSynth Reverb Configuration
```python
self.synth.set_reverb(
    roomsize=0.6,   # Medium room
    damping=0.5,    # Moderate damping
    width=0.8,      # Wide stereo
    level=0.2       # 20% wet mix
)
```

### GC Tuning
```python
import gc
gc.set_threshold(10000, 20, 20)  # Reduce GC frequency
```

### Adaptive Buffer Size Selection
```python
if jitter_ema < 10:
    target_buffer_size = 3  # 300ms
elif jitter_ema < 25:
    target_buffer_size = 5  # 500ms
elif jitter_ema < 50:
    target_buffer_size = 7  # 700ms
else:
    target_buffer_size = 10  # 1000ms
```

### WebSocket Message Structure
```json
{
  "type": "audio_chunk",
  "seq": 12345,
  "timestamp": 1672531200.123,
  "duration_ms": 100,
  "sample_rate": 44100,
  "channels": 2,
  "audio_data": "BASE64_ENCODED_PCM..."
}
```

---

## Open Questions

**Resolved**:
- ✅ Which SoundFonts? → Salamander + FluidR3_GM
- ✅ Include reverb? → Yes, minimal FluidSynth reverb
- ✅ Percussion in MVP? → No, strictly excluded

**Remaining**:
- ❓ Default intensity/BPM validation → User testing needed
- ❓ Firefox support? → Test after Chrome/Edge/Safari proven
- ❓ Network bandwidth warnings? → Implement if underruns >5/min

---

## Next Steps

1. **Generate Implementation Plan**: Run `/speckit.plan` for detailed technical design
2. **Generate Tasks**: Run `/speckit.tasks` for actionable task breakdown
3. **Environment Setup**: Install FluidSynth, download SoundFonts
4. **Core Implementation**: Build synthesis engine, ring buffer, streaming server
5. **Client Development**: AudioWorklet, adaptive buffering, controls
6. **Testing & Benchmarking**: Verify latency targets, memory stability

---

**For Full Details**: See `/Users/andrew/Develop/auralis/docs/technical-research.md`
