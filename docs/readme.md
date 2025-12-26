## Key Highlights

### Architecture Overview
A **three-layer system** separating concerns:
1. **Generation Layer** (async) - Composition engine for melodies, chords, percussion
2. **Synthesis Layer** (blocking) - Real-time audio rendering via torchsynth
3. **Streaming Layer** (async) - FastAPI WebSocket for client delivery

### Critical Design Decisions

**Real-Time Streaming Approach:**
- WebSocket + base64-encoded PCM (MVP simplicity)
- 100ms chunks @ 44.1kHz (balance between latency & efficiency)
- Ring buffers on both server & client to prevent dropouts
- Target end-to-end latency: 400-700ms (acceptable for ambient, non-interactive)

**Generation Strategy:**
- **Chords**: Markov chains (fast, predictable) for MVP
- **Melody**: Constraint-based generation (melodies must fit chord tones)
- **Percussion**: Sparse texture + rule-based patterns (ambient rarely needs drums)
- **Pre-generation**: Always queue 2-3 phrases ahead to avoid latency hiccups

**Synthesis Tech:**
- **Primary**: `torchsynth` (16,000x real-time speed, GPU-accelerated on M4)
- **Fallback**: Pedalboard for VST hosting (Serum, Vital, etc.)
- **Threading model**: Synthesis in dedicated thread (blocking), WebSocket in asyncio (non-blocking)

### Addressed Gaps

1. **Melody Generation** - Three options from simple to advanced (Markov → fine-tuned Transformer → DDSP)
2. **Harmony Constraint** - Notes must respect chord progressions; can't just generate random pitches
3. **Ambient Textures** - Not "drums"; instead granular swells, sparse bells, minimal kicks
4. **Network Latency** - Client-side adaptive buffering to handle jitter
5. **GPU Memory** - Batch rendering + Metal cache invalidation for M4 Mac
6. **Streaming Protocol** - Start with base64 PCM, upgrade to Opus if bandwidth is critical

### Implementation Roadmap

**Phase 1 (MVP)**: Markov chords + constraint melody + torchsynth rendering + FastAPI streaming
**Phase 2**: Transformer-based melody + percussion + control API
**Phase 3**: GPU optimization + Opus codec + effects
**Phase 4**: Cloud deployment + monitoring