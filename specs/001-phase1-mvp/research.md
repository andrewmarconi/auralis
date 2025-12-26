# Phase 0 Research: Phase 1 MVP Implementation

**Created**: 2024-12-26  
**Feature**: Real-time Ambient Music Streaming (001-phase1-mvp)

## Research Summary

This document consolidates research findings for Phase 1 MVP implementation, focusing on Markov chord generation, torchsynth integration, and WebSocket audio streaming best practices.

---

## Markov Chord Generation Research

### Decision: Bigram Markov Chain with Ambient-Biased Transition Matrix

**Rationale**: 
- Sufficient for basic ambient progressions with real-time performance requirements
- Simple implementation with predictable resource usage
- Lower complexity than higher-order chains, reducing failure risk for MVP

**Key Findings**:
- Ambient music benefits from diagonal-dominant transition matrices (higher self-transition probabilities)
- Prioritize I→IV, I→vi, IV→V, V→I harmonic movements
- 8-bar phrases with 2-4 notes per bar create appropriate ambient pacing

**Implementation Approach**:
```python
# Ambient-optimized chord state space: [i, iv, V, VI, III]
transition_matrix = np.array([
    [0.60, 0.05, 0.02, 0.15, 0.08],  # From i (favor staying, IV, VI)
    [0.20, 0.50, 0.05, 0.10, 0.05],  # From iv  
    [0.10, 0.10, 0.55, 0.15, 0.05],  # From V
    [0.18, 0.05, 0.07, 0.45, 0.15],  # From VI
    [0.25, 0.05, 0.03, 0.12, 0.40],  # From III
])
```

**Alternatives Considered**:
- Trigram Markov chains: Better musical coherence but exponential state complexity
- LSTM-based generation: More sophisticated but overkill for MVP
- Rule-based generation: Too predictable for ambient music

---

## torchsynth Integration Research

### Decision: Monophonic Voice with ADSR for Lead + Pad Layers

**Rationale**:
- torchsynth achieves 16,200x real-time throughput (major performance advantage)
- Monophonic synthesis simplifies implementation while maintaining quality
- GPU acceleration via Metal/CUDA supports real-time requirements

**Key Findings**:
- Control rate optimization (100x lower than audio rate) crucial for performance
- Separate voice instances for lead (brighter) and pad (warmer) timbres
- Batch generation and GPU acceleration meet <5 second phrase generation target

**Implementation Approach**:
```python
# Lead voice: SquareSawVCO, fast ADSR (10-50ms attack)
lead_voice = Voice()
lead_voice.vco = SquareSawVCO()
lead_voice.adsr.set_attack(0.01)  # 10ms

# Pad voice: SineVCO, slow ADSR (0.5-2.0s attack)  
pad_voice = Voice()
pad_voice.vco = SineVCO()
pad_voice.adsr.set_attack(1.0)  # 1 second

# GPU acceleration
if torch.cuda.is_available():
    lead_voice = lead_voice.to("cuda")
    pad_voice = pad_voice.to("cuda")
```

**Alternatives Considered**:
- Polyphonic synthesis: More complex but unnecessary for MVP ambient needs
- Custom DSP implementation: Higher risk and longer development time
- Sample-based synthesis: Limited flexibility and larger memory footprint

---

## WebSocket Audio Streaming Research

### Decision: JSON Protocol with Base64-Encoded PCM16 Chunks

**Rationale**:
- Simplifies error handling and metadata transmission
- Compatible with Chrome/Edge WebSocket implementations
- 100ms chunks at 44.1kHz balance latency and bandwidth

**Key Findings**:
- 8,820 bytes per chunk (44.1kHz × 100ms × 2 bytes × 1 channel)
- Client-side buffering of 2-3 chunks prevents playback interruptions
- Metadata protocol enables proper timing synchronization

**Implementation Approach**:
```javascript
// WebSocket message format
{
  "type": "audio_chunk",
  "timestamp": 1703123456789,
  "sequence": 12345,
  "sample_rate": 44100,
  "format": "pcm16",
  "data": "base64_encoded_audio"
}

// Client buffering with Web Audio API
function processAudioChunk(pcmData) {
  const audioBuffer = audioContext.createBuffer(1, pcmData.length/2, 44100);
  const channelData = audioBuffer.getChannelData(0);
  
  // Convert Int16 to Float32 for Web Audio API
  for (let i = 0; i < pcmData.length/2; i++) {
    channelData[i] = pcmData[i] / 32768.0;
  }
  
  bufferQueue.push(audioBuffer);
  if (!isPlaying && bufferQueue.length >= 2) {
    startPlayback();
  }
}
```

**Alternatives Considered**:
- Binary WebSocket frames: 33% overhead reduction but more complex error handling
- Opus compression: Significant bandwidth savings but adds codec complexity
- WebRTC data channels: Better browser support but unnecessary for single-user MVP

---

## Performance Targets Validation

**Research-Confirmed Targets**:
- **End-to-end latency**: 500-800ms (MVP acceptable)
- **Synthesis throughput**: >50x real-time with GPU acceleration
- **Chunk generation**: <100ms for 8-bar chord progressions
- **Phrase synthesis**: <5 seconds for complete audio rendering
- **WebSocket delivery**: ~100ms intervals with minimal jitter

**Technical Feasibility**: All targets achievable with researched approaches and constitutional constraints.

---

## Implementation Readiness

**RESOLVED**: All NEEDS CLARIFICATION items from Technical Context
- **Markov generation strategy**: Bigram with ambient bias confirmed
- **torchsynth integration**: Monophonic voices with GPU acceleration planned
- **WebSocket protocol**: JSON + Base64 PCM16 for Chrome/Edge compatibility
- **Performance targets**: Research validates <800ms end-to-end latency achievable

**READY**: Proceed to Phase 1 design and implementation.