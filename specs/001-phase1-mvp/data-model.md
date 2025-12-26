# Data Model: Phase 1 MVP - Real-time Ambient Music Streaming

**Created**: 2024-12-26  
**Feature**: 001-phase1-mvp  
**Scope**: Real-time audio generation and streaming system

---

## Core Entities

### AudioChunk
**Purpose**: 100ms segment of audio data transmitted over WebSocket

**Fields**:
- `data`: string (base64-encoded 16-bit PCM audio)
- `timestamp`: integer (Unix timestamp for synchronization)
- `sequence`: integer (sequential packet number for ordering)
- `sample_rate`: integer (44100 - audio sampling rate)
- `format`: string ("pcm16" - audio format identifier)

**Validation Rules**:
- Base64 data must decode to valid 16-bit PCM array
- Sequence numbers must be monotonically increasing
- Audio data length must equal 8,820 bytes (44.1kHz × 100ms × 2 bytes)

### ChordProgression
**Purpose**: 8-bar harmonic sequence for ambient music generation

**Fields**:
- `chords`: array of strings (chord symbols: ["i", "iv", "V", "VI", "III"])
- `length_bars`: integer (8 bars - phrase length)
- `root_midi`: integer (57 - A3 MIDI note reference)
- `transition_matrix`: 2D array (5×5 probability matrix for Markov chain)

**Validation Rules**:
- Chord symbols must belong to ambient chord vocabulary
- Length must be exactly 8 bars for MVP
- Transition matrix rows must sum to 1.0 (probability distribution)
- Root MIDI must be within valid range (21-127)

### MelodyPhrase
**Purpose**: Series of MIDI notes conforming to harmonic constraints

**Fields**:
- `notes`: array of tuples (onset_sec, pitch_midi, velocity, duration_sec)
- `bars`: integer (8 - matching chord progression length)
- `scale_intervals`: array of integers ([0, 2, 3, 5, 7, 8, 10] for A minor)

**Validation Rules**:
- All MIDI pitches must be within valid range (21-127)
- Velocities must be between 0.0 and 1.0
- Notes must fit within current chord harmony (constraint validation)
- Onset times must be within bar boundaries

### RingBuffer
**Purpose**: Thread-safe audio buffer for continuous streaming

**Fields**:
- `capacity_samples`: integer (88,200 samples - 2 seconds at 44.1kHz)
- `write_position`: integer (current write cursor position)
- `read_position`: integer (current read cursor position)
- `buffer_data`: array of float32 (stereo audio samples)

**Validation Rules**:
- Capacity must be multiple of chunk size (8,820 samples)
- Write/read positions must stay within buffer bounds
- Buffer state must detect underflow/overflow conditions

### SynthesisParameters
**Purpose**: Audio generation configuration settings

**Fields**:
- `key`: string ("A" - musical key)
- `bpm`: integer (70 - tempo for ambient music)
- `intensity`: float (0.5 - generation density control)

**Validation Rules**:
- Key must be valid musical key (A-G with minor/major)
- BPM must be between 40-120 (reasonable tempo range)
- Intensity must be between 0.0 and 1.0

---

## Entity Relationships

### Generation Pipeline
```
SynthesisParameters → ChordProgressionGenerator → ChordProgression
                                     ↓
MelodyPhraseGenerator → MelodyPhrase
                                     ↓
                              AmbientSynth → AudioChunk
                                     ↓
                              RingBuffer → WebSocket → Client
```

### State Transitions
- **SynthesisParameters**: Configuration changes update generation behavior
- **ChordProgression**: Generated every 8 bars, influences melody constraints
- **MelodyPhrase**: Generated per chord progression, synchronized timing
- **RingBuffer**: Continuous write/read cycle with thread safety
- **AudioChunk**: Generated every 100ms, transmitted via WebSocket

---

## Data Flow

### Real-time Generation Cycle
1. **Configuration**: User sets SynthesisParameters via web interface
2. **Chord Generation**: Markov chain creates 8-bar ChordProgression (<100ms)
3. **Melody Generation**: Constraint-based MelodyPhrase created for each chord
4. **Audio Synthesis**: torchsynth renders complete phrase (<5 seconds)
5. **Buffer Management**: AudioChunk written to RingBuffer
6. **Streaming**: RingBuffer read in 100ms chunks, encoded as base64
7. **Transmission**: WebSocket sends AudioChunk to web client
8. **Playback**: Web Audio API converts PCM to audio output

### Error Handling States
- **GPU Unavailable**: Fallback to CPU synthesis, update SynthesisParameters
- **Buffer Underflow**: Generate silence chunk, notify user
- **Connection Drop**: Retry with exponential backoff, maintain RingBuffer state
- **Synthesis Error**: Generate fallback phrase, log error details

---

## Performance Constraints

### Real-time Requirements
- **Chord Generation**: <100ms for 8-bar progression
- **Audio Synthesis**: <5 seconds for complete phrase
- **Chunk Encoding**: <10ms for base64 conversion
- **WebSocket Latency**: <50ms delivery time
- **Client Playback**: <100ms buffer management

### Memory Limits
- **Ring Buffer**: 2 seconds audio capacity (88,200 stereo samples)
- **Phrase Buffer**: 5 seconds rendered audio (220,500 stereo samples)
- **Chunk Queue**: Maximum 10 pending chunks (1 second backup)

### CPU/GPU Utilization
- **GPU Acceleration**: Prioritized for torchsynth operations
- **CPU Fallback**: Available when GPU unavailable
- **Thread Utilization**: Separate threads for synthesis and streaming
- **Memory Usage**: Optimized for continuous operation (no leaks)