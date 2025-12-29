# Data Model: Auralis MVP v2.0

**Feature**: Real-Time Generative Ambient Music Streaming Engine
**Branch**: `001-mvp-v2`
**Date**: 2025-12-28

## Overview

This document defines the data entities, relationships, and validation rules for the Auralis MVP v2.0 implementation. The data model is organized into five layers:

1. **Composition Layer**: Musical generation entities
2. **Synthesis Layer**: Audio rendering entities
3. **Streaming Layer**: Network transport entities
4. **Client Layer**: Browser playback entities
5. **Monitoring Layer**: Performance tracking entities

## Composition Layer

### ChordProgression

**Purpose**: Represents a generated sequence of chord changes over time.

**Attributes**:
- `chords`: List of chord events, each containing:
  - `onset_time` (int): Sample offset from phrase start (0-based)
  - `root_pitch` (int): MIDI note number (0-127, typically 48-72 for ambient bass)
  - `chord_type` (str): Chord quality ("major", "minor", "sus2", "sus4", "add9", "maj7")
- `duration_samples` (int): Total phrase length in samples (calculated from BPM and bar count)
- `bpm` (float): Tempo in beats per minute (40-90 range)
- `mode` (str): Modal context ("aeolian", "dorian", "lydian", "phrygian")

**Constraints**:
- Chord onset times must be monotonically increasing
- Root pitch must be valid MIDI note (0-127)
- Duration must match phrase rendering expectations (8-16 bars)
- Chord type must be from supported set

**Generation Logic**:
- Markov chain (bigram, order 2) considers 1 previous chord
- Harmonic transitions occur at 30-90 second intervals
- Modal constraints enforce scale degree relationships

**Example**:
```python
ChordProgression(
    chords=[
        {"onset_time": 0, "root_pitch": 60, "chord_type": "minor"},      # C minor
        {"onset_time": 44100, "root_pitch": 65, "chord_type": "major"},  # F major (1 sec later)
        {"onset_time": 88200, "root_pitch": 67, "chord_type": "sus4"},   # G sus4 (2 sec)
    ],
    duration_samples=176400,  # 4 seconds at 44.1kHz
    bpm=60.0,
    mode="aeolian"
)
```

---

### MelodyPhrase

**Purpose**: Represents a generated melodic line with note timing, pitch, velocity, and duration.

**Attributes**:
- `notes`: List of note events, each containing:
  - `onset_time` (int): Sample offset from phrase start
  - `pitch` (int): MIDI note number (48-96 range for ambient melodies)
  - `velocity` (int): Note velocity (20-100, humanized randomization)
  - `duration` (int): Note length in samples
- `duration_samples` (int): Total phrase length in samples
- `chord_context` (ChordProgression): Reference to harmonic structure

**Constraints**:
- Note onsets must be within phrase duration
- Pitch must be valid MIDI note (0-127, typically 48-96)
- Velocity must be in range 20-100
- Duration must be positive and less than phrase length
- 70% of notes must be chord tones (validated post-generation)
- 25% must be scale notes (non-chord)
- 5% may be chromatic passing tones

**Generation Logic**:
- Constraint-based selection weighted by chord/scale membership
- Note probability (50-80%) creates sparse texture
- Velocity randomized within 20-100 range for humanization

**Example**:
```python
MelodyPhrase(
    notes=[
        {"onset_time": 0, "pitch": 60, "velocity": 72, "duration": 22050},       # C, soft, 0.5s
        {"onset_time": 44100, "pitch": 63, "velocity": 55, "duration": 44100},  # Eb, quieter, 1s
        {"onset_time": 132300, "pitch": 67, "velocity": 80, "duration": 22050}, # G, louder, 0.5s
    ],
    duration_samples=176400,  # 4 seconds
    chord_context=chord_progression  # Reference to ChordProgression above
)
```

---

### MusicalContext

**Purpose**: Encapsulates current generative parameters controlling composition and synthesis.

**Attributes**:
- `key` (int): Root MIDI pitch (e.g., 60 = C, 62 = D, 64 = E, 67 = G, 69 = A)
- `mode` (str): Scale mode ("aeolian", "dorian", "lydian", "phrygian")
- `bpm` (float): Tempo in beats per minute (40-90 range, default 60)
- `intensity` (float): Note density multiplier (0.0-1.0, default 0.5)
- `key_signature` (str): Human-readable key (e.g., "C minor", "G major")

**Validation Rules**:
- Key must be in supported set: C (60), D (62), E (64), G (67), A (69)
- Mode must be one of: "aeolian", "dorian", "lydian", "phrygian"
- BPM must be in range [40, 90]
- Intensity must be in range [0.0, 1.0]

**State Transitions**:
- Updated via WebSocket JSON control messages
- Changes apply at next phrase boundary (not mid-phrase)
- Persisted to client localStorage

**Preset Mapping**:
- **Focus**: mode="dorian", bpm=60, intensity=0.5
- **Meditation**: mode="aeolian", bpm=50, intensity=0.3
- **Sleep**: mode="phrygian", bpm=40, intensity=0.2
- **Bright**: mode="lydian", bpm=70, intensity=0.6

**Example**:
```python
MusicalContext(
    key=60,  # C
    mode="aeolian",
    bpm=60.0,
    intensity=0.5,
    key_signature="C minor"
)
```

---

## Synthesis Layer

### FluidSynthVoice

**Purpose**: Wrapper around FluidSynth sample-based synthesis engine, managing SoundFont loading and audio rendering.

**Attributes**:
- `synth` (fluidsynth.Synth): FluidSynth synthesizer instance
- `sample_rate` (int): Audio sample rate (44,100 Hz fixed)
- `soundfont_id` (int): Loaded SoundFont identifier
- `preset_channels` (dict): Mapping of instrument names to MIDI channels
  - Example: `{"piano": 0, "pad": 1}`
- `reverb_config` (dict): Reverb settings
  - `enabled` (bool): True for MVP (minimal reverb)
  - `room_size` (float): 0.5 typical
  - `damping` (float): 0.5 typical
  - `wet_level` (float): 0.2 (20% wet)

**Operations**:
- `load_soundfont(file_path: str) -> int`: Load SF2 file, return SoundFont ID
- `select_preset(channel: int, preset_num: int)`: Assign preset to MIDI channel
- `note_on(channel: int, pitch: int, velocity: int)`: Trigger note
- `note_off(channel: int, pitch: int)`: Release note
- `render(num_samples: int) -> np.ndarray`: Generate audio (2, num_samples) float32

**Lifecycle**:
1. **Initialization**: Create synth, set sample rate (44.1kHz)
2. **SoundFont Loading**: Load Piano SF2 (preset 0), Pad SF2 (preset 88-90)
3. **Configuration**: Enable reverb (3s decay, 20% wet), set polyphony (32 voices)
4. **Rendering**: Process note events, generate stereo PCM
5. **Cleanup**: Delete synth, free SoundFont memory

**Example**:
```python
voice = FluidSynthVoice(sample_rate=44100)
voice.load_soundfont("soundfonts/Salamander-Grand-Piano.sf2")
voice.select_preset(channel=0, preset_num=0)  # Piano on channel 0
voice.note_on(channel=0, pitch=60, velocity=80)
audio = voice.render(num_samples=44100)  # 1 second of audio
voice.note_off(channel=0, pitch=60)
```

---

### SoundFontPreset

**Purpose**: Maps instrument names to SoundFont file paths and General MIDI preset numbers.

**Attributes**:
- `instrument_name` (str): Logical name ("piano", "pad", "strings")
- `sf2_file_path` (str): Absolute path to SoundFont file
- `preset_number` (int): General MIDI preset (0-127)
- `bank_number` (int): Bank select (typically 0 for GM)
- `file_size_mb` (float): SoundFont file size for memory tracking

**Constraints**:
- File path must exist and be readable
- Preset number must be valid (0-127)
- Total SoundFont memory < 500MB

**MVP Configuration**:
```python
SOUNDFONT_PRESETS = [
    SoundFontPreset(
        instrument_name="piano",
        sf2_file_path="/path/to/Salamander-Grand-Piano.sf2",
        preset_number=0,  # Acoustic Grand Piano
        bank_number=0,
        file_size_mb=200.0
    ),
    SoundFontPreset(
        instrument_name="pad",
        sf2_file_path="/path/to/Arachno-Warm-Pad.sf2",
        preset_number=88,  # Warm Pad
        bank_number=0,
        file_size_mb=150.0
    )
]
```

---

### AudioBuffer

**Purpose**: Represents rendered stereo PCM audio data ready for chunking and streaming.

**Attributes**:
- `data` (np.ndarray): NumPy array, shape (2, num_samples), dtype float32
  - Channel 0: Left channel
  - Channel 1: Right channel
- `sample_rate` (int): 44,100 Hz
- `duration_sec` (float): Duration in seconds (num_samples / sample_rate)

**Constraints**:
- Shape must be (2, N) where N > 0
- Data type must be float32
- Values must be in range [-1.0, 1.0] (soft clipping applied)

**Operations**:
- `to_int16() -> np.ndarray`: Convert float32 [-1.0, 1.0] → int16 [-32768, 32767]
- `chunk(chunk_size: int) -> List[np.ndarray]`: Split into fixed-size chunks
- `apply_limiting(threshold: float)`: Prevent clipping with soft limiter

**Example**:
```python
# Rendered audio from FluidSynth
audio_buffer = AudioBuffer(
    data=np.array([[0.5, 0.3, ...], [0.4, 0.2, ...]]),  # (2, 44100)
    sample_rate=44100,
    duration_sec=1.0
)

# Convert to int16 for base64 encoding
int16_data = audio_buffer.to_int16()  # Shape (2, 44100), dtype int16

# Chunk into 100ms pieces (4,410 samples)
chunks = audio_buffer.chunk(chunk_size=4410)  # 10 chunks of 100ms each
```

---

## Streaming Layer

### RingBuffer

**Purpose**: Thread-safe circular buffer for audio chunks, enabling continuous streaming without dropouts.

**Attributes**:
- `capacity` (int): Maximum number of chunks (10-20 for 1-2 second capacity)
- `buffer` (list): Pre-allocated list of AudioChunk slots
- `write_cursor` (int): Next write position (0 to capacity-1)
- `read_cursor` (int): Next read position (0 to capacity-1)
- `lock` (threading.Lock): Thread synchronization primitive
- `depth` (int): Current number of buffered chunks (write_cursor - read_cursor)

**Operations**:
- `write(chunk: AudioChunk) -> bool`: Add chunk to buffer, return False if full
- `read() -> Optional[AudioChunk]`: Remove chunk from buffer, return None if empty
- `get_depth() -> int`: Return current buffer depth
- `is_full() -> bool`: Check if write would block
- `is_empty() -> bool`: Check if read would block

**Constraints**:
- Depth must be in range [0, capacity]
- Write cursor and read cursor wrap at capacity
- Thread-safe operations (lock acquired for all mutations)

**Back-Pressure Logic**:
- If depth < 2 chunks: Sleep 10ms before next write attempt
- Prevents buffer underruns by slowing generation

**Example**:
```python
ring_buffer = RingBuffer(capacity=20)  # 2-second capacity at 100ms chunks

# Producer thread (synthesis)
chunk = AudioChunk(data=pcm_data, seq=123, timestamp=time.time())
if ring_buffer.write(chunk):
    print(f"Buffered chunk {chunk.seq}, depth: {ring_buffer.get_depth()}")
else:
    time.sleep(0.01)  # Back-pressure: wait 10ms

# Consumer thread (WebSocket streaming)
chunk = ring_buffer.read()
if chunk:
    await websocket.send_bytes(chunk.to_base64())
```

---

### WebSocketConnection

**Purpose**: Represents per-client WebSocket connection state and metrics.

**Attributes**:
- `client_id` (str): Unique identifier (UUID)
- `websocket` (WebSocket): FastAPI WebSocket instance
- `connected_at` (datetime): Connection timestamp
- `last_chunk_seq` (int): Last delivered chunk sequence number
- `buffer_health` (str): Current state ("healthy", "low", "emergency", "full")
- `musical_context` (MusicalContext): Current parameter settings
- `latency_ms` (float): Most recent end-to-end latency measurement

**Operations**:
- `send_audio_chunk(chunk: AudioChunk) -> None`: Send base64 PCM via WebSocket
- `send_control_message(msg_type: str, data: dict) -> None`: Send status/error JSON
- `receive_control_message() -> dict`: Parse incoming JSON (key, BPM, etc.)
- `update_metrics(latency: float, buffer_depth: int) -> None`: Track performance

**State Machine**:
- **Connected**: Initial state, audio streaming active
- **Buffering**: Buffer depth < 2, displaying indicator to user
- **Disconnected**: Connection lost, attempting reconnect
- **Error**: Unrecoverable error, connection closed

**Example**:
```python
connection = WebSocketConnection(
    client_id="a1b2c3d4-...",
    websocket=websocket_instance,
    connected_at=datetime.now(),
    last_chunk_seq=0,
    buffer_health="healthy",
    musical_context=MusicalContext(key=60, mode="aeolian", bpm=60, intensity=0.5)
)

# Send audio chunk
chunk = AudioChunk(data=pcm_data, seq=connection.last_chunk_seq + 1, timestamp=time.time())
await connection.send_audio_chunk(chunk)
connection.last_chunk_seq += 1

# Receive control message (parameter change)
msg = await connection.receive_control_message()
if msg["type"] == "update_params":
    connection.musical_context.bpm = msg["bpm"]
    connection.musical_context.intensity = msg["intensity"]
```

---

### AudioChunk

**Purpose**: Represents a 100ms segment of PCM audio with timing metadata for streaming.

**Attributes**:
- `data` (np.ndarray): Int16 PCM data, shape (2, 4410) for 100ms at 44.1kHz
- `seq` (int): Sequence number (monotonically increasing)
- `timestamp` (float): Server-side generation timestamp (Unix epoch)
- `duration_ms` (float): Chunk duration (100ms fixed)
- `sample_rate` (int): 44,100 Hz
- `channels` (int): 2 (stereo)

**Constraints**:
- Data shape must be (2, 4410) exactly
- Data type must be int16 (range -32768 to 32767)
- Sequence numbers must increment without gaps
- Timestamp must be valid Unix time

**Operations**:
- `to_base64() -> str`: Encode PCM as base64 string (~23.5kB)
- `to_json() -> dict`: Serialize as JSON with metadata
- `from_base64(b64: str) -> AudioChunk`: Decode base64 to AudioChunk

**Serialization Format** (WebSocket message):
```json
{
  "type": "audio",
  "seq": 123,
  "timestamp": 1735401234.567,
  "duration_ms": 100,
  "sample_rate": 44100,
  "channels": 2,
  "data": "base64-encoded-pcm-data-here..."
}
```

**Example**:
```python
# Create chunk from AudioBuffer
chunk = AudioChunk(
    data=audio_buffer.to_int16()[:, :4410],  # First 100ms
    seq=1,
    timestamp=time.time(),
    duration_ms=100.0,
    sample_rate=44100,
    channels=2
)

# Serialize for WebSocket
json_msg = chunk.to_json()
await websocket.send_json(json_msg)

# Client-side: Deserialize
received_chunk = AudioChunk.from_base64(json_msg["data"])
```

---

## Client Layer

### AudioContext (Web Audio API)

**Purpose**: Browser-side audio playback engine managing AudioContext and AudioWorklet.

**Attributes** (JavaScript):
- `context` (AudioContext): Web Audio API context
- `sampleRate` (number): 44,100 Hz (matches server)
- `workletNode` (AudioWorkletNode): Audio processing thread
- `ringBuffer` (AdaptiveBuffer): Client-side buffer (300-500ms target)
- `connected` (boolean): WebSocket connection state

**Operations**:
- `init() -> Promise<void>`: Initialize AudioContext and load AudioWorklet
- `decodeChunk(base64: string) -> Float32Array`: Decode base64 PCM → float32
- `queueChunk(chunk: Float32Array) -> void`: Add to adaptive buffer
- `start() -> void`: Begin playback (auto-play on page load)
- `stop() -> void`: Pause playback

**Lifecycle**:
1. User navigates to page → `init()` called
2. WebSocket connects → `connected = true`
3. Chunks arrive → `decodeChunk()` → `queueChunk()`
4. AudioWorklet processes buffer → audio output
5. Disconnect → `connected = false`, attempt reconnect

---

### AdaptiveBuffer (Client-Side)

**Purpose**: Client-side ring buffer with dynamic size adjustment based on network jitter.

**Attributes**:
- `buffer` (Float32Array[]): Array of decoded PCM chunks
- `targetDepth` (number): Desired buffer size (3-5 chunks, 300-500ms)
- `currentDepth` (number): Actual buffered chunks
- `jitterTracker` (JitterTracker): EMA-based jitter measurement
- `bufferHealth` (string): "emergency", "low", "healthy", "full"

**Buffer Tiers**:
- **Emergency** (<1 chunk): Display buffering indicator, request faster delivery
- **Low** (1-2 chunks): Increase target to 5 chunks, tolerate higher latency
- **Healthy** (3-4 chunks): Optimal latency (~300-400ms)
- **Full** (5+ chunks): Reduce target to 3 chunks, minimize latency

**Operations**:
- `enqueue(chunk: Float32Array) -> void`: Add chunk, update health
- `dequeue() -> Float32Array | null`: Remove chunk for playback
- `updateTarget(jitter: number) -> void`: Adjust target based on variance
- `getHealth() -> string`: Return current tier

**Example** (JavaScript):
```javascript
const buffer = new AdaptiveBuffer(targetDepth: 3);

// WebSocket receives chunk
websocket.onmessage = (msg) => {
  const chunk = decodeBase64(msg.data.data);
  buffer.enqueue(chunk);

  if (buffer.getHealth() === 'emergency') {
    displayBufferingIndicator();
  }
};

// AudioWorklet requests chunk
worklet.port.onmessage = () => {
  const chunk = buffer.dequeue();
  if (chunk) {
    worklet.port.postMessage({ chunk });
  }
};
```

---

### ControlState (Client-Side)

**Purpose**: Manages user settings and persistence via localStorage.

**Attributes**:
- `key` (number): Current root MIDI pitch (60, 62, 64, 67, 69)
- `mode` (string): Current scale mode ("aeolian", "dorian", "lydian", "phrygian")
- `intensity` (number): Note density (0.0-1.0)
- `bpm` (number): Tempo (40-90)
- `preset` (string | null): Active preset name ("focus", "meditation", "sleep", "bright")

**Operations**:
- `save() -> void`: Persist to localStorage
- `load() -> ControlState`: Restore from localStorage or defaults
- `applyPreset(name: string) -> void`: Load preset configuration
- `sendToServer(websocket: WebSocket) -> void`: Send JSON update message

**localStorage Format**:
```json
{
  "auralis_key": 60,
  "auralis_mode": "aeolian",
  "auralis_intensity": 0.5,
  "auralis_bpm": 60,
  "auralis_preset": null
}
```

**Example** (JavaScript):
```javascript
const state = ControlState.load();  // Restore saved settings

// User adjusts slider
intensitySlider.oninput = (e) => {
  state.intensity = parseFloat(e.target.value);
  state.save();  // Persist to localStorage
  state.sendToServer(websocket);  // Notify server
};

// User clicks preset
focusButton.onclick = () => {
  state.applyPreset('focus');  // mode="dorian", bpm=60, intensity=0.5
  state.save();
  state.sendToServer(websocket);
};
```

---

## Monitoring Layer

### PerformanceMetrics

**Purpose**: Tracks latency histograms, event counters, and memory usage for observability.

**Attributes**:
- `synthesis_latency` (LatencyHistogram): <100ms target, tracks avg/p50/p95/p99
- `network_latency` (LatencyHistogram): Per-client WebSocket delivery times
- `end_to_end_latency` (LatencyHistogram): Total generation → playback time
- `buffer_underruns` (int): Count of buffer depth < 1 events
- `buffer_overflows` (int): Count of write-blocked events
- `disconnects` (int): Count of WebSocket disconnections
- `memory_usage_mb` (float): Current process memory footprint
- `last_updated` (datetime): Most recent metrics update timestamp

**Operations**:
- `record_synthesis(latency_ms: float) -> void`: Add synthesis timing
- `record_network(client_id: str, latency_ms: float) -> void`: Add network timing
- `record_buffer_event(event_type: str) -> void`: Increment underrun/overflow counter
- `snapshot() -> dict`: Return current metrics as JSON (for `/api/metrics`)
- `reset() -> void`: Clear counters (typically not used)

**Histogram Implementation**:
- Circular buffer (1000 samples)
- Calculate avg, p50, p95, p99 on-demand
- Memory overhead < 10MB

**Example**:
```python
metrics = PerformanceMetrics()

# Record synthesis timing
start = time.time()
audio = synthesize_phrase(chords, melody)
latency_ms = (time.time() - start) * 1000
metrics.record_synthesis(latency_ms)

# Record buffer underrun
if ring_buffer.is_empty():
    metrics.record_buffer_event("underrun")

# Export to /api/metrics endpoint
@app.get("/api/metrics")
async def get_metrics():
    return metrics.snapshot()
```

**Snapshot Format** (`/api/metrics` response):
```json
{
  "synthesis_latency_ms": {
    "avg": 45.3,
    "p50": 42.1,
    "p95": 78.5,
    "p99": 92.3
  },
  "network_latency_ms": {
    "avg": 120.5,
    "p50": 115.2,
    "p95": 200.1,
    "p99": 285.3
  },
  "end_to_end_latency_ms": {
    "avg": 520.4,
    "p50": 495.7,
    "p95": 750.2,
    "p99": 820.1
  },
  "buffer_underruns": 2,
  "buffer_overflows": 0,
  "disconnects": 1,
  "memory_usage_mb": 345.2,
  "last_updated": "2025-12-28T13:45:23.123Z"
}
```

---

### SystemStatus

**Purpose**: Provides real-time operational state for `/api/status` endpoint.

**Attributes**:
- `uptime_sec` (float): Server uptime in seconds
- `active_connections` (int): Current WebSocket client count
- `buffer_depth` (int): Current ring buffer depth (chunks available)
- `device` (str): GPU/CPU device ("Metal", "CUDA", "CPU")
- `soundfont_loaded` (bool): Whether SoundFonts successfully loaded
- `synthesis_active` (bool): Whether generation loop is running

**Operations**:
- `update() -> void`: Refresh all status fields
- `to_json() -> dict`: Serialize as JSON response

**Example**:
```python
status = SystemStatus(
    uptime_sec=3600.5,  # 1 hour
    active_connections=5,
    buffer_depth=12,  # 1.2 seconds of audio buffered
    device="Metal",
    soundfont_loaded=True,
    synthesis_active=True
)

@app.get("/api/status")
async def get_status():
    status.update()
    return status.to_json()
```

**Response Format** (`/api/status`):
```json
{
  "uptime_sec": 3600.5,
  "active_connections": 5,
  "buffer_depth": 12,
  "device": "Metal",
  "soundfont_loaded": true,
  "synthesis_active": true,
  "timestamp": "2025-12-28T13:45:23.123Z"
}
```

---

## Entity Relationships

### Dependency Graph

```
MusicalContext
    ↓
ChordGenerator → ChordProgression
    ↓
MelodyGenerator → MelodyPhrase
    ↓                ↓
FluidSynthVoice (combines chords + melody)
    ↓
AudioBuffer
    ↓
AudioChunk (chunked)
    ↓
RingBuffer
    ↓
WebSocketConnection → Client (Browser)
    ↓
AudioContext → AdaptiveBuffer → Speakers
```

### Layer Boundaries

**Composition → Synthesis**:
- Input: ChordProgression + MelodyPhrase
- Output: AudioBuffer (NumPy float32 stereo)

**Synthesis → Streaming**:
- Input: AudioBuffer
- Output: List[AudioChunk] (base64-encoded PCM)

**Streaming → Client**:
- Input: AudioChunk (JSON over WebSocket)
- Output: Float32Array (decoded PCM in browser)

**Client → Monitoring**:
- Input: Latency measurements, buffer health
- Output: PerformanceMetrics updates

---

## Validation Rules Summary

### Composition Layer
- Chord onset times: Monotonically increasing, within phrase duration
- Melody note distribution: 70% chord tones, 25% scale, 5% chromatic (validated post-generation)
- Musical context: BPM [40, 90], intensity [0.0, 1.0], supported keys/modes only

### Synthesis Layer
- Audio data: Shape (2, N), dtype float32, range [-1.0, 1.0]
- SoundFont files: Must exist, readable, total size < 500MB
- Reverb settings: Reasonable decay (1-10s), wet level (0.0-1.0)

### Streaming Layer
- Ring buffer: Depth [0, capacity], thread-safe operations
- AudioChunk: Shape (2, 4410), dtype int16, monotonic sequence numbers
- WebSocket: Valid JSON messages, required fields present

### Client Layer
- Sample rate: 44,100 Hz (must match server)
- Buffer health: Emergency/low/healthy/full tiers enforced
- Control state: Valid MIDI pitches, modes, BPM/intensity ranges

### Monitoring Layer
- Latency histograms: Positive values only, reasonable upper bounds (<10s)
- Memory usage: <500MB total footprint warning threshold
- Metrics update frequency: 1 second interval

---

## State Transitions

### WebSocketConnection States

```
        connect()
          │
          ▼
    ┌──────────┐
    │Connected │ ◄─── Auto-play starts
    └──────────┘
          │
          │ buffer_depth < 2
          ▼
    ┌──────────┐
    │Buffering │ ──── Display indicator to user
    └──────────┘
          │
          │ buffer_depth ≥ 2
          ▼
    ┌──────────┐
    │Connected │
    └──────────┘
          │
          │ disconnect()
          ▼
    ┌──────────┐
    │Disconnected│ ──── Exponential backoff reconnect
    └──────────┘
          │
          │ reconnect() success
          ▼
    ┌──────────┐
    │Connected │
    └──────────┘
          │
          │ unrecoverable error
          ▼
    ┌──────────┐
    │  Error   │ ──── Close connection
    └──────────┘
```

### Buffer Health Transitions

```
   Emergency (<1 chunk)
        ↕
    Low (1-2 chunks) ──── Increase target to 5
        ↕
  Healthy (3-4 chunks) ──── Optimal latency
        ↕
    Full (5+ chunks) ──── Reduce target to 3
```

---

## Performance Considerations

### Memory Footprint Budget

| Component | Allocation | Notes |
|-----------|------------|-------|
| SoundFonts | 200-350MB | Salamander (200MB) + Arachno (150MB) |
| Ring Buffer | ~3.5MB | 20 chunks × 17.6kB × 10 clients |
| NumPy Arrays | ~10MB | Pre-allocated synthesis buffers |
| FluidSynth | ~50MB | Voice memory, reverb buffers |
| Metrics | <10MB | Circular histograms |
| **Total** | **<500MB** | MVP target |

### Latency Budget Breakdown

| Stage | Target | Measured (Apple M4) |
|-------|--------|---------------------|
| Composition | <50ms | ~15ms (Markov + constraints) |
| Synthesis | <100ms | ~31ms (FluidSynth, 8 bars) |
| Chunking | <5ms | ~2ms (NumPy slicing) |
| Base64 Encoding | <10ms | ~5ms per chunk |
| Network | <200ms | ~100ms (local), varies |
| Client Buffering | 300-500ms | ~350ms adaptive |
| **End-to-End** | **<800ms** | **~503ms** (MVP acceptable) |

---

## Next Steps

This data model defines all entities for implementation. See:
- [contracts/](contracts/) for API specifications (WebSocket, REST, internal interfaces)
- [quickstart.md](quickstart.md) for testing scenarios
- [plan.md](plan.md) for implementation phases

**Status**: ✅ Data model complete, ready for contract definition
