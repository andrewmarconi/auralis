# WebSocket API Contract: Auralis Streaming Protocol

**Feature**: Real-Time Audio Streaming
**Branch**: `001-mvp-v2`
**Protocol Version**: 1.0.0
**Date**: 2025-12-28

## Overview

The Auralis WebSocket API provides bidirectional real-time communication between the server (Python/FastAPI) and browser clients (JavaScript/Web Audio API) for:

1. **Audio Streaming**: Server → Client (44.1kHz/16-bit PCM in 100ms chunks)
2. **Control Messages**: Client → Server (parameter updates, presets)
3. **Status Events**: Server → Client (connection state, buffer health)

**Endpoint**: `ws://localhost:8000/ws/stream` (development) or `wss://auralis.example.com/ws/stream` (production)

---

## Connection Lifecycle

### 1. Client Initiates Connection

```javascript
const websocket = new WebSocket('ws://localhost:8000/ws/stream');

websocket.onopen = () => {
  console.log('Connected to Auralis streaming server');
  // Audio auto-play begins immediately
};

websocket.onerror = (error) => {
  console.error('WebSocket error:', error);
  // Display error to user
};

websocket.onclose = (event) => {
  console.log('Disconnected:', event.code, event.reason);
  // Attempt reconnection with exponential backoff
};
```

### 2. Server Accepts Connection

- Assigns unique `client_id` (UUID)
- Initializes MusicalContext with defaults (C Aeolian, 60 BPM, 0.5 intensity)
- Starts audio generation loop
- Begins streaming chunks immediately (auto-play)

### 3. Audio Streaming Begins

- Server sends `AudioChunk` messages at ~100ms intervals
- Client receives, decodes, and queues for playback
- No acknowledgment required (fire-and-forget for low latency)

### 4. Connection Termination

- **Clean Close**: Client sends close frame, server cleanup
- **Disconnect**: Server detects disconnect, removes client from active list
- **Reconnection**: Client uses exponential backoff (100ms → 200ms → 400ms → 800ms → max 5s)

---

## Message Types

### Server → Client Messages

#### 1. Audio Chunk (`type: "audio"`)

**Purpose**: Deliver 100ms of stereo PCM audio for playback.

**Frequency**: ~10 messages/second (100ms intervals)

**Format**:
```json
{
  "type": "audio",
  "seq": 123,
  "timestamp": 1735401234.567,
  "duration_ms": 100,
  "sample_rate": 44100,
  "channels": 2,
  "data": "base64-encoded-pcm-data..."
}
```

**Fields**:
- `type` (string): Always `"audio"`
- `seq` (integer): Monotonically increasing sequence number (starts at 0)
- `timestamp` (float): Server-side generation timestamp (Unix epoch seconds)
- `duration_ms` (float): Chunk duration in milliseconds (always 100.0)
- `sample_rate` (integer): Sample rate in Hz (always 44100)
- `channels` (integer): Number of audio channels (always 2 for stereo)
- `data` (string): Base64-encoded int16 PCM data
  - Raw format: NumPy array shape (2, 4410), dtype int16
  - Interleaved: [L0, R0, L1, R1, ..., L4409, R4409]
  - Bytes: 4,410 samples × 2 channels × 2 bytes = 17,640 bytes
  - Base64 length: ~23,520 characters

**Client Handling**:
```javascript
websocket.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === 'audio') {
    // Decode base64 → int16 → float32
    const int16Data = base64ToInt16(msg.data);
    const float32Data = int16ToFloat32(int16Data);  // Scale to [-1.0, 1.0]

    // Queue for playback
    adaptiveBuffer.enqueue(float32Data, msg.seq, msg.timestamp);
  }
};
```

**Binary Format Details**:
```
Int16 PCM (raw):
[Left channel: 4,410 samples] [Right channel: 4,410 samples]
Each sample: 2 bytes, range -32768 to 32767

Base64 Encoding:
base64.b64encode(int16_array.tobytes())
Decoding: np.frombuffer(base64.b64decode(b64_string), dtype=np.int16).reshape(2, 4410)
```

---

#### 2. Connection Status (`type: "status"`)

**Purpose**: Notify client of connection state changes.

**Frequency**: On state change only

**Format**:
```json
{
  "type": "status",
  "state": "connected",
  "message": "Audio streaming started",
  "buffer_depth": 12,
  "timestamp": 1735401234.567
}
```

**Fields**:
- `type` (string): Always `"status"`
- `state` (string): One of `"connected"`, `"buffering"`, `"disconnected"`, `"error"`
- `message` (string): Human-readable status description
- `buffer_depth` (integer): Current server-side ring buffer depth (chunks available)
- `timestamp` (float): Server timestamp

**States**:
- `"connected"`: Normal operation, audio streaming
- `"buffering"`: Buffer depth < 2, pausing generation
- `"disconnected"`: Client connection lost (not typically sent, as client is already disconnected)
- `"error"`: Server-side error (e.g., SoundFont load failure)

**Client Handling**:
```javascript
if (msg.type === 'status') {
  if (msg.state === 'buffering') {
    displayBufferingIndicator();
  } else if (msg.state === 'connected') {
    hideBufferingIndicator();
  } else if (msg.state === 'error') {
    displayError(msg.message);
  }

  updateBufferHealthUI(msg.buffer_depth);
}
```

---

#### 3. Error Message (`type: "error"`)

**Purpose**: Report server-side errors to client.

**Frequency**: As errors occur

**Format**:
```json
{
  "type": "error",
  "code": "SOUNDFONT_LOAD_FAILED",
  "message": "Failed to load SoundFont: Salamander-Grand-Piano.sf2 not found",
  "recoverable": false,
  "timestamp": 1735401234.567
}
```

**Fields**:
- `type` (string): Always `"error"`
- `code` (string): Error code (uppercase snake_case)
- `message` (string): Human-readable error description
- `recoverable` (boolean): Whether client can recover (retry vs fatal)
- `timestamp` (float): Server timestamp

**Error Codes**:
- `SOUNDFONT_LOAD_FAILED`: SoundFont file not found or corrupted
- `SYNTHESIS_TIMEOUT`: FluidSynth rendering exceeded 100ms latency
- `BUFFER_OVERFLOW`: Ring buffer full, cannot write
- `GPU_NOT_AVAILABLE`: GPU initialization failed (not fatal, falls back to CPU)
- `INTERNAL_ERROR`: Unhandled exception in server code

**Client Handling**:
```javascript
if (msg.type === 'error') {
  console.error(`[${msg.code}] ${msg.message}`);

  if (msg.recoverable) {
    // Attempt reconnection
    scheduleReconnect();
  } else {
    // Display fatal error, stop playback
    displayFatalError(msg.message);
    websocket.close();
  }
}
```

---

### Client → Server Messages

#### 1. Parameter Update (`type: "update_params"`)

**Purpose**: Change musical generation parameters (key, mode, BPM, intensity).

**Frequency**: User interaction (slider change, dropdown selection)

**Format**:
```json
{
  "type": "update_params",
  "params": {
    "key": 60,
    "mode": "aeolian",
    "bpm": 70.0,
    "intensity": 0.6
  },
  "timestamp": 1735401234.567
}
```

**Fields**:
- `type` (string): Always `"update_params"`
- `params` (object): Key-value pairs of parameters to update
  - `key` (integer, optional): Root MIDI pitch (60=C, 62=D, 64=E, 67=G, 69=A)
  - `mode` (string, optional): Scale mode ("aeolian", "dorian", "lydian", "phrygian")
  - `bpm` (float, optional): Tempo (40.0-90.0)
  - `intensity` (float, optional): Note density (0.0-1.0)
- `timestamp` (float): Client-side timestamp (Unix epoch seconds)

**Validation**:
- Server validates all parameters before applying
- Invalid values are ignored (no error sent to client)
- Changes apply at next phrase boundary (~8 bars, ~10-30 seconds)

**Server Handling**:
```python
async def handle_update_params(websocket, msg):
    params = msg["params"]

    # Validate and update MusicalContext
    if "key" in params and params["key"] in [60, 62, 64, 67, 69]:
        websocket.musical_context.key = params["key"]

    if "mode" in params and params["mode"] in ["aeolian", "dorian", "lydian", "phrygian"]:
        websocket.musical_context.mode = params["mode"]

    if "bpm" in params and 40.0 <= params["bpm"] <= 90.0:
        websocket.musical_context.bpm = params["bpm"]

    if "intensity" in params and 0.0 <= params["intensity"] <= 1.0:
        websocket.musical_context.intensity = params["intensity"]

    # Log parameter change
    logger.info(f"Client {websocket.client_id} updated params: {params}")
```

**Client Sending**:
```javascript
function updateServerParams(key, mode, bpm, intensity) {
  const msg = {
    type: 'update_params',
    params: {
      key: key,
      mode: mode,
      bpm: bpm,
      intensity: intensity
    },
    timestamp: Date.now() / 1000  // Unix epoch
  };

  websocket.send(JSON.stringify(msg));
}

// Example: User adjusts BPM slider
bpmSlider.oninput = (e) => {
  const bpm = parseFloat(e.target.value);
  updateServerParams(null, null, bpm, null);  // Only update BPM
};
```

---

#### 2. Preset Selection (`type: "select_preset"`)

**Purpose**: Apply predefined musical preset (Focus, Meditation, Sleep, Bright).

**Frequency**: User clicks preset button

**Format**:
```json
{
  "type": "select_preset",
  "preset": "focus",
  "timestamp": 1735401234.567
}
```

**Fields**:
- `type` (string): Always `"select_preset"`
- `preset` (string): Preset name ("focus", "meditation", "sleep", "bright")
- `timestamp` (float): Client-side timestamp

**Preset Definitions** (server-side):
- **focus**: mode="dorian", bpm=60, intensity=0.5
- **meditation**: mode="aeolian", bpm=50, intensity=0.3
- **sleep**: mode="phrygian", bpm=40, intensity=0.2
- **bright**: mode="lydian", bpm=70, intensity=0.6

**Server Handling**:
```python
PRESETS = {
    "focus": {"mode": "dorian", "bpm": 60.0, "intensity": 0.5},
    "meditation": {"mode": "aeolian", "bpm": 50.0, "intensity": 0.3},
    "sleep": {"mode": "phrygian", "bpm": 40.0, "intensity": 0.2},
    "bright": {"mode": "lydian", "bpm": 70.0, "intensity": 0.6}
}

async def handle_select_preset(websocket, msg):
    preset_name = msg["preset"]

    if preset_name in PRESETS:
        preset = PRESETS[preset_name]
        websocket.musical_context.mode = preset["mode"]
        websocket.musical_context.bpm = preset["bpm"]
        websocket.musical_context.intensity = preset["intensity"]

        logger.info(f"Client {websocket.client_id} selected preset: {preset_name}")
    else:
        logger.warning(f"Unknown preset: {preset_name}")
```

**Client Sending**:
```javascript
focusButton.onclick = () => {
  websocket.send(JSON.stringify({
    type: 'select_preset',
    preset: 'focus',
    timestamp: Date.now() / 1000
  }));

  // Optimistically update UI
  updateUIForPreset('focus');
};
```

---

#### 3. Ping/Heartbeat (`type: "ping"`)

**Purpose**: Keep WebSocket connection alive, measure round-trip latency.

**Frequency**: Every 30 seconds (client-initiated)

**Format**:
```json
{
  "type": "ping",
  "timestamp": 1735401234.567
}
```

**Fields**:
- `type` (string): Always `"ping"`
- `timestamp` (float): Client-side timestamp

**Server Response**:
```json
{
  "type": "pong",
  "client_timestamp": 1735401234.567,
  "server_timestamp": 1735401234.590
}
```

**Latency Calculation** (client-side):
```javascript
let pingTimestamp;

function sendPing() {
  pingTimestamp = Date.now() / 1000;
  websocket.send(JSON.stringify({
    type: 'ping',
    timestamp: pingTimestamp
  }));
}

websocket.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === 'pong') {
    const now = Date.now() / 1000;
    const roundTripLatency = (now - msg.client_timestamp) * 1000;  // ms
    console.log(`Round-trip latency: ${roundTripLatency.toFixed(1)}ms`);

    updateLatencyUI(roundTripLatency);
  }
};

// Send ping every 30 seconds
setInterval(sendPing, 30000);
```

---

## Error Handling

### Reconnection Strategy

**Exponential Backoff**:
```javascript
let reconnectDelay = 100;  // Start at 100ms
const MAX_DELAY = 5000;     // Cap at 5 seconds

function reconnect() {
  setTimeout(() => {
    console.log(`Reconnecting in ${reconnectDelay}ms...`);
    websocket = new WebSocket('ws://localhost:8000/ws/stream');

    websocket.onopen = () => {
      console.log('Reconnected!');
      reconnectDelay = 100;  // Reset delay on success
    };

    websocket.onerror = () => {
      reconnectDelay = Math.min(reconnectDelay * 2, MAX_DELAY);
      reconnect();
    };
  }, reconnectDelay);
}

websocket.onclose = reconnect;
```

### Server-Side Disconnect Detection

```python
async def stream_audio(websocket: WebSocket):
    try:
        while True:
            chunk = await ring_buffer.read()
            await websocket.send_json(chunk.to_json())
            await asyncio.sleep(0.1)  # 100ms interval
    except WebSocketDisconnect:
        logger.info(f"Client {websocket.client_id} disconnected")
        cleanup_client(websocket.client_id)
    except Exception as e:
        logger.error(f"Error streaming to {websocket.client_id}: {e}")
        await websocket.close(code=1011, reason="Internal server error")
```

---

## Performance Guarantees

### Latency Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Chunk delivery interval | 100ms ± 10ms | 100ms ± 5ms |
| Server processing time | <10ms/chunk | ~2ms/chunk |
| Network latency (local) | <50ms | ~10-30ms |
| Client decode + queue | <5ms | ~2ms |
| End-to-end (generation → playback) | <800ms | ~500ms |

### Bandwidth Usage

- **Audio stream**: ~250 kbps (base64 PCM overhead included)
  - Raw PCM: 44,100 Hz × 16 bits × 2 channels = 1,411 kbps
  - Chunked + base64: ~250 kbps (80% compression from chunking efficiency)
- **Control messages**: Negligible (<1 kbps)
- **Total**: ~250 kbps per connected client

### Concurrency

- **MVP Target**: 10+ concurrent clients per server instance
- **Expected**: Single server handles 20-50 clients on modern hardware
- **Scaling**: Horizontal scaling via load balancer (post-MVP)

---

## Security Considerations

### MVP (Local Development)

- **Protocol**: `ws://` (unencrypted WebSocket)
- **Authentication**: None (local-only access)
- **CORS**: Allow all origins

### Production (Post-MVP)

- **Protocol**: `wss://` (TLS-encrypted WebSocket)
- **Authentication**: JWT tokens or session cookies
- **CORS**: Restrict to known domains
- **Rate Limiting**: Max 1 connection per IP per minute
- **DDoS Protection**: Cloudflare or equivalent

---

## Testing & Validation

### Client-Side Test Cases

1. **Connection**: Verify WebSocket connects within 2 seconds
2. **Audio Streaming**: Receive 100ms chunks at ~100ms intervals for 30+ minutes
3. **Parameter Updates**: Change BPM, verify applied within 5 seconds (next phrase)
4. **Preset Selection**: Click preset button, verify parameters update
5. **Reconnection**: Disconnect network, verify exponential backoff reconnect
6. **Error Handling**: Receive error message, verify UI displays warning

### Server-Side Test Cases

1. **Client Management**: Connect 10 clients, verify all receive unique streams
2. **Chunk Delivery**: Measure timing variance, ensure <10ms jitter
3. **Parameter Validation**: Send invalid values, verify ignored without crash
4. **Graceful Shutdown**: Disconnect client, verify cleanup (no memory leak)
5. **Error Recovery**: Simulate SoundFont load failure, verify error message sent

### Integration Test Example

```python
import asyncio
import websockets
import json

async def test_audio_streaming():
    uri = "ws://localhost:8000/ws/stream"

    async with websockets.connect(uri) as ws:
        # Wait for first audio chunk
        msg = json.loads(await ws.recv())
        assert msg["type"] == "audio"
        assert msg["sample_rate"] == 44100
        assert msg["channels"] == 2

        # Receive 10 chunks (1 second of audio)
        for i in range(10):
            msg = json.loads(await ws.recv())
            assert msg["type"] == "audio"
            assert msg["seq"] == i + 1

        # Update parameters
        await ws.send(json.dumps({
            "type": "update_params",
            "params": {"bpm": 80.0, "intensity": 0.7},
            "timestamp": time.time()
        }))

        # Continue receiving chunks
        for i in range(10):
            msg = json.loads(await ws.recv())
            assert msg["type"] == "audio"

        print("✅ Audio streaming test passed")

asyncio.run(test_audio_streaming())
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-28 | Initial WebSocket API specification for MVP |

---

**Next Steps**:
- Implement server-side WebSocket endpoint in `server/streaming_server.py`
- Implement client-side WebSocket handler in `client/audio_client_worklet.js`
- Write integration tests in `tests/integration/test_smooth_streaming.py`

**Related Contracts**:
- [http-api.md](http-api.md) - REST endpoints for status and metrics
- [internal-interfaces.md](internal-interfaces.md) - Python internal interfaces
