# WebSocket API Contract: Real-time Audio Streaming

**Version**: 1.0.0  
**Feature**: Phase 1 MVP Ambient Music Streaming  
**Created**: 2024-12-26

---

## Connection Endpoint

### WebSocket URL
```
ws://localhost:8000/ws/stream
```

### Connection Handshake
```javascript
const socket = new WebSocket('ws://localhost:8000/ws/stream');
```

### Supported Browsers
- Chrome (last 3 years)
- Edge (last 3 years)
- Requires Web Audio API support

---

## Message Protocol

### Audio Chunk Message
**Type**: `audio_chunk`

**Schema**:
```json
{
  "type": "audio_chunk",
  "timestamp": 1703123456789,
  "sequence": 12345,
  "sample_rate": 44100,
  "format": "pcm16",
  "data": "base64_encoded_audio_string"
}
```

**Field Descriptions**:
- `type`: Message type identifier
- `timestamp`: Unix timestamp in milliseconds for synchronization
- `sequence`: Sequential packet number (starts at 0, increments by 1)
- `sample_rate`: Audio sampling rate (always 44100 for MVP)
- `format`: Audio format identifier (always "pcm16" for MVP)
- `data`: Base64-encoded 16-bit PCM audio data

**Constraints**:
- Base64 data must decode to exactly 8,820 bytes (100ms × 44.1kHz × 2 bytes)
- Sequence numbers must be monotonically increasing
- Timestamps must be within ±100ms of server generation time

### Control Message
**Type**: `stream_control`

**Schema**:
```json
{
  "type": "stream_control",
  "action": "start|stop|pause|resume",
  "timestamp": 1703123456789,
  "parameters": {
    "buffer_ms": 100,
    "sample_rate": 44100
  }
}
```

**Field Descriptions**:
- `action`: Control action to perform
- `timestamp`: Unix timestamp for synchronization
- `parameters`: Optional control parameters

### Error Message
**Type**: `error`

**Schema**:
```json
{
  "type": "error",
  "code": "GPU_UNAVAILABLE|BUFFER_ERROR|SYNTHESIS_ERROR",
  "message": "Human-readable error description",
  "timestamp": 1703123456789,
  "recoverable": true
}
```

**Error Codes**:
- `GPU_UNAVAILABLE`: GPU acceleration failed, falling back to CPU
- `BUFFER_ERROR`: Ring buffer underflow/overflow detected
- `SYNTHESIS_ERROR`: Audio synthesis failed, using fallback phrase
- `CONNECTION_ERROR`: WebSocket connection issues
- `BROWSER_ERROR`: Browser compatibility problem

---

## Client Requirements

### WebSocket Events
```javascript
socket.onmessage = function(event) {
  const message = JSON.parse(event.data);
  
  switch(message.type) {
    case 'audio_chunk':
      handleAudioChunk(message);
      break;
    case 'error':
      handleError(message);
      break;
    case 'stream_control':
      handleControl(message);
      break;
  }
};

socket.onerror = function(event) {
  // Handle connection errors
};

socket.onclose = function(event) {
  // Handle connection drops with reconnection logic
};
```

### Audio Processing
```javascript
function handleAudioChunk(message) {
  // Decode base64 to Int16Array
  const base64Data = message.data;
  const binaryString = atob(base64Data);
  const uint8Array = new Uint8Array(binaryString.length);
  
  for (let i = 0; i < binaryString.length; i++) {
    uint8Array[i] = binaryString.charCodeAt(i);
  }
  
  const int16Array = new Int16Array(uint8Array.buffer);
  
  // Convert to Float32 for Web Audio API
  const float32Array = new Float32Array(int16Array.length);
  for (let i = 0; i < int16Array.length; i++) {
    float32Array[i] = int16Array[i] / 32768.0;
  }
  
  // Schedule playback with AudioContext
  const audioBuffer = audioContext.createBuffer(1, float32Array.length, 44100);
  audioBuffer.getChannelData(0).set(float32Array);
  
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);
  source.start(audioContext.currentTime + 0.05); // 50ms buffer
}
```

---

## Server Implementation Guidelines

### Streaming Loop
```python
async def streaming_loop():
    """Continuous audio streaming to connected clients."""
    sequence = 0
    
    while True:
        # Read from ring buffer (100ms chunk)
        chunk_samples = 4410  # 44.1kHz × 100ms
        audio_chunk = ring_buffer.read(chunk_samples)
        
        if audio_chunk is not None:
            # Convert to 16-bit PCM
            audio_int16 = (np.clip(audio_chunk, -1, 1) * 32767).astype(np.int16)
            
            # Base64 encode
            encoded_data = base64.b64encode(audio_int16.tobytes()).decode()
            
            # Create message
            message = {
                "type": "audio_chunk",
                "timestamp": int(time.time() * 1000),
                "sequence": sequence,
                "sample_rate": 44100,
                "format": "pcm16",
                "data": encoded_data
            }
            
            # Send to client
            await websocket.send_json(message)
            sequence += 1
        
        await asyncio.sleep(0.1)  # 100ms interval
```

### Error Handling
```python
async def handle_synthesis_error(error):
    """Handle synthesis errors with graceful fallback."""
    
    # Generate silence chunk
    silence = np.zeros(4410, dtype=np.float32)
    
    # Send error message
    error_message = {
        "type": "error",
        "code": "SYNTHESIS_ERROR",
        "message": str(error),
        "timestamp": int(time.time() * 1000),
        "recoverable": True
    }
    await websocket.send_json(error_message)
    
    # Resume with fallback generation
    await resume_synth_fallback()
```

---

## Performance Requirements

### Timing Constraints
- **Message Interval**: Exactly 100ms between audio chunks
- **Message Size**: ~12KB per audio chunk (including JSON overhead)
- **Latency Target**: <50ms network delivery time
- **Sequence Accuracy**: No gaps or duplicates in sequence numbers

### Connection Management
- **Single User**: MVP supports one WebSocket connection per server
- **Reconnection**: Exponential backoff (1s, 2s, 4s, 8s, 16s)
- **Heartbeat**: Optional ping/pong for connection health
- **Graceful Shutdown**: Proper WebSocket close on server shutdown

---

## Security Considerations

### Input Validation
- All JSON messages validated against schemas
- Base64 data must be valid and within expected size limits
- Sequence numbers checked for monotonic progression
- Timestamps validated for reasonable ranges

### Error Information
- Error messages do not expose sensitive system details
- Stack traces filtered in production logs
- Recovery capabilities clearly indicated to client