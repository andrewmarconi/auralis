# WebSocket API Contract - Performance Optimizations

**Feature**: Performance Optimizations
**Branch**: `003-performance-optimizations`
**Date**: 2025-12-26
**Version**: 1.0.0

## Overview

This document defines the WebSocket API contract for the performance-optimized audio streaming implementation. The API supports concurrent clients with per-client buffer cursors, adaptive buffering, and graceful degradation.

---

## Endpoint

### Audio Streaming WebSocket

**URL**: `ws://{host}:{port}/ws/audio/{client_id}`

**Method**: WebSocket

**Path Parameters**:
- `client_id` (string, required): Unique identifier for this client connection
  - Format: UUID v4 or alphanumeric string
  - Max length: 64 characters
  - Example: `"client_123"`, `"a1b2c3d4-e5f6-7890-abcd-ef1234567890"`

**Query Parameters**:
- `buffer_tier` (string, optional): Initial buffer tier preference
  - Values: `"minimal"`, `"normal"`, `"stable"`, `"defensive"`
  - Default: `"normal"`
  - Example: `ws://localhost:8000/ws/audio/client_123?buffer_tier=stable`

**Headers**:
- `Sec-WebSocket-Protocol` (optional): Subprotocol version
  - Value: `"auralis-v1"`

---

## Connection Lifecycle

### 1. Connection Establishment

**Client → Server**:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/audio/client_123?buffer_tier=normal');

ws.addEventListener('open', (event) => {
    console.log('WebSocket connection established');
});
```

**Server → Client** (Initial message after connection):

```json
{
  "type": "connection_established",
  "client_id": "client_123",
  "server_time": 1735228800.123,
  "buffer_config": {
    "initial_tier": "normal",
    "target_buffer_ms": 1000,
    "chunk_duration_ms": 100,
    "samples_per_chunk": 8820
  },
  "server_version": "0.3.0"
}
```

**Response Fields**:
- `type` (string): Message type (`"connection_established"`)
- `client_id` (string): Echo of the client's ID
- `server_time` (float): Unix timestamp (seconds since epoch)
- `buffer_config` (object): Initial buffer configuration
  - `initial_tier` (string): Starting buffer tier
  - `target_buffer_ms` (int): Target buffer duration in milliseconds
  - `chunk_duration_ms` (int): Duration of each audio chunk
  - `samples_per_chunk` (int): Number of stereo samples per chunk
- `server_version` (string): Server version

---

### 2. Audio Streaming

**Server → Client** (Periodic audio chunks, every ~100ms):

```json
{
  "type": "audio",
  "chunk_id": 12345,
  "data": "AQEBAQEBAQEBAQEBAQE...",
  "timestamp": 1735228801.234,
  "buffer_depth": 10,
  "current_tier": "normal"
}
```

**Message Fields**:
- `type` (string): Message type (`"audio"`)
- `chunk_id` (int): Monotonically increasing chunk identifier
- `data` (string): Base64-encoded PCM audio data
  - Format: 16-bit PCM, stereo, 44.1kHz
  - Samples per chunk: 4410 per channel (100ms)
  - Total bytes: 4410 × 2 channels × 2 bytes = 17,640 bytes
  - Base64 encoded: ~23,520 characters
- `timestamp` (float): Server-side chunk generation time (Unix timestamp)
- `buffer_depth` (int): Current server-side buffer depth for this client (chunks)
- `current_tier` (string): Current adaptive buffer tier

**Audio Data Decoding** (Client-side):

```javascript
ws.addEventListener('message', async (event) => {
    const message = JSON.parse(event.data);

    if (message.type === 'audio') {
        // Decode base64 to ArrayBuffer
        const binaryString = atob(message.data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        // Convert to Float32Array for Web Audio API
        const int16Array = new Int16Array(bytes.buffer);
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0;
        }

        // Enqueue to AudioWorklet
        await audioWorklet.enqueueChunk(float32Array);
    }
});
```

---

### 3. Buffer Status Updates

**Server → Client** (Periodic buffer health updates, every ~5 seconds):

```json
{
  "type": "buffer_status",
  "client_id": "client_123",
  "timestamp": 1735228805.678,
  "buffer_health": {
    "current_depth_chunks": 10,
    "target_depth_chunks": 10,
    "current_tier": "normal",
    "jitter_ms": 12.5,
    "underrun_rate": 0.002,
    "health_status": "healthy"
  },
  "performance": {
    "chunks_sent": 1234,
    "bytes_sent": 21758160,
    "underruns": 2,
    "connection_uptime_sec": 123.4
  }
}
```

**Message Fields**:
- `type` (string): Message type (`"buffer_status"`)
- `client_id` (string): Client identifier
- `timestamp` (float): Status snapshot time
- `buffer_health` (object): Buffer health metrics
  - `current_depth_chunks` (int): Current buffer depth
  - `target_depth_chunks` (int): Target depth for current tier
  - `current_tier` (string): Active buffer tier
  - `jitter_ms` (float): Current mean jitter (milliseconds)
  - `underrun_rate` (float): Underrun rate (0.0-1.0)
  - `health_status` (string): `"healthy"`, `"warning"`, or `"critical"`
- `performance` (object): Connection performance metrics
  - `chunks_sent` (int): Total chunks sent
  - `bytes_sent` (int): Total bytes sent
  - `underruns` (int): Total underrun events
  - `connection_uptime_sec` (float): Connection duration

---

### 4. Client Control Messages

**Client → Server** (Optional buffer control):

```json
{
  "type": "buffer_control",
  "requested_tier": "stable",
  "timestamp": 1735228806.123
}
```

**Request Fields**:
- `type` (string): Message type (`"buffer_control"`)
- `requested_tier` (string): Desired buffer tier
- `timestamp` (float): Client timestamp

**Server → Client** (Acknowledgment):

```json
{
  "type": "buffer_control_ack",
  "previous_tier": "normal",
  "new_tier": "stable",
  "timestamp": 1735228806.125
}
```

**Response Fields**:
- `type` (string): Message type (`"buffer_control_ack"`)
- `previous_tier` (string): Tier before change
- `new_tier` (string): Tier after change
- `timestamp` (float): Server timestamp

---

### 5. Error Messages

**Server → Client** (Error notification):

```json
{
  "type": "error",
  "error_code": "BUFFER_OVERRUN",
  "error_message": "Client fell too far behind, skipping to safe position",
  "severity": "warning",
  "timestamp": 1735228807.456
}
```

**Error Codes**:

| Code | Severity | Description | Client Action |
|------|----------|-------------|---------------|
| `BUFFER_OVERRUN` | warning | Client fell too far behind | Auto-corrected, continue |
| `RATE_LIMIT_EXCEEDED` | warning | Client consuming chunks too fast | Throttled, reduce rate |
| `PERSISTENT_UNDERRUNS` | error | >10 underruns detected | Disconnecting soon |
| `SERVER_OVERLOAD` | error | Server at capacity | Retry later |
| `INVALID_MESSAGE` | error | Malformed client message | Fix message format |

**Error Fields**:
- `type` (string): Message type (`"error"`)
- `error_code` (string): Machine-readable error code
- `error_message` (string): Human-readable description
- `severity` (string): `"info"`, `"warning"`, or `"error"`
- `timestamp` (float): Error occurrence time

---

### 6. Graceful Disconnection

**Server → Client** (Shutdown notification):

```json
{
  "type": "shutdown_warning",
  "reason": "Server maintenance",
  "drain_period_sec": 5.0,
  "timestamp": 1735228900.000
}
```

**Shutdown Fields**:
- `type` (string): Message type (`"shutdown_warning"`)
- `reason` (string): Shutdown reason
- `drain_period_sec` (float): Time until disconnect (seconds)
- `timestamp` (float): Warning time

**Client → Server** (Explicit disconnect):

```javascript
// Client initiates graceful disconnect
ws.close(1000, "Client closing connection");
```

**WebSocket Close Codes**:

| Code | Meaning | Initiated By |
|------|---------|--------------|
| 1000 | Normal closure | Client or Server |
| 1001 | Going away | Client navigating away |
| 1002 | Protocol error | Server (invalid message) |
| 1008 | Policy violation | Server (rate limit) |
| 1011 | Unexpected condition | Server (internal error) |

---

## Message Timing Constraints

### Server-to-Client Guarantees

1. **Audio Chunks**:
   - Delivery interval: `100ms ± 10ms` (target 100ms)
   - Jitter tolerance: `<50ms` p95
   - Max consecutive drops: 0 (99% reliability)

2. **Buffer Status**:
   - Update interval: `5 seconds ± 1 second`
   - Not sent if no active connection

3. **Error Messages**:
   - Sent immediately upon detection
   - Max latency: `<500ms`

### Client-to-Server Expectations

1. **Message Frequency**:
   - Buffer control: Max 1 request per second
   - Heartbeat (if implemented): Every 30 seconds

2. **Response Timeout**:
   - Acknowledgments: Within 100ms

---

## Rate Limiting

### Per-Client Limits

**Token Bucket Configuration**:
- Capacity: 10 chunks (1 second burst)
- Refill rate: 10 chunks/second
- Enforcement: Server-side

**Behavior on Limit Exceeded**:
1. Server sends `RATE_LIMIT_EXCEEDED` error
2. Client connection throttled (no chunk delivery until tokens refill)
3. After 3 consecutive violations: Connection closed with code 1008

**Bypass** (if implemented):
- Premium clients: 2× capacity, 2× refill rate
- Requires authentication

---

## Data Format Specifications

### Audio Chunk Binary Layout

**PCM Format**:
```
Sample Rate: 44,100 Hz
Bit Depth: 16-bit signed integer
Channels: 2 (stereo)
Endianness: Little-endian
Interleaving: LRLRLR... (left-right alternating)
```

**Chunk Structure**:
```
Chunk Duration: 100ms
Samples per channel: 4,410
Total samples: 8,820 (stereo)
Bytes per sample: 2
Total bytes: 17,640
Base64 encoded: ~23,520 characters
```

**Example Raw Data**:
```
[L0_LSB, L0_MSB, R0_LSB, R0_MSB, L1_LSB, L1_MSB, R1_LSB, R1_MSB, ...]
```

**Base64 Encoding**:
```javascript
// Server-side (Python)
import base64
chunk_bytes = audio_chunk.tobytes()  # np.int16 array
encoded = base64.b64encode(chunk_bytes).decode('utf-8')

// Client-side (JavaScript)
const binaryString = atob(encoded);
const bytes = new Uint8Array(binaryString.length);
for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
}
const int16Array = new Int16Array(bytes.buffer);
```

---

## Backward Compatibility

### Version Negotiation

**Subprotocol Header**:
```
Sec-WebSocket-Protocol: auralis-v1
```

**Server Support**:
- `auralis-v1`: Performance-optimized API (this spec)
- `auralis-v0`: Legacy API (Phase 1/2, no per-client cursors)

**Fallback Behavior**:
- If client doesn't specify subprotocol → use `auralis-v1`
- If client requests unsupported version → reject with 400 Bad Request

### Migration Path

**Breaking Changes from Phase 2**:

| Phase 2 (auralis-v0) | Phase 3 (auralis-v1) | Migration |
|----------------------|----------------------|-----------|
| Single read cursor | Per-client cursors | Client ID required in URL |
| Fixed buffer | Adaptive buffer tiers | Auto-managed, transparent |
| No buffer status | Periodic status updates | Client can ignore |
| Sequential encoding | Broadcast encoding | Transparent to client |

**Non-Breaking Changes**:
- Audio data format: Unchanged
- Chunk timing: Unchanged
- Close codes: Unchanged

---

## Security Considerations

### Authentication

**Optional JWT Token** (if implemented):

```
GET /ws/audio/client_123?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Token Validation**:
- Signature verification using secret key
- Expiration check
- Client ID claim must match URL parameter

### Rate Limiting

**Per-IP Limits** (to prevent DoS):
- Max 5 concurrent connections per IP
- Max 10 connection attempts per minute

**Action on Violation**:
- Reject new connections with 429 Too Many Requests
- Existing connections unaffected

### Input Validation

**Client ID**:
- Max length: 64 characters
- Allowed characters: `[a-zA-Z0-9_-]`
- Reserved IDs: `"admin"`, `"system"`, `"test"`

**Message Validation**:
- Max message size: 1 KB (client → server)
- Reject messages with unknown `type`
- Validate JSON schema

---

## Error Handling

### Client-Side Recommendations

**Connection Loss**:
```javascript
ws.addEventListener('close', (event) => {
    if (event.code === 1000) {
        console.log('Normal closure');
    } else {
        console.error('Unexpected close:', event.code, event.reason);
        // Retry with exponential backoff
        setTimeout(() => reconnect(), Math.min(1000 * Math.pow(2, retryCount), 30000));
    }
});
```

**Message Processing Errors**:
```javascript
ws.addEventListener('message', (event) => {
    try {
        const message = JSON.parse(event.data);
        handleMessage(message);
    } catch (error) {
        console.error('Failed to process message:', error);
        // Log to error tracking service, but don't disconnect
    }
});
```

**Buffer Underruns**:
```javascript
// If client-side audio buffer underruns detected
function handleUnderrun() {
    // Request higher buffer tier
    ws.send(JSON.stringify({
        type: 'buffer_control',
        requested_tier: 'stable',
        timestamp: Date.now() / 1000
    }));
}
```

### Server-Side Error Recovery

**Synthesis Failure**:
1. Log error with context
2. Send empty chunk (zeros) to prevent underrun
3. Attempt recovery on next chunk
4. If 3 consecutive failures: Disconnect client with code 1011

**Memory Pressure**:
1. Trigger garbage collection
2. Clear GPU cache
3. Temporarily reduce max concurrent clients
4. Send `SERVER_OVERLOAD` error to new connections

---

## Performance SLAs

### Latency Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Chunk delivery interval | 100ms ± 10ms | Server-to-client time |
| Chunk jitter (p95) | <50ms | Deviation from expected time |
| Chunk jitter (p99) | <100ms | Worst-case deviation |
| Total audio latency | <300ms | Generation + buffering + transmission |

### Reliability Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Chunk delivery success | 99% | Chunks delivered on time |
| Concurrent clients | 10+ | Without degradation |
| Connection uptime | 99.9% | Excluding planned maintenance |
| Buffer underruns | <1 per hour | Per client |

### Resource Constraints

| Resource | Limit | Enforcement |
|----------|-------|-------------|
| Max concurrent clients | 20 | Reject new connections |
| Max connection duration | Unlimited | - |
| Max bandwidth per client | ~180 KB/s | Audio stream only |
| Total server bandwidth | ~3.6 MB/s | 20 clients × 180 KB/s |

---

## Testing Scenarios

### 1. Nominal Operation

**Setup**:
- Single client connects
- Stable network (no packet loss, <10ms jitter)

**Expected Behavior**:
- Connection established within 100ms
- Audio chunks arrive every 100ms ± 5ms
- Buffer tier remains "normal"
- Zero underruns

**Validation**:
```javascript
// Client-side test
let chunkCount = 0;
let lastChunkTime = null;

ws.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'audio') {
        const now = Date.now();
        if (lastChunkTime) {
            const interval = now - lastChunkTime;
            assert(interval >= 90 && interval <= 110, 'Chunk interval within 100ms ± 10ms');
        }
        lastChunkTime = now;
        chunkCount++;
    }
});

// After 10 seconds
assert(chunkCount >= 95 && chunkCount <= 105, '~100 chunks received in 10 seconds');
```

### 2. High Network Jitter

**Setup**:
- Single client connects
- Simulated network jitter: ±50ms

**Expected Behavior**:
- Buffer tier escalates to "stable" or "defensive"
- Jitter metrics reflect network conditions
- Zero underruns (adaptive buffer compensates)

**Validation**:
```python
# Server-side test
async def test_jitter_adaptation():
    client = await connect_with_jitter(jitter_ms=50)
    await asyncio.sleep(10)  # Allow tier adjustment

    status = await client.get_buffer_status()
    assert status['current_tier'] in ['stable', 'defensive']
    assert status['underrun_rate'] < 0.01
```

### 3. Concurrent Clients

**Setup**:
- 10 clients connect simultaneously
- Stable network

**Expected Behavior**:
- All clients receive chunks every 100ms ± 10ms
- No client experiences >5% latency increase
- Server CPU usage <90%

**Validation**:
```python
async def test_concurrent_clients():
    clients = [await connect_client(f'client_{i}') for i in range(10)]

    # Measure latency for each client
    latencies = await asyncio.gather(*[measure_latency(c) for c in clients])

    for latency in latencies:
        assert latency < 105, 'Latency <105ms for all clients'
```

### 4. Graceful Shutdown

**Setup**:
- Client connected and streaming
- Server initiates shutdown

**Expected Behavior**:
- Client receives `shutdown_warning` message
- Drain period: 5 seconds
- All buffered chunks delivered before disconnect
- WebSocket closed with code 1000

**Validation**:
```javascript
ws.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'shutdown_warning') {
        console.log('Shutdown in', message.drain_period_sec, 'seconds');
        // Prepare to reconnect after drain period
    }
});

ws.addEventListener('close', (event) => {
    assert(event.code === 1000, 'Normal closure');
});
```

---

## API Changelog

### Version 1.0.0 (2025-12-26) - Performance Optimizations

**Added**:
- Per-client buffer cursors
- Adaptive buffer tier system
- Buffer status updates
- Client buffer control messages
- Graceful shutdown notifications
- Rate limiting with token bucket

**Changed**:
- Client ID now required in WebSocket URL
- Audio messages include `buffer_depth` and `current_tier` fields
- Error messages expanded with severity levels

**Deprecated**:
- None (first optimized version)

**Removed**:
- None

**Fixed**:
- N/A (new API)

**Security**:
- Added per-IP connection limits
- Enhanced input validation

---

## References

- WebSocket Protocol: [RFC 6455](https://tools.ietf.org/html/rfc6455)
- WebRTC Jitter Buffer: [RFC 3550](https://tools.ietf.org/html/rfc3550)
- Base64 Encoding: [RFC 4648](https://tools.ietf.org/html/rfc4648)
- PCM Audio Format: [WAVE PCM soundfile format](http://soundfile.sapp.org/doc/WaveFormat/)

---

## Contact

For API questions or issues:
- GitHub Issues: https://github.com/yourusername/auralis/issues
- Documentation: https://github.com/yourusername/auralis/docs
