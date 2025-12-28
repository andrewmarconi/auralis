# API Contracts: FluidSynth Integration

**Feature**: FluidSynth Sample-Based Instrument Synthesis
**Branch**: `004-fluidsynth-integration`

## Summary

**No API contract changes required for this feature.**

The FluidSynth integration is a purely **internal synthesis layer modification**. All existing API endpoints and WebSocket protocols remain unchanged.

---

## Rationale

This feature replaces the **synthesis implementation** (oscillator-based → sample-based) while preserving all external interfaces:

### Unchanged APIs

#### 1. WebSocket Streaming Endpoint

**Endpoint**: `/stream` (WebSocket)

**No Changes**:
- Stream initiation protocol unchanged
- Audio chunk format unchanged (base64-encoded PCM, 100ms chunks)
- Chunk metadata structure unchanged
- Client integration unchanged

**Why**: FluidSynth generates the same stereo PCM output format (44.1kHz, 16-bit) as existing PyTorch synthesis. The streaming infrastructure is format-agnostic.

#### 2. REST API Control Endpoints

**Endpoints**:
- `GET /api/status` - Server health check
- `POST /api/control` - Playback control (start/stop)
- `GET /api/metrics` - Prometheus metrics

**No Changes**:
- Request/response schemas unchanged
- Control commands unchanged (start, stop, get status)
- Metrics exposed remain the same

**Why**: Control plane is independent of synthesis method. FluidSynth changes only affect audio rendering, not API contracts.

#### 3. Configuration Endpoints (if any)

**No Changes**:
- Environment variables (`.env`) may add new optional settings (e.g., `AURALIS_SOUNDFONT_DIR`), but existing vars unchanged
- No new required configuration exposed via API

---

## Internal Changes Only

The following changes are **internal implementation details** not exposed via API:

### Synthesis Layer (server/synthesis_engine.py)

**Before** (Oscillator-based):
```python
class SynthesisEngine:
    def render_phrase(chords, melody, percussion, duration_sec) -> np.ndarray:
        # PyTorch oscillator synthesis for all voices
        ...
        return stereo_audio  # (2, num_samples), float32
```

**After** (FluidSynth hybrid):
```python
class SynthesisEngine:
    def render_phrase(chords, melody, percussion, duration_sec) -> np.ndarray:
        # FluidSynth for piano/pads/choir
        # PyTorch for kicks
        ...
        return stereo_audio  # (2, num_samples), float32  <-- Same output format
```

**Key Point**: Return type and audio format **unchanged** - both return stereo float32 arrays.

### New Internal Modules (No API Exposure)

**New Files**:
- `server/fluidsynth_renderer.py` - FluidSynth audio rendering
- `server/soundfont_manager.py` - SoundFont loading/validation
- `server/voice_manager.py` - Polyphony management (optional)

**None exposed via HTTP/WebSocket**. Purely internal synthesis pipeline components.

---

## Client Impact: None

### Web Client (client/)

**No Changes Required**:
- Audio decoding unchanged (base64 PCM → AudioBuffer)
- Web Audio API integration unchanged
- UI controls unchanged (key, BPM, intensity)

**Why**: Client receives identical audio stream format. Synthesis method is transparent to the client.

### Example Client Code (Unchanged)

```javascript
// client/audio_client_worklet.js

// WebSocket message handling - NO CHANGES
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const audioChunk = base64ToFloat32Array(data.audio);  // Same format
    audioQueue.enqueue(audioChunk);  // Same processing
};

// Audio playback - NO CHANGES
workletNode.port.onmessage = (event) => {
    const samples = event.data;
    // Render to speakers (format unchanged)
};
```

---

## Testing Contract Compliance

### Integration Tests

Verify API contracts remain unchanged:

```python
# tests/integration/test_api_contracts_unchanged.py

import pytest
from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)

def test_stream_endpoint_contract_unchanged():
    """Verify /stream WebSocket returns same audio format."""
    with client.websocket_connect("/stream") as websocket:
        # Receive audio chunk
        data = websocket.receive_json()

        # Verify chunk structure unchanged
        assert "audio" in data  # Base64 PCM chunk
        assert "timestamp" in data
        assert "duration_ms" in data

        # Verify audio format unchanged
        audio_bytes = base64.b64decode(data["audio"])
        assert len(audio_bytes) == 17640  # 100ms stereo 16-bit @ 44.1kHz

        # NOTE: Sound quality improved, but format identical

def test_control_endpoint_contract_unchanged():
    """Verify /api/control request/response schema unchanged."""
    response = client.post("/api/control", json={"action": "start"})

    assert response.status_code == 200
    assert response.json() == {"status": "started"}  # Same response

def test_status_endpoint_contract_unchanged():
    """Verify /api/status response schema unchanged."""
    response = client.get("/api/status")

    assert response.status_code == 200
    data = response.json()

    # Existing fields unchanged
    assert "server_status" in data
    assert "active_streams" in data

    # New internal field (optional, backwards compatible)
    # assert "soundfont_loaded" in data  # Non-breaking addition
```

### WebSocket Protocol Test

```python
def test_websocket_audio_chunk_format():
    """Verify WebSocket audio chunks maintain backward compatibility."""
    with client.websocket_connect("/stream") as websocket:
        chunk = websocket.receive_json()

        # Decode audio
        audio_data = base64.b64decode(chunk["audio"])
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_stereo = audio_np.reshape(-1, 2).T

        # Verify format
        assert audio_stereo.shape == (2, 4410)  # Stereo, 100ms @ 44.1kHz
        assert audio_stereo.dtype == np.int16
        assert -32768 <= audio_stereo.min() <= 32767
        assert -32768 <= audio_stereo.max() <= 32767
```

---

## Documentation Updates

### README.md

**Minimal Changes**:
- Add SoundFont download instructions
- Update "How It Works" section to mention sample-based synthesis (optional)
- No breaking changes to setup/usage instructions

### API Documentation (if exists)

**No Updates Required**:
- OpenAPI/Swagger schema unchanged (if present)
- WebSocket protocol documentation unchanged

---

## Deployment Considerations

### Backwards Compatibility

**100% Backward Compatible**:
- Existing clients continue working without modification
- No API version bump required
- No migration scripts needed

### Rollout Strategy

**Safe Deployment**:
1. Deploy new server with FluidSynth integration
2. Existing clients immediately benefit from improved sound quality
3. No client-side updates required
4. No database migrations needed (no persistence changes)

### Environment Variables (Optional)

**New Optional Configuration** (backward compatible):

```bash
# .env (optional additions, non-breaking)
AURALIS_SOUNDFONT_DIR=/path/to/soundfonts  # Default: ./soundfonts/
AURALIS_FLUIDSYNTH_POLYPHONY=20           # Default: 20
AURALIS_FLUIDSYNTH_INTERPOLATION=4thorder # Default: 4thorder
```

**Existing Variables Unchanged**:
- `AURALIS_HOST`
- `AURALIS_PORT`
- `AURALIS_DEVICE` (still used for PyTorch kick synthesis)
- All other existing config

---

## Conclusion

**API Contract Status**: ✅ No Changes Required

This feature is a **transparent internal upgrade**:
- **Externally**: Same APIs, same protocols, same audio format
- **Internally**: Better sound quality via sample-based synthesis
- **Client Impact**: Zero (improved audio quality, no code changes)

**Contract Validation**: Existing API integration tests provide sufficient coverage to ensure backward compatibility.

---

**Contract Review Status**: ✅ COMPLETE
**API Breaking Changes**: NONE
**Client Migration Required**: NO
**Deployment Risk**: LOW (backward compatible)
