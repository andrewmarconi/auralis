# Technical Specifications

## 1. Hardware Detection & Backend Selection

### 1.1 Device Detection Strategy

```python
# auralis/core/device_manager.py
import torch
import platform
from typing import Literal, Tuple
from loguru import logger

DeviceType = Literal["mps", "cuda", "cpu"]

class DeviceManager:
    """Manages hardware detection and optimal backend selection."""

    @staticmethod
    def detect_optimal_device() -> Tuple[DeviceType, str]:
        """
        Detect and return optimal compute device.

        Priority:
        1. Apple Metal (MPS) on macOS with Apple Silicon
        2. NVIDIA CUDA on Linux/Windows with compatible GPU
        3. CPU fallback

        Returns:
            Tuple of (device_type, device_string)
            Examples: ("mps", "mps"), ("cuda", "cuda:0"), ("cpu", "cpu")
        """
        # Check for Apple Metal (M1/M2/M3/M4)
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            logger.info("Using Apple Metal (MPS) backend")
            return "mps", "mps"

        # Check for NVIDIA CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA backend: {device_name} ({device_count} devices)")
            return "cuda", "cuda:0"

        # CPU fallback
        logger.warning("No GPU detected. Using CPU backend (performance will be degraded)")
        return "cpu", "cpu"

    @staticmethod
    def get_device_info(device: str) -> dict:
        """Get detailed device information."""
        info = {
            "device": device,
            "platform": platform.system(),
            "python_version": platform.python_version(),
        }

        if device == "mps":
            info.update({
                "backend": "Metal Performance Shaders",
                "available": torch.backends.mps.is_available(),
            })
        elif device.startswith("cuda"):
            info.update({
                "backend": "NVIDIA CUDA",
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
            })
        else:
            info.update({
                "backend": "CPU",
                "cpu_count": torch.get_num_threads(),
            })

        return info
```

### 1.2 Performance Targets by Backend

| Backend | Target Real-Time Factor | Latency (8 bars) | Concurrent Clients |
|---------|------------------------|------------------|-------------------|
| **Apple MPS (M4)** | > 50× | < 500ms | 10+ |
| **NVIDIA CUDA (RTX 3080)** | > 100× | < 250ms | 20+ |
| **CPU (8-core)** | > 5× | < 5s | 2-3 |

### 1.3 Fallback Strategy

1. **Primary**: Try torchsynth on detected GPU
2. **Fallback 1**: Try torchsynth on CPU (degraded performance)
3. **Fallback 2**: Use pedalboard with pre-rendered samples
4. **Fallback 3**: Stream silence + log critical error

---

## 2. Audio Format Specification

### 2.1 PCM Format

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample Rate | 44,100 Hz | Standard CD quality |
| Bit Depth | 16-bit signed integer | Range: -32768 to 32767 |
| Channels | 2 (Stereo) | Left and Right |
| Byte Order | Little-endian | Standard for x86/ARM |
| Interleaving | LRLRLR... | Channel samples alternating |

### 2.2 Encoding Pipeline

```
Float32 [-1.0, 1.0]  (synthesis output)
    ↓
Clip to [-1.0, 1.0]  (safety)
    ↓
Multiply by 32767    (scale to int16 range)
    ↓
Convert to int16     (quantize)
    ↓
Interleave L/R       (LRLRLR...)
    ↓
Little-endian bytes  (platform standard)
    ↓
Base64 encode        (for WebSocket JSON transport)
```

### 2.3 Chunk Specification

| Parameter | Value | Calculation |
|-----------|-------|-------------|
| Chunk Duration | 100ms | Balances latency vs overhead |
| Samples per Chunk | 4,410 samples/channel | 44,100 Hz × 0.1s |
| Stereo Samples | 8,820 samples | 4,410 × 2 channels |
| Bytes (int16) | 17,640 bytes | 8,820 × 2 bytes |
| Base64 Encoded | ~23,520 chars | 17,640 × 4/3 |

### 2.4 Sample Rate Mismatch Handling

**Server**: Always outputs 44,100 Hz

**Client**:
- If client AudioContext sample rate ≠ 44,100 Hz:
  - Use linear interpolation for resampling
  - Example: 48 kHz client → upsample by 48/44.1 = 1.088

```javascript
// Client-side resampling
function resample(input, inputRate, outputRate) {
    const ratio = outputRate / inputRate;
    const outputLength = Math.floor(input.length * ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
        const srcIndex = i / ratio;
        const srcIndexFloor = Math.floor(srcIndex);
        const t = srcIndex - srcIndexFloor;

        // Linear interpolation
        const sample1 = input[srcIndexFloor] || 0;
        const sample2 = input[srcIndexFloor + 1] || 0;
        output[i] = sample1 * (1 - t) + sample2 * t;
    }

    return output;
}
```

---

## 3. Music Theory Mappings

### 3.1 Key to Root MIDI Note Mapping

```python
# auralis/music/theory.py
from typing import Dict

KEY_TO_ROOT_MIDI: Dict[str, int] = {
    # Minor keys (preferred for ambient)
    "A minor": 57,   # A3
    "B minor": 59,   # B3
    "C minor": 60,   # C4
    "D minor": 62,   # D4
    "E minor": 64,   # E4
    "F minor": 65,   # F4
    "G minor": 67,   # G4

    # Major keys (optional)
    "C major": 60,   # C4
    "D major": 62,   # D4
    "E major": 64,   # E4
    "F major": 65,   # F4
    "G major": 67,   # G4
    "A major": 57,   # A3
    "B major": 59,   # B3
}

# Scale intervals (semitones from root)
SCALE_INTERVALS: Dict[str, list[int]] = {
    "aeolian": [0, 2, 3, 5, 7, 8, 10],      # Natural minor
    "dorian": [0, 2, 3, 5, 7, 9, 10],       # Dorian mode
    "phrygian": [0, 1, 3, 5, 7, 8, 10],     # Phrygian mode
    "major": [0, 2, 4, 5, 7, 9, 11],        # Ionian/major
}

# Chord type to intervals
CHORD_INTERVALS: Dict[str, list[int]] = {
    "i": [0, 3, 7],          # Minor triad
    "ii": [0, 3, 7],         # Minor triad
    "III": [0, 4, 7],        # Major triad
    "iv": [0, 3, 7],         # Minor triad
    "v": [0, 3, 7],          # Minor triad (ambient often uses minor v)
    "V": [0, 4, 7],          # Major triad
    "VI": [0, 4, 7],         # Major triad
    "VII": [0, 4, 7],        # Major triad

    # Extended chords (optional)
    "i7": [0, 3, 7, 10],     # Minor 7th
    "imaj7": [0, 3, 7, 11],  # Minor major 7th
    "VI9": [0, 4, 7, 10, 14], # Major 9th
}

def get_scale_notes(root_midi: int, scale_type: str = "aeolian") -> list[int]:
    """Get all MIDI notes in a scale across 2 octaves."""
    intervals = SCALE_INTERVALS.get(scale_type, SCALE_INTERVALS["aeolian"])
    notes = []
    for octave_offset in range(3):  # 3 octaves
        for interval in intervals:
            notes.append(root_midi + octave_offset * 12 + interval)
    return notes
```

### 3.2 Chord Progression Transition Matrix (Aeolian/Minor)

```python
import numpy as np

# State space: [i, ii, III, iv, v, VI, VII]
# Optimized for ambient minor key progressions
AMBIENT_MINOR_TRANSITIONS = np.array([
    #     i     ii    III   iv    v     VI    VII
    [0.15, 0.05, 0.10, 0.30, 0.05, 0.30, 0.05],  # From i
    [0.25, 0.10, 0.05, 0.15, 0.25, 0.15, 0.05],  # From ii
    [0.20, 0.05, 0.15, 0.10, 0.10, 0.30, 0.10],  # From III
    [0.25, 0.10, 0.10, 0.10, 0.20, 0.15, 0.10],  # From iv
    [0.40, 0.10, 0.05, 0.20, 0.05, 0.15, 0.05],  # From v (often resolves to i)
    [0.25, 0.05, 0.20, 0.15, 0.05, 0.10, 0.20],  # From VI
    [0.30, 0.05, 0.25, 0.10, 0.05, 0.20, 0.05],  # From VII
])

# Verify rows sum to 1.0
assert np.allclose(AMBIENT_MINOR_TRANSITIONS.sum(axis=1), 1.0)
```

### 3.3 BPM to Duration Calculations

```python
def bpm_to_bar_duration(bpm: int, time_signature: tuple[int, int] = (4, 4)) -> float:
    """
    Convert BPM to bar duration in seconds.

    Args:
        bpm: Beats per minute
        time_signature: (numerator, denominator), e.g., (4, 4) = 4/4 time

    Returns:
        Duration of one bar in seconds
    """
    beats_per_bar = time_signature[0]
    seconds_per_beat = 60.0 / bpm
    return beats_per_bar * seconds_per_beat

# Examples:
# 70 BPM, 4/4 time: (60/70) * 4 = 3.43 seconds per bar
# 120 BPM, 4/4 time: (60/120) * 4 = 2.0 seconds per bar
```

---

## 4. WebSocket Protocol Specification

### 4.1 Message Schemas

#### Server → Client Messages

**Audio Chunk**
```json
{
  "type": "audio",
  "data": "base64_encoded_pcm_string",
  "timestamp": "2025-01-15T10:30:45.123456Z",
  "chunk_index": 12345,
  "sample_rate": 44100,
  "channels": 2,
  "format": "int16_le"
}
```

**Server Status**
```json
{
  "type": "status",
  "buffer_depth_ms": 150.5,
  "active_clients": 3,
  "synthesis_latency_ms": 45.2,
  "server_time": "2025-01-15T10:30:45.123456Z"
}
```

**Error**
```json
{
  "type": "error",
  "code": "SYNTHESIS_FAILED",
  "message": "Synthesis engine error: GPU memory exhausted",
  "severity": "warning",
  "timestamp": "2025-01-15T10:30:45.123456Z"
}
```

**Generation Update**
```json
{
  "type": "generation_update",
  "phrase_id": "phrase_789",
  "key": "A minor",
  "bpm": 70,
  "intensity": 0.6,
  "next_phrase_in_ms": 18500
}
```

#### Client → Server Messages

**Control Update**
```json
{
  "type": "control",
  "key": "A minor",
  "bpm": 75,
  "intensity": 0.7,
  "timestamp": "2025-01-15T10:30:45.123456Z"
}
```

**Heartbeat (Keepalive)**
```json
{
  "type": "ping",
  "client_time": "2025-01-15T10:30:45.123456Z"
}
```

**Server responds with:**
```json
{
  "type": "pong",
  "client_time": "2025-01-15T10:30:45.123456Z",
  "server_time": "2025-01-15T10:30:45.150000Z"
}
```

**Status Request**
```json
{
  "type": "status_request"
}
```

### 4.2 Connection Lifecycle

```
CLIENT                          SERVER
  |                               |
  |-------- WS Connect ---------->|
  |<------- Accept (101) ---------|
  |                               |
  |<------ Audio chunks --------- | (every 100ms)
  |                               |
  |---- Control message -------->|
  |<-- Generation update --------|
  |                               |
  |-------- Ping -------------->|
  |<------- Pong ---------------|
  |                               |
  |-------- Close ------------->|
  |<------- Close (1000) --------|
```

### 4.3 Error Codes

| Code | Severity | Meaning | Client Action |
|------|----------|---------|---------------|
| `SYNTHESIS_FAILED` | warning | Temporary synthesis error | Continue, expect silence |
| `BUFFER_UNDERRUN` | warning | Server buffer empty | Expect audio gaps |
| `BUFFER_OVERRUN` | info | Client too slow | Speed up playback rate |
| `INVALID_CONTROL` | error | Malformed control message | Check parameter types |
| `GENERATION_ERROR` | error | Composition failed | Continue with previous settings |
| `SERVER_OVERLOAD` | critical | Too many clients | Disconnect and retry later |

### 4.4 Heartbeat/Keepalive

- **Interval**: Every 30 seconds
- **Timeout**: 60 seconds (disconnect if no response)
- **Purpose**: Detect network failures, prevent idle disconnects

---

## 5. Multi-Client Architecture

### 5.1 Isolation Strategy

**Shared Components** (All clients receive same audio):
- Single synthesis engine
- Single composition engine
- Single ring buffer
- Global generation state (key, BPM, intensity)

**Per-Client Components**:
- WebSocket connection
- Audio send queue (5-chunk buffer)
- Statistics tracking (chunks sent, dropped)
- Read cursor into ring buffer

### 5.2 Control Parameter Behavior

**Current Implementation** (Phase 1-3):
- Control changes affect ALL clients
- Last write wins
- Changes apply to next phrase (~20s latency)

**Future Enhancement** (Phase 5+):
- Per-client isolated streams
- Independent key/BPM/intensity per client
- Requires multiple synthesis engines

### 5.3 Client Limits

| Configuration | Max Clients | Notes |
|--------------|-------------|-------|
| **Development** (CPU) | 2 | Limited by CPU synthesis speed |
| **Production** (M4 Mac MPS) | 10 | Bandwidth-limited (~500 kbps/client) |
| **Production** (NVIDIA GPU) | 20+ | GPU can handle it, network is limit |

### 5.4 Resource Management

```python
# server/client_manager.py
from typing import Dict, Set
import asyncio
from loguru import logger

class ClientManager:
    """Manages connected clients and enforces limits."""

    MAX_CLIENTS = 10  # Configurable via env var

    def __init__(self):
        self.clients: Dict[str, dict] = {}
        self.client_limit_lock = asyncio.Lock()

    async def register_client(self, client_id: str) -> bool:
        """
        Register a new client connection.

        Returns:
            True if client was accepted, False if limit reached
        """
        async with self.client_limit_lock:
            if len(self.clients) >= self.MAX_CLIENTS:
                logger.warning(f"Client limit reached ({self.MAX_CLIENTS}), rejecting {client_id}")
                return False

            self.clients[client_id] = {
                "connected_at": asyncio.get_event_loop().time(),
                "chunks_sent": 0,
                "chunks_dropped": 0,
            }

            logger.info(f"Client {client_id} registered ({len(self.clients)}/{self.MAX_CLIENTS})")
            return True

    async def unregister_client(self, client_id: str):
        """Remove client on disconnect."""
        async with self.client_limit_lock:
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"Client {client_id} unregistered ({len(self.clients)} remaining)")
```

---

## 6. Configuration & Environment Variables

### 6.1 Environment Variables

Create `.env` file in project root:

```bash
# Server Configuration
AURALIS_HOST=0.0.0.0
AURALIS_PORT=8000
AURALIS_ENV=development  # development | production

# Audio Configuration
AURALIS_SAMPLE_RATE=44100
AURALIS_CHUNK_DURATION_MS=100
AURALIS_BUFFER_CAPACITY_SEC=2.0

# Hardware Configuration
AURALIS_DEVICE=auto  # auto | mps | cuda | cpu
AURALIS_TORCH_THREADS=8  # CPU thread count

# Generation Defaults
AURALIS_DEFAULT_KEY="A minor"
AURALIS_DEFAULT_BPM=70
AURALIS_DEFAULT_INTENSITY=0.5

# Client Limits
AURALIS_MAX_CLIENTS=10

# Feature Flags
AURALIS_ENABLE_OPUS=false  # Enable Opus compression (Phase 3)
AURALIS_ENABLE_TRANSFORMER=false  # Use transformer melody gen (Phase 2)
AURALIS_ENABLE_EFFECTS=false  # Enable reverb/delay (Phase 3)

# Logging
AURALIS_LOG_LEVEL=INFO  # DEBUG | INFO | WARNING | ERROR
AURALIS_LOG_FILE=logs/auralis.log

# Security (Production)
AURALIS_CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
AURALIS_ENABLE_SSL=false
AURALIS_SSL_CERT_PATH=/path/to/cert.pem
AURALIS_SSL_KEY_PATH=/path/to/key.pem
```

### 6.2 Configuration Loading

```python
# auralis/core/config.py
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Literal

class AuralisConfig(BaseSettings):
    """Application configuration with validation."""

    # Server
    host: str = Field(default="0.0.0.0", env="AURALIS_HOST")
    port: int = Field(default=8000, env="AURALIS_PORT")
    env: Literal["development", "production"] = Field(default="development", env="AURALIS_ENV")

    # Audio
    sample_rate: int = Field(default=44100, env="AURALIS_SAMPLE_RATE")
    chunk_duration_ms: int = Field(default=100, env="AURALIS_CHUNK_DURATION_MS")
    buffer_capacity_sec: float = Field(default=2.0, env="AURALIS_BUFFER_CAPACITY_SEC")

    # Hardware
    device: Literal["auto", "mps", "cuda", "cpu"] = Field(default="auto", env="AURALIS_DEVICE")
    torch_threads: int = Field(default=8, env="AURALIS_TORCH_THREADS")

    # Generation
    default_key: str = Field(default="A minor", env="AURALIS_DEFAULT_KEY")
    default_bpm: int = Field(default=70, env="AURALIS_DEFAULT_BPM")
    default_intensity: float = Field(default=0.5, env="AURALIS_DEFAULT_INTENSITY")

    # Limits
    max_clients: int = Field(default=10, env="AURALIS_MAX_CLIENTS")

    # Features
    enable_opus: bool = Field(default=False, env="AURALIS_ENABLE_OPUS")
    enable_transformer: bool = Field(default=False, env="AURALIS_ENABLE_TRANSFORMER")
    enable_effects: bool = Field(default=False, env="AURALIS_ENABLE_EFFECTS")

    # Logging
    log_level: str = Field(default="INFO", env="AURALIS_LOG_LEVEL")
    log_file: str = Field(default="logs/auralis.log", env="AURALIS_LOG_FILE")

    # Security
    cors_origins: list[str] = Field(default=["http://localhost:3000"], env="AURALIS_CORS_ORIGINS")
    enable_ssl: bool = Field(default=False, env="AURALIS_ENABLE_SSL")
    ssl_cert_path: str | None = Field(default=None, env="AURALIS_SSL_CERT_PATH")
    ssl_key_path: str | None = Field(default=None, env="AURALIS_SSL_KEY_PATH")

    @validator("default_bpm")
    def validate_bpm(cls, v):
        if not 40 <= v <= 200:
            raise ValueError("BPM must be between 40 and 200")
        return v

    @validator("default_intensity")
    def validate_intensity(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global config instance
config = AuralisConfig()
```

---

## 7. Parameter Validation & API Specification

### 7.1 Control Parameter Constraints

```python
# auralis/api/validation.py
from pydantic import BaseModel, Field, validator
from typing import Literal

class ControlParameters(BaseModel):
    """Validated control parameters from client."""

    key: str = Field(
        default="A minor",
        description="Musical key for generation",
        examples=["A minor", "C minor", "D minor"]
    )

    bpm: int = Field(
        default=70,
        ge=40,
        le=200,
        description="Beats per minute (tempo)"
    )

    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Generation intensity/density (0=sparse, 1=dense)"
    )

    @validator("key")
    def validate_key(cls, v):
        from auralis.music.theory import KEY_TO_ROOT_MIDI
        if v not in KEY_TO_ROOT_MIDI:
            raise ValueError(
                f"Invalid key: {v}. Must be one of: {list(KEY_TO_ROOT_MIDI.keys())}"
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "key": "A minor",
                "bpm": 70,
                "intensity": 0.6
            }
        }

class GenerationStatus(BaseModel):
    """Server generation status response."""

    current_key: str
    current_bpm: int
    current_intensity: float
    buffer_depth_ms: float
    active_clients: int
    synthesis_latency_ms: float
    phrases_generated: int
```

### 7.2 REST API Endpoints

```python
# server/api/routes.py
from fastapi import APIRouter, HTTPException
from auralis.api.validation import ControlParameters, GenerationStatus

router = APIRouter(prefix="/api/v1", tags=["control"])

@router.post("/control", response_model=dict)
async def update_control(params: ControlParameters):
    """
    Update generation parameters.

    Changes take effect on the next phrase (~20 seconds).
    """
    try:
        await composition_engine.update_params(params.dict())
        return {"status": "updated", "params": params.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=GenerationStatus)
async def get_status():
    """Get current server status and generation parameters."""
    return GenerationStatus(
        current_key=composition_engine.current_key,
        current_bpm=composition_engine.current_bpm,
        current_intensity=composition_engine.current_intensity,
        buffer_depth_ms=ring_buffer.buffer_depth_ms(),
        active_clients=len(client_manager.clients),
        synthesis_latency_ms=metrics.avg_synthesis_latency_ms,
        phrases_generated=metrics.total_phrases_generated,
    )

@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "auralis"}
```

---

This specification provides concrete, implementable details for the technical gaps identified. Would you like me to continue with the remaining specifications?
