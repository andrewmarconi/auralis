# Internal Interfaces Contract - Performance Optimizations

**Feature**: Performance Optimizations
**Branch**: `003-performance-optimizations`
**Date**: 2025-12-26
**Version**: 1.0.0

## Overview

This document defines internal component interfaces for the performance optimization implementation. These contracts ensure loose coupling, testability, and maintainability across buffer management, GPU synthesis, WebSocket concurrency, and monitoring components.

---

## Table of Contents

1. [Buffer Management Interfaces](#1-buffer-management-interfaces)
2. [WebSocket Concurrency Interfaces](#2-websocket-concurrency-interfaces)
3. [GPU Synthesis Interfaces](#3-gpu-synthesis-interfaces)
4. [Memory Monitoring Interfaces](#4-memory-monitoring-interfaces)
5. [Performance Metrics Interfaces](#5-performance-metrics-interfaces)

---

## 1. Buffer Management Interfaces

### 1.1 IRingBuffer (Abstract Interface)

**Purpose**: Define contract for ring buffer implementations (adaptive, broadcast, etc.)

**Module**: `server/ring_buffer.py`

**Interface**:

```python
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class IRingBuffer(ABC):
    """Abstract interface for ring buffer implementations."""

    @abstractmethod
    def write_chunk(self, chunk: np.ndarray) -> int:
        """
        Write audio chunk to buffer.

        Args:
            chunk: Audio data (int16, stereo interleaved)

        Returns:
            Chunk ID (write position)

        Raises:
            BufferFullError: If buffer is full and cannot accept writes
        """
        pass

    @abstractmethod
    def read_chunk(self) -> Optional[np.ndarray]:
        """
        Read next audio chunk from buffer.

        Returns:
            Audio chunk or None if no data available

        Raises:
            BufferUnderrunError: If buffer is empty when read expected
        """
        pass

    @abstractmethod
    def get_depth(self) -> int:
        """
        Get current buffer depth.

        Returns:
            Number of chunks available for reading
        """
        pass

    @abstractmethod
    def get_capacity(self) -> int:
        """
        Get buffer capacity.

        Returns:
            Maximum number of chunks buffer can hold
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all buffered data."""
        pass
```

**Implementing Classes**:
- `AdaptiveRingBuffer`: Adaptive tier-based buffering
- `BroadcastRingBuffer`: Per-client cursor implementation
- `FixedRingBuffer`: Legacy fixed-size buffer (Phase 1/2 compatibility)

---

### 1.2 IJitterTracker (Abstract Interface)

**Purpose**: Define contract for jitter measurement implementations

**Module**: `server/buffer_management.py`

**Interface**:

```python
from abc import ABC, abstractmethod

class IJitterTracker(ABC):
    """Abstract interface for jitter tracking."""

    @abstractmethod
    def record_chunk(self, expected_time: float, actual_time: float) -> None:
        """
        Record chunk delivery timing.

        Args:
            expected_time: Expected arrival time (Unix timestamp)
            actual_time: Actual arrival time (Unix timestamp)
        """
        pass

    @abstractmethod
    def record_underrun(self) -> None:
        """Record buffer underrun event."""
        pass

    @abstractmethod
    def get_current_jitter(self) -> float:
        """
        Get current mean jitter.

        Returns:
            Mean jitter in milliseconds
        """
        pass

    @abstractmethod
    def get_jitter_std(self) -> float:
        """
        Get jitter standard deviation.

        Returns:
            Standard deviation in milliseconds
        """
        pass

    @abstractmethod
    def get_underrun_rate(self) -> float:
        """
        Get underrun rate.

        Returns:
            Fraction of chunks that underran (0.0-1.0)
        """
        pass

    @abstractmethod
    def get_recommended_buffer_ms(self, confidence: float = 0.95) -> float:
        """
        Calculate recommended buffer size.

        Args:
            confidence: Confidence level (0.95 or 0.99)

        Returns:
            Recommended buffer duration in milliseconds
        """
        pass
```

**Implementing Classes**:
- `EMAJitterTracker`: Exponential Moving Average implementation
- `WindowedJitterTracker`: Sliding window implementation
- `MockJitterTracker`: Test double for unit testing

---

### 1.3 ITokenBucket (Abstract Interface)

**Purpose**: Define contract for rate limiting implementations

**Module**: `server/rate_limiting.py`

**Interface**:

```python
from abc import ABC, abstractmethod

class ITokenBucket(ABC):
    """Abstract interface for token bucket rate limiter."""

    @abstractmethod
    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if insufficient
        """
        pass

    @abstractmethod
    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time until tokens available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds (0 if tokens available now)
        """
        pass

    @abstractmethod
    def get_token_count(self) -> float:
        """
        Get current token count.

        Returns:
            Number of tokens currently available
        """
        pass
```

**Implementing Classes**:
- `TokenBucket`: Standard token bucket algorithm
- `LeakyBucket`: Leaky bucket variant (constant rate)
- `MockTokenBucket`: Test double

---

## 2. WebSocket Concurrency Interfaces

### 2.1 IStreamingServer (Abstract Interface)

**Purpose**: Define contract for WebSocket streaming server implementations

**Module**: `server/streaming_server.py`

**Interface**:

```python
from abc import ABC, abstractmethod
from typing import Dict
from fastapi import WebSocket

class IStreamingServer(ABC):
    """Abstract interface for WebSocket streaming server."""

    @abstractmethod
    async def connect_client(self, websocket: WebSocket, client_id: str) -> None:
        """
        Register new WebSocket client.

        Args:
            websocket: FastAPI WebSocket connection
            client_id: Unique client identifier

        Raises:
            ConnectionError: If client already connected
            ValueError: If client_id invalid
        """
        pass

    @abstractmethod
    async def disconnect_client(self, client_id: str) -> None:
        """
        Unregister WebSocket client.

        Args:
            client_id: Client identifier

        Raises:
            KeyError: If client not found
        """
        pass

    @abstractmethod
    async def broadcast_chunk(self, chunk: np.ndarray) -> int:
        """
        Broadcast audio chunk to all connected clients.

        Args:
            chunk: Audio data to broadcast

        Returns:
            Number of clients that received chunk

        Raises:
            EncodingError: If chunk encoding fails
        """
        pass

    @abstractmethod
    async def send_chunk_to_client(
        self,
        client_id: str,
        chunk: np.ndarray
    ) -> bool:
        """
        Send audio chunk to specific client.

        Args:
            client_id: Target client identifier
            chunk: Audio data

        Returns:
            True if sent successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_client_count(self) -> int:
        """
        Get number of connected clients.

        Returns:
            Active client count
        """
        pass

    @abstractmethod
    def get_client_stats(self, client_id: str) -> Dict:
        """
        Get statistics for specific client.

        Args:
            client_id: Client identifier

        Returns:
            Dictionary with client statistics

        Raises:
            KeyError: If client not found
        """
        pass
```

**Implementing Classes**:
- `BroadcastStreamingServer`: Optimized broadcast implementation
- `SequentialStreamingServer`: Legacy sequential implementation
- `MockStreamingServer`: Test double

---

### 2.2 IChunkEncoder (Abstract Interface)

**Purpose**: Define contract for audio chunk encoding implementations

**Module**: `server/encoding.py`

**Interface**:

```python
from abc import ABC, abstractmethod
import numpy as np

class IChunkEncoder(ABC):
    """Abstract interface for audio chunk encoding."""

    @abstractmethod
    def encode_chunk(self, chunk: np.ndarray) -> str:
        """
        Encode audio chunk to string format.

        Args:
            chunk: Audio data (int16, stereo)

        Returns:
            Encoded string (typically base64)

        Raises:
            EncodingError: If encoding fails
        """
        pass

    @abstractmethod
    def decode_chunk(self, encoded: str) -> np.ndarray:
        """
        Decode string to audio chunk.

        Args:
            encoded: Encoded string

        Returns:
            Audio data (int16, stereo)

        Raises:
            DecodingError: If decoding fails
        """
        pass

    @abstractmethod
    def get_encoding_overhead(self) -> float:
        """
        Get encoding overhead ratio.

        Returns:
            Overhead factor (e.g., 1.33 for base64)
        """
        pass
```

**Implementing Classes**:
- `Base64ChunkEncoder`: Standard base64 encoding
- `PooledChunkEncoder`: Object-pooled base64 encoding
- `MockChunkEncoder`: Test double

---

## 3. GPU Synthesis Interfaces

### 3.1 ISynthesisEngine (Abstract Interface)

**Purpose**: Define contract for audio synthesis engine implementations

**Module**: `server/synthesis_engine.py`

**Interface**:

```python
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class ISynthesisEngine(ABC):
    """Abstract interface for audio synthesis engine."""

    @abstractmethod
    def render_phrase(
        self,
        chords: List[Tuple[int, int, str]],
        melody: List[Tuple[int, int, float, float]],
        duration_sec: float
    ) -> np.ndarray:
        """
        Render musical phrase to audio.

        Args:
            chords: List of (onset_sample, root_midi, chord_type)
            melody: List of (onset_sample, pitch_midi, velocity, duration_sec)
            duration_sec: Total phrase duration

        Returns:
            Stereo audio array (2, num_samples), int16

        Raises:
            SynthesisError: If rendering fails
        """
        pass

    @abstractmethod
    def get_device_info(self) -> Dict:
        """
        Get GPU device information.

        Returns:
            Dictionary with device details (type, name, memory, etc.)
        """
        pass

    @abstractmethod
    def get_render_stats(self) -> Dict:
        """
        Get rendering performance statistics.

        Returns:
            Dictionary with latency, throughput, etc.
        """
        pass

    @abstractmethod
    def clear_gpu_cache(self) -> None:
        """Clear GPU memory cache (CUDA/MPS)."""
        pass
```

**Implementing Classes**:
- `OptimizedSynthesisEngine`: Batch processing + pre-allocation
- `CompiledSynthesisEngine`: torch.compile optimization
- `StreamedSynthesisEngine`: CUDA stream optimization
- `LegacySynthesisEngine`: Phase 1/2 implementation
- `MockSynthesisEngine`: Test double

---

### 3.2 IDeviceSelector (Abstract Interface)

**Purpose**: Define contract for GPU device selection

**Module**: `server/device_management.py`

**Interface**:

```python
from abc import ABC, abstractmethod
import torch

class IDeviceSelector(ABC):
    """Abstract interface for GPU device selection."""

    @abstractmethod
    def select_device(self) -> torch.device:
        """
        Select optimal GPU device.

        Returns:
            PyTorch device (mps, cuda, or cpu)

        Priority:
            1. Metal (MPS) - Apple Silicon
            2. CUDA - NVIDIA GPUs
            3. CPU - Fallback
        """
        pass

    @abstractmethod
    def get_device_capabilities(self, device: torch.device) -> Dict:
        """
        Get device capabilities and characteristics.

        Args:
            device: PyTorch device

        Returns:
            Dictionary with memory, compute capability, etc.
        """
        pass

    @abstractmethod
    def validate_device(self, device: torch.device) -> bool:
        """
        Validate device is available and functional.

        Args:
            device: PyTorch device

        Returns:
            True if device valid
        """
        pass
```

**Implementing Classes**:
- `AutoDeviceSelector`: Automatic selection based on availability
- `ManualDeviceSelector`: User-specified device
- `MockDeviceSelector`: Test double

---

## 4. Memory Monitoring Interfaces

### 4.1 IMemoryMonitor (Abstract Interface)

**Purpose**: Define contract for memory monitoring implementations

**Module**: `server/memory_monitoring.py`

**Interface**:

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class IMemoryMonitor(ABC):
    """Abstract interface for memory monitoring."""

    @abstractmethod
    def snapshot(self, label: str = "") -> Dict:
        """
        Take memory usage snapshot.

        Args:
            label: Descriptive label for this snapshot

        Returns:
            Dictionary with memory metrics (python_mb, gpu_mb, rss_mb, etc.)
        """
        pass

    @abstractmethod
    def detect_leak(self, threshold_mb_per_hour: float = 50.0) -> bool:
        """
        Detect memory leak using linear regression.

        Args:
            threshold_mb_per_hour: Growth rate threshold

        Returns:
            True if leak detected
        """
        pass

    @abstractmethod
    def get_snapshots(self) -> List[Dict]:
        """
        Get all recorded snapshots.

        Returns:
            List of snapshot dictionaries
        """
        pass

    @abstractmethod
    def print_top_allocations(self, limit: int = 10) -> None:
        """
        Print top memory allocations by line.

        Args:
            limit: Number of top allocations to print
        """
        pass
```

**Implementing Classes**:
- `TracemallocMemoryMonitor`: tracemalloc-based implementation
- `PsutilMemoryMonitor`: psutil-based implementation
- `MockMemoryMonitor`: Test double

---

### 4.2 IGCConfig (Abstract Interface)

**Purpose**: Define contract for garbage collection configuration

**Module**: `server/gc_tuning.py`

**Interface**:

```python
from abc import ABC, abstractmethod

class IGCConfig(ABC):
    """Abstract interface for GC configuration."""

    @abstractmethod
    def configure(self) -> None:
        """Apply garbage collection configuration."""
        pass

    @abstractmethod
    def manual_collect(self) -> int:
        """
        Manually trigger garbage collection.

        Returns:
            Number of objects collected
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """
        Get GC statistics.

        Returns:
            Dictionary with collection counts, thresholds, etc.
        """
        pass
```

**Implementing Classes**:
- `RealTimeGCConfig`: Tuned for real-time audio
- `DefaultGCConfig`: Python default configuration
- `MockGCConfig`: Test double

---

## 5. Performance Metrics Interfaces

### 5.1 IMetricsCollector (Abstract Interface)

**Purpose**: Define contract for metrics collection implementations

**Module**: `server/metrics.py`

**Interface**:

```python
from abc import ABC, abstractmethod

class IMetricsCollector(ABC):
    """Abstract interface for metrics collection."""

    @abstractmethod
    def record_synthesis_latency(self, latency_sec: float) -> None:
        """
        Record synthesis latency.

        Args:
            latency_sec: Latency in seconds
        """
        pass

    @abstractmethod
    def record_buffer_depth(self, client_id: str, depth: int) -> None:
        """
        Record buffer depth for client.

        Args:
            client_id: Client identifier
            depth: Buffer depth in chunks
        """
        pass

    @abstractmethod
    def record_underrun(self, client_id: str) -> None:
        """
        Record buffer underrun event.

        Args:
            client_id: Client identifier
        """
        pass

    @abstractmethod
    def update_memory_metrics(self) -> None:
        """Update all memory-related metrics."""
        pass

    @abstractmethod
    def export_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus text exposition format
        """
        pass
```

**Implementing Classes**:
- `PrometheusMetricsCollector`: Prometheus client implementation
- `InMemoryMetricsCollector`: In-memory for testing
- `MockMetricsCollector`: Test double

---

## Component Dependencies

### Dependency Graph

```
┌─────────────────────────┐
│  FastAPI Application    │
└────────────┬────────────┘
             │
             ├──► IStreamingServer
             │    └──► IBroadcastRingBuffer
             │         ├──► IChunkEncoder
             │         └──► ITokenBucket
             │
             ├──► ISynthesisEngine
             │    ├──► IDeviceSelector
             │    └──► IMemoryMonitor
             │
             ├──► IMetricsCollector
             │
             └──► IGCConfig
```

### Dependency Injection Pattern

**Container Configuration** (`server/di_container.py`):

```python
from typing import Protocol
import torch

class DIContainer:
    """Dependency injection container for component wiring."""

    def __init__(self):
        # Core components
        self.device_selector: IDeviceSelector = AutoDeviceSelector()
        self.device: torch.device = self.device_selector.select_device()

        # Synthesis
        self.synthesis_engine: ISynthesisEngine = OptimizedSynthesisEngine(
            device=self.device
        )

        # Buffering
        self.jitter_tracker: IJitterTracker = EMAJitterTracker()
        self.ring_buffer: IRingBuffer = BroadcastRingBuffer(
            capacity_chunks=30,
            jitter_tracker=self.jitter_tracker
        )

        # Encoding
        self.chunk_encoder: IChunkEncoder = PooledChunkEncoder(pool_size=20)

        # Streaming
        self.streaming_server: IStreamingServer = BroadcastStreamingServer(
            ring_buffer=self.ring_buffer,
            chunk_encoder=self.chunk_encoder
        )

        # Monitoring
        self.memory_monitor: IMemoryMonitor = TracemallocMemoryMonitor()
        self.gc_config: IGCConfig = RealTimeGCConfig()
        self.metrics_collector: IMetricsCollector = PrometheusMetricsCollector()

        # Configure GC
        self.gc_config.configure()

# Global container instance
container = DIContainer()
```

**Usage in Application**:

```python
# server/main.py
from server.di_container import container

app = FastAPI()

@app.on_event("startup")
async def startup():
    """Initialize components from DI container."""
    await container.streaming_server.start()

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint using injected streaming server."""
    await container.streaming_server.connect_client(websocket, client_id)
```

---

## Interface Contracts

### Pre-conditions and Post-conditions

**IRingBuffer.write_chunk**:

**Pre-conditions**:
- `chunk` must be valid numpy array (int16, shape: (8820,))
- Buffer must not be full

**Post-conditions**:
- Chunk written at write_cursor position
- write_cursor advanced by 1 (mod capacity)
- Chunk ID returned equals pre-write write_cursor

**Invariants**:
- `0 <= write_cursor < capacity`
- `0 <= (write_cursor - read_cursor) % capacity <= capacity`

---

**ISynthesisEngine.render_phrase**:

**Pre-conditions**:
- `chords` non-empty list
- `melody` may be empty
- `duration_sec > 0`
- GPU device initialized

**Post-conditions**:
- Returns stereo audio array (2, num_samples)
- Audio in int16 range [-32768, 32767]
- num_samples = int(duration_sec * sample_rate)

**Invariants**:
- GPU memory allocated < device capacity
- No gradient tracking active after return

---

## Error Handling Contracts

### Exception Hierarchy

```python
class AuralisError(Exception):
    """Base exception for all Auralis errors."""
    pass

class BufferError(AuralisError):
    """Base exception for buffer-related errors."""
    pass

class BufferFullError(BufferError):
    """Raised when attempting to write to full buffer."""
    pass

class BufferUnderrunError(BufferError):
    """Raised when buffer underruns during read."""
    pass

class SynthesisError(AuralisError):
    """Base exception for synthesis-related errors."""
    pass

class DeviceError(SynthesisError):
    """Raised when GPU device initialization fails."""
    pass

class EncodingError(AuralisError):
    """Raised when chunk encoding/decoding fails."""
    pass

class ConnectionError(AuralisError):
    """Raised when WebSocket connection fails."""
    pass

class RateLimitError(AuralisError):
    """Raised when rate limit exceeded."""
    pass
```

### Error Handling Policy

**Critical Errors** (propagate to caller):
- `DeviceError`: GPU initialization failure
- `SynthesisError`: Rendering failure

**Recoverable Errors** (handle and continue):
- `BufferUnderrunError`: Log, adjust tier, continue
- `EncodingError`: Log, skip chunk, continue
- `ConnectionError`: Disconnect client, continue

**Transient Errors** (retry with backoff):
- `RateLimitError`: Wait and retry

---

## Testing Contracts

### Test Doubles (Mocks/Stubs)

**MockSynthesisEngine**:

```python
class MockSynthesisEngine(ISynthesisEngine):
    """Test double for synthesis engine."""

    def __init__(self, latency_ms: float = 50.0):
        self.latency_ms = latency_ms
        self.render_count = 0
        self.device_info = {
            "device_type": "mock",
            "device_name": "Mock GPU",
            "total_memory_mb": 1000.0
        }

    def render_phrase(self, chords, melody, duration_sec):
        """Return silent audio after simulated latency."""
        time.sleep(self.latency_ms / 1000.0)
        self.render_count += 1

        num_samples = int(duration_sec * 44100)
        return np.zeros((2, num_samples), dtype=np.int16)

    def get_device_info(self):
        return self.device_info

    def get_render_stats(self):
        return {
            "render_count": self.render_count,
            "mean_latency_ms": self.latency_ms
        }

    def clear_gpu_cache(self):
        pass
```

**Usage in Tests**:

```python
def test_synthesis_latency():
    """Test synthesis latency measurement."""
    engine = MockSynthesisEngine(latency_ms=50.0)

    start = time.time()
    audio = engine.render_phrase(test_chords, test_melody, 8.0)
    elapsed = time.time() - start

    assert elapsed >= 0.050  # Simulated latency
    assert audio.shape == (2, 352800)  # 8 sec @ 44.1kHz
```

---

## Versioning and Compatibility

### Interface Versioning

**Version Format**: `{major}.{minor}.{patch}`

**Semantic Versioning**:
- **Major**: Breaking changes to interface signatures
- **Minor**: Backward-compatible additions (new methods)
- **Patch**: Bug fixes, documentation updates

**Current Version**: 1.0.0

**Compatibility Promise**:
- Interfaces remain stable within major version
- Deprecated methods supported for 1 minor version
- Breaking changes announced 2 releases in advance

---

### Deprecation Policy

**Example Deprecation**:

```python
class IRingBuffer(ABC):
    @abstractmethod
    def write_chunk(self, chunk: np.ndarray) -> int:
        """Write chunk to buffer."""
        pass

    @deprecated(version="1.1.0", alternative="write_chunk")
    def write(self, chunk: np.ndarray) -> int:
        """
        DEPRECATED: Use write_chunk instead.

        Will be removed in version 2.0.0.
        """
        return self.write_chunk(chunk)
```

---

## Performance Contracts

### Latency Budgets

| Interface Method | Latency Budget | Measurement |
|------------------|----------------|-------------|
| `IRingBuffer.write_chunk` | <100µs | Per-call overhead |
| `IRingBuffer.read_chunk` | <100µs | Per-call overhead |
| `ISynthesisEngine.render_phrase` | <100ms | 8-sec phrase rendering |
| `IChunkEncoder.encode_chunk` | <5ms | Per-chunk encoding |
| `IStreamingServer.broadcast_chunk` | <50ms | All clients delivery |

### Memory Contracts

| Component | Memory Budget | Measurement |
|-----------|---------------|-------------|
| `BroadcastRingBuffer` | <10 MB | Buffer allocation |
| `OptimizedSynthesisEngine` | <20 MB | GPU buffers |
| `AudioChunkPool` | <5 MB | Encoding pool |

---

## Contact

For internal interface questions:
- Architecture Documentation: [docs/system_architecture.md](../../../docs/system_architecture.md)
- GitHub Issues: https://github.com/yourusername/auralis/issues
