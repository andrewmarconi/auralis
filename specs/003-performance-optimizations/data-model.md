# Data Model: Performance Optimizations

**Feature**: Performance Optimizations
**Branch**: `003-performance-optimizations`
**Date**: 2025-12-26
**Phase**: 1 - Design

## Overview

This document defines all data structures, entities, and state management for the performance optimization implementation. These models support adaptive buffering, concurrent WebSocket streaming, GPU-optimized synthesis, memory leak prevention, and performance monitoring.

---

## Table of Contents

1. [Buffer Management Models](#1-buffer-management-models)
2. [WebSocket Concurrency Models](#2-websocket-concurrency-models)
3. [GPU Optimization Models](#3-gpu-optimization-models)
4. [Memory Monitoring Models](#4-memory-monitoring-models)
5. [Performance Metrics Models](#5-performance-metrics-models)
6. [Configuration Models](#6-configuration-models)

---

## 1. Buffer Management Models

### 1.1 AdaptiveRingBuffer

**Purpose**: Ring buffer with adaptive tier-based buffering based on observed network conditions.

**Attributes**:

```python
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import threading
import time

@dataclass
class BufferTier:
    """Configuration for a specific buffer tier."""
    target_ms: int
    description: str

class AdaptiveRingBuffer:
    """Ring buffer with adaptive tier-based buffering."""

    # Class-level tier definitions
    TIERS: Dict[str, BufferTier] = {
        "minimal": BufferTier(target_ms=500, description="Stable network, low latency"),
        "normal": BufferTier(target_ms=1000, description="Default for new connections"),
        "stable": BufferTier(target_ms=2000, description="Occasional jitter"),
        "defensive": BufferTier(target_ms=3000, description="High jitter/unstable network")
    }

    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_duration_ms: int = 100
    ):
        """
        Initialize adaptive ring buffer.

        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_duration_ms: Duration of each chunk in milliseconds
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.samples_per_chunk = int((chunk_duration_ms / 1000) * sample_rate)

        # Current tier state
        self.current_tier = "normal"
        self.target_buffer_ms = self.TIERS["normal"].target_ms

        # Buffer allocation (max size = defensive tier)
        max_buffer_ms = self.TIERS["defensive"].target_ms
        self.max_chunks = int(max_buffer_ms / chunk_duration_ms)
        self.buffer = np.zeros((self.max_chunks, self.samples_per_chunk * 2), dtype=np.int16)

        # Cursors
        self.write_cursor = 0
        self.read_cursor = 0
        self._lock = threading.Lock()

        # Jitter tracking
        self.jitter_tracker = JitterTracker(window_size=50)
```

**Key Methods**:
- `adjust_tier()`: Escalate or demote tier based on observed jitter and underrun rate
- `get_buffer_health() -> dict`: Return current buffer health metrics
- `write_chunk(chunk: np.ndarray)`: Write audio chunk to buffer
- `read_chunk() -> Optional[np.ndarray]`: Read next chunk from buffer
- `get_depth() -> int`: Get current buffer depth in chunks

**State Transitions**:
```
minimal ←→ normal ←→ stable ←→ defensive
  ↑                                 ↓
  └─────────────────────────────────┘
  (Based on jitter_ms and underrun_rate)
```

**Tier Promotion Rules**:
- Promote if `underrun_rate > 5%`
- Demote if `underrun_rate < 1%` AND `jitter_ms < 10ms` (or `< 5ms` for minimal)

---

### 1.2 JitterTracker

**Purpose**: Track chunk delivery jitter using Exponential Moving Average (EMA) for adaptive buffer sizing.

**Attributes**:

```python
from collections import deque
from dataclasses import dataclass
import numpy as np

@dataclass
class ChunkTimestamp:
    """Tracks individual chunk delivery timing."""
    chunk_id: int
    expected_time: float  # Unix timestamp
    actual_time: float    # Unix timestamp

    @property
    def jitter_ms(self) -> float:
        """Calculate jitter for this chunk in milliseconds."""
        return abs(self.actual_time - self.expected_time) * 1000

class JitterTracker:
    """Exponential Moving Average (EMA) based jitter tracking."""

    def __init__(self, window_size: int = 50, alpha: float = 0.1):
        """
        Initialize jitter tracker.

        Args:
            window_size: Number of recent chunks to retain
            alpha: EMA smoothing factor (0-1, lower = more smoothing)
        """
        self.window_size = window_size
        self.alpha = alpha

        # Circular buffer for recent timestamps
        self.timestamps: deque[ChunkTimestamp] = deque(maxlen=window_size)

        # EMA state
        self.mean_jitter_ms: float = 0.0
        self.variance_jitter_ms: float = 0.0

        # Statistics
        self.total_chunks: int = 0
        self.underrun_count: int = 0
```

**Key Methods**:
- `record_chunk(expected_time: float, actual_time: float)`: Record chunk arrival and update EMA
- `record_underrun()`: Increment underrun counter
- `get_current_jitter() -> float`: Return current mean jitter (ms)
- `get_jitter_std() -> float`: Return standard deviation of jitter
- `get_recommended_buffer_ms(confidence: float = 0.95) -> float`: Calculate buffer size using mean + k*σ
- `get_underrun_rate() -> float`: Return underrun rate as fraction

**EMA Update Formula**:
```python
# For jitter value x_t at time t:
mean_t = mean_{t-1} + α * (x_t - mean_{t-1})
variance_t = (1 - α) * (variance_{t-1} + α * (x_t - mean_{t-1})^2)
```

**Statistical Buffer Sizing**:
- 95% confidence: `buffer_ms = mean + 2 * std`
- 99% confidence: `buffer_ms = mean + 3 * std`

---

### 1.3 TokenBucket

**Purpose**: Rate limiter for flow control to prevent slow clients from consuming chunks faster than generation rate.

**Attributes**:

```python
@dataclass
class TokenBucket:
    """Token bucket rate limiter for flow control."""

    capacity: int          # Maximum burst size (tokens)
    refill_rate: float     # Tokens per second

    # Runtime state (initialized in __post_init__)
    tokens: float = 0.0
    last_refill: float = 0.0

    def __post_init__(self):
        """Initialize runtime state."""
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()
```

**Key Methods**:
- `consume(tokens: int = 1) -> bool`: Attempt to consume tokens (returns True if successful)
- `_refill()`: Refill tokens based on elapsed time
- `get_wait_time(tokens: int = 1) -> float`: Calculate wait time before tokens available

**Refill Algorithm**:
```python
elapsed = now - last_refill
tokens = min(capacity, tokens + elapsed * refill_rate)
```

**Typical Configuration**:
- `capacity=10` (1 second burst for 100ms chunks)
- `refill_rate=10/sec` (matches chunk generation rate)

---

### 1.4 AudioChunkPool

**Purpose**: Object pool for audio chunk encoding to reduce garbage collection pressure.

**Attributes**:

```python
from typing import List
import numpy as np

class AudioChunkPool:
    """Object pool for audio chunk encoding to reduce GC pressure."""

    def __init__(self, pool_size: int = 20, samples_per_chunk: int = 8820):
        """
        Initialize chunk pool.

        Args:
            pool_size: Number of pre-allocated buffers
            samples_per_chunk: Stereo samples (4410 samples × 2 channels)
        """
        self.pool_size = pool_size
        self.samples_per_chunk = samples_per_chunk

        # Pre-allocated numpy arrays
        self.chunk_buffers: List[np.ndarray] = [
            np.zeros(samples_per_chunk, dtype=np.int16)
            for _ in range(pool_size)
        ]

        # Pre-allocated bytes buffers
        bytes_per_chunk = samples_per_chunk * 2
        self.bytes_buffers: List[bytearray] = [
            bytearray(bytes_per_chunk)
            for _ in range(pool_size)
        ]

        # Pre-allocated base64 output buffers
        base64_size = ((bytes_per_chunk + 2) // 3) * 4
        self.base64_buffers: List[bytearray] = [
            bytearray(base64_size)
            for _ in range(pool_size)
        ]

        # Round-robin cursor
        self.current_index: int = 0
```

**Key Methods**:
- `encode_chunk(audio_chunk: np.ndarray) -> str`: Encode chunk using pre-allocated buffers

**Memory Layout**:
```
Pool Size: 20 buffers
Per Buffer:
  - chunk_buffer: 8820 samples × 2 bytes = 17,640 bytes
  - bytes_buffer: 17,640 bytes
  - base64_buffer: 23,520 bytes (base64 expansion)
Total per buffer: ~59 KB
Total pool: ~1.2 MB
```

---

## 2. WebSocket Concurrency Models

### 2.1 ClientCursor

**Purpose**: Per-client read position in shared ring buffer for lock-free concurrent reads.

**Attributes**:

```python
@dataclass
class ClientCursor:
    """Per-client read position in shared ring buffer."""
    client_id: str
    read_position: int         # Chunk index in ring buffer
    last_read_time: float      # Unix timestamp
    chunks_read: int           # Total chunks read
    buffer_underruns: int      # Underrun count
```

**State Tracking**:
- `read_position`: Current position in ring buffer (0 to capacity-1)
- `last_read_time`: Used to detect idle/disconnected clients
- `chunks_read`: Total chunks successfully read (for throughput metrics)
- `buffer_underruns`: Count of underrun events (for quality metrics)

---

### 2.2 BroadcastRingBuffer

**Purpose**: Ring buffer with per-client cursors for lock-free concurrent reads.

**Attributes**:

```python
from typing import Dict
import threading

class BroadcastRingBuffer:
    """Ring buffer with per-client cursors for concurrent reads."""

    def __init__(self, capacity_chunks: int = 30, samples_per_chunk: int = 8820):
        """
        Initialize broadcast ring buffer.

        Args:
            capacity_chunks: Maximum number of chunks to buffer
            samples_per_chunk: Stereo samples per chunk
        """
        self.capacity = capacity_chunks
        self.samples_per_chunk = samples_per_chunk

        # Shared buffer (write-once, read-many)
        self.buffer = np.zeros((capacity_chunks, samples_per_chunk), dtype=np.int16)

        # Single write cursor (updated by synthesis thread)
        self.write_position: int = 0
        self._write_lock = threading.Lock()

        # Per-client cursors (no shared lock needed for reads)
        self.client_cursors: Dict[str, ClientCursor] = {}
        self._clients_lock = threading.Lock()  # Only for registration/unregistration

        # Chunk metadata
        self.chunk_timestamps: List[float] = [0.0] * capacity_chunks
```

**Key Methods**:
- `register_client(client_id: str, start_position: Optional[int])`: Add client with initial cursor
- `unregister_client(client_id: str)`: Remove client cursor
- `write_chunk(chunk: np.ndarray) -> int`: Write chunk (single writer)
- `read_chunk(client_id: str) -> Optional[np.ndarray]`: Read next chunk for client (lock-free)
- `get_client_stats(client_id: str) -> dict`: Get per-client buffer statistics

**Lock-Free Read Pattern**:
```python
# No lock needed for read - write_position only advances forward
chunks_available = (write_position - read_position) % capacity
if chunks_available > 0:
    chunk = buffer[read_position].copy()
    read_position = (read_position + 1) % capacity
```

**Catchup Logic**:
- If `chunks_available >= capacity - 1`: Client fell too far behind
- Action: Jump to `write_position - 5` (5 chunks behind for buffering)

---

### 2.3 WebSocketClientState

**Purpose**: Track state for individual WebSocket client connections.

**Attributes**:

```python
from enum import Enum
from fastapi import WebSocket
from typing import Optional

class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    ACTIVE = "active"
    DRAINING = "draining"
    DISCONNECTED = "disconnected"

@dataclass
class WebSocketClientState:
    """State for individual WebSocket client."""
    client_id: str
    websocket: WebSocket
    connection_state: ConnectionState
    connected_at: float         # Unix timestamp
    last_message_at: float      # Unix timestamp
    messages_sent: int
    bytes_sent: int
    errors: int

    # Adaptive buffering state
    current_buffer_tier: str = "normal"
    jitter_tracker: Optional[JitterTracker] = None
```

**State Lifecycle**:
```
CONNECTING → ACTIVE → DRAINING → DISCONNECTED
     ↓          ↓
     └──────────┴─→ DISCONNECTED (error)
```

---

### 2.4 AsyncSynthesisEngine

**Purpose**: Synthesis engine wrapper with thread pool offloading for non-blocking GPU operations.

**Attributes**:

```python
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import numpy as np

class AsyncSynthesisEngine:
    """Synthesis engine with thread pool offloading."""

    def __init__(self, max_workers: int = 4):
        """
        Initialize async synthesis engine.

        Args:
            max_workers: Thread pool size (2-4 recommended)
        """
        self.sync_engine = SynthesisEngine()  # Original engine
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Performance tracking
        self.render_count: int = 0
        self.total_render_time: float = 0.0
```

**Key Methods**:
- `async render_phrase_async(...) -> np.ndarray`: Offload render to thread pool
- `async shutdown()`: Gracefully shutdown thread pool

**Thread Pool Sizing**:
- **2 workers**: Single synthesis stream (1 rendering, 1 preparing next)
- **4 workers**: Overlapped synthesis + composition + encoding
- **>4 workers**: Diminishing returns (GPU bottleneck)

---

## 3. GPU Optimization Models

### 3.1 OptimizedSynthesisEngine

**Purpose**: Synthesis engine with pre-allocated GPU buffers and batch processing.

**Attributes**:

```python
import torch

class OptimizedSynthesisEngine:
    """Synthesis engine with pre-allocated GPU buffers."""

    def __init__(self, sample_rate: int = 44100):
        """Initialize optimized synthesis engine."""
        self.sample_rate = sample_rate

        # Device selection
        self.device = self._select_device()

        # Pre-allocated buffers
        self.max_phrase_samples = int(30.0 * sample_rate)
        self.audio_buffer = torch.zeros(
            self.max_phrase_samples,
            device=self.device,
            dtype=torch.float32
        )

        self.max_voice_duration = int(10.0 * sample_rate)
        self.voice_buffer = torch.zeros(
            self.max_voice_duration,
            device=self.device,
            dtype=torch.float32
        )

        self.time_buffer = torch.arange(
            self.max_voice_duration,
            device=self.device,
            dtype=torch.float32
        ) / sample_rate

        # Performance tracking
        self.render_count: int = 0
```

**Memory Allocation**:

| Buffer | Size | Memory |
|--------|------|--------|
| audio_buffer | 30s × 44100 Hz × 4 bytes | ~5.3 MB |
| voice_buffer | 10s × 44100 Hz × 4 bytes | ~1.8 MB |
| time_buffer | 10s × 44100 Hz × 4 bytes | ~1.8 MB |
| **Total** | | **~9 MB** |

**Device Selection Priority**:
1. Metal (MPS) - Apple Silicon
2. CUDA - NVIDIA GPUs
3. CPU (fallback)

---

### 3.2 DeviceInfo

**Purpose**: Track GPU device capabilities and characteristics.

**Attributes**:

```python
@dataclass
class DeviceInfo:
    """GPU device information and capabilities."""
    device_type: str           # "mps", "cuda", "cpu"
    device_name: str           # e.g., "Apple M4", "NVIDIA RTX 3090"
    total_memory_mb: float     # Total device memory
    available_memory_mb: float # Available memory
    compute_capability: Optional[str] = None  # CUDA compute capability
    supports_fp16: bool = False
    supports_bf16: bool = False
    unified_memory: bool = False  # True for Metal
```

**Example Values**:

**Metal (Apple M4)**:
```python
DeviceInfo(
    device_type="mps",
    device_name="Apple M4",
    total_memory_mb=16384,  # Shared with system
    available_memory_mb=12000,
    unified_memory=True,
    supports_fp16=True,
    supports_bf16=False
)
```

**CUDA (NVIDIA RTX 3090)**:
```python
DeviceInfo(
    device_type="cuda",
    device_name="NVIDIA GeForce RTX 3090",
    total_memory_mb=24576,
    available_memory_mb=23000,
    compute_capability="8.6",
    supports_fp16=True,
    supports_bf16=True,
    unified_memory=False
)
```

---

### 3.3 BatchRenderConfig

**Purpose**: Configuration for batch rendering operations.

**Attributes**:

```python
@dataclass
class BatchRenderConfig:
    """Configuration for GPU batch rendering."""
    max_batch_size: int = 16        # Maximum voices in single batch
    enable_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    warmup_iterations: int = 3
    use_cuda_streams: bool = True   # CUDA only
```

**Compile Modes**:
- `"default"`: Balanced optimization
- `"reduce-overhead"`: Minimize kernel launch overhead (best for repetitive calls)
- `"max-autotune"`: Maximum optimization (longer compile time)

---

## 4. Memory Monitoring Models

### 4.1 MemorySnapshot

**Purpose**: Point-in-time memory usage snapshot.

**Attributes**:

```python
@dataclass
class MemorySnapshot:
    """Point-in-time memory usage snapshot."""
    timestamp: float          # Unix timestamp
    label: str               # Descriptive label

    # Python heap
    python_mb: float
    python_peak_mb: float

    # GPU memory
    gpu_mb: float

    # System memory
    rss_mb: float            # Resident Set Size
    vms_mb: float            # Virtual Memory Size

    # GC stats
    gc_gen0_count: int
    gc_gen1_count: int
    gc_gen2_count: int
```

**Derived Metrics**:
```python
@property
def memory_growth_mb(self, baseline: MemorySnapshot) -> float:
    """Calculate memory growth since baseline."""
    return self.rss_mb - baseline.rss_mb
```

---

### 4.2 MemoryMonitor

**Purpose**: Track memory usage over time for leak detection.

**Attributes**:

```python
class MemoryMonitor:
    """Monitor memory usage over time for leak detection."""

    def __init__(self):
        self.baseline_mb: Optional[float] = None
        self.snapshots: List[MemorySnapshot] = []

        # Start tracemalloc
        import tracemalloc
        tracemalloc.start(25)  # Track 25 stack frames
```

**Key Methods**:
- `snapshot(label: str) -> MemorySnapshot`: Take memory snapshot
- `print_top_allocations(limit: int = 10)`: Print top memory allocations
- `detect_leak(threshold_mb: float = 50) -> bool`: Use linear regression to detect sustained growth

**Leak Detection Algorithm**:
```python
# Fit line: rss = slope * time + intercept
times = [(s.timestamp - baseline_time) / 3600 for s in snapshots]  # hours
rss_values = [s.rss_mb for s in snapshots]
slope, intercept = np.polyfit(times, rss_values, 1)

# Detect leak if growth > threshold_mb per hour
is_leak = slope > threshold_mb
```

---

### 4.3 MemoryGrowthTracker

**Purpose**: Real-time memory growth rate tracking with alerting.

**Attributes**:

```python
@dataclass
class MemoryMeasurement:
    """Single memory measurement."""
    timestamp: float
    rss_mb: float
    label: str

class MemoryGrowthTracker:
    """Track memory growth rate over time with alerting."""

    def __init__(self, alert_threshold_mb_per_hour: float = 20):
        self.alert_threshold = alert_threshold_mb_per_hour
        self.measurements: List[MemoryMeasurement] = []
        self.start_time: float = time.time()
```

**Key Methods**:
- `record(label: str)`: Record current memory usage
- `get_growth_rate() -> float`: Calculate MB/hour growth rate
- `check_alert() -> Optional[str]`: Check if growth exceeds threshold

**Growth Rate Calculation**:
```python
hours = (timestamps - start_time) / 3600
slope, _ = np.polyfit(hours, rss_mb_values, 1)
growth_rate_mb_per_hour = slope
```

---

### 4.4 GCConfig

**Purpose**: Garbage collection configuration for real-time workloads.

**Attributes**:

```python
@dataclass
class GCConfig:
    """Garbage collection configuration."""
    gen0_threshold: int = 5000  # vs default 700
    gen1_threshold: int = 50    # vs default 10
    gen2_threshold: int = 50    # vs default 10
    auto_gc_enabled: bool = False
    manual_collect_interval: int = 100  # Collect every N renders
```

**Application**:
```python
import gc

def apply_gc_config(config: GCConfig):
    """Apply GC configuration."""
    gc.set_threshold(
        config.gen0_threshold,
        config.gen1_threshold,
        config.gen2_threshold
    )

    if not config.auto_gc_enabled:
        gc.disable()
```

---

## 5. Performance Metrics Models

### 5.1 PrometheusMetrics

**Purpose**: Prometheus metric definitions for performance monitoring.

**Attributes**:

```python
from prometheus_client import Counter, Gauge, Histogram

class PerformanceMetrics:
    """Prometheus metrics for Auralis performance monitoring."""

    def __init__(self):
        # Synthesis metrics
        self.synthesis_latency = Histogram(
            'synthesis_latency_seconds',
            'Time to render audio phrase',
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
        )

        # Buffer metrics
        self.buffer_depth = Gauge(
            'buffer_depth_chunks',
            'Current ring buffer depth',
            ['client_id']
        )

        self.chunk_jitter = Histogram(
            'chunk_delivery_jitter_ms',
            'Chunk delivery timing variance',
            buckets=[1, 2, 5, 10, 20, 30, 50, 75, 100, 200]
        )

        self.buffer_underruns = Counter(
            'buffer_underruns_total',
            'Buffer underrun events',
            ['client_id']
        )

        # WebSocket metrics
        self.active_connections = Gauge(
            'active_websocket_connections',
            'Number of connected WebSocket clients'
        )

        self.websocket_errors = Counter(
            'websocket_send_errors_total',
            'Failed WebSocket send operations',
            ['error_type']
        )

        # Memory metrics
        self.memory_usage = Gauge(
            'memory_usage_mb',
            'Process memory usage (RSS)'
        )

        self.gpu_memory = Gauge(
            'gpu_memory_allocated_mb',
            'GPU memory allocated'
        )

        # Encoding metrics
        self.encoding_duration = Histogram(
            'chunk_encoding_duration_seconds',
            'base64 encoding duration',
            buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
        )

        # GC metrics
        self.gc_collections = Counter(
            'gc_collections_total',
            'Garbage collection count',
            ['generation']
        )

        # Generation metrics
        self.generation_rate = Gauge(
            'phrase_generation_rate_hz',
            'Phrases generated per second'
        )
```

**Metric Types**:

| Metric | Type | Unit | Labels |
|--------|------|------|--------|
| synthesis_latency_seconds | Histogram | seconds | - |
| buffer_depth_chunks | Gauge | chunks | client_id |
| chunk_delivery_jitter_ms | Histogram | milliseconds | - |
| buffer_underruns_total | Counter | count | client_id |
| active_websocket_connections | Gauge | count | - |
| websocket_send_errors_total | Counter | count | error_type |
| memory_usage_mb | Gauge | megabytes | - |
| gpu_memory_allocated_mb | Gauge | megabytes | - |
| chunk_encoding_duration_seconds | Histogram | seconds | - |
| gc_collections_total | Counter | count | generation |
| phrase_generation_rate_hz | Gauge | hertz | - |

---

### 5.2 PerformanceBenchmark

**Purpose**: Store benchmark results for regression testing.

**Attributes**:

```python
@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    benchmark_name: str
    timestamp: float

    # Latency metrics
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float

    # Throughput metrics
    operations_per_second: float

    # Resource metrics
    cpu_percent: float
    memory_mb: float
    gpu_utilization_percent: Optional[float] = None

    # Configuration
    device_type: str          # "mps", "cuda", "cpu"
    batch_size: Optional[int] = None
    concurrent_clients: Optional[int] = None
```

**Example Benchmark**:
```python
PerformanceBenchmark(
    benchmark_name="synthesis_latency_metal",
    timestamp=1735228800.0,
    mean_latency_ms=45.2,
    p50_latency_ms=42.0,
    p95_latency_ms=65.0,
    p99_latency_ms=80.0,
    max_latency_ms=95.0,
    operations_per_second=22.1,
    cpu_percent=25.0,
    memory_mb=350.0,
    gpu_utilization_percent=75.0,
    device_type="mps",
    batch_size=None,
    concurrent_clients=1
)
```

---

## 6. Configuration Models

### 6.1 PerformanceConfig

**Purpose**: Centralized configuration for all performance optimization settings.

**Attributes**:

```python
from pydantic import BaseModel, Field

class BufferConfig(BaseModel):
    """Buffer management configuration."""
    enable_adaptive_tiers: bool = True
    enable_jitter_tracking: bool = True
    enable_token_bucket: bool = True
    enable_object_pooling: bool = True

    # Adaptive buffer settings
    initial_tier: str = "normal"
    tier_adjustment_interval_chunks: int = 50

    # Jitter tracking settings
    jitter_window_size: int = 50
    jitter_ema_alpha: float = 0.1

    # Token bucket settings
    token_bucket_capacity: int = 10
    token_bucket_refill_rate: float = 10.0

    # Object pool settings
    chunk_pool_size: int = 20

class ConcurrencyConfig(BaseModel):
    """WebSocket concurrency configuration."""
    enable_per_client_cursors: bool = True
    enable_broadcast: bool = True
    enable_thread_pool: bool = True

    # Thread pool settings
    thread_pool_workers: int = Field(default=2, ge=1, le=8)

    # Broadcast settings
    broadcast_interval_ms: int = 100
    max_concurrent_clients: int = 20

    # Graceful shutdown
    drain_timeout_sec: float = 5.0

class GPUConfig(BaseModel):
    """GPU optimization configuration."""
    enable_memory_prealloc: bool = True
    enable_batch_processing: bool = True
    enable_torch_compile: bool = True
    enable_cuda_streams: bool = True

    # Batch processing
    max_batch_size: int = 16

    # torch.compile settings
    compile_mode: str = "reduce-overhead"
    warmup_iterations: int = 3

    # Memory settings
    max_phrase_duration_sec: float = 30.0
    max_voice_duration_sec: float = 10.0

    # Periodic cleanup
    cleanup_interval_renders: int = 100

class MemoryConfig(BaseModel):
    """Memory leak prevention configuration."""
    enable_tracemalloc: bool = True
    enable_gc_tuning: bool = True
    enable_periodic_cleanup: bool = True

    # GC settings
    gc_gen0_threshold: int = 5000
    gc_gen1_threshold: int = 50
    gc_gen2_threshold: int = 50
    auto_gc_enabled: bool = False

    # Memory monitoring
    snapshot_interval_sec: int = 300  # 5 minutes
    leak_detection_threshold_mb_per_hour: float = 20.0

class MonitoringConfig(BaseModel):
    """Performance monitoring configuration."""
    enable_prometheus: bool = True
    enable_grafana: bool = True

    # Metrics collection
    metrics_collection_interval_sec: int = 5
    prometheus_port: int = 9090

    # Alerting
    enable_alerts: bool = True
    alert_severity_levels: List[str] = ["critical", "high", "medium"]

class PerformanceConfig(BaseModel):
    """Master performance optimization configuration."""
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Global settings
    enable_performance_mode: bool = True
    target_latency_ms: float = 100.0
    target_concurrent_users: int = 10
```

**Configuration Loading**:

```python
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

# Usage:
# PERFORMANCE__BUFFER__ENABLE_ADAPTIVE_TIERS=false
# PERFORMANCE__GPU__COMPILE_MODE=max-autotune
settings = Settings()
```

---

### 6.2 AlertConfig

**Purpose**: Prometheus alert rule configuration.

**Attributes**:

```python
from enum import Enum

class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class AlertRule:
    """Prometheus alert rule definition."""
    name: str
    severity: AlertSeverity
    expr: str                # PromQL expression
    for_duration: str        # e.g., "2m", "5m"
    summary: str
    description: str

class AlertConfig(BaseModel):
    """Alert configuration."""
    rules: List[AlertRule] = Field(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add default rules
        self.rules.extend([
            AlertRule(
                name="BufferUnderrun",
                severity=AlertSeverity.CRITICAL,
                expr="rate(buffer_underruns_total[1m]) > 0.0167",
                for_duration="2m",
                summary="Buffer underruns detected",
                description="Client experiencing underruns"
            ),
            AlertRule(
                name="SynthesisLatencyHigh",
                severity=AlertSeverity.CRITICAL,
                expr="histogram_quantile(0.99, synthesis_latency_seconds_bucket) > 0.1",
                for_duration="5m",
                summary="Synthesis latency p99 exceeds 100ms",
                description="p99 latency too high"
            ),
            AlertRule(
                name="MemoryLeak",
                severity=AlertSeverity.CRITICAL,
                expr="deriv(memory_usage_mb[1h]) > 20",
                for_duration="1h",
                summary="Memory leak detected",
                description="Memory growing > 20 MB/hour"
            ),
        ])
```

---

## Data Flow Diagrams

### Adaptive Buffer Flow

```
┌─────────────────┐
│  Audio Chunks   │
│   (from GPU)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  AdaptiveRingBuffer    │
│  - Current tier: normal │
│  - Target: 1000ms      │
└────────┬────────────────┘
         │
         ├──► JitterTracker
         │    - Record chunk timing
         │    - Calculate EMA jitter
         │    - Update underrun rate
         │
         ├──► adjust_tier()
         │    - Check underrun_rate
         │    - Check jitter_ms
         │    - Promote/demote tier
         │
         ▼
┌─────────────────┐
│  Client Reads   │
│  - Via cursors  │
└─────────────────┘
```

### WebSocket Broadcast Flow

```
┌──────────────────┐
│ Synthesis Engine │
└────────┬─────────┘
         │
         ▼
┌───────────────────────┐
│ BroadcastRingBuffer  │
│ - Single write cursor │
│ - Per-client cursors  │
└────────┬──────────────┘
         │
         ├──► Client A cursor (position: 10)
         ├──► Client B cursor (position: 12)
         └──► Client C cursor (position: 8)
                │
                ▼
        ┌─────────────────┐
        │  AudioChunkPool │
        │  - Encode once  │
        └────────┬────────┘
                 │
                 ├──► WebSocket A
                 ├──► WebSocket B
                 └──► WebSocket C
```

### GPU Optimization Flow

```
┌──────────────┐
│ Phrase Data  │
│ (chords, mel)│
└──────┬───────┘
       │
       ▼
┌─────────────────────────┐
│ OptimizedSynthesisEngine│
│                         │
│ torch.no_grad():        │
│   ┌──────────────────┐  │
│   │ Pre-allocated    │  │
│   │ audio_buffer     │  │
│   └──────────────────┘  │
│          │              │
│          ▼              │
│   ┌──────────────────┐  │
│   │ Batch Rendering  │  │
│   │ - All chord      │  │
│   │   voices in      │  │
│   │   parallel       │  │
│   └──────────────────┘  │
│          │              │
│          ▼              │
│   ┌──────────────────┐  │
│   │ torch.compile    │  │
│   │ (JIT optimized)  │  │
│   └──────────────────┘  │
└─────────┬───────────────┘
          │
          ▼
    ┌──────────┐
    │ Audio Out│
    └──────────┘
```

---

## State Invariants

### Buffer State Invariants

1. **Write-Read Relationship**:
   ```python
   0 <= (write_cursor - read_cursor) % capacity <= capacity
   ```

2. **Tier Ordering**:
   ```python
   minimal.target_ms < normal.target_ms < stable.target_ms < defensive.target_ms
   ```

3. **Token Bucket**:
   ```python
   0 <= tokens <= capacity
   ```

4. **Client Cursor Bounds**:
   ```python
   0 <= client_cursor.read_position < capacity
   ```

### Memory State Invariants

1. **Memory Growth**:
   ```python
   # Over time t, memory should be bounded:
   memory(t) < memory(0) + leak_threshold * t
   ```

2. **GPU Memory**:
   ```python
   allocated_memory <= total_device_memory * 0.9  # 90% max
   ```

3. **Pool Utilization**:
   ```python
   current_index < pool_size
   ```

---

## Schema Validation

All Pydantic models include automatic validation:

```python
# Example: Invalid configuration raises validation error
try:
    config = ConcurrencyConfig(thread_pool_workers=0)  # Must be >= 1
except ValidationError as e:
    print(e)
    # ValidationError: thread_pool_workers must be >= 1

# Example: Automatic type coercion
config = BufferConfig(jitter_ema_alpha="0.1")  # String → float
assert isinstance(config.jitter_ema_alpha, float)
```

---

## Conclusion

This data model provides a comprehensive foundation for implementing production-grade performance optimizations in Auralis. Key design principles:

1. **Type Safety**: All models use Python dataclasses or Pydantic for type validation
2. **Observability**: Rich metrics and monitoring models for debugging
3. **Configuration**: Centralized, environment-variable-friendly configuration
4. **State Management**: Clear state transitions and invariants
5. **Scalability**: Per-client state for concurrent operations

Next Phase: Generate API contracts defining interfaces between these models.
