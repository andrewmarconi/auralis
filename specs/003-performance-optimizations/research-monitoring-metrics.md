# Phase 0 Research: Performance Monitoring & Metrics

**Feature**: Performance Optimizations (003)
**Date**: 2025-12-26
**Research Focus**: Real-time audio streaming performance monitoring, metrics collection, and observability without impacting <100ms latency requirement

## Executive Summary

This research investigates production-grade performance monitoring for the Auralis real-time audio streaming system. The primary challenge is measuring and tracking streaming quality metrics (latency, jitter, buffer health) while supporting 10+ concurrent connections without introducing performance overhead that violates the <100ms audio processing latency constraint.

**Key Findings**:
- **Prometheus** is the industry-standard time-series database for real-time metrics with native FastAPI integration
- **Asynchronous metrics collection** is essential to avoid blocking the audio pipeline
- **Ring buffer instrumentation** provides critical visibility into streaming health
- **OpenTelemetry** offers future-proof observability but adds complexity for current scope
- **Grafana** provides production-ready dashboards with minimal configuration

**Recommendation**: Implement Prometheus metrics with `prometheus-fastapi-instrumentator` library, using asynchronous collection and sampling strategies to preserve real-time performance.

---

## 1. Key Performance Indicators (KPIs) for Real-Time Audio Streaming

### 1.1 Audio-Specific Metrics

#### Latency Metrics
**Definition**: Time delay between audio generation and delivery to client.

| Metric Name | Type | Description | Target Value | Alert Threshold |
|-------------|------|-------------|--------------|-----------------|
| `audio_synthesis_latency_ms` | Histogram | Time to render audio phrase (GPU/CPU) | <100ms p99 | >150ms |
| `audio_chunk_delivery_latency_ms` | Histogram | Time from buffer write to WebSocket send | <50ms p99 | >100ms |
| `audio_end_to_end_latency_ms` | Histogram | Total latency from synthesis to client playback | <200ms p99 | >300ms |
| `audio_startup_latency_ms` | Histogram | Time from connection to first audio chunk | <2000ms | >3000ms |

**Collection Method**:
```python
# Use time.perf_counter() for high-resolution timing
start_time = time.perf_counter()
audio_data = synthesis_engine.render_phrase(...)
synthesis_time_ms = (time.perf_counter() - start_time) * 1000
SYNTHESIS_LATENCY.observe(synthesis_time_ms)
```

#### Jitter Metrics
**Definition**: Variance in audio chunk delivery timing.

| Metric Name | Type | Description | Target Value | Alert Threshold |
|-------------|------|-------------|--------------|-----------------|
| `audio_chunk_jitter_ms` | Histogram | Deviation from expected chunk delivery interval | <20ms p95 | >50ms |
| `audio_chunk_delivery_variance_ms` | Gauge | Standard deviation of delivery intervals | <15ms | >30ms |

**Collection Method**:
```python
# Track inter-arrival time variance
expected_interval_ms = 100  # 100ms chunks
actual_interval_ms = (current_timestamp - last_timestamp) * 1000
jitter_ms = abs(actual_interval_ms - expected_interval_ms)
CHUNK_JITTER.observe(jitter_ms)
```

#### Buffer Health Metrics
**Definition**: Ring buffer state and quality indicators.

| Metric Name | Type | Description | Target Value | Alert Threshold |
|-------------|------|-------------|--------------|-----------------|
| `ring_buffer_depth_ms` | Gauge | Current buffer fill level in milliseconds | 500-1000ms | <200ms or >1800ms |
| `ring_buffer_underruns_total` | Counter | Total buffer underflow events | 0 | >0 |
| `ring_buffer_overruns_total` | Counter | Total buffer overflow events | 0 | >5 per hour |
| `ring_buffer_utilization_percent` | Gauge | Buffer capacity used (0-100%) | 25-50% | <10% or >80% |

**Collection Method**:
```python
# Sample buffer depth periodically (not every chunk)
buffer_depth_ms = ring_buffer.get_buffer_depth_ms()
BUFFER_DEPTH.set(buffer_depth_ms)

# Increment counters on events
if audio_data is None:  # Underrun
    BUFFER_UNDERRUNS.inc()
```

#### Packet Loss and Delivery
**Definition**: WebSocket transmission reliability.

| Metric Name | Type | Description | Target Value | Alert Threshold |
|-------------|------|-------------|--------------|-----------------|
| `websocket_chunks_sent_total` | Counter | Total audio chunks sent | N/A | N/A |
| `websocket_send_errors_total` | Counter | Failed chunk transmissions | 0 | >10 per hour |
| `websocket_chunk_delivery_success_rate` | Gauge | Percentage of successful deliveries | >99% | <95% |

### 1.2 System-Wide Metrics

#### Resource Utilization
**Definition**: Hardware resource consumption.

| Metric Name | Type | Description | Target Value | Alert Threshold |
|-------------|------|-------------|--------------|-----------------|
| `cpu_utilization_percent` | Gauge | CPU usage (overall) | <80% | >90% |
| `gpu_utilization_percent` | Gauge | GPU usage (Metal/CUDA) | <80% | >90% |
| `memory_usage_mb` | Gauge | Process memory consumption | Stable | >2GB or +10% growth/hour |
| `gpu_memory_allocated_mb` | Gauge | GPU memory allocated (CUDA only) | <50% GPU capacity | >80% |

**Collection Method**:
```python
import psutil

# CPU usage (sampled every 5 seconds to avoid overhead)
cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
CPU_UTILIZATION.set(cpu_percent)

# Memory usage
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
MEMORY_USAGE.set(memory_mb)

# GPU memory (CUDA only)
if torch.cuda.is_available():
    gpu_memory_mb = torch.cuda.memory_allocated(0) / 1024 / 1024
    GPU_MEMORY.set(gpu_memory_mb)
```

#### Network Bandwidth
**Definition**: Data transmission metrics.

| Metric Name | Type | Description | Target Value | Alert Threshold |
|-------------|------|-------------|--------------|-----------------|
| `network_bytes_sent_total` | Counter | Total bytes transmitted | N/A | N/A |
| `network_bandwidth_mbps` | Gauge | Current transmission rate (Mbps) | ~1.4 Mbps/stream | >10 Mbps/stream |

**Calculation**:
- 44,100 samples/sec × 2 channels × 2 bytes = 176,400 bytes/sec
- 176,400 bytes/sec × 8 bits/byte = 1,411,200 bps = ~1.4 Mbps raw
- Base64 encoding adds ~33% overhead = ~1.9 Mbps per stream

#### Connection Metrics
**Definition**: WebSocket session tracking.

| Metric Name | Type | Description | Target Value | Alert Threshold |
|-------------|------|-------------|--------------|-----------------|
| `websocket_active_connections` | Gauge | Current active WebSocket connections | N/A | >50 |
| `websocket_connections_total` | Counter | Total connections established | N/A | N/A |
| `websocket_disconnections_total` | Counter | Total disconnections | N/A | N/A |
| `websocket_connection_duration_seconds` | Histogram | Duration of connection sessions | N/A | N/A |

---

## 2. Prometheus Integration Patterns for FastAPI

### 2.1 Recommended Library: `prometheus-fastapi-instrumentator`

**Why Prometheus**:
- Industry-standard time-series database for real-time metrics
- Native support for histograms, counters, gauges, summaries
- Excellent FastAPI integration via `prometheus-fastapi-instrumentator`
- Pull-based model reduces impact on audio pipeline
- Powerful query language (PromQL) for analysis
- Battle-tested in production audio streaming systems (Spotify, SoundCloud)

**Installation**:
```bash
uv add prometheus-client prometheus-fastapi-instrumentator psutil
```

### 2.2 Basic Integration Pattern

```python
# server/metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time

# Create custom metrics registry
registry = CollectorRegistry()

# Audio synthesis metrics
SYNTHESIS_LATENCY = Histogram(
    'audio_synthesis_latency_ms',
    'Audio synthesis processing time in milliseconds',
    buckets=[10, 25, 50, 75, 100, 150, 200, 300, 500, 1000],
    registry=registry
)

CHUNK_DELIVERY_LATENCY = Histogram(
    'audio_chunk_delivery_latency_ms',
    'WebSocket chunk delivery latency in milliseconds',
    buckets=[10, 25, 50, 75, 100, 150, 200],
    registry=registry
)

CHUNK_JITTER = Histogram(
    'audio_chunk_jitter_ms',
    'Audio chunk delivery jitter (deviation from 100ms)',
    buckets=[5, 10, 20, 30, 50, 75, 100, 150],
    registry=registry
)

# Buffer health metrics
BUFFER_DEPTH = Gauge(
    'ring_buffer_depth_ms',
    'Current ring buffer depth in milliseconds',
    registry=registry
)

BUFFER_UNDERRUNS = Counter(
    'ring_buffer_underruns_total',
    'Total buffer underrun events',
    registry=registry
)

BUFFER_OVERRUNS = Counter(
    'ring_buffer_overruns_total',
    'Total buffer overrun events',
    registry=registry
)

# WebSocket metrics
WS_CHUNKS_SENT = Counter(
    'websocket_chunks_sent_total',
    'Total audio chunks sent',
    registry=registry
)

WS_SEND_ERRORS = Counter(
    'websocket_send_errors_total',
    'Failed chunk transmissions',
    registry=registry
)

ACTIVE_CONNECTIONS = Gauge(
    'websocket_active_connections',
    'Current active WebSocket connections',
    registry=registry
)

# System metrics
CPU_UTILIZATION = Gauge(
    'cpu_utilization_percent',
    'CPU utilization percentage',
    registry=registry
)

MEMORY_USAGE = Gauge(
    'memory_usage_mb',
    'Process memory usage in MB',
    registry=registry
)

GPU_MEMORY = Gauge(
    'gpu_memory_allocated_mb',
    'GPU memory allocated in MB (CUDA only)',
    registry=registry
)

GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    registry=registry
)
```

### 2.3 FastAPI Application Integration

```python
# server/main.py
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import make_asgi_app
from server.metrics import registry

# Create FastAPI app
app = FastAPI(...)

# Initialize Prometheus instrumentator (automatic HTTP metrics)
Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False  # Hide from OpenAPI docs
)

# Alternative: Mount metrics app manually for custom registry
# metrics_app = make_asgi_app(registry=registry)
# app.mount("/metrics", metrics_app)
```

### 2.4 WebSocket Metrics Integration

```python
# server/streaming_server.py
import time
from server.metrics import (
    WS_CHUNKS_SENT,
    WS_SEND_ERRORS,
    ACTIVE_CONNECTIONS,
    CHUNK_DELIVERY_LATENCY,
    CHUNK_JITTER,
    BUFFER_UNDERRUNS
)

class StreamingServer:
    def __init__(self, ring_buffer: RingBuffer):
        self.ring_buffer = ring_buffer
        self.active_connections: set[WebSocket] = set()
        self.sequence_counter = 0
        self.last_send_time = None  # For jitter calculation

    async def handle_client(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)
        ACTIVE_CONNECTIONS.inc()  # Increment gauge

        logger.info(f"Client connected. Total active: {len(self.active_connections)}")

        try:
            await self._stream_audio(websocket)
        except WebSocketDisconnect:
            logger.info("Client disconnected gracefully")
        except Exception as e:
            logger.error(f"Error in client handler: {e}")
        finally:
            self.active_connections.discard(websocket)
            ACTIVE_CONNECTIONS.dec()  # Decrement gauge
            logger.info(f"Client removed. Total active: {len(self.active_connections)}")

    async def _stream_audio(self, websocket: WebSocket) -> None:
        while True:
            # Start timing for delivery latency
            start_time = time.perf_counter()

            # Read from buffer
            audio_data = await asyncio.to_thread(
                self.ring_buffer.read_blocking,
                num_samples=self.ring_buffer.chunk_size,
                timeout=0.5,
            )

            if audio_data is None:
                # Buffer underflow - track metric
                BUFFER_UNDERRUNS.inc()
                logger.warning("Ring buffer underflow - sending silence")
                audio_data = np.zeros((2, self.ring_buffer.chunk_size), dtype=np.float32)

                await websocket.send_json({
                    "type": "warning",
                    "message": "Buffer underflow - temporary silence",
                })

            # Create and send audio chunk
            chunk = AudioChunk(audio_data, self.sequence_counter)
            self.sequence_counter += 1

            try:
                await websocket.send_text(chunk.to_json())
                WS_CHUNKS_SENT.inc()

                # Record delivery latency
                delivery_time_ms = (time.perf_counter() - start_time) * 1000
                CHUNK_DELIVERY_LATENCY.observe(delivery_time_ms)

                # Calculate and record jitter
                current_time = time.perf_counter()
                if self.last_send_time is not None:
                    actual_interval_ms = (current_time - self.last_send_time) * 1000
                    expected_interval_ms = 100  # 100ms chunks
                    jitter_ms = abs(actual_interval_ms - expected_interval_ms)
                    CHUNK_JITTER.observe(jitter_ms)
                self.last_send_time = current_time

            except Exception as e:
                WS_SEND_ERRORS.inc()
                logger.error(f"Failed to send chunk: {e}")
                raise

            # Check for client control messages (non-blocking)
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.001,
                )
                await self._handle_control_message(websocket, message)
            except asyncio.TimeoutError:
                pass
```

### 2.5 Synthesis Engine Metrics

```python
# server/synthesis_engine.py
import time
from server.metrics import SYNTHESIS_LATENCY

class SynthesisEngine:
    def render_phrase(
        self,
        chords: list[tuple[int, int, str]],
        melody: list[tuple[int, int, float, float]],
        duration_sec: float,
        kicks: list[tuple[int, float]] | None = None,
        swells: list[tuple[int, float, float]] | None = None,
    ) -> NDArray[np.float32]:
        """Render a musical phrase with latency tracking."""
        start_time = time.perf_counter()

        # Existing rendering logic...
        num_samples = int(duration_sec * self.sample_rate)
        audio = torch.zeros(num_samples, device=self.device)
        audio = self._render_chord_pads(audio, chords, num_samples)
        audio = self._render_melody_lead(audio, melody, num_samples)

        if kicks:
            audio = self._render_kicks(audio, kicks, num_samples)
        if swells:
            audio = self._render_swells(audio, swells, num_samples)

        audio = torch.tanh(audio * 0.7)
        audio_np = audio.cpu().numpy()
        stereo = np.stack([audio_np, audio_np], axis=0)

        # Record synthesis latency
        synthesis_time_ms = (time.perf_counter() - start_time) * 1000
        SYNTHESIS_LATENCY.observe(synthesis_time_ms)

        return stereo.astype(np.float32)
```

---

## 3. Metrics Collection Strategies (Non-Blocking)

### 3.1 Asynchronous Collection Patterns

**Challenge**: Metrics collection must not block the audio synthesis or streaming threads.

**Solution**: Use sampling, background tasks, and non-blocking gauges.

#### Pattern 1: Periodic Sampling (Background Task)

```python
# server/main.py
import asyncio
import psutil
import torch
from server.metrics import CPU_UTILIZATION, MEMORY_USAGE, BUFFER_DEPTH, GPU_MEMORY, ACTIVE_CONNECTIONS

async def metrics_collection_loop():
    """
    Background task that samples system metrics every 5 seconds.
    Does NOT block audio pipeline.
    """
    logger.info("Starting metrics collection loop...")

    while True:
        try:
            # Sample CPU (non-blocking)
            cpu_percent = psutil.cpu_percent(interval=None)
            CPU_UTILIZATION.set(cpu_percent)

            # Sample memory
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            MEMORY_USAGE.set(memory_mb)

            # Sample buffer depth
            buffer_depth_ms = app_state.ring_buffer.get_buffer_depth_ms()
            BUFFER_DEPTH.set(buffer_depth_ms)

            # Sample GPU memory (CUDA only)
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated(0) / 1024 / 1024
                GPU_MEMORY.set(gpu_memory_mb)

            # Sample active connections (already tracked by gauge)
            # ACTIVE_CONNECTIONS is updated in real-time by StreamingServer

            # Wait 5 seconds before next sample
            await asyncio.sleep(5.0)

        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            await asyncio.sleep(5.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Auralis server...")

    # Initialize components...
    app_state.ring_buffer = RingBuffer(...)
    app_state.synthesis_engine = SynthesisEngine(...)
    app_state.streaming_server = StreamingServer(app_state.ring_buffer)

    # Start background tasks
    app_state.synthesis_task = asyncio.create_task(synthesis_loop())
    app_state.metrics_task = asyncio.create_task(metrics_collection_loop())

    logger.info("Server ready for connections")

    yield

    # Cleanup
    logger.info("Shutting down Auralis server...")
    app_state.synthesis_task.cancel()
    app_state.metrics_task.cancel()
    try:
        await app_state.synthesis_task
    except asyncio.CancelledError:
        pass
    try:
        await app_state.metrics_task
    except asyncio.CancelledError:
        pass

    logger.info("Server stopped")
```

#### Pattern 2: Event-Based Increments (Zero-Overhead)

```python
# Counters and gauges have negligible overhead (<1μs)
# Safe to call in hot paths

# Example: Increment counter on buffer underrun
if audio_data is None:
    BUFFER_UNDERRUNS.inc()  # Atomic operation, <1μs overhead

# Example: Track WebSocket send
try:
    await websocket.send_text(chunk.to_json())
    WS_CHUNKS_SENT.inc()  # Safe in hot path
except Exception:
    WS_SEND_ERRORS.inc()  # Safe even in error path
```

#### Pattern 3: Histogram Sampling (Conditional)

```python
# Only observe histograms every Nth event to reduce overhead
synthesis_counter = 0

async def synthesis_loop():
    global synthesis_counter

    while True:
        start_time = time.perf_counter()
        audio_data = app_state.synthesis_engine.render_phrase(...)
        synthesis_time_ms = (time.perf_counter() - start_time) * 1000

        # Sample every 10th synthesis for latency histogram
        synthesis_counter += 1
        if synthesis_counter % 10 == 0:
            SYNTHESIS_LATENCY.observe(synthesis_time_ms)

        # Alternatively: Always observe (overhead is ~10μs, acceptable)
        SYNTHESIS_LATENCY.observe(synthesis_time_ms)
```

### 3.2 Lock-Free Metrics (Atomic Operations)

Prometheus Python client uses thread-safe atomic operations for counters/gauges, making them safe to use in concurrent code without explicit locking.

```python
# Thread-safe without locks (atomic operations)
CHUNKS_SENT.inc()  # Atomic increment
BUFFER_DEPTH.set(value)  # Atomic set
LATENCY.observe(value)  # Thread-safe histogram update
```

### 3.3 Performance Impact Analysis

| Metric Type | Overhead per Operation | Frequency | Total Impact |
|-------------|------------------------|-----------|--------------|
| **Counter.inc()** | <1μs | 10/sec (chunks sent) | <0.01ms/sec |
| **Gauge.set()** | <1μs | 1/5sec (background sample) | <0.2μs/5sec |
| **Histogram.observe()** | ~10μs | 1/sec (sampled synthesis) | ~10μs/sec |
| **Prometheus scrape** | ~50ms | 1/15sec (scrape interval) | ~3.3ms/sec |

**Total Estimated Overhead**: <0.1% of CPU time (negligible compared to <100ms latency requirement)

---

## 4. Time-Series Database Comparison

### 4.1 Prometheus vs. Alternatives

| Feature | Prometheus | InfluxDB | TimescaleDB | Recommendation |
|---------|-----------|----------|-------------|----------------|
| **Data Model** | Metrics (time-series) | General time-series | SQL + time-series | Prometheus (metrics-focused) |
| **Query Language** | PromQL | InfluxQL / Flux | SQL | Prometheus (powerful aggregations) |
| **Storage Model** | Local disk (TSDB) | Local disk | PostgreSQL | Prometheus (embedded) |
| **Retention** | Configurable (15d default) | Configurable | Configurable | All equal |
| **Alerting** | Native (Alertmanager) | Kapacitor / external | External (Grafana) | Prometheus (integrated) |
| **Grafana Integration** | Excellent | Excellent | Excellent | All equal |
| **Python Client** | `prometheus-client` | `influxdb-client` | `psycopg2` + TimescaleDB | Prometheus (mature) |
| **FastAPI Integration** | `prometheus-fastapi-instrumentator` | Manual | Manual | Prometheus (native) |
| **Operational Complexity** | Low (single binary) | Low (single binary) | Medium (PostgreSQL) | Prometheus (simplest) |
| **Performance (Metrics)** | Excellent | Good | Good | Prometheus (optimized) |
| **Cardinality Handling** | Good (labels) | Good | Moderate | Prometheus (metrics) |
| **Production Maturity** | Excellent (CNCF) | Good | Good | Prometheus (industry standard) |

**Verdict**: **Prometheus** is the optimal choice for audio streaming metrics due to:
1. Native FastAPI integration (`prometheus-fastapi-instrumentator`)
2. Pull-based model minimizes impact on audio pipeline
3. Powerful PromQL for latency percentiles and aggregations
4. Integrated alerting via Alertmanager
5. Industry standard for real-time monitoring (Spotify, SoundCloud)

### 4.2 Prometheus Configuration for Audio Streaming

```yaml
# prometheus.yml
global:
  scrape_interval: 15s  # Scrape metrics every 15 seconds
  evaluation_interval: 15s  # Evaluate rules every 15 seconds

scrape_configs:
  - job_name: 'auralis'
    static_configs:
      - targets: ['localhost:8000']  # FastAPI /metrics endpoint
    metrics_path: '/metrics'
    scrape_interval: 5s  # Scrape more frequently for real-time data
```

### 4.3 Alternative: OpenTelemetry (Future-Proof)

**OpenTelemetry** is the emerging standard for observability but adds complexity:
- **Pros**: Unified metrics, traces, and logs; vendor-neutral; future-proof
- **Cons**: Requires additional infrastructure (collector, backend); higher learning curve
- **Recommendation**: Stick with Prometheus for current scope; migrate to OTel if adding distributed tracing

---

## 5. Alerting Thresholds for Audio Quality Degradation

### 5.1 Alert Severity Levels

| Severity | Description | Response Time | Example |
|----------|-------------|---------------|---------|
| **Critical** | Service down or severe degradation | Immediate (page on-call) | All connections failing |
| **High** | Major degradation affecting users | <15 minutes | >50% chunks late |
| **Medium** | Performance degradation | <1 hour | CPU >90% sustained |
| **Low** | Warning signs, no user impact | <24 hours | Memory growth trend |

### 5.2 Audio-Specific Alert Rules

#### Critical Alerts

```yaml
# prometheus/alerts/audio_critical.yml
groups:
  - name: audio_critical
    interval: 30s
    rules:
      # Buffer underruns indicate audio glitches
      - alert: AudioBufferUnderrun
        expr: increase(ring_buffer_underruns_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Audio buffer underrun detected"
          description: "Ring buffer underruns in last 5 minutes: {{ $value }}"

      # Synthesis latency exceeds real-time constraint
      - alert: SynthesisLatencyHigh
        expr: histogram_quantile(0.99, rate(audio_synthesis_latency_ms_bucket[5m])) > 150
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Audio synthesis latency too high"
          description: "P99 synthesis latency: {{ $value }}ms (threshold: 150ms)"

      # All WebSocket connections failing
      - alert: NoActiveConnections
        expr: websocket_active_connections == 0 AND increase(websocket_connections_total[5m]) > 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "No active WebSocket connections despite attempts"
          description: "Connections attempted but none active"
```

#### High Priority Alerts

```yaml
  - name: audio_high
    interval: 1m
    rules:
      # Chunk delivery latency degraded
      - alert: ChunkDeliveryLatencyHigh
        expr: histogram_quantile(0.99, rate(audio_chunk_delivery_latency_ms_bucket[5m])) > 100
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "Audio chunk delivery latency degraded"
          description: "P99 chunk delivery: {{ $value }}ms (threshold: 100ms)"

      # Jitter excessive
      - alert: AudioJitterHigh
        expr: histogram_quantile(0.95, rate(audio_chunk_jitter_ms_bucket[5m])) > 50
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "Audio jitter too high"
          description: "P95 jitter: {{ $value }}ms (threshold: 50ms)"

      # Buffer depth critically low
      - alert: RingBufferDepthLow
        expr: ring_buffer_depth_ms < 200
        for: 2m
        labels:
          severity: high
        annotations:
          summary: "Ring buffer depth critically low"
          description: "Buffer depth: {{ $value }}ms (threshold: 200ms)"
```

#### Medium Priority Alerts

```yaml
  - name: audio_medium
    interval: 2m
    rules:
      # CPU utilization high
      - alert: CPUUtilizationHigh
        expr: cpu_utilization_percent > 90
        for: 10m
        labels:
          severity: medium
        annotations:
          summary: "CPU utilization high"
          description: "CPU usage: {{ $value }}% (threshold: 90%)"

      # Memory growth detected
      - alert: MemoryGrowth
        expr: deriv(memory_usage_mb[1h]) > 10
        for: 1h
        labels:
          severity: medium
        annotations:
          summary: "Memory usage growing"
          description: "Memory growth rate: {{ $value }}MB/hour"
```

### 5.3 Alertmanager Configuration

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

  routes:
    # Critical alerts: page immediately
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true

    # High priority: Slack notification
    - match:
        severity: high
      receiver: 'slack'

    # Medium/Low: Email digest
    - match:
        severity: medium
      receiver: 'email'

receivers:
  - name: 'default'
    email_configs:
      - to: 'ops@auralis.io'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '<pagerduty_key>'

  - name: 'slack'
    slack_configs:
      - api_url: '<slack_webhook>'
        channel: '#auralis-alerts'

  - name: 'email'
    email_configs:
      - to: 'ops@auralis.io'
```

---

## 6. Performance Dashboard Design

### 6.1 Grafana Dashboard Layout

**Recommended Structure**: 4-row layout with real-time focus

#### Row 1: Audio Quality Metrics (Most Critical)
- **Panel 1**: Audio Synthesis Latency (P50, P95, P99) - Line graph
- **Panel 2**: Chunk Delivery Latency (P50, P95, P99) - Line graph
- **Panel 3**: Buffer Underruns (Counter) - Stat panel (should be 0)
- **Panel 4**: Active WebSocket Connections - Gauge

#### Row 2: Buffer Health
- **Panel 5**: Ring Buffer Depth Over Time - Area graph
- **Panel 6**: Buffer Utilization Percentage - Gauge
- **Panel 7**: Audio Chunk Jitter - Histogram heatmap
- **Panel 8**: WebSocket Chunk Send Rate - Line graph

#### Row 3: System Resources
- **Panel 9**: CPU Utilization - Line graph
- **Panel 10**: GPU Utilization (if available) - Line graph
- **Panel 11**: Memory Usage - Area graph
- **Panel 12**: GPU Memory (CUDA only) - Area graph

#### Row 4: Network and Connections
- **Panel 13**: Network Bandwidth - Area graph
- **Panel 14**: WebSocket Connection Duration - Histogram
- **Panel 15**: Chunks Sent vs. Errors - Line graph (dual-axis)
- **Panel 16**: Active Connections Timeline - Line graph

### 6.2 Key PromQL Queries

#### Audio Synthesis Latency (P99)
```promql
histogram_quantile(0.99,
  rate(audio_synthesis_latency_ms_bucket[5m])
)
```

#### Chunk Delivery Success Rate
```promql
(
  rate(websocket_chunks_sent_total[5m]) -
  rate(websocket_send_errors_total[5m])
) / rate(websocket_chunks_sent_total[5m]) * 100
```

#### Buffer Depth (Moving Average)
```promql
avg_over_time(ring_buffer_depth_ms[1m])
```

#### Jitter (P95)
```promql
histogram_quantile(0.95,
  rate(audio_chunk_jitter_ms_bucket[5m])
)
```

#### CPU/GPU Utilization (Average)
```promql
avg_over_time(cpu_utilization_percent[5m])
avg_over_time(gpu_utilization_percent[5m])
```

### 6.3 Real-Time Dashboard Features

1. **Auto-Refresh**: Set to 5-second intervals for real-time monitoring
2. **Alert Annotations**: Display alerts on graphs when triggered
3. **Variable Filters**: Allow filtering by connection ID or time range
4. **Drill-Down Links**: Click metrics to view detailed logs
5. **Mobile View**: Responsive layout for on-call monitoring

---

## 7. Lightweight Logging Patterns (Non-Blocking)

### 7.1 Structured Logging with Loguru

**Current Implementation**: Auralis uses `loguru` for logging.

**Best Practices**:
1. **Async handlers** for file/network logging
2. **Log levels** to control verbosity
3. **Structured fields** for queryability
4. **Sampling** for high-frequency events

### 7.2 Async Logging Configuration

```python
# server/logging_config.py
from loguru import logger
import sys

def configure_logging(level: str = "INFO", async_enabled: bool = True):
    """
    Configure non-blocking logging for real-time audio system.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        async_enabled: Use async handlers to avoid blocking audio threads
    """
    # Remove default handler
    logger.remove()

    # Console logging (synchronous for development, async for production)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        enqueue=async_enabled,  # Enable async queue
        backtrace=True,
        diagnose=True
    )

    # File logging (async, rotated daily)
    logger.add(
        "logs/auralis_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        compression="zip",
        level=level,
        enqueue=True,  # Always async for file I/O
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        serialize=True  # JSON format for log aggregation
    )

    # Error-only log file
    logger.add(
        "logs/auralis_errors_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="90 days",
        level="ERROR",
        enqueue=True,
        serialize=True
    )

# Initialize in main.py
from server.logging_config import configure_logging
configure_logging(level="INFO", async_enabled=True)
```

### 7.3 Sampling High-Frequency Events

```python
# Avoid logging every audio chunk (100ms = 10 logs/second)
# Sample every 10th chunk instead

chunk_counter = 0

async def _stream_audio(self, websocket: WebSocket) -> None:
    global chunk_counter

    while True:
        audio_data = await asyncio.to_thread(...)
        chunk = AudioChunk(audio_data, self.sequence_counter)
        await websocket.send_text(chunk.to_json())

        chunk_counter += 1

        # Log every 10th chunk (1 log/second instead of 10/second)
        if chunk_counter % 10 == 0:
            logger.debug(
                f"Streamed chunk #{self.sequence_counter}",
                extra={
                    "buffer_depth_ms": self.ring_buffer.get_buffer_depth_ms(),
                    "active_connections": len(self.active_connections)
                }
            )
```

### 7.4 Structured Logging for Metrics

```python
# Use structured fields for log aggregation (ELK, Loki)
logger.info(
    "Audio synthesis completed",
    extra={
        "synthesis_time_ms": synthesis_time_ms,
        "phrase_duration_sec": duration_sec,
        "num_chords": len(chord_events),
        "num_melody_notes": len(melody_events),
        "device_type": self.device.type
    }
)

# Query logs in Loki/Elasticsearch:
# {synthesis_time_ms > 100}
```

### 7.5 Log Levels for Audio Pipeline

| Component | Log Level | Rationale |
|-----------|-----------|-----------|
| **Synthesis Loop** | INFO (sampled) | Log every 10th phrase, not every synthesis |
| **WebSocket Streaming** | DEBUG (sampled) | Log every 10th chunk, not every send |
| **Buffer Underruns** | WARNING | Always log underruns (rare, critical) |
| **Connection Events** | INFO | Log all connect/disconnect (low frequency) |
| **Errors** | ERROR | Always log errors |
| **Performance Metrics** | DEBUG | High-frequency data goes to Prometheus, not logs |

---

## 8. Architecture Recommendations

### 8.1 Metrics Collection Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Auralis FastAPI Server                    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Synthesis   │  │ Ring Buffer  │  │  WebSocket   │      │
│  │    Engine    │  │              │  │   Streaming  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         │ observe()        │ set()           │ inc()        │
│         v                  v                  v              │
│  ┌────────────────────────────────────────────────────┐     │
│  │         Prometheus Metrics Registry                │     │
│  │  - audio_synthesis_latency_ms (histogram)         │     │
│  │  - ring_buffer_depth_ms (gauge)                   │     │
│  │  - websocket_chunks_sent_total (counter)          │     │
│  │  - cpu_utilization_percent (gauge)                │     │
│  └────────────────┬───────────────────────────────────┘     │
│                   │                                          │
│                   │ /metrics endpoint                        │
│                   v                                          │
│  ┌────────────────────────────────────────────────────┐     │
│  │     FastAPI /metrics Endpoint (HTTP GET)           │     │
│  └────────────────┬───────────────────────────────────┘     │
└───────────────────┼──────────────────────────────────────────┘
                    │
                    │ HTTP scrape (every 5s)
                    v
         ┌──────────────────────┐
         │   Prometheus Server  │
         │   - Time-series DB   │
         │   - PromQL queries   │
         │   - Alert evaluation │
         └──────────┬───────────┘
                    │
                    │ PromQL queries
                    v
         ┌──────────────────────┐
         │  Grafana Dashboard   │
         │  - Real-time graphs  │
         │  - Alert annotations │
         │  - Mobile view       │
         └──────────────────────┘
                    │
                    │ Alerts
                    v
         ┌──────────────────────┐
         │   Alertmanager       │
         │  - PagerDuty         │
         │  - Slack             │
         │  - Email             │
         └──────────────────────┘
```

---

## 9. Implementation Priorities

### Phase 1: Core Metrics (Week 1)
1. Install `prometheus-client`, `prometheus-fastapi-instrumentator`, and `psutil` via uv
2. Define core audio metrics (synthesis latency, buffer depth, underruns) in `server/metrics.py`
3. Instrument synthesis loop in `server/main.py`
4. Instrument WebSocket streaming in `server/streaming_server.py`
5. Create `/metrics` endpoint
6. Deploy Prometheus server and scrape metrics

### Phase 2: System Metrics (Week 2)
1. Add background metrics collection task in `lifespan()`
2. Implement CPU/GPU/memory sampling
3. Add network bandwidth tracking
4. Test metrics collection overhead (<1% CPU)

### Phase 3: Alerting (Week 3)
1. Configure Prometheus alert rules
2. Deploy Alertmanager
3. Integrate Slack/PagerDuty
4. Test alert thresholds with load testing

### Phase 4: Dashboards (Week 4)
1. Create Grafana dashboard with 4-row layout
2. Implement PromQL queries for all panels
3. Configure auto-refresh and alert annotations
4. Mobile-responsive design

---

## 10. Open Questions

1. **Metric Retention**: How long should metrics be retained? (Recommend: 15 days local, 90 days remote storage)
2. **Distributed Tracing**: Should we add OpenTelemetry for request tracing? (Recommend: No for current scope, add in Phase 4)
3. **Log Aggregation**: Should logs be sent to ELK/Loki? (Recommend: Yes for production, optional for MVP)
4. **Cost Analysis**: What is the infrastructure cost for Prometheus/Grafana? (Recommend: Self-hosted = $0, Grafana Cloud = ~$50/month)
5. **Mobile Dashboard**: Do we need a dedicated mobile app for monitoring? (Recommend: No, use responsive Grafana)

---

## 11. Next Steps

1. **Review this research** with stakeholders and validate approach
2. **Run `/speckit.plan`** to continue implementation plan generation
3. **Implement Phase 1** (Core Metrics) in tasks.md
4. **Load test** with 10 concurrent connections to validate metrics accuracy
5. **Deploy Prometheus/Grafana** in staging environment
6. **Create alerts** based on production thresholds
7. **Document runbooks** for alert response procedures

---

**Research Completed**: 2025-12-26
**Next Command**: `/speckit.plan` to continue Phase 1 artifacts
