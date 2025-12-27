# Metrics API Contract - Performance Optimizations

**Feature**: Performance Optimizations
**Branch**: `003-performance-optimizations`
**Date**: 2025-12-26
**Version**: 1.0.0

## Overview

This document defines the Prometheus metrics API for monitoring Auralis performance. The API exposes real-time metrics for synthesis latency, buffer health, memory usage, and concurrent connections.

---

## Endpoint

### Prometheus Metrics Endpoint

**URL**: `http://{host}:{port}/metrics`

**Method**: GET

**Response Format**: Prometheus text-based exposition format

**Response Headers**:
```
Content-Type: text/plain; version=0.0.4; charset=utf-8
```

**Authentication**: None (intended for internal monitoring network)

**Rate Limiting**: None

---

## Metrics Catalog

### 1. Synthesis Metrics

#### synthesis_latency_seconds

**Type**: Histogram

**Description**: Time to render audio phrase (from phrase data to numpy output)

**Labels**: None

**Buckets**: `[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]`

**Units**: seconds

**Example**:
```prometheus
# HELP synthesis_latency_seconds Time to render audio phrase
# TYPE synthesis_latency_seconds histogram
synthesis_latency_seconds_bucket{le="0.01"} 0
synthesis_latency_seconds_bucket{le="0.025"} 15
synthesis_latency_seconds_bucket{le="0.05"} 250
synthesis_latency_seconds_bucket{le="0.075"} 980
synthesis_latency_seconds_bucket{le="0.1"} 1200
synthesis_latency_seconds_bucket{le="0.15"} 1205
synthesis_latency_seconds_bucket{le="+Inf"} 1210
synthesis_latency_seconds_sum 58.5
synthesis_latency_seconds_count 1210
```

**Query Examples**:
```promql
# p99 synthesis latency
histogram_quantile(0.99, rate(synthesis_latency_seconds_bucket[5m]))

# Average latency
rate(synthesis_latency_seconds_sum[5m]) / rate(synthesis_latency_seconds_count[5m])

# Phrases per second
rate(synthesis_latency_seconds_count[1m])
```

---

### 2. Buffer Metrics

#### buffer_depth_chunks

**Type**: Gauge

**Description**: Current ring buffer depth for each client

**Labels**:
- `client_id` (string): Unique client identifier

**Units**: chunks

**Example**:
```prometheus
# HELP buffer_depth_chunks Current ring buffer depth
# TYPE buffer_depth_chunks gauge
buffer_depth_chunks{client_id="client_123"} 10
buffer_depth_chunks{client_id="client_456"} 12
buffer_depth_chunks{client_id="client_789"} 8
```

**Query Examples**:
```promql
# Average buffer depth across all clients
avg(buffer_depth_chunks)

# Minimum buffer depth (closest to underrun)
min(buffer_depth_chunks)

# Clients with low buffer (<5 chunks)
buffer_depth_chunks < 5
```

---

#### chunk_delivery_jitter_ms

**Type**: Histogram

**Description**: Chunk delivery timing variance (deviation from expected 100ms interval)

**Labels**: None

**Buckets**: `[1, 2, 5, 10, 20, 30, 50, 75, 100, 200]`

**Units**: milliseconds

**Example**:
```prometheus
# HELP chunk_delivery_jitter_ms Chunk delivery timing variance
# TYPE chunk_delivery_jitter_ms histogram
chunk_delivery_jitter_ms_bucket{le="1"} 50
chunk_delivery_jitter_ms_bucket{le="2"} 200
chunk_delivery_jitter_ms_bucket{le="5"} 800
chunk_delivery_jitter_ms_bucket{le="10"} 1500
chunk_delivery_jitter_ms_bucket{le="20"} 2200
chunk_delivery_jitter_ms_bucket{le="50"} 2450
chunk_delivery_jitter_ms_bucket{le="+Inf"} 2500
chunk_delivery_jitter_ms_sum 25000
chunk_delivery_jitter_ms_count 2500
```

**Query Examples**:
```promql
# p95 jitter
histogram_quantile(0.95, rate(chunk_delivery_jitter_ms_bucket[5m]))

# Jitter increase over time
delta(chunk_delivery_jitter_ms_sum[10m]) / delta(chunk_delivery_jitter_ms_count[10m])
```

---

#### buffer_underruns_total

**Type**: Counter

**Description**: Total buffer underrun events per client

**Labels**:
- `client_id` (string): Unique client identifier

**Units**: count

**Example**:
```prometheus
# HELP buffer_underruns_total Buffer underrun events
# TYPE buffer_underruns_total counter
buffer_underruns_total{client_id="client_123"} 0
buffer_underruns_total{client_id="client_456"} 2
buffer_underruns_total{client_id="client_789"} 15
```

**Query Examples**:
```promql
# Underrun rate (per second)
rate(buffer_underruns_total[5m])

# Total underruns across all clients
sum(buffer_underruns_total)

# Clients with recent underruns
rate(buffer_underruns_total[1m]) > 0
```

---

### 3. WebSocket Metrics

#### active_websocket_connections

**Type**: Gauge

**Description**: Number of currently connected WebSocket clients

**Labels**: None

**Units**: count

**Example**:
```prometheus
# HELP active_websocket_connections Number of connected WebSocket clients
# TYPE active_websocket_connections gauge
active_websocket_connections 10
```

**Query Examples**:
```promql
# Current connections
active_websocket_connections

# Connection growth rate
deriv(active_websocket_connections[5m])

# Peak connections in last hour
max_over_time(active_websocket_connections[1h])
```

---

#### websocket_send_errors_total

**Type**: Counter

**Description**: Failed WebSocket send operations

**Labels**:
- `error_type` (string): Error classification
  - Values: `"disconnect"`, `"timeout"`, `"encoding_error"`, `"unknown"`

**Units**: count

**Example**:
```prometheus
# HELP websocket_send_errors_total Failed WebSocket send operations
# TYPE websocket_send_errors_total counter
websocket_send_errors_total{error_type="disconnect"} 5
websocket_send_errors_total{error_type="timeout"} 0
websocket_send_errors_total{error_type="encoding_error"} 1
```

**Query Examples**:
```promql
# Error rate by type
rate(websocket_send_errors_total[5m])

# Total error rate
sum(rate(websocket_send_errors_total[5m]))

# Errors per active connection
sum(rate(websocket_send_errors_total[5m])) / active_websocket_connections
```

---

### 4. Memory Metrics

#### memory_usage_mb

**Type**: Gauge

**Description**: Process memory usage (Resident Set Size)

**Labels**: None

**Units**: megabytes

**Example**:
```prometheus
# HELP memory_usage_mb Process memory usage (RSS)
# TYPE memory_usage_mb gauge
memory_usage_mb 450.5
```

**Query Examples**:
```promql
# Current memory usage
memory_usage_mb

# Memory growth rate (MB/hour)
deriv(memory_usage_mb[1h]) * 3600

# Memory leak detection (sustained growth)
deriv(memory_usage_mb[1h]) > 20  # Alert if >20 MB/hour
```

---

#### gpu_memory_allocated_mb

**Type**: Gauge

**Description**: GPU memory allocated (CUDA/MPS)

**Labels**: None

**Units**: megabytes

**Example**:
```prometheus
# HELP gpu_memory_allocated_mb GPU memory allocated
# TYPE gpu_memory_allocated_mb gauge
gpu_memory_allocated_mb 512.0
```

**Query Examples**:
```promql
# Current GPU memory
gpu_memory_allocated_mb

# GPU memory utilization (if total known)
gpu_memory_allocated_mb / 24576 * 100  # For RTX 3090 (24GB)

# GPU memory growth
delta(gpu_memory_allocated_mb[10m])
```

---

### 5. Encoding Metrics

#### chunk_encoding_duration_seconds

**Type**: Histogram

**Description**: base64 encoding duration per chunk

**Labels**: None

**Buckets**: `[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]`

**Units**: seconds

**Example**:
```prometheus
# HELP chunk_encoding_duration_seconds base64 encoding duration
# TYPE chunk_encoding_duration_seconds histogram
chunk_encoding_duration_seconds_bucket{le="0.0001"} 10
chunk_encoding_duration_seconds_bucket{le="0.0005"} 500
chunk_encoding_duration_seconds_bucket{le="0.001"} 1200
chunk_encoding_duration_seconds_bucket{le="0.002"} 1450
chunk_encoding_duration_seconds_bucket{le="+Inf"} 1500
chunk_encoding_duration_seconds_sum 1.2
chunk_encoding_duration_seconds_count 1500
```

**Query Examples**:
```promql
# p99 encoding latency
histogram_quantile(0.99, rate(chunk_encoding_duration_seconds_bucket[5m]))

# Average encoding time
rate(chunk_encoding_duration_seconds_sum[5m]) / rate(chunk_encoding_duration_seconds_count[5m])
```

---

### 6. Garbage Collection Metrics

#### gc_collections_total

**Type**: Counter

**Description**: Garbage collection count by generation

**Labels**:
- `generation` (string): GC generation (`"0"`, `"1"`, `"2"`)

**Units**: count

**Example**:
```prometheus
# HELP gc_collections_total Garbage collection count
# TYPE gc_collections_total counter
gc_collections_total{generation="0"} 1500
gc_collections_total{generation="1"} 50
gc_collections_total{generation="2"} 3
```

**Query Examples**:
```promql
# GC rate by generation
rate(gc_collections_total[5m])

# Gen-2 collections (full GC, expensive)
rate(gc_collections_total{generation="2"}[5m])

# Total GC events
sum(rate(gc_collections_total[5m]))
```

---

### 7. Generation Metrics

#### phrase_generation_rate_hz

**Type**: Gauge

**Description**: Phrases generated per second (long-term average)

**Labels**: None

**Units**: hertz (1/seconds)

**Example**:
```prometheus
# HELP phrase_generation_rate_hz Phrases generated per second
# TYPE phrase_generation_rate_hz gauge
phrase_generation_rate_hz 0.125
```

**Query Examples**:
```promql
# Current generation rate
phrase_generation_rate_hz

# Phrases per minute
phrase_generation_rate_hz * 60

# Generation rate drop
delta(phrase_generation_rate_hz[10m]) < -0.05
```

---

## Alert Rules

### Critical Alerts

#### BufferUnderrun

**Condition**: Buffer underruns >1 per minute for any client

**PromQL**:
```promql
rate(buffer_underruns_total[1m]) > 0.0167
```

**For**: 2 minutes

**Severity**: critical

**Annotations**:
- Summary: `"Buffer underruns detected for {{ $labels.client_id }}"`
- Description: `"Client {{ $labels.client_id }} experiencing {{ $value }} underruns/sec"`

**Remediation**:
1. Check client network conditions
2. Verify buffer tier escalation
3. Review server resource usage

---

#### SynthesisLatencyHigh

**Condition**: p99 synthesis latency >100ms

**PromQL**:
```promql
histogram_quantile(0.99, synthesis_latency_seconds_bucket) > 0.1
```

**For**: 5 minutes

**Severity**: critical

**Annotations**:
- Summary: `"Synthesis latency p99 exceeds 100ms"`
- Description: `"p99 latency: {{ $value }}s (target: <0.1s)"`

**Remediation**:
1. Check GPU utilization
2. Verify batch processing is enabled
3. Review memory allocations

---

#### MemoryLeak

**Condition**: Memory growth >20 MB/hour

**PromQL**:
```promql
deriv(memory_usage_mb[1h]) > 20
```

**For**: 1 hour

**Severity**: critical

**Annotations**:
- Summary: `"Memory leak detected"`
- Description: `"Memory growing at {{ $value }} MB/hour"`

**Remediation**:
1. Review tracemalloc snapshots
2. Check for unclosed resources
3. Trigger manual GC collection

---

### High Severity Alerts

#### HighChunkJitter

**Condition**: p95 jitter >50ms

**PromQL**:
```promql
histogram_quantile(0.95, chunk_delivery_jitter_ms_bucket) > 50
```

**For**: 10 minutes

**Severity**: high

**Annotations**:
- Summary: `"High chunk delivery jitter"`
- Description: `"p95 jitter: {{ $value }}ms (target: <50ms)"`

**Remediation**:
1. Check network conditions
2. Verify buffer tier is adaptive
3. Review server CPU usage

---

#### GPUMemoryHigh

**Condition**: GPU memory >90% of capacity

**PromQL**:
```promql
gpu_memory_allocated_mb > 0.9 * 24576  # For RTX 3090
```

**For**: 5 minutes

**Severity**: high

**Annotations**:
- Summary: `"GPU memory usage > 90%"`
- Description: `"GPU memory: {{ $value }} MB"`

**Remediation**:
1. Trigger GPU cache clear
2. Review memory pre-allocation sizes
3. Check for memory leaks

---

### Medium Severity Alerts

#### HighCPUUsage

**Condition**: CPU usage >90% for 10 minutes

**PromQL**:
```promql
rate(process_cpu_seconds_total[5m]) > 0.9
```

**For**: 10 minutes

**Severity**: medium

**Annotations**:
- Summary: `"CPU usage > 90% for 10 minutes"`

**Remediation**:
1. Check concurrent client count
2. Verify thread pool configuration
3. Review encoding optimizations

---

#### FrequentGCCollections

**Condition**: >1 gen-2 GC per 5 minutes

**PromQL**:
```promql
rate(gc_collections_total{generation="2"}[5m]) > 0.0033
```

**For**: 10 minutes

**Severity**: medium

**Annotations**:
- Summary: `"Frequent gen-2 GC collections"`
- Description: `"{{ $value }} gen-2 collections per second"`

**Remediation**:
1. Review GC threshold configuration
2. Check for memory churn
3. Verify object pooling

---

## Grafana Dashboard Queries

### Synthesis Performance Panel

**Title**: "Synthesis Latency (p50, p95, p99)"

**Queries**:
```promql
# p50
histogram_quantile(0.50, rate(synthesis_latency_seconds_bucket[5m]))

# p95
histogram_quantile(0.95, rate(synthesis_latency_seconds_bucket[5m]))

# p99
histogram_quantile(0.99, rate(synthesis_latency_seconds_bucket[5m]))
```

**Visualization**: Time series graph

**Thresholds**:
- Green: <75ms
- Yellow: 75-100ms
- Red: >100ms

---

### Buffer Health Panel

**Title**: "Buffer Depth (per client)"

**Query**:
```promql
buffer_depth_chunks
```

**Visualization**: Time series graph

**Legend**: `{{ client_id }}`

**Thresholds**:
- Red: <2 chunks (critical)
- Yellow: 2-5 chunks (warning)
- Green: >5 chunks (healthy)

---

### Memory Usage Panel

**Title**: "Memory Usage"

**Queries**:
```promql
# System memory
memory_usage_mb

# GPU memory
gpu_memory_allocated_mb
```

**Visualization**: Time series graph

**Legend**:
- `memory_usage_mb`: "System RAM"
- `gpu_memory_allocated_mb`: "GPU VRAM"

---

### Underrun Rate Panel

**Title**: "Buffer Underruns (rate)"

**Query**:
```promql
sum by (client_id) (rate(buffer_underruns_total[5m]))
```

**Visualization**: Time series graph

**Legend**: `{{ client_id }}`

**Thresholds**:
- Green: 0 underruns/sec
- Yellow: 0-0.01 underruns/sec
- Red: >0.01 underruns/sec

---

### Active Connections Panel

**Title**: "Active WebSocket Connections"

**Query**:
```promql
active_websocket_connections
```

**Visualization**: Stat panel

**Color**: Dynamic based on value
- Green: 0-10 clients
- Yellow: 11-15 clients
- Red: >15 clients (approaching limit)

---

## Metrics Collection Strategy

### Collection Intervals

**Frequency**: Every 5 seconds

**Implementation**:
```python
class MetricsCollectionServer:
    def __init__(self):
        self.metrics_interval_sec = 5

    async def _metrics_collection_loop(self):
        """Background task to collect metrics."""
        while True:
            await asyncio.sleep(self.metrics_interval_sec)
            self._collect_metrics()

    def _collect_metrics(self):
        """Collect all metrics (non-blocking)."""
        # Update gauges
        metrics.memory_usage.set(get_memory_usage_mb())
        metrics.gpu_memory.set(get_gpu_memory_mb())
        metrics.active_connections.set(len(self.active_clients))

        # Update per-client gauges
        for client_id in self.active_clients:
            stats = self.ring_buffer.get_client_stats(client_id)
            metrics.buffer_depth.labels(client_id=client_id).set(
                stats.get("chunks_buffered", 0)
            )
```

**Overhead**:
- CPU: <0.1% per collection
- Memory: Negligible (metrics pre-allocated)
- Total: ~0.02% CPU overhead

---

### Histogram Bucket Selection Rationale

**Synthesis Latency** (`[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]`):
- Fine-grained around target (100ms)
- Captures p95 and p99 accurately
- Wide range for outlier detection

**Chunk Jitter** (`[1, 2, 5, 10, 20, 30, 50, 75, 100, 200]`):
- Linear distribution for jitter measurement
- Critical threshold: 50ms (covers p95 target)
- Upper bound: 200ms (severe jitter detection)

**Encoding Duration** (`[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]`):
- Sub-millisecond precision
- Logarithmic distribution
- Target: <5ms for encoding

---

## Metrics Retention

**Prometheus Configuration**:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

storage:
  tsdb:
    retention.time: 30d
    retention.size: 10GB
```

**Retention Policy**:
- Short-term: 15 days (full resolution)
- Long-term: 30 days (downsampled to 1m)
- Archive: Export to external storage (optional)

---

## Security and Access Control

### Network Restrictions

**Bind Address**: `0.0.0.0:8000` (configure via environment)

**Firewall Rules**:
- Allow: Internal monitoring network (e.g., `10.0.0.0/8`)
- Deny: Public internet

**Best Practice**: Use reverse proxy (nginx) with authentication for external access

---

### Authentication (Optional)

**Basic Auth** (if exposed externally):

```nginx
# nginx.conf
location /metrics {
    auth_basic "Metrics";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:8000/metrics;
}
```

**Alternative**: Prometheus service discovery with TLS mutual authentication

---

## API Changelog

### Version 1.0.0 (2025-12-26) - Initial Release

**Added**:
- 11 core performance metrics
- Prometheus exposition format endpoint
- Alert rule definitions
- Grafana dashboard queries

**Changed**: N/A (initial release)

**Deprecated**: None

**Removed**: None

**Fixed**: N/A

**Security**: Recommend internal-only access

---

## References

- Prometheus Exposition Format: [OpenMetrics](https://github.com/OpenObservability/OpenMetrics/blob/main/specification/OpenMetrics.md)
- Prometheus Best Practices: [Metric and Label Naming](https://prometheus.io/docs/practices/naming/)
- Histogram Bucket Guidance: [Histograms and Summaries](https://prometheus.io/docs/practices/histograms/)

---

## Contact

For metrics API questions:
- Documentation: [Prometheus Python Client](https://github.com/prometheus/client_python)
- GitHub Issues: https://github.com/yourusername/auralis/issues
