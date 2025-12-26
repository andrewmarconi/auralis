# Monitoring & Metrics Specification

## Overview

This document defines the metrics collection, monitoring dashboards, and alerting strategy for Auralis.

---

## 1. Key Metrics

### 1.1 Synthesis Metrics

| Metric | Unit | Target | Alert Threshold |
|--------|------|--------|-----------------|
| **Synthesis Latency** | ms | <500ms | >2000ms |
| **Real-Time Factor** | × | >20× | <5× |
| **Synthesis Failure Rate** | % | 0% | >5% |
| **GPU Memory Usage** | MB | <2GB | >6GB |
| **Audio Quality** (peak amplitude) | dB | -6dB to 0dB | >0dB (clipping) |

### 1.2 Streaming Metrics

| Metric | Unit | Target | Alert Threshold |
|--------|------|--------|-----------------|
| **Buffer Depth** | ms | 150-500ms | <50ms or >1000ms |
| **Chunk Send Rate** | chunks/sec | 10/sec | <8/sec |
| **WebSocket Latency** | ms | <100ms | >500ms |
| **Dropped Chunks** | count | 0 | >10/min |
| **Active Connections** | count | Variable | >max_clients |

### 1.3 Generation Metrics

| Metric | Unit | Target | Alert Threshold |
|--------|------|--------|-----------------|
| **Phrase Generation Time** | ms | <2000ms | >5000ms |
| **Phrase Queue Depth** | count | 1-3 | 0 (underrun) |
| **Generation Failures** | count | 0 | >5/hour |
| **Average Notes per Phrase** | count | 8-16 | N/A |

### 1.4 System Metrics

| Metric | Unit | Target | Alert Threshold |
|--------|------|--------|-----------------|
| **CPU Usage** | % | <60% | >90% |
| **Memory Usage** | MB | <1GB | >4GB |
| **Disk I/O** | MB/s | <10 MB/s | >100 MB/s |
| **Network Bandwidth** | Mbps | <5 Mbps/client | >50 Mbps total |

---

## 2. Metrics Collection

### 2.1 Metrics Collector

```python
# auralis/monitoring/metrics.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict
import time
import psutil
from collections import deque
from loguru import logger


@dataclass
class MetricsSnapshot:
    """Single point-in-time metrics snapshot."""

    timestamp: datetime
    synthesis_latency_ms: float
    real_time_factor: float
    buffer_depth_ms: float
    active_clients: int
    phrase_queue_depth: int
    cpu_percent: float
    memory_mb: float
    chunks_sent: int
    chunks_dropped: int


class MetricsCollector:
    """
    Centralized metrics collection and aggregation.

    Collects metrics from various components and provides
    aggregated statistics for monitoring.
    """

    def __init__(self, history_size: int = 1000):
        """
        Args:
            history_size: Number of snapshots to keep in memory
        """
        self.history: deque = deque(maxlen=history_size)

        # Counters
        self.total_phrases_generated = 0
        self.total_synthesis_errors = 0
        self.total_chunks_sent = 0
        self.total_chunks_dropped = 0

        # Timing
        self._last_synthesis_start = None

        logger.info("MetricsCollector initialized")

    def record_synthesis_start(self):
        """Mark start of synthesis operation."""
        self._last_synthesis_start = time.perf_counter()

    def record_synthesis_end(self, success: bool = True, phrase_duration_sec: float = 22.0):
        """
        Record synthesis completion.

        Args:
            success: Whether synthesis succeeded
            phrase_duration_sec: Duration of rendered phrase
        """
        if self._last_synthesis_start is None:
            logger.warning("Synthesis end recorded without start")
            return

        elapsed = time.perf_counter() - self._last_synthesis_start
        latency_ms = elapsed * 1000
        rtf = phrase_duration_sec / elapsed if elapsed > 0 else 0

        if success:
            self.total_phrases_generated += 1
        else:
            self.total_synthesis_errors += 1

        logger.debug(f"Synthesis: {latency_ms:.0f}ms ({rtf:.1f}× real-time)")

        return latency_ms, rtf

    def record_chunk_sent(self):
        """Record successful chunk send."""
        self.total_chunks_sent += 1

    def record_chunk_dropped(self):
        """Record dropped chunk (client queue full)."""
        self.total_chunks_dropped += 1

    def take_snapshot(
        self,
        buffer_depth_ms: float,
        active_clients: int,
        phrase_queue_depth: int,
    ) -> MetricsSnapshot:
        """
        Take a snapshot of current metrics.

        Args:
            buffer_depth_ms: Current audio buffer depth
            active_clients: Number of connected clients
            phrase_queue_depth: Phrases in composition queue

        Returns:
            MetricsSnapshot instance
        """
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

        # Calculate recent synthesis latency (avg of last 10 phrases)
        recent_latencies = [s.synthesis_latency_ms for s in list(self.history)[-10:]]
        avg_synthesis_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0

        # Calculate recent RTF
        recent_rtfs = [s.real_time_factor for s in list(self.history)[-10:]]
        avg_rtf = sum(recent_rtfs) / len(recent_rtfs) if recent_rtfs else 0

        snapshot = MetricsSnapshot(
            timestamp=datetime.utcnow(),
            synthesis_latency_ms=avg_synthesis_latency,
            real_time_factor=avg_rtf,
            buffer_depth_ms=buffer_depth_ms,
            active_clients=active_clients,
            phrase_queue_depth=phrase_queue_depth,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            chunks_sent=self.total_chunks_sent,
            chunks_dropped=self.total_chunks_dropped,
        )

        self.history.append(snapshot)

        return snapshot

    def get_summary_stats(self) -> Dict:
        """
        Get aggregated summary statistics.

        Returns:
            Dictionary of summary metrics
        """
        if not self.history:
            return {}

        recent = list(self.history)[-100:]  # Last 100 snapshots

        return {
            "uptime_seconds": (datetime.utcnow() - self.history[0].timestamp).total_seconds()
            if self.history
            else 0,
            "total_phrases_generated": self.total_phrases_generated,
            "total_synthesis_errors": self.total_synthesis_errors,
            "synthesis_error_rate": self.total_synthesis_errors
            / max(self.total_phrases_generated, 1),
            "total_chunks_sent": self.total_chunks_sent,
            "total_chunks_dropped": self.total_chunks_dropped,
            "chunk_drop_rate": self.total_chunks_dropped / max(self.total_chunks_sent, 1),
            "avg_synthesis_latency_ms": sum(s.synthesis_latency_ms for s in recent)
            / len(recent),
            "avg_rtf": sum(s.real_time_factor for s in recent) / len(recent),
            "avg_buffer_depth_ms": sum(s.buffer_depth_ms for s in recent) / len(recent),
            "avg_cpu_percent": sum(s.cpu_percent for s in recent) / len(recent),
            "avg_memory_mb": sum(s.memory_mb for s in recent) / len(recent),
            "current_active_clients": recent[-1].active_clients if recent else 0,
        }

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        stats = self.get_summary_stats()

        metrics = f"""
# HELP auralis_phrases_total Total phrases generated
# TYPE auralis_phrases_total counter
auralis_phrases_total {stats.get('total_phrases_generated', 0)}

# HELP auralis_synthesis_errors_total Total synthesis errors
# TYPE auralis_synthesis_errors_total counter
auralis_synthesis_errors_total {stats.get('total_synthesis_errors', 0)}

# HELP auralis_synthesis_latency_ms Average synthesis latency in milliseconds
# TYPE auralis_synthesis_latency_ms gauge
auralis_synthesis_latency_ms {stats.get('avg_synthesis_latency_ms', 0):.2f}

# HELP auralis_real_time_factor Synthesis real-time factor
# TYPE auralis_real_time_factor gauge
auralis_real_time_factor {stats.get('avg_rtf', 0):.2f}

# HELP auralis_buffer_depth_ms Audio buffer depth in milliseconds
# TYPE auralis_buffer_depth_ms gauge
auralis_buffer_depth_ms {stats.get('avg_buffer_depth_ms', 0):.2f}

# HELP auralis_active_clients Number of active WebSocket clients
# TYPE auralis_active_clients gauge
auralis_active_clients {stats.get('current_active_clients', 0)}

# HELP auralis_cpu_percent CPU usage percentage
# TYPE auralis_cpu_percent gauge
auralis_cpu_percent {stats.get('avg_cpu_percent', 0):.2f}

# HELP auralis_memory_mb Memory usage in megabytes
# TYPE auralis_memory_mb gauge
auralis_memory_mb {stats.get('avg_memory_mb', 0):.2f}

# HELP auralis_chunks_sent_total Total audio chunks sent
# TYPE auralis_chunks_sent_total counter
auralis_chunks_sent_total {stats.get('total_chunks_sent', 0)}

# HELP auralis_chunks_dropped_total Total audio chunks dropped
# TYPE auralis_chunks_dropped_total counter
auralis_chunks_dropped_total {stats.get('total_chunks_dropped', 0)}
"""
        return metrics.strip()


# Global metrics collector
metrics_collector = MetricsCollector()
```

---

## 3. Metrics API Endpoints

```python
# server/api/metrics_routes.py
from fastapi import APIRouter
from auralis.monitoring.metrics import metrics_collector

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


@router.get("/")
async def get_metrics():
    """
    Get current metrics summary.

    Returns JSON summary of all key metrics.
    """
    return metrics_collector.get_summary_stats()


@router.get("/prometheus")
async def get_prometheus_metrics():
    """
    Get metrics in Prometheus format.

    Returns:
        Plain text Prometheus metrics
    """
    from fastapi.responses import PlainTextResponse

    metrics_text = metrics_collector.export_prometheus()
    return PlainTextResponse(metrics_text, media_type="text/plain")


@router.get("/health")
async def get_health():
    """
    Simplified health check.

    Returns:
        200: Healthy
        503: Unhealthy
    """
    stats = metrics_collector.get_summary_stats()

    # Check key health indicators
    is_healthy = (
        stats.get("synthesis_error_rate", 1.0) < 0.1
        and stats.get("avg_rtf", 0) > 2.0
        and stats.get("avg_cpu_percent", 100) < 90
    )

    status_code = 200 if is_healthy else 503

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "synthesis_error_rate": stats.get("synthesis_error_rate", 0),
        "rtf": stats.get("avg_rtf", 0),
        "cpu_percent": stats.get("avg_cpu_percent", 0),
    }
```

---

## 4. Periodic Metrics Reporting

```python
# server/monitoring_worker.py
import asyncio
from loguru import logger
from auralis.monitoring.metrics import metrics_collector


async def metrics_reporter(
    ring_buffer,
    client_manager,
    composition_engine,
    interval_sec: int = 10,
):
    """
    Periodically collect and log metrics.

    Args:
        interval_sec: Reporting interval in seconds
    """
    while True:
        await asyncio.sleep(interval_sec)

        try:
            # Take snapshot
            snapshot = metrics_collector.take_snapshot(
                buffer_depth_ms=ring_buffer.buffer_depth_ms(),
                active_clients=len(client_manager.clients),
                phrase_queue_depth=composition_engine.phrase_queue.qsize(),
            )

            # Log summary
            logger.info(
                f"Metrics: "
                f"clients={snapshot.active_clients}, "
                f"buffer={snapshot.buffer_depth_ms:.0f}ms, "
                f"cpu={snapshot.cpu_percent:.0f}%, "
                f"mem={snapshot.memory_mb:.0f}MB, "
                f"chunks_sent={snapshot.chunks_sent}, "
                f"dropped={snapshot.chunks_dropped}"
            )

            # Check for alerts
            check_alert_conditions(snapshot)

        except Exception as e:
            logger.error(f"Metrics reporting error: {e}", exc_info=True)


def check_alert_conditions(snapshot: MetricsSnapshot):
    """Check metrics against alert thresholds."""

    # High CPU
    if snapshot.cpu_percent > 90:
        logger.warning(f"⚠️ HIGH CPU: {snapshot.cpu_percent:.0f}%")

    # Low buffer
    if snapshot.buffer_depth_ms < 50:
        logger.warning(f"⚠️ LOW BUFFER: {snapshot.buffer_depth_ms:.0f}ms")

    # High buffer (client not consuming)
    if snapshot.buffer_depth_ms > 1000:
        logger.warning(f"⚠️ HIGH BUFFER: {snapshot.buffer_depth_ms:.0f}ms (client slow?)")

    # Low RTF
    if snapshot.real_time_factor < 5:
        logger.warning(f"⚠️ LOW RTF: {snapshot.real_time_factor:.1f}×")

    # Memory
    if snapshot.memory_mb > 4000:  # 4GB
        logger.warning(f"⚠️ HIGH MEMORY: {snapshot.memory_mb:.0f}MB")
```

---

## 5. Client-Side Metrics

### 5.1 Client Metrics Collection

```javascript
// client/metrics.js

class ClientMetrics {
    constructor() {
        this.chunksReceived = 0;
        this.chunkErrors = 0;
        this.bufferUnderruns = 0;
        this.latencySamples = [];
    }

    recordChunkReceived(timestamp) {
        this.chunksReceived++;

        // Calculate latency (if server includes timestamp)
        const now = Date.now();
        const serverTime = new Date(timestamp).getTime();
        const latency = now - serverTime;

        this.latencySamples.push(latency);

        // Keep only last 100 samples
        if (this.latencySamples.length > 100) {
            this.latencySamples.shift();
        }
    }

    recordChunkError() {
        this.chunkErrors++;
    }

    recordBufferUnderrun() {
        this.bufferUnderruns++;
    }

    getStats() {
        const avgLatency = this.latencySamples.length > 0
            ? this.latencySamples.reduce((a, b) => a + b, 0) / this.latencySamples.length
            : 0;

        return {
            chunksReceived: this.chunksReceived,
            chunkErrors: this.chunkErrors,
            bufferUnderruns: this.bufferUnderruns,
            avgLatencyMs: avgLatency,
            errorRate: this.chunkErrors / Math.max(this.chunksReceived, 1),
        };
    }

    // Send metrics to server periodically
    async sendToServer(ws) {
        const stats = this.getStats();

        ws.send(JSON.stringify({
            type: 'client_metrics',
            metrics: stats,
        }));
    }
}
```

---

## 6. Monitoring Dashboard (Simple)

### 6.1 HTML Dashboard

```html
<!-- client/dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Auralis Monitoring Dashboard</title>
    <style>
        body {
            font-family: monospace;
            background: #1a1a1a;
            color: #00ff00;
            padding: 20px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        .metric-card {
            background: #2a2a2a;
            border: 1px solid #00ff00;
            padding: 15px;
            border-radius: 5px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
        }
        .metric-label {
            color: #888;
            font-size: 0.9em;
        }
        .status-ok { color: #00ff00; }
        .status-warning { color: #ffaa00; }
        .status-error { color: #ff0000; }
    </style>
</head>
<body>
    <h1>Auralis Monitoring Dashboard</h1>

    <div class="metric-grid" id="metrics"></div>

    <script>
        async function fetchMetrics() {
            const response = await fetch('http://localhost:8000/api/v1/metrics');
            const data = await response.json();
            displayMetrics(data);
        }

        function displayMetrics(data) {
            const container = document.getElementById('metrics');

            const metrics = [
                { label: 'Active Clients', value: data.current_active_clients, unit: '' },
                { label: 'Phrases Generated', value: data.total_phrases_generated, unit: '' },
                { label: 'Synthesis Latency', value: data.avg_synthesis_latency_ms?.toFixed(0), unit: 'ms' },
                { label: 'Real-Time Factor', value: data.avg_rtf?.toFixed(1), unit: '×' },
                { label: 'Buffer Depth', value: data.avg_buffer_depth_ms?.toFixed(0), unit: 'ms' },
                { label: 'CPU Usage', value: data.avg_cpu_percent?.toFixed(0), unit: '%' },
                { label: 'Memory Usage', value: data.avg_memory_mb?.toFixed(0), unit: 'MB' },
                { label: 'Error Rate', value: (data.synthesis_error_rate * 100)?.toFixed(1), unit: '%' },
            ];

            container.innerHTML = metrics.map(m => `
                <div class="metric-card">
                    <div class="metric-label">${m.label}</div>
                    <div class="metric-value">${m.value}${m.unit}</div>
                </div>
            `).join('');
        }

        // Update every 2 seconds
        setInterval(fetchMetrics, 2000);
        fetchMetrics();  // Initial load
    </script>
</body>
</html>
```

---

## 7. Alerting (Future: Integration with External Systems)

### 7.1 Alert Manager (Conceptual)

```python
# auralis/monitoring/alerts.py
from enum import Enum
from loguru import logger


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertManager:
    """
    Send alerts to external systems.

    Future integrations:
    - Email (SMTP)
    - Slack webhooks
    - PagerDuty
    - Prometheus Alertmanager
    """

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url

    def send_alert(self, severity: AlertSeverity, title: str, message: str):
        """
        Send alert to configured destination.

        Args:
            severity: Alert severity level
            title: Short alert title
            message: Detailed message
        """
        logger.log(severity.value.upper(), f"ALERT: {title} - {message}")

        # TODO: Implement webhook sending
        if self.webhook_url:
            # Send to Slack/Discord/etc
            pass


# Example usage
alert_manager = AlertManager()

# Trigger alert on critical metrics
if synthesis_error_rate > 0.1:
    alert_manager.send_alert(
        AlertSeverity.ERROR,
        "High Synthesis Error Rate",
        f"Synthesis failures: {synthesis_error_rate * 100:.1f}%",
    )
```

---

## 8. Summary

**Metrics Collection**: Centralized collector tracks all key metrics
**Reporting**: Periodic snapshots logged every 10 seconds
**API**: REST endpoints for metrics (`/api/v1/metrics`)
**Dashboard**: Simple HTML dashboard for visualization
**Alerting**: Threshold-based alerts logged to console/file
**Future**: Prometheus integration, Grafana dashboards, external alerting

---

This monitoring system provides comprehensive visibility into Auralis performance and health.
