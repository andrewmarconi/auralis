# HTTP API Contract: Auralis REST Endpoints

**Feature**: System Status & Metrics
**Branch**: `001-mvp-v2`
**API Version**: 1.0.0
**Date**: 2025-12-28

## Overview

The Auralis HTTP API provides RESTful endpoints for:

1. **System Status** (`/api/status`): Real-time operational state
2. **Performance Metrics** (`/api/metrics`): Latency histograms, buffer health, memory usage
3. **Static Assets** (`/`, `/client/*`): Serve HTML/JS client files

**Base URL**: `http://localhost:8000` (development) or `https://auralis.example.com` (production)

---

## Endpoints

### 1. GET `/api/status`

**Purpose**: Retrieve current system operational state.

**Authentication**: None (MVP), JWT required (production)

**Request**:
```http
GET /api/status HTTP/1.1
Host: localhost:8000
Accept: application/json
```

**Response** (200 OK):
```json
{
  "uptime_sec": 3600.5,
  "active_connections": 5,
  "buffer_depth": 12,
  "device": "Metal",
  "soundfont_loaded": true,
  "synthesis_active": true,
  "timestamp": "2025-12-28T13:45:23.123Z"
}
```

**Fields**:
- `uptime_sec` (float): Server uptime in seconds since startup
- `active_connections` (integer): Current WebSocket client count
- `buffer_depth` (integer): Server-side ring buffer depth (chunks available, 0-20)
- `device` (string): GPU/CPU device in use ("Metal", "CUDA", "CPU")
- `soundfont_loaded` (boolean): Whether SoundFonts loaded successfully
- `synthesis_active` (boolean): Whether generation loop is running
- `timestamp` (string): ISO 8601 timestamp

**Error Responses**:
- `500 Internal Server Error`: Server exception (with error details)

**Example Usage**:
```bash
curl http://localhost:8000/api/status
```

```javascript
fetch('/api/status')
  .then(res => res.json())
  .then(data => {
    console.log(`Uptime: ${data.uptime_sec}s`);
    console.log(`Active clients: ${data.active_connections}`);
    console.log(`Device: ${data.device}`);
  });
```

**Use Cases**:
- Health check endpoint for monitoring (Prometheus, Grafana)
- Debug UI to display server state
- Verify SoundFont loading before connecting WebSocket
- Check GPU availability

---

### 2. GET `/api/metrics`

**Purpose**: Retrieve detailed performance metrics (latency, buffer health, memory).

**Authentication**: None (MVP), JWT required (production)

**Request**:
```http
GET /api/metrics HTTP/1.1
Host: localhost:8000
Accept: application/json
```

**Response** (200 OK):
```json
{
  "synthesis_latency_ms": {
    "avg": 45.3,
    "p50": 42.1,
    "p95": 78.5,
    "p99": 92.3,
    "samples": 1000
  },
  "network_latency_ms": {
    "avg": 120.5,
    "p50": 115.2,
    "p95": 200.1,
    "p99": 285.3,
    "samples": 5000
  },
  "end_to_end_latency_ms": {
    "avg": 520.4,
    "p50": 495.7,
    "p95": 750.2,
    "p99": 820.1,
    "samples": 5000
  },
  "buffer_underruns": 2,
  "buffer_overflows": 0,
  "disconnects": 1,
  "memory_usage_mb": 345.2,
  "gc_collections": {
    "gen0": 50,
    "gen1": 5,
    "gen2": 1
  },
  "timestamp": "2025-12-28T13:45:23.123Z"
}
```

**Fields**:

**Latency Histograms** (synthesis_latency_ms, network_latency_ms, end_to_end_latency_ms):
- `avg` (float): Average latency in milliseconds
- `p50` (float): Median (50th percentile)
- `p95` (float): 95th percentile (target threshold)
- `p99` (float): 99th percentile (worst-case outliers)
- `samples` (integer): Number of samples in histogram

**Event Counters**:
- `buffer_underruns` (integer): Count of ring buffer empty events (target: <1 per 30-min session)
- `buffer_overflows` (integer): Count of ring buffer full events (should be 0)
- `disconnects` (integer): Count of WebSocket disconnections since startup

**Memory**:
- `memory_usage_mb` (float): Current process memory footprint in MB (target: <500MB)
- `gc_collections` (object): Python garbage collection stats
  - `gen0` (integer): Generation 0 collections (short-lived objects)
  - `gen1` (integer): Generation 1 collections
  - `gen2` (integer): Generation 2 collections (long-lived objects, expensive)

**Error Responses**:
- `500 Internal Server Error`: Server exception

**Example Usage**:
```bash
curl http://localhost:8000/api/metrics | jq .
```

```javascript
fetch('/api/metrics')
  .then(res => res.json())
  .then(data => {
    console.log(`Synthesis p95: ${data.synthesis_latency_ms.p95}ms`);
    console.log(`End-to-end p95: ${data.end_to_end_latency_ms.p95}ms`);
    console.log(`Memory usage: ${data.memory_usage_mb}MB`);

    if (data.synthesis_latency_ms.p95 > 100) {
      console.warn('⚠️  Synthesis latency exceeds 100ms target');
    }

    if (data.end_to_end_latency_ms.p95 > 800) {
      console.warn('⚠️  End-to-end latency exceeds 800ms target');
    }

    if (data.buffer_underruns > 0) {
      console.warn(`⚠️  ${data.buffer_underruns} buffer underruns detected`);
    }
  });
```

**Use Cases**:
- Performance monitoring dashboard
- Automated alerts (Prometheus/Grafana)
- Debugging latency issues
- Memory leak detection (track memory_usage_mb over time)
- Capacity planning (measure latency under load)

**Prometheus Export Format** (Post-MVP):
```prometheus
# HELP auralis_synthesis_latency_ms FluidSynth synthesis latency
# TYPE auralis_synthesis_latency_ms histogram
auralis_synthesis_latency_ms_sum 45300.0
auralis_synthesis_latency_ms_count 1000
auralis_synthesis_latency_ms_bucket{le="50"} 650
auralis_synthesis_latency_ms_bucket{le="100"} 950
auralis_synthesis_latency_ms_bucket{le="+Inf"} 1000

# HELP auralis_memory_usage_mb Process memory usage
# TYPE auralis_memory_usage_mb gauge
auralis_memory_usage_mb 345.2
```

---

### 3. GET `/` (Root)

**Purpose**: Serve main HTML client interface.

**Request**:
```http
GET / HTTP/1.1
Host: localhost:8000
Accept: text/html
```

**Response** (200 OK):
```html
<!DOCTYPE html>
<html>
<head>
  <title>Auralis - Generative Ambient Music</title>
  <link rel="stylesheet" href="/client/styles.css">
</head>
<body>
  <div id="app">
    <!-- UI controls, presets, status indicator -->
  </div>
  <script src="/client/audio_client_worklet.js"></script>
</body>
</html>
```

**Headers**:
- `Content-Type: text/html; charset=utf-8`
- `Cache-Control: no-cache` (MVP), `max-age=3600` (production)

**Use Cases**:
- Primary entry point for users
- Auto-play initiates WebSocket connection

---

### 4. GET `/client/{filename}`

**Purpose**: Serve static client assets (JavaScript, CSS).

**Request**:
```http
GET /client/audio_client_worklet.js HTTP/1.1
Host: localhost:8000
Accept: application/javascript
```

**Response** (200 OK):
```javascript
// audio_client_worklet.js content
const websocket = new WebSocket('ws://localhost:8000/ws/stream');
// ...
```

**Headers**:
- `Content-Type: application/javascript` (for .js files)
- `Content-Type: text/css` (for .css files)
- `Cache-Control: no-cache` (MVP)

**Files Served**:
- `/client/audio_client_worklet.js`: AudioContext setup, WebSocket handler
- `/client/audio_worklet_processor.js`: AudioWorklet thread logic
- `/client/styles.css`: UI styling
- `/client/debug.html`: Debug interface (optional)

**Error Responses**:
- `404 Not Found`: File does not exist

---

### 5. GET `/debug` (Optional Debug UI)

**Purpose**: Display real-time metrics visualization for debugging.

**Request**:
```http
GET /debug HTTP/1.1
Host: localhost:8000
Accept: text/html
```

**Response** (200 OK):
```html
<!DOCTYPE html>
<html>
<head>
  <title>Auralis Debug - Metrics Dashboard</title>
</head>
<body>
  <h1>Auralis Metrics Dashboard</h1>
  <div id="metrics-chart">
    <!-- Real-time latency chart (Chart.js or similar) -->
  </div>
  <pre id="status-json"></pre>
  <pre id="metrics-json"></pre>

  <script>
    setInterval(() => {
      fetch('/api/status').then(res => res.json()).then(data => {
        document.getElementById('status-json').textContent = JSON.stringify(data, null, 2);
      });

      fetch('/api/metrics').then(res => res.json()).then(data => {
        document.getElementById('metrics-json').textContent = JSON.stringify(data, null, 2);
        updateLatencyChart(data);
      });
    }, 1000);  // Update every second
  </script>
</body>
</html>
```

**Use Cases**:
- Visual debugging during development
- Demo performance characteristics
- Troubleshoot latency issues

---

## FastAPI Implementation

### Server Setup

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from server.metrics import PerformanceMetrics
from server.status import SystemStatus

app = FastAPI(title="Auralis API", version="1.0.0")

# Serve static files
app.mount("/client", StaticFiles(directory="client"), name="client")

# Global state
metrics = PerformanceMetrics()
status = SystemStatus()

@app.get("/", response_class=HTMLResponse)
async def serve_root():
    with open("client/index.html") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/status")
async def get_status():
    status.update()
    return status.to_json()

@app.get("/api/metrics")
async def get_metrics():
    return metrics.snapshot()

@app.get("/debug", response_class=HTMLResponse)
async def serve_debug():
    with open("client/debug.html") as f:
        return HTMLResponse(content=f.read())
```

### CORS Configuration (Production)

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://auralis.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## Response Schemas (Pydantic Models)

### SystemStatus

```python
from pydantic import BaseModel
from datetime import datetime

class SystemStatus(BaseModel):
    uptime_sec: float
    active_connections: int
    buffer_depth: int
    device: str
    soundfont_loaded: bool
    synthesis_active: bool
    timestamp: datetime
```

### PerformanceMetrics

```python
class LatencyHistogram(BaseModel):
    avg: float
    p50: float
    p95: float
    p99: float
    samples: int

class GCStats(BaseModel):
    gen0: int
    gen1: int
    gen2: int

class PerformanceMetrics(BaseModel):
    synthesis_latency_ms: LatencyHistogram
    network_latency_ms: LatencyHistogram
    end_to_end_latency_ms: LatencyHistogram
    buffer_underruns: int
    buffer_overflows: int
    disconnects: int
    memory_usage_mb: float
    gc_collections: GCStats
    timestamp: datetime
```

---

## Testing

### Unit Tests

```python
from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)

def test_get_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_sec" in data
    assert "active_connections" in data
    assert "device" in data

def test_get_metrics():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "synthesis_latency_ms" in data
    assert "memory_usage_mb" in data
    assert data["buffer_underruns"] >= 0

def test_serve_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html>" in response.text.lower()

def test_serve_static():
    response = client.get("/client/audio_client_worklet.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
```

### Integration Tests

```python
import pytest
import httpx

@pytest.mark.asyncio
async def test_status_endpoint_live():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/status")
        assert response.status_code == 200

        data = response.json()
        assert data["soundfont_loaded"] is True
        assert data["synthesis_active"] is True
        assert data["active_connections"] >= 0

@pytest.mark.asyncio
async def test_metrics_latency_targets():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/metrics")
        data = response.json()

        # Verify latency targets
        assert data["synthesis_latency_ms"]["p95"] < 100  # <100ms target
        assert data["end_to_end_latency_ms"]["p95"] < 800  # <800ms target

        # Verify memory budget
        assert data["memory_usage_mb"] < 500  # <500MB target
```

---

## Rate Limiting (Post-MVP)

### Configuration

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.get("/api/status")
@limiter.limit("10/minute")  # Max 10 requests per minute per IP
async def get_status(request: Request):
    status.update()
    return status.to_json()

@app.get("/api/metrics")
@limiter.limit("5/minute")  # Max 5 requests per minute per IP
async def get_metrics(request: Request):
    return metrics.snapshot()
```

**Error Response** (429 Too Many Requests):
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 30
}
```

---

## Security Headers (Production)

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

# HTTPS redirect
app.add_middleware(HTTPSRedirectMiddleware)

# Trusted hosts
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["auralis.example.com"])

# Security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-28 | Initial HTTP API specification for MVP |

---

**Next Steps**:
- Implement endpoints in `server/main.py`
- Create Pydantic models in `server/metrics.py`, `server/status.py`
- Write tests in `tests/integration/test_http_api.py`

**Related Contracts**:
- [websocket-api.md](websocket-api.md) - WebSocket streaming protocol
- [internal-interfaces.md](internal-interfaces.md) - Python internal interfaces
