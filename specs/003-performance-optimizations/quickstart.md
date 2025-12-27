# Quick Start Testing Guide
## Performance Optimizations - Specs 003

This guide provides step-by-step instructions for testing the performance optimizations implemented in the Auralis real-time audio streaming system.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Test Environment Setup](#test-environment-setup)
4. [Testing Scenarios](#testing-scenarios)
   - [Scenario 1: Adaptive Buffer Management](#scenario-1-adaptive-buffer-management)
   - [Scenario 2: WebSocket Concurrency](#scenario-2-websocket-concurrency)
   - [Scenario 3: GPU Optimization](#scenario-3-gpu-optimization)
   - [Scenario 4: Memory Leak Detection](#scenario-4-memory-leak-detection)
   - [Scenario 5: Performance Metrics Validation](#scenario-5-performance-metrics-validation)
5. [Integration Testing](#integration-testing)
6. [Load Testing](#load-testing)
7. [Performance Benchmarking](#performance-benchmarking)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This testing guide validates the following performance optimizations:

- **Adaptive Buffer Management**: Tier-based buffering (minimal/normal/stable/defensive)
- **WebSocket Concurrency**: Broadcast architecture with per-client cursors
- **GPU Optimization**: Batch processing, torch.compile, kernel fusion
- **Memory Management**: Pre-allocation, GC tuning, leak detection
- **Performance Monitoring**: Prometheus metrics, real-time dashboards

**Performance Targets**:
- ✅ <100ms total audio latency (p99)
- ✅ 99% chunk delivery within 50ms
- ✅ 10+ concurrent users without degradation
- ✅ 30% resource reduction vs Phase 1 baseline
- ✅ <10MB memory growth over 8+ hours
- ✅ Zero buffer underruns under normal conditions

---

## Prerequisites

### System Requirements
- **Python**: 3.12+
- **UV Package Manager**: Latest version
- **GPU**: Metal (Apple Silicon) or CUDA (NVIDIA) recommended
- **OS**: macOS, Linux, or Windows
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 2GB free space for test data

### Required Tools
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version

# Install testing dependencies
uv pip install -e ".[dev]"

# Install monitoring tools
uv pip install prometheus-client grafana-client psutil
```

### Environment Variables
Create `.env` file in project root:
```bash
# Performance Configuration
AURALIS_ENV=testing
AURALIS_HOST=0.0.0.0
AURALIS_PORT=8000
AURALIS_DEVICE=auto          # auto | mps | cuda | cpu
AURALIS_LOG_LEVEL=DEBUG

# Buffer Configuration
BUFFER_INITIAL_TIER=normal   # minimal | normal | stable | defensive
BUFFER_MAX_CAPACITY_MS=3000
BUFFER_MIN_CAPACITY_MS=200

# WebSocket Configuration
WS_MAX_CONNECTIONS=50
WS_RATE_LIMIT_CHUNKS_PER_SEC=50

# GPU Configuration
GPU_BATCH_SIZE=8
GPU_ENABLE_COMPILE=true

# Monitoring Configuration
METRICS_ENABLED=true
METRICS_PORT=9090
```

---

## Test Environment Setup

### 1. Clone and Install
```bash
# Navigate to project directory
cd /Users/andrew/Develop/auralis

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install in development mode
uv pip install -e ".[dev]"
```

### 2. Start Prometheus (Optional)
```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.darwin-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*

# Create prometheus.yml
cat > prometheus.yml <<EOF
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'auralis'
    static_configs:
      - targets: ['localhost:9090']
EOF

# Start Prometheus
./prometheus --config.file=prometheus.yml
```

### 3. Verify GPU Availability
```bash
# Check GPU device
python -c "
import torch
print(f'Metal (MPS): {torch.backends.mps.is_available()}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cuda\" if torch.cuda.is_available() else \"cpu\")}')
"
```

Expected output (macOS with Apple Silicon):
```
Metal (MPS): True
CUDA: False
Device: mps
```

---

## Testing Scenarios

### Scenario 1: Adaptive Buffer Management

**Objective**: Verify that the adaptive buffer system correctly adjusts tier levels based on network jitter.

#### Test 1.1: Buffer Tier Promotion (Stable → Defensive)

**Setup**:
```bash
# Start server with normal tier
BUFFER_INITIAL_TIER=stable uvicorn server.main:app --reload --port 8000
```

**Test Script** (`tests/integration/test_adaptive_buffer.py`):
```python
import asyncio
import websockets
import json
import time
import random

async def test_tier_promotion():
    """Simulate network jitter to trigger tier promotion."""
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        # Send initial control message
        await websocket.send(json.dumps({
            "type": "control",
            "action": "start",
            "params": {"key": "C", "bpm": 70, "intensity": 0.5}
        }))

        chunk_count = 0
        delayed_chunks = 0

        for _ in range(100):  # Receive 100 chunks (~10 seconds)
            # Simulate network delay randomly
            if random.random() < 0.2:  # 20% chance of delay
                await asyncio.sleep(random.uniform(0.05, 0.15))  # 50-150ms delay
                delayed_chunks += 1

            message = await websocket.recv()
            data = json.loads(message)

            if data["type"] == "audio":
                chunk_count += 1
                current_tier = data.get("current_tier", "unknown")
                buffer_depth = data.get("buffer_depth", 0)

                print(f"Chunk {chunk_count}: tier={current_tier}, buffer_depth={buffer_depth}")

                # After 50 chunks, should promote to defensive tier
                if chunk_count == 50:
                    assert current_tier == "defensive", \
                        f"Expected 'defensive' tier after {delayed_chunks} delays, got '{current_tier}'"
                    print(f"✅ Tier promoted to 'defensive' after {delayed_chunks} delayed chunks")

        # Send disconnect
        await websocket.send(json.dumps({"type": "control", "action": "stop"}))

if __name__ == "__main__":
    asyncio.run(test_tier_promotion())
```

**Run Test**:
```bash
python tests/integration/test_adaptive_buffer.py
```

**Expected Output**:
```
Chunk 1: tier=stable, buffer_depth=15
Chunk 2: tier=stable, buffer_depth=14
...
Chunk 48: tier=stable, buffer_depth=12
Chunk 49: tier=defensive, buffer_depth=25
Chunk 50: tier=defensive, buffer_depth=24
✅ Tier promoted to 'defensive' after 19 delayed chunks
```

**Success Criteria**:
- ✅ Buffer tier promotes from stable → defensive after ≥5% underruns
- ✅ Buffer depth increases when tier promotes (15 chunks → 25 chunks)
- ✅ No underruns occur after tier promotion

---

#### Test 1.2: Jitter Tracking with EMA

**Objective**: Verify that jitter is tracked using Exponential Moving Average (EMA).

**Test Script** (`tests/unit/test_jitter_tracker.py`):
```python
import pytest
from server.jitter_tracker import JitterTracker, ChunkTimestamp

def test_ema_jitter_tracking():
    """Test EMA-based jitter calculation."""
    tracker = JitterTracker(window_size=50, alpha=0.1)

    # Simulate 100 chunks with varying jitter
    base_time = 1735228800.0
    chunk_interval = 0.1  # 100ms per chunk

    for i in range(100):
        expected_time = base_time + (i * chunk_interval)

        # Add realistic jitter (±5-20ms)
        import random
        actual_time = expected_time + random.uniform(-0.020, 0.020)

        timestamp = ChunkTimestamp(
            chunk_id=i,
            expected_time=expected_time,
            actual_time=actual_time
        )

        tracker.record_chunk(timestamp, underrun=False)

    # Verify jitter statistics
    mean_jitter = tracker.get_mean_jitter_ms()
    jitter_std = tracker.get_jitter_std()
    recommended_buffer = tracker.get_recommended_buffer_ms(confidence=0.95)

    print(f"Mean jitter: {mean_jitter:.2f}ms")
    print(f"Jitter std dev: {jitter_std:.2f}ms")
    print(f"Recommended buffer (95% confidence): {recommended_buffer:.2f}ms")

    # Assertions
    assert 0 <= mean_jitter <= 30, f"Mean jitter out of range: {mean_jitter}ms"
    assert 200 <= recommended_buffer <= 3000, f"Buffer recommendation out of range: {recommended_buffer}ms"
    assert tracker.get_underrun_rate() == 0.0, "Should have zero underruns"

    print("✅ EMA jitter tracking working correctly")

if __name__ == "__main__":
    test_ema_jitter_tracking()
```

**Run Test**:
```bash
pytest tests/unit/test_jitter_tracker.py -v
```

**Expected Output**:
```
Mean jitter: 12.34ms
Jitter std dev: 8.21ms
Recommended buffer (95% confidence): 345.67ms
✅ EMA jitter tracking working correctly
PASSED
```

---

### Scenario 2: WebSocket Concurrency

**Objective**: Verify that the broadcast architecture handles 10+ concurrent clients efficiently.

#### Test 2.1: Concurrent Client Connections

**Test Script** (`tests/integration/test_concurrent_clients.py`):
```python
import asyncio
import websockets
import json
import time
from typing import List

async def client_worker(client_id: int, duration_sec: int = 10) -> dict:
    """Simulates a single WebSocket client."""
    uri = "ws://localhost:8000/ws"
    chunks_received = 0
    underruns = 0
    latencies = []

    try:
        async with websockets.connect(uri) as websocket:
            # Start streaming
            await websocket.send(json.dumps({
                "type": "control",
                "action": "start",
                "params": {"key": "C", "bpm": 70, "intensity": 0.5}
            }))

            start_time = time.time()

            while time.time() - start_time < duration_sec:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                data = json.loads(message)

                if data["type"] == "audio":
                    chunks_received += 1

                    # Calculate latency
                    server_timestamp = data.get("timestamp", 0)
                    client_timestamp = time.time()
                    latency_ms = (client_timestamp - server_timestamp) * 1000
                    latencies.append(latency_ms)

                    if data.get("buffer_depth", 100) == 0:
                        underruns += 1

            # Disconnect
            await websocket.send(json.dumps({"type": "control", "action": "stop"}))

    except Exception as e:
        print(f"Client {client_id} error: {e}")
        return {"client_id": client_id, "error": str(e)}

    # Calculate statistics
    p50_latency = sorted(latencies)[len(latencies) // 2] if latencies else 0
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

    return {
        "client_id": client_id,
        "chunks_received": chunks_received,
        "underruns": underruns,
        "p50_latency_ms": p50_latency,
        "p99_latency_ms": p99_latency,
    }

async def test_concurrent_clients(num_clients: int = 10, duration_sec: int = 10):
    """Test multiple concurrent WebSocket clients."""
    print(f"Starting {num_clients} concurrent clients for {duration_sec} seconds...")

    # Launch all clients concurrently
    tasks = [client_worker(i, duration_sec) for i in range(num_clients)]
    results = await asyncio.gather(*tasks)

    # Aggregate results
    total_chunks = sum(r.get("chunks_received", 0) for r in results)
    total_underruns = sum(r.get("underruns", 0) for r in results)
    avg_p99_latency = sum(r.get("p99_latency_ms", 0) for r in results) / num_clients

    print("\n" + "="*60)
    print("CONCURRENT CLIENT TEST RESULTS")
    print("="*60)

    for result in results:
        if "error" in result:
            print(f"❌ Client {result['client_id']}: ERROR - {result['error']}")
        else:
            print(f"✅ Client {result['client_id']}: "
                  f"{result['chunks_received']} chunks, "
                  f"{result['underruns']} underruns, "
                  f"p99 latency: {result['p99_latency_ms']:.1f}ms")

    print("="*60)
    print(f"Total chunks delivered: {total_chunks}")
    print(f"Total underruns: {total_underruns}")
    print(f"Average p99 latency: {avg_p99_latency:.1f}ms")
    print("="*60)

    # Assertions
    assert total_chunks >= num_clients * duration_sec * 9, \
        f"Expected at least {num_clients * duration_sec * 9} chunks, got {total_chunks}"
    assert total_underruns == 0, f"Expected zero underruns, got {total_underruns}"
    assert avg_p99_latency < 100, f"Expected p99 < 100ms, got {avg_p99_latency:.1f}ms"

    print("✅ All concurrent client tests passed!")

if __name__ == "__main__":
    asyncio.run(test_concurrent_clients(num_clients=10, duration_sec=10))
```

**Run Test**:
```bash
# Start server in one terminal
uvicorn server.main:app --reload --port 8000

# Run test in another terminal
python tests/integration/test_concurrent_clients.py
```

**Expected Output**:
```
Starting 10 concurrent clients for 10 seconds...

============================================================
CONCURRENT CLIENT TEST RESULTS
============================================================
✅ Client 0: 98 chunks, 0 underruns, p99 latency: 45.2ms
✅ Client 1: 97 chunks, 0 underruns, p99 latency: 47.8ms
✅ Client 2: 99 chunks, 0 underruns, p99 latency: 43.1ms
✅ Client 3: 98 chunks, 0 underruns, p99 latency: 46.5ms
✅ Client 4: 97 chunks, 0 underruns, p99 latency: 48.9ms
✅ Client 5: 98 chunks, 0 underruns, p99 latency: 44.7ms
✅ Client 6: 99 chunks, 0 underruns, p99 latency: 42.3ms
✅ Client 7: 97 chunks, 0 underruns, p99 latency: 49.2ms
✅ Client 8: 98 chunks, 0 underruns, p99 latency: 45.6ms
✅ Client 9: 99 chunks, 0 underruns, p99 latency: 43.8ms
============================================================
Total chunks delivered: 980
Total underruns: 0
Average p99 latency: 45.7ms
============================================================
✅ All concurrent client tests passed!
```

**Success Criteria**:
- ✅ All 10 clients connect successfully
- ✅ Each client receives ≥90 chunks in 10 seconds (~9 chunks/sec expected)
- ✅ Zero buffer underruns across all clients
- ✅ Average p99 latency <100ms

---

#### Test 2.2: Broadcast Encoding Efficiency

**Objective**: Verify that audio chunks are encoded once and broadcast to multiple clients.

**Test Script** (`tests/performance/test_broadcast_encoding.py`):
```python
import asyncio
import time
from server.streaming_server import StreamingServer
from server.chunk_encoder import ChunkEncoder
from unittest.mock import MagicMock, patch

async def test_broadcast_encoding():
    """Verify that chunks are encoded once for N clients."""

    # Mock encoder to count encode calls
    encode_count = 0

    def mock_encode(audio_data):
        nonlocal encode_count
        encode_count += 1
        return "base64_encoded_data"

    with patch.object(ChunkEncoder, 'encode', side_effect=mock_encode):
        server = StreamingServer()

        # Simulate 10 connected clients
        num_clients = 10
        mock_clients = [MagicMock() for _ in range(num_clients)]
        server.clients = mock_clients

        # Send 50 chunks
        num_chunks = 50
        for i in range(num_chunks):
            fake_audio_data = b'\x00' * 17640  # 100ms stereo PCM
            await server.broadcast_chunk(fake_audio_data)

        # Calculate encoding efficiency
        expected_encodes = num_chunks  # Should encode once per chunk
        actual_encodes = encode_count
        efficiency_ratio = num_clients / (actual_encodes / num_chunks)

        print(f"Chunks broadcasted: {num_chunks}")
        print(f"Clients: {num_clients}")
        print(f"Expected encode calls: {expected_encodes}")
        print(f"Actual encode calls: {actual_encodes}")
        print(f"Encoding efficiency: {efficiency_ratio:.1f}x (1 encode → {num_clients} sends)")

        # Assertions
        assert actual_encodes == expected_encodes, \
            f"Expected {expected_encodes} encodes, got {actual_encodes}"
        assert efficiency_ratio == num_clients, \
            f"Expected {num_clients}x efficiency, got {efficiency_ratio:.1f}x"

        print("✅ Broadcast encoding is efficient (1× encode, N× send)")

if __name__ == "__main__":
    asyncio.run(test_broadcast_encoding())
```

**Run Test**:
```bash
pytest tests/performance/test_broadcast_encoding.py -v
```

**Expected Output**:
```
Chunks broadcasted: 50
Clients: 10
Expected encode calls: 50
Actual encode calls: 50
Encoding efficiency: 10.0x (1 encode → 10 sends)
✅ Broadcast encoding is efficient (1× encode, N× send)
PASSED
```

---

### Scenario 3: GPU Optimization

**Objective**: Verify that GPU optimizations (batch processing, torch.compile, kernel fusion) improve synthesis performance.

#### Test 3.1: GPU Device Selection

**Test Script** (`tests/unit/test_device_selection.py`):
```python
import torch
from server.device_selector import AutoDeviceSelector

def test_device_selection():
    """Test automatic GPU device selection."""
    selector = AutoDeviceSelector()
    device = selector.select_device()

    print(f"Selected device: {device}")
    print(f"Device type: {device.type}")

    # Verify device is GPU if available
    if torch.backends.mps.is_available():
        assert device.type == "mps", f"Expected MPS device on Apple Silicon, got {device.type}"
        print("✅ Metal (MPS) device selected correctly")
    elif torch.cuda.is_available():
        assert device.type == "cuda", f"Expected CUDA device, got {device.type}"
        print("✅ CUDA device selected correctly")
    else:
        assert device.type == "cpu", f"Expected CPU fallback, got {device.type}"
        print("⚠️  CPU fallback (no GPU available)")

if __name__ == "__main__":
    test_device_selection()
```

**Run Test**:
```bash
python tests/unit/test_device_selection.py
```

**Expected Output (macOS with Apple Silicon)**:
```
Selected device: mps
Device type: mps
✅ Metal (MPS) device selected correctly
```

---

#### Test 3.2: Batch Processing Performance

**Objective**: Compare single-voice vs. batch rendering performance.

**Test Script** (`tests/performance/test_batch_synthesis.py`):
```python
import time
import torch
import numpy as np
from server.synthesis_engine import OptimizedSynthesisEngine

def benchmark_batch_synthesis():
    """Benchmark batch vs. sequential synthesis."""

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    engine = OptimizedSynthesisEngine(device=device, batch_size=8)

    # Generate test phrase
    chords = [
        (0, 60, "major"),      # C major
        (22050, 65, "minor"),  # F minor
        (44100, 67, "major"),  # G major
    ]

    melody = [
        (0, 72, 0.8, 0.5),
        (11025, 74, 0.7, 0.5),
        (22050, 76, 0.8, 0.5),
        (33075, 77, 0.7, 0.5),
    ]

    duration_sec = 2.0
    num_iterations = 20

    # Benchmark with batching
    print("Benchmarking with GPU batch processing...")
    start_time = time.time()
    for _ in range(num_iterations):
        audio = engine.render_phrase(chords, melody, duration_sec)
    batch_time = (time.time() - start_time) / num_iterations

    # Benchmark without batching (simulate sequential)
    print("Benchmarking without batching (sequential)...")
    engine_seq = OptimizedSynthesisEngine(device=device, batch_size=1)
    start_time = time.time()
    for _ in range(num_iterations):
        audio = engine_seq.render_phrase(chords, melody, duration_sec)
    sequential_time = (time.time() - start_time) / num_iterations

    # Calculate speedup
    speedup = sequential_time / batch_time

    print("\n" + "="*60)
    print("BATCH SYNTHESIS BENCHMARK")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: 8")
    print(f"Phrase duration: {duration_sec}s")
    print(f"Iterations: {num_iterations}")
    print("-"*60)
    print(f"Batch processing time: {batch_time*1000:.2f}ms")
    print(f"Sequential processing time: {sequential_time*1000:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")
    print("="*60)

    # Assertions
    assert batch_time < 0.1, f"Batch synthesis should be <100ms, got {batch_time*1000:.2f}ms"
    assert speedup > 1.5, f"Batching should provide >1.5x speedup, got {speedup:.2f}x"

    print("✅ GPU batch processing provides significant speedup")

if __name__ == "__main__":
    benchmark_batch_synthesis()
```

**Run Test**:
```bash
python tests/performance/test_batch_synthesis.py
```

**Expected Output**:
```
Benchmarking with GPU batch processing...
Benchmarking without batching (sequential)...

============================================================
BATCH SYNTHESIS BENCHMARK
============================================================
Device: mps
Batch size: 8
Phrase duration: 2.0s
Iterations: 20
------------------------------------------------------------
Batch processing time: 45.32ms
Sequential processing time: 87.91ms
Speedup: 1.94x
============================================================
✅ GPU batch processing provides significant speedup
```

---

#### Test 3.3: torch.compile Kernel Fusion

**Objective**: Verify that torch.compile improves synthesis performance through kernel fusion.

**Test Script** (`tests/performance/test_torch_compile.py`):
```python
import time
import torch
from server.synthesis_engine import OptimizedSynthesisEngine

def benchmark_torch_compile():
    """Benchmark synthesis with and without torch.compile."""

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    # Test data
    chords = [(0, 60, "major"), (22050, 65, "minor")]
    melody = [(0, 72, 0.8, 0.5), (11025, 74, 0.7, 0.5)]
    duration_sec = 1.0
    num_iterations = 50

    # Without torch.compile
    print("Benchmarking without torch.compile...")
    engine_no_compile = OptimizedSynthesisEngine(device=device, enable_compile=False)

    start_time = time.time()
    for _ in range(num_iterations):
        audio = engine_no_compile.render_phrase(chords, melody, duration_sec)
    time_no_compile = (time.time() - start_time) / num_iterations

    # With torch.compile
    print("Benchmarking with torch.compile...")
    engine_compile = OptimizedSynthesisEngine(device=device, enable_compile=True)

    # Warm-up compile
    _ = engine_compile.render_phrase(chords, melody, duration_sec)

    start_time = time.time()
    for _ in range(num_iterations):
        audio = engine_compile.render_phrase(chords, melody, duration_sec)
    time_compile = (time.time() - start_time) / num_iterations

    # Calculate speedup
    speedup = time_no_compile / time_compile

    print("\n" + "="*60)
    print("TORCH.COMPILE BENCHMARK")
    print("="*60)
    print(f"Device: {device}")
    print(f"Iterations: {num_iterations}")
    print("-"*60)
    print(f"Without torch.compile: {time_no_compile*1000:.2f}ms")
    print(f"With torch.compile: {time_compile*1000:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")
    print("="*60)

    # Assertions
    assert time_compile < 0.1, f"Compiled synthesis should be <100ms, got {time_compile*1000:.2f}ms"
    assert speedup > 1.2, f"torch.compile should provide >1.2x speedup, got {speedup:.2f}x"

    print("✅ torch.compile provides kernel fusion speedup")

if __name__ == "__main__":
    benchmark_torch_compile()
```

**Run Test**:
```bash
python tests/performance/test_torch_compile.py
```

**Expected Output**:
```
Benchmarking without torch.compile...
Benchmarking with torch.compile...

============================================================
TORCH.COMPILE BENCHMARK
============================================================
Device: mps
Iterations: 50
------------------------------------------------------------
Without torch.compile: 52.34ms
With torch.compile: 38.71ms
Speedup: 1.35x
============================================================
✅ torch.compile provides kernel fusion speedup
```

---

### Scenario 4: Memory Leak Detection

**Objective**: Verify that the system maintains stable memory usage over extended periods.

#### Test 4.1: Long-Running Memory Stability Test

**Test Script** (`tests/integration/test_memory_stability.py`):
```python
import asyncio
import psutil
import time
import numpy as np
from server.memory_monitor import MemoryMonitor
from server.synthesis_engine import OptimizedSynthesisEngine

async def test_memory_stability(duration_hours: float = 0.5):
    """Run synthesis for extended period and monitor memory growth."""

    monitor = MemoryMonitor(sample_interval_sec=10)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    engine = OptimizedSynthesisEngine(device=device)

    # Test data
    chords = [(0, 60, "major"), (22050, 65, "minor")]
    melody = [(0, 72, 0.8, 0.5), (11025, 74, 0.7, 0.5)]
    duration_sec = 1.0

    # Start monitoring
    monitor.start()

    print(f"Running memory stability test for {duration_hours} hours...")
    print("Press Ctrl+C to stop early\n")

    start_time = time.time()
    duration_sec_total = duration_hours * 3600
    iteration = 0

    try:
        while time.time() - start_time < duration_sec_total:
            # Render phrase
            audio = engine.render_phrase(chords, melody, duration_sec)

            iteration += 1

            # Log every 100 iterations
            if iteration % 100 == 0:
                snapshot = monitor.get_current_snapshot()
                elapsed_min = (time.time() - start_time) / 60
                print(f"[{elapsed_min:.1f}m] Iteration {iteration}: "
                      f"RSS={snapshot.rss_mb:.1f}MB, "
                      f"GPU={snapshot.gpu_allocated_mb:.1f}MB")

            # Small delay to simulate realistic usage
            await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")

    # Stop monitoring and analyze
    monitor.stop()
    growth_report = monitor.get_memory_growth_report()

    print("\n" + "="*60)
    print("MEMORY STABILITY TEST RESULTS")
    print("="*60)
    print(f"Test duration: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Iterations: {iteration}")
    print("-"*60)
    print(f"Initial RSS memory: {growth_report['initial_rss_mb']:.1f}MB")
    print(f"Final RSS memory: {growth_report['final_rss_mb']:.1f}MB")
    print(f"Memory growth: {growth_report['growth_mb']:.1f}MB")
    print(f"Growth rate: {growth_report['growth_rate_mb_per_hour']:.2f}MB/hour")
    print("-"*60)
    print(f"Leak detected: {growth_report['leak_detected']}")
    print("="*60)

    # Assertions
    assert growth_report['growth_mb'] < 10, \
        f"Memory growth should be <10MB, got {growth_report['growth_mb']:.1f}MB"
    assert not growth_report['leak_detected'], \
        "Memory leak detected!"

    print("✅ Memory remains stable over extended period")

if __name__ == "__main__":
    import torch
    asyncio.run(test_memory_stability(duration_hours=0.5))  # 30 minute test
```

**Run Test**:
```bash
# Quick test (30 minutes)
python tests/integration/test_memory_stability.py

# Full test (8 hours) - run overnight
# python tests/integration/test_memory_stability.py --duration 8.0
```

**Expected Output**:
```
Running memory stability test for 0.5 hours...
Press Ctrl+C to stop early

[1.0m] Iteration 100: RSS=145.2MB, GPU=87.3MB
[2.0m] Iteration 200: RSS=145.8MB, GPU=87.3MB
[3.0m] Iteration 300: RSS=146.1MB, GPU=87.3MB
...
[28.0m] Iteration 2800: RSS=148.7MB, GPU=87.3MB
[29.0m] Iteration 2900: RSS=149.1MB, GPU=87.3MB

============================================================
MEMORY STABILITY TEST RESULTS
============================================================
Test duration: 0.50 hours
Iterations: 3000
------------------------------------------------------------
Initial RSS memory: 145.2MB
Final RSS memory: 149.1MB
Memory growth: 3.9MB
Growth rate: 7.80MB/hour
------------------------------------------------------------
Leak detected: False
============================================================
✅ Memory remains stable over extended period
```

**Success Criteria**:
- ✅ Memory growth <10MB over test duration
- ✅ Linear regression slope <20MB/hour
- ✅ No memory leak detected

---

#### Test 4.2: GC Configuration Effectiveness

**Objective**: Verify that GC tuning reduces pause times.

**Test Script** (`tests/unit/test_gc_config.py`):
```python
import gc
import time
import sys
from server.gc_config import RealTimeGCConfig

def measure_gc_pause():
    """Measure GC collection pause time."""
    # Disable automatic GC
    gc.disable()

    # Allocate garbage
    garbage = []
    for _ in range(10000):
        garbage.append([0] * 1000)

    # Manually trigger GC and measure pause
    start_time = time.perf_counter()
    gc.collect(generation=0)
    pause_ms = (time.perf_counter() - start_time) * 1000

    # Re-enable GC
    gc.enable()

    return pause_ms

def test_gc_configuration():
    """Test that GC configuration reduces pause times."""

    # Measure baseline GC pause (default settings)
    gc.set_threshold(700, 10, 10)  # Python defaults
    baseline_pause = measure_gc_pause()

    # Apply optimized GC config
    config = RealTimeGCConfig()
    config.configure()

    # Measure optimized GC pause
    optimized_pause = measure_gc_pause()

    # Calculate improvement
    improvement = (baseline_pause - optimized_pause) / baseline_pause * 100

    print("="*60)
    print("GC CONFIGURATION TEST")
    print("="*60)
    print(f"Baseline GC pause (default): {baseline_pause:.2f}ms")
    print(f"Optimized GC pause (tuned): {optimized_pause:.2f}ms")
    print(f"Improvement: {improvement:.1f}%")
    print("="*60)

    # Verify thresholds
    thresholds = gc.get_threshold()
    print(f"GC thresholds: {thresholds}")
    assert thresholds[0] == 50000, f"Expected gen0 threshold 50000, got {thresholds[0]}"
    assert thresholds[1] == 500, f"Expected gen1 threshold 500, got {thresholds[1]}"
    assert thresholds[2] == 1000, f"Expected gen2 threshold 1000, got {thresholds[2]}"

    print("✅ GC configuration reduces pause times")

if __name__ == "__main__":
    test_gc_configuration()
```

**Run Test**:
```bash
python tests/unit/test_gc_config.py
```

**Expected Output**:
```
============================================================
GC CONFIGURATION TEST
============================================================
Baseline GC pause (default): 8.34ms
Optimized GC pause (tuned): 2.17ms
Improvement: 74.0%
============================================================
GC thresholds: (50000, 500, 1000)
✅ GC configuration reduces pause times
```

---

### Scenario 5: Performance Metrics Validation

**Objective**: Verify that Prometheus metrics are correctly collected and exposed.

#### Test 5.1: Metrics Collection

**Test Script** (`tests/integration/test_metrics_collection.py`):
```python
import asyncio
import requests
from server.metrics import PrometheusMetrics

async def test_metrics_collection():
    """Verify Prometheus metrics are collected correctly."""

    metrics = PrometheusMetrics()

    # Simulate synthesis operations
    print("Simulating synthesis operations...")
    for i in range(10):
        # Record synthesis latency
        latency_sec = 0.045 + (i * 0.001)  # 45-54ms
        metrics.synthesis_latency_seconds.observe(latency_sec)

        # Record buffer depth
        buffer_depth = 15 - (i % 3)  # 13-15 chunks
        metrics.buffer_depth_chunks.labels(client_id=f"client_{i}").set(buffer_depth)

        # Record jitter
        jitter_ms = 10 + (i * 0.5)  # 10-14.5ms
        metrics.chunk_delivery_jitter_ms.observe(jitter_ms)

    # Increment counters
    metrics.active_websocket_connections.set(5)
    metrics.buffer_underruns_total.labels(client_id="client_1").inc()
    metrics.websocket_send_errors_total.labels(error_type="timeout").inc(2)

    # Fetch metrics from endpoint
    print("Fetching metrics from /metrics endpoint...")
    response = requests.get("http://localhost:9090/metrics")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    metrics_text = response.text

    # Verify key metrics present
    assert "synthesis_latency_seconds" in metrics_text
    assert "buffer_depth_chunks" in metrics_text
    assert "chunk_delivery_jitter_ms" in metrics_text
    assert "buffer_underruns_total" in metrics_text
    assert "active_websocket_connections" in metrics_text

    print("\n" + "="*60)
    print("METRICS COLLECTION TEST")
    print("="*60)
    print("Metrics endpoint: http://localhost:9090/metrics")
    print(f"Response status: {response.status_code}")
    print(f"Metrics found: {len([line for line in metrics_text.split('\\n') if line and not line.startswith('#')])}")
    print("="*60)

    # Show sample metrics
    print("\nSample metrics:")
    for line in metrics_text.split('\n'):
        if 'synthesis_latency' in line or 'buffer_depth' in line:
            print(f"  {line}")

    print("\n✅ Prometheus metrics collected correctly")

if __name__ == "__main__":
    asyncio.run(test_metrics_collection())
```

**Run Test**:
```bash
# Start server with metrics enabled
METRICS_ENABLED=true uvicorn server.main:app --port 8000

# Run test
python tests/integration/test_metrics_collection.py
```

**Expected Output**:
```
Simulating synthesis operations...
Fetching metrics from /metrics endpoint...

============================================================
METRICS COLLECTION TEST
============================================================
Metrics endpoint: http://localhost:9090/metrics
Response status: 200
Metrics found: 47
============================================================

Sample metrics:
  synthesis_latency_seconds_bucket{le="0.05"} 4.0
  synthesis_latency_seconds_bucket{le="0.1"} 10.0
  synthesis_latency_seconds_sum 0.495
  synthesis_latency_seconds_count 10
  buffer_depth_chunks{client_id="client_0"} 15.0
  buffer_depth_chunks{client_id="client_1"} 14.0

✅ Prometheus metrics collected correctly
```

---

#### Test 5.2: Alert Rules Validation

**Objective**: Verify that alert rules trigger correctly for performance degradation.

**Test Script** (`tests/integration/test_alert_rules.py`):
```python
import requests
import time
from server.metrics import PrometheusMetrics

def trigger_high_latency_alert():
    """Simulate conditions that should trigger SynthesisLatencyHigh alert."""

    metrics = PrometheusMetrics()

    print("Simulating high synthesis latency (>100ms)...")

    # Record 20 samples with high latency (>100ms)
    for i in range(20):
        latency_sec = 0.11 + (i * 0.001)  # 110-129ms
        metrics.synthesis_latency_seconds.observe(latency_sec)
        time.sleep(0.1)  # Small delay between samples

    # Wait for Prometheus to scrape and evaluate
    print("Waiting for Prometheus to evaluate alert rules (30s)...")
    time.sleep(30)

    # Query Prometheus alerts API
    response = requests.get("http://localhost:9090/api/v1/alerts")
    alerts = response.json()['data']['alerts']

    # Find SynthesisLatencyHigh alert
    high_latency_alert = next(
        (alert for alert in alerts if alert['labels']['alertname'] == 'SynthesisLatencyHigh'),
        None
    )

    print("\n" + "="*60)
    print("ALERT RULES TEST")
    print("="*60)

    if high_latency_alert:
        state = high_latency_alert['state']
        value = high_latency_alert.get('value', 'N/A')
        print(f"Alert: SynthesisLatencyHigh")
        print(f"State: {state}")
        print(f"Value: {value}")
        print("="*60)

        assert state in ['pending', 'firing'], f"Expected alert to be active, got state '{state}'"
        print("✅ Alert rule triggered correctly for high latency")
    else:
        print("❌ SynthesisLatencyHigh alert not found")
        print(f"Active alerts: {[a['labels']['alertname'] for a in alerts]}")
        raise AssertionError("Expected SynthesisLatencyHigh alert to trigger")

if __name__ == "__main__":
    trigger_high_latency_alert()
```

**Run Test**:
```bash
# Ensure Prometheus is running with Auralis alert rules
python tests/integration/test_alert_rules.py
```

**Expected Output**:
```
Simulating high synthesis latency (>100ms)...
Waiting for Prometheus to evaluate alert rules (30s)...

============================================================
ALERT RULES TEST
============================================================
Alert: SynthesisLatencyHigh
State: firing
Value: 0.1198
============================================================
✅ Alert rule triggered correctly for high latency
```

---

## Integration Testing

### Full System Integration Test

**Objective**: Test the entire pipeline from composition → synthesis → streaming → client playback.

**Test Script** (`tests/integration/test_full_pipeline.py`):
```python
import asyncio
import websockets
import json
import base64
import numpy as np
import time

async def test_full_pipeline():
    """Test complete audio pipeline end-to-end."""

    uri = "ws://localhost:8000/ws"
    chunks_received = 0
    audio_data_bytes = bytearray()

    print("Connecting to WebSocket server...")
    async with websockets.connect(uri) as websocket:
        # Start streaming
        print("Sending start control message...")
        await websocket.send(json.dumps({
            "type": "control",
            "action": "start",
            "params": {
                "key": "C",
                "bpm": 70,
                "intensity": 0.5
            }
        }))

        # Receive 50 chunks (~5 seconds)
        print("Receiving audio chunks...\n")

        for _ in range(50):
            message = await websocket.recv()
            data = json.loads(message)

            if data["type"] == "audio":
                chunks_received += 1

                # Decode audio data
                chunk_data_b64 = data["data"]
                chunk_data_bytes = base64.b64decode(chunk_data_b64)
                audio_data_bytes.extend(chunk_data_bytes)

                # Log progress
                if chunks_received % 10 == 0:
                    buffer_depth = data.get("buffer_depth", 0)
                    tier = data.get("current_tier", "unknown")
                    print(f"Chunk {chunks_received}: buffer_depth={buffer_depth}, tier={tier}")

        # Stop streaming
        print("\nSending stop control message...")
        await websocket.send(json.dumps({"type": "control", "action": "stop"}))

    # Validate received audio
    print("\n" + "="*60)
    print("FULL PIPELINE TEST RESULTS")
    print("="*60)
    print(f"Chunks received: {chunks_received}")
    print(f"Total audio data: {len(audio_data_bytes)} bytes ({len(audio_data_bytes) / 1024:.1f} KB)")
    print(f"Expected data: {50 * 17640} bytes ({50 * 17640 / 1024:.1f} KB)")
    print("="*60)

    # Convert to numpy array
    audio_array = np.frombuffer(audio_data_bytes, dtype=np.int16)
    audio_stereo = audio_array.reshape(-1, 2)

    print(f"Audio shape: {audio_stereo.shape}")
    print(f"Duration: {audio_stereo.shape[0] / 44100:.2f} seconds")
    print(f"Sample rate: 44,100 Hz")
    print(f"Channels: 2 (stereo)")
    print("="*60)

    # Assertions
    assert chunks_received == 50, f"Expected 50 chunks, got {chunks_received}"
    assert len(audio_data_bytes) == 50 * 17640, \
        f"Expected {50 * 17640} bytes, got {len(audio_data_bytes)}"
    assert audio_stereo.shape[1] == 2, f"Expected stereo (2 channels), got {audio_stereo.shape[1]}"

    print("✅ Full pipeline test passed successfully")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
```

**Run Test**:
```bash
# Start server
uvicorn server.main:app --reload --port 8000

# Run full pipeline test
python tests/integration/test_full_pipeline.py
```

**Expected Output**:
```
Connecting to WebSocket server...
Sending start control message...
Receiving audio chunks...

Chunk 10: buffer_depth=15, tier=normal
Chunk 20: buffer_depth=14, tier=normal
Chunk 30: buffer_depth=15, tier=normal
Chunk 40: buffer_depth=14, tier=normal
Chunk 50: buffer_depth=15, tier=normal

Sending stop control message...

============================================================
FULL PIPELINE TEST RESULTS
============================================================
Chunks received: 50
Total audio data: 882000 bytes (861.3 KB)
Expected data: 882000 bytes (861.3 KB)
============================================================
Audio shape: (220500, 2)
Duration: 5.00 seconds
Sample rate: 44,100 Hz
Channels: 2 (stereo)
============================================================
✅ Full pipeline test passed successfully
```

---

## Load Testing

### Multi-Client Load Test

**Objective**: Stress test the system with 20+ concurrent clients to validate scalability.

**Test Script** (`tests/load/test_load_20_clients.py`):
```python
import asyncio
import websockets
import json
import time
from statistics import mean, stdev

async def client_load_worker(client_id: int, duration_sec: int = 60):
    """Simulates a high-load WebSocket client."""
    uri = "ws://localhost:8000/ws"
    chunks = []
    errors = []

    try:
        async with websockets.connect(uri) as websocket:
            # Start streaming
            await websocket.send(json.dumps({
                "type": "control",
                "action": "start",
                "params": {"key": "C", "bpm": 70, "intensity": 0.5}
            }))

            start_time = time.time()

            while time.time() - start_time < duration_sec:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)

                    if data["type"] == "audio":
                        recv_time = time.time()
                        server_time = data.get("timestamp", recv_time)
                        latency_ms = (recv_time - server_time) * 1000

                        chunks.append({
                            "latency_ms": latency_ms,
                            "buffer_depth": data.get("buffer_depth", 0),
                            "tier": data.get("current_tier", "unknown")
                        })

                except asyncio.TimeoutError:
                    errors.append("recv_timeout")

            # Disconnect
            await websocket.send(json.dumps({"type": "control", "action": "stop"}))

    except Exception as e:
        errors.append(str(e))

    # Calculate statistics
    if chunks:
        latencies = [c["latency_ms"] for c in chunks]
        p50 = sorted(latencies)[len(latencies) // 2]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        underruns = sum(1 for c in chunks if c["buffer_depth"] == 0)
    else:
        p50 = p99 = underruns = 0

    return {
        "client_id": client_id,
        "chunks_received": len(chunks),
        "errors": len(errors),
        "p50_latency_ms": p50,
        "p99_latency_ms": p99,
        "underruns": underruns
    }

async def load_test_20_clients(num_clients: int = 20, duration_sec: int = 60):
    """Run load test with 20 concurrent clients."""

    print(f"Starting load test: {num_clients} clients for {duration_sec} seconds")
    print("="*60)

    # Launch all clients
    tasks = [client_load_worker(i, duration_sec) for i in range(num_clients)]
    results = await asyncio.gather(*tasks)

    # Aggregate results
    total_chunks = sum(r["chunks_received"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_underruns = sum(r["underruns"] for r in results)
    avg_p99_latency = mean([r["p99_latency_ms"] for r in results if r["chunks_received"] > 0])

    print("\nLOAD TEST RESULTS")
    print("="*60)
    print(f"Clients: {num_clients}")
    print(f"Duration: {duration_sec} seconds")
    print(f"Total chunks delivered: {total_chunks}")
    print(f"Total errors: {total_errors}")
    print(f"Total underruns: {total_underruns}")
    print(f"Average p99 latency: {avg_p99_latency:.1f}ms")
    print("="*60)

    # Per-client breakdown
    print("\nPer-Client Results:")
    for r in results[:5]:  # Show first 5 clients
        print(f"  Client {r['client_id']}: {r['chunks_received']} chunks, "
              f"{r['errors']} errors, p99={r['p99_latency_ms']:.1f}ms, "
              f"{r['underruns']} underruns")
    if num_clients > 5:
        print(f"  ... ({num_clients - 5} more clients)")

    print("="*60)

    # Assertions
    min_expected_chunks = num_clients * duration_sec * 8  # ~8 chunks/sec per client
    assert total_chunks >= min_expected_chunks, \
        f"Expected ≥{min_expected_chunks} chunks, got {total_chunks}"
    assert total_underruns < total_chunks * 0.01, \
        f"Underrun rate {total_underruns/total_chunks*100:.2f}% exceeds 1% threshold"
    assert avg_p99_latency < 100, \
        f"Average p99 latency {avg_p99_latency:.1f}ms exceeds 100ms target"

    print("✅ Load test passed: System handles 20+ concurrent clients")

if __name__ == "__main__":
    asyncio.run(load_test_20_clients(num_clients=20, duration_sec=60))
```

**Run Test**:
```bash
# Start server
uvicorn server.main:app --port 8000 --workers 4

# Run load test (takes ~60 seconds)
python tests/load/test_load_20_clients.py
```

**Expected Output**:
```
Starting load test: 20 clients for 60 seconds
============================================================

LOAD TEST RESULTS
============================================================
Clients: 20
Duration: 60 seconds
Total chunks delivered: 10,234
Total errors: 0
Total underruns: 0
Average p99 latency: 67.3ms
============================================================

Per-Client Results:
  Client 0: 512 chunks, 0 errors, p99=65.2ms, 0 underruns
  Client 1: 511 chunks, 0 errors, p99=68.1ms, 0 underruns
  Client 2: 513 chunks, 0 errors, p99=66.7ms, 0 underruns
  Client 3: 510 chunks, 0 errors, p99=69.4ms, 0 underruns
  Client 4: 512 chunks, 0 errors, p99=64.8ms, 0 underruns
  ... (15 more clients)
============================================================
✅ Load test passed: System handles 20+ concurrent clients
```

---

## Performance Benchmarking

### Comprehensive Performance Benchmark Suite

**Objective**: Measure all performance metrics against Phase 1 baseline and optimization targets.

**Test Script** (`tests/performance/benchmark_suite.py`):
```python
import asyncio
import time
import psutil
import torch
import numpy as np
from server.synthesis_engine import OptimizedSynthesisEngine
from server.memory_monitor import MemoryMonitor
from statistics import mean, stdev

def benchmark_synthesis_latency(iterations: int = 100):
    """Benchmark synthesis latency."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    engine = OptimizedSynthesisEngine(device=device, batch_size=8, enable_compile=True)

    chords = [(0, 60, "major"), (22050, 65, "minor")]
    melody = [(0, 72, 0.8, 0.5), (11025, 74, 0.7, 0.5)]
    duration_sec = 1.0

    # Warm-up
    for _ in range(10):
        engine.render_phrase(chords, melody, duration_sec)

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        engine.render_phrase(chords, melody, duration_sec)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

    p50 = sorted(latencies)[len(latencies) // 2]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    avg = mean(latencies)

    return {
        "metric": "Synthesis Latency",
        "p50_ms": p50,
        "p99_ms": p99,
        "avg_ms": avg,
        "target_p99_ms": 100,
        "pass": p99 < 100
    }

def benchmark_memory_usage():
    """Benchmark memory usage."""
    process = psutil.Process()

    # Initial memory
    initial_rss = process.memory_info().rss / 1024 / 1024  # MB

    # Run synthesis for 100 iterations
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    engine = OptimizedSynthesisEngine(device=device)

    chords = [(0, 60, "major"), (22050, 65, "minor")]
    melody = [(0, 72, 0.8, 0.5), (11025, 74, 0.7, 0.5)]

    for _ in range(100):
        engine.render_phrase(chords, melody, 1.0)

    # Final memory
    final_rss = process.memory_info().rss / 1024 / 1024  # MB
    memory_growth = final_rss - initial_rss

    return {
        "metric": "Memory Growth (100 iterations)",
        "initial_mb": initial_rss,
        "final_mb": final_rss,
        "growth_mb": memory_growth,
        "target_growth_mb": 5,
        "pass": memory_growth < 5
    }

def benchmark_gpu_utilization():
    """Benchmark GPU memory utilization."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if device.type == "mps":
        # Metal GPU memory
        initial_allocated = torch.mps.current_allocated_memory() / 1024 / 1024

        engine = OptimizedSynthesisEngine(device=device, batch_size=8)
        chords = [(0, 60, "major"), (22050, 65, "minor")]
        melody = [(0, 72, 0.8, 0.5), (11025, 74, 0.7, 0.5)]

        for _ in range(50):
            engine.render_phrase(chords, melody, 1.0)

        peak_allocated = torch.mps.current_allocated_memory() / 1024 / 1024
        gpu_usage = peak_allocated - initial_allocated

        return {
            "metric": "GPU Memory Usage",
            "peak_allocated_mb": peak_allocated,
            "usage_mb": gpu_usage,
            "target_mb": 500,
            "pass": gpu_usage < 500
        }
    else:
        return {
            "metric": "GPU Memory Usage",
            "error": "No GPU available",
            "pass": False
        }

async def run_full_benchmark_suite():
    """Run complete benchmark suite."""

    print("="*60)
    print("PERFORMANCE BENCHMARK SUITE")
    print("="*60)
    print()

    # Run benchmarks
    print("Running synthesis latency benchmark...")
    latency_result = benchmark_synthesis_latency(iterations=100)

    print("Running memory usage benchmark...")
    memory_result = benchmark_memory_usage()

    print("Running GPU utilization benchmark...")
    gpu_result = benchmark_gpu_utilization()

    # Display results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)

    results = [latency_result, memory_result, gpu_result]

    for result in results:
        metric = result["metric"]
        status = "✅ PASS" if result.get("pass", False) else "❌ FAIL"

        print(f"\n{metric}: {status}")
        print("-" * 40)

        for key, value in result.items():
            if key not in ["metric", "pass"]:
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

    print("\n" + "="*60)

    # Overall pass/fail
    all_passed = all(r.get("pass", False) for r in results)

    if all_passed:
        print("✅ ALL BENCHMARKS PASSED")
    else:
        failed_count = sum(1 for r in results if not r.get("pass", False))
        print(f"❌ {failed_count} BENCHMARK(S) FAILED")

    print("="*60)

if __name__ == "__main__":
    asyncio.run(run_full_benchmark_suite())
```

**Run Benchmark**:
```bash
python tests/performance/benchmark_suite.py
```

**Expected Output**:
```
============================================================
PERFORMANCE BENCHMARK SUITE
============================================================

Running synthesis latency benchmark...
Running memory usage benchmark...
Running GPU utilization benchmark...

============================================================
BENCHMARK RESULTS
============================================================

Synthesis Latency: ✅ PASS
----------------------------------------
  p50_ms: 42.31
  p99_ms: 78.92
  avg_ms: 45.67
  target_p99_ms: 100

Memory Growth (100 iterations): ✅ PASS
----------------------------------------
  initial_mb: 145.23
  final_mb: 147.81
  growth_mb: 2.58
  target_growth_mb: 5

GPU Memory Usage: ✅ PASS
----------------------------------------
  peak_allocated_mb: 87.34
  usage_mb: 12.45
  target_mb: 500

============================================================
✅ ALL BENCHMARKS PASSED
============================================================
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: High Synthesis Latency (>100ms)

**Symptoms**:
- `synthesis_latency_seconds` p99 metric exceeds 0.1
- SynthesisLatencyHigh alert firing in Prometheus
- Client playback stuttering

**Diagnosis**:
```bash
# Check GPU availability
python -c "import torch; print(f'Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')"

# Check if torch.compile is enabled
grep "GPU_ENABLE_COMPILE" .env

# Profile synthesis
python -m cProfile -s cumtime server/synthesis_engine.py
```

**Solutions**:
1. **Enable GPU**: Ensure `AURALIS_DEVICE=auto` in `.env` and GPU is available
2. **Enable torch.compile**: Set `GPU_ENABLE_COMPILE=true` in `.env`
3. **Increase batch size**: Set `GPU_BATCH_SIZE=16` (if GPU has sufficient memory)
4. **Reduce phrase complexity**: Lower `intensity` parameter (0.3 instead of 0.7)

---

#### Issue 2: Buffer Underruns

**Symptoms**:
- `buffer_underruns_total` counter increasing
- Audio playback gaps/silence
- Client-side buffer empty warnings

**Diagnosis**:
```bash
# Check buffer depth metrics
curl http://localhost:9090/metrics | grep buffer_depth_chunks

# Check jitter metrics
curl http://localhost:9090/metrics | grep chunk_delivery_jitter_ms

# Check current tier
# (Look for "current_tier" in WebSocket messages)
```

**Solutions**:
1. **Increase initial buffer tier**: Set `BUFFER_INITIAL_TIER=stable` in `.env`
2. **Check network latency**: Run ping test to server
3. **Reduce concurrent clients**: If under load, reduce `WS_MAX_CONNECTIONS`
4. **Investigate jitter source**: Check if issue is network or synthesis latency

---

#### Issue 3: Memory Leak

**Symptoms**:
- `memory_usage_mb` metric steadily increasing
- Process RSS memory growing >20MB/hour
- MemoryLeakDetected alert firing

**Diagnosis**:
```bash
# Run memory stability test
python tests/integration/test_memory_stability.py --duration 1.0

# Enable tracemalloc profiling
python -c "
import tracemalloc
tracemalloc.start()
# ... run server ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
"
```

**Solutions**:
1. **Check GC configuration**: Verify `RealTimeGCConfig` is applied
2. **Review object pooling**: Ensure `AudioChunkPool` is reusing buffers
3. **Check for circular references**: Use `gc.garbage` to find uncollectable objects
4. **Update PyTorch**: Ensure latest version (memory leak fixes in 2.5.0+)

---

#### Issue 4: WebSocket Connection Failures

**Symptoms**:
- `websocket_send_errors_total` counter increasing
- Clients disconnecting unexpectedly
- "Connection closed" errors in logs

**Diagnosis**:
```bash
# Check active connections
curl http://localhost:9090/metrics | grep active_websocket_connections

# Check error types
curl http://localhost:9090/metrics | grep websocket_send_errors_total

# Check server logs
tail -f server.log | grep -i "websocket\|error"
```

**Solutions**:
1. **Check rate limiting**: Verify `WS_RATE_LIMIT_CHUNKS_PER_SEC` is not too restrictive
2. **Increase connection limit**: Set `WS_MAX_CONNECTIONS=100` if needed
3. **Check firewall rules**: Ensure WebSocket port (8000) is open
4. **Review client timeout**: Increase client-side connection timeout

---

#### Issue 5: GPU Not Detected

**Symptoms**:
- Device selection falls back to CPU
- Synthesis latency very high (>500ms)
- `gpu_memory_allocated_mb` metric reports 0

**Diagnosis**:
```bash
# Check PyTorch GPU support
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Check system GPU (macOS)
system_profiler SPDisplaysDataType | grep "Chipset Model"

# Check CUDA (Linux/Windows)
nvidia-smi
```

**Solutions**:
1. **Install correct PyTorch**: For Apple Silicon: `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
2. **Update GPU drivers**: For NVIDIA, install latest CUDA drivers
3. **Check macOS version**: Metal requires macOS 12.3+ for Apple Silicon
4. **Force CPU mode**: Set `AURALIS_DEVICE=cpu` in `.env` (performance will be degraded)

---

## Quick Reference Commands

### Start Server
```bash
# Development (with auto-reload)
uvicorn server.main:app --reload --port 8000

# Production (multi-worker)
uvicorn server.main:app --port 8000 --workers 4

# With metrics enabled
METRICS_ENABLED=true uvicorn server.main:app --port 8000
```

### Run Tests
```bash
# All tests
pytest

# Specific test suite
pytest tests/integration/
pytest tests/performance/
pytest tests/unit/

# With coverage
pytest --cov=server --cov-report=html

# Verbose output
pytest -v -s
```

### Monitoring
```bash
# View Prometheus metrics
curl http://localhost:9090/metrics

# Check active alerts
curl http://localhost:9090/api/v1/alerts

# Memory profiling
python -m memory_profiler server/main.py

# CPU profiling
python -m cProfile -o profile.stats server/main.py
```

### Performance Benchmarks
```bash
# Full benchmark suite
python tests/performance/benchmark_suite.py

# Synthesis latency only
python tests/performance/test_batch_synthesis.py

# Memory stability (30 min)
python tests/integration/test_memory_stability.py

# Load test (20 clients)
python tests/load/test_load_20_clients.py
```

---

## Success Criteria Summary

| **Metric** | **Target** | **Test** |
|------------|------------|----------|
| Synthesis latency (p99) | <100ms | `test_batch_synthesis.py` |
| Chunk delivery jitter (p99) | <50ms | `test_concurrent_clients.py` |
| Concurrent users | 10+ | `test_concurrent_clients.py` |
| Memory growth (8 hours) | <10MB | `test_memory_stability.py` |
| Buffer underruns | 0 | `test_adaptive_buffer.py` |
| GPU speedup | >1.5x | `test_batch_synthesis.py` |
| torch.compile speedup | >1.2x | `test_torch_compile.py` |
| Broadcast efficiency | Nx | `test_broadcast_encoding.py` |

---

## Next Steps

After completing these tests:

1. **Review Metrics**: Check Prometheus dashboards for any anomalies
2. **Tune Parameters**: Adjust buffer tiers, batch sizes based on results
3. **Document Findings**: Record baseline vs. optimized performance
4. **Deploy to Staging**: Test in staging environment before production
5. **Monitor Production**: Set up alerts and dashboards for ongoing monitoring

---

**End of Quick Start Testing Guide**
