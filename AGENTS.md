# Auralis - Agent Guidelines

## Project Commands
- **Run:** `uv run python main.py`
- **Install Dependencies:** `uv add fastapi torch torchsynth`
- **Format:** No specific formatter configured
- **Lint:** No specific linter configured
- **Test:** Integration tests for audio performance via `uv run pytest tests/integration/`
- **Performance Tests:** `uv run pytest tests/performance/` (synthesis latency, GPU benchmarks)
- **Load Tests:** `uv run python tests/load/test_load_20_clients.py` (concurrent client stress tests)
- **Memory Tests:** `uv run python tests/integration/test_memory_stability.py` (leak detection)
- **Metrics:** Access Prometheus metrics at `http://localhost:9090/metrics`

## Code Style Guidelines
- **Python Version:** 3.12+ (managed through uv)
- **UV Management:** All dependencies via `uv add`/`uv run`, no manual venvs
- **Project Type:** Real-time generative ambient music engine
- **Architecture:** FastAPI + PyTorch + WebSockets + Web Audio API
- **Structure:** Modular design with `server/`, `composition/`, and `client/` directories
- **Audio:** 44.1kHz, 16-bit PCM, 100ms chunks over WebSockets
- **Key Libraries:** FastAPI, asyncio, PyTorch, torchsynth, numpy, prometheus-client, psutil
- **Real-time:** Low latency streaming with adaptive buffering (<100ms latency)
- **Composition:** Markov chains + constraint-based melodies
- **Naming:** Use descriptive names for audio/composition functions
- **GPU:** Prioritize Metal/CUDA acceleration where possible
- **Error Handling:** Focus on seamless audio streaming over exhaustive error recovery
- **Performance Patterns:**
  - **Client-Side Adaptive Buffering:**
    - 4-tier system (minimal/normal/stable/defensive: 300-1200ms)
    - EMA jitter tracking (α=0.1) with variance calculation
    - Auto-escalation on jitter>50ms or underruns>5%
    - Auto-de-escalation on jitter<20ms and underruns<1%
  - **Server-Side Optimizations:**
    - Object pooling: Pre-allocated numpy buffers (10 pairs) for audio encoding
    - GC tuning: (gen0: 50000, gen1: 500, gen2: 1000) for real-time constraints
    - CPU affinity: Pin synthesis to performance cores (first 50% of cores)
    - Memory pre-allocation: 32-second audio buffer to prevent fragmentation
  - **GPU Acceleration:**
    - Kernel fusion: @torch.jit.script for dual-osc + LFO operations
    - torch.compile: CUDA-only (reduce-overhead mode) for 10-20% boost
    - Batch processing: Auto-tuned batch sizes (CUDA:32, Metal:16, CPU:4)
    - torch.no_grad() context: Prevents gradient tracking memory overhead
    - Cache management: Clear GPU cache every 100 renders
  - **Monitoring & Detection:**
    - Memory leak detection: Linear regression on 24-hour samples (20MB/hour threshold)
    - Prometheus metrics: Synthesis latency, jitter histograms, GPU memory, GC stats
    - Buffer health tracking: Per-client depth monitoring with warnings
    - Graceful shutdown: 5-second drain period for active WebSocket connections

## Active Technologies
- **Core Stack:** Python 3.12+ via uv, FastAPI, PyTorch 2.5+, torchsynth, asyncio, WebSockets (001-phase1-mvp)
- **Audio Pipeline:** Ring buffers with 100ms chunks, base64-encoded 16-bit PCM streaming (001-phase1-mvp)
- **Client-Side:** AudioWorklet processor, 4-tier adaptive buffering, EMA jitter tracking (003-performance-optimizations)
- **Server-Side:** Object pooling, GC tuning, CPU affinity, memory pre-allocation (003-performance-optimizations)
- **GPU Acceleration:** Device selector (Metal/CUDA/CPU priority), kernel fusion (@torch.jit.script), torch.compile (CUDA-only), batch processing with auto-tuning (003-performance-optimizations)
- **Monitoring:** Prometheus metrics (synthesis latency, jitter histograms, memory, GPU), memory leak detection with linear regression (003-performance-optimizations)
- **Dependencies:** FastAPI 0.127+, PyTorch 2.5+, torchsynth 1.0.2+, prometheus-client, psutil, numpy 1.26+

## Recent Changes
- **003-performance-optimizations (Phase 3):**
  - Client: AudioWorklet processor, 4-tier adaptive buffering (300-1200ms), EMA jitter tracking (α=0.1)
  - Server: Object pooling (10 buffer pairs), GC tuning (50000/500/1000), CPU affinity (performance cores)
  - GPU: Kernel fusion (@torch.jit.script), torch.compile (CUDA-only), batch processing (auto-tuned sizes)
  - Memory: Pre-allocation (32s buffer), leak detection (linear regression, 20MB/hour threshold)
  - Monitoring: Prometheus metrics (synthesis latency, jitter, memory, GPU, GC stats)
  - Achieved: 31ms synthesis latency, <10MB memory growth over 8 hours, 30% resource reduction
- **002-enhanced-generation-controls:** Added real-time parameter controls (key, BPM, intensity, melody complexity)
- **001-phase1-mvp:** Initial FastAPI + PyTorch + WebSockets implementation
