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
  - Adaptive buffer tiers (minimal/normal/stable/defensive) based on jitter
  - Statistical jitter tracking with Exponential Moving Average (EMA)
  - Token bucket rate limiting for WebSocket flow control
  - Broadcast architecture (1× encode, N× send) for concurrent clients
  - GPU batch processing with torch.compile kernel fusion
  - Object pooling for zero-allocation audio encoding
  - GC tuning for real-time constraints (gen0: 50000, gen1: 500, gen2: 1000)
  - Memory pre-allocation to prevent fragmentation
  - Linear regression-based memory leak detection

## Active Technologies
- Python 3.12+ via uv + FastAPI, PyTorch, torchsynth, asyncio, WebSockets (001-phase1-mvp)
- Audio ring buffers, temporary in-memory storage (001-phase1-mvp)
- Adaptive buffer management with 4-tier system and EMA jitter tracking (003-performance-optimizations)
- Broadcast WebSocket architecture with per-client cursors and token bucket rate limiting (003-performance-optimizations)
- GPU optimization: batch processing, torch.compile, kernel fusion, Metal/CUDA support (003-performance-optimizations)
- Memory management: pre-allocation, GC tuning, leak detection with linear regression (003-performance-optimizations)
- Prometheus metrics: synthesis latency, buffer depth, jitter, memory, GPU utilization (003-performance-optimizations)
- Object pooling for base64 encoding and graceful shutdown with drain periods (003-performance-optimizations)

## Recent Changes
- 003-performance-optimizations: Added adaptive buffer tiers, EMA jitter tracking, token bucket rate limiting, broadcast WebSocket architecture, GPU batch processing with torch.compile, memory leak detection, Prometheus metrics, object pooling, GC tuning
- 001-phase1-mvp: Added Python 3.12+ via uv + FastAPI, PyTorch, torchsynth, asyncio, WebSockets
