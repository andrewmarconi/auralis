# Auralis 
### Real-time Generative Ambient Music in Python  

Auralis is an open-source **generative ambient music engine** built with **Python 3**, **FastAPI**, and **PyTorch**.  
It composes and streams evolving, atmospheric soundscapes — complete with synth pads, minimalist percussion, and slowly shifting melodies — **in real time** to connected web clients.

***

## 🎧 Overview

Auralis generates continuous ambient sound that never repeats, blending algorithmic composition, differentiable synthesis, and real-time streaming.

- **Generative Composition:** Markov chord progressions + constraint-based or transformer melodies.  
- **Real-Time Rendering:** GPU-accelerated **torchsynth** synthesis for pads, leads, and textures.  
- **Adaptive Streaming:** FastAPI WebSocket server sends audio at 100 ms intervals with low-latency buffering.  
- **Web Playback:** Web Audio API client handles playback, jitter correction, and user controls.

***

## 🧠 System Architecture

```
┌──────────────┐        ┌──────────────────┐        ┌───────────────┐
│ Composition  │  --->  │  Synthesis Core  │  --->  │ Ring Buffer   │
│  (Markov &   │        │  (torchsynth)    │        │  + Encoder    │
│  Constraints)│        └──────────────────┘        └──────┬────────┘
└──────────────┘                                           │
                                                           ▼
                                               ┌──────────────────┐
                                               │  FastAPI Server  │
                                               │ (WebSocket out)  │
                                               └──────┬───────────┘
                                                      │  base64 PCM chunks
                                                      ▼
                                      ┌────────────────────────────────┐
                                      │ Web Client (Web Audio API)     │
                                      │ Adaptive buffering + playback  │
                                      └────────────────────────────────┘
```

***

## ✨ Features

| Layer | Description |
|-------|--------------|
| 🎼 **Composition** | Chord progressions driven by Markov chains and constraint-based melody generation |
| 🎛️ **Synthesis** | Real-time GPU synthesizer using *torchsynth* with kernel fusion, batch processing, and torch.compile |
| 🔄 **Streaming** | 100 ms audio chunks streamed over WebSockets via FastAPI |
| 🧠 **Adaptive Buffering** | Client adjusts playback rate to maintain seamless streaming |
| 💻 **Live Controls** | Change key, BPM, or mood intensity from the browser |
| 📊 **Metrics** | Real-time performance monitoring via REST `/api/metrics` |

***

## ⚡ Performance Optimizations

Auralis achieves production-grade performance through comprehensive optimizations across the entire stack:

### Client-Side
- **4-Tier Adaptive Buffering:** Auto-adjusts between minimal (300ms), normal (500ms), stable (800ms), and defensive (1200ms) based on network conditions
- **EMA Jitter Tracking:** Exponential moving average (α=0.1) with variance calculation for intelligent tier escalation/de-escalation
- **AudioWorklet Processing:** Low-latency audio rendering in dedicated audio thread with sub-3ms overhead

### Server-Side
- **Object Pooling:** Pre-allocated numpy buffer pairs (10× reusable) eliminate GC pressure during audio encoding
- **GC Tuning:** Optimized thresholds (gen0: 50000, gen1: 500, gen2: 1000) prevent pauses during real-time synthesis
- **CPU Affinity:** Process pinned to performance cores (first 50% of available cores) for cache locality
- **Memory Pre-Allocation:** 32-second audio buffer prevents fragmentation and dynamic allocation overhead

### GPU Acceleration
- **Kernel Fusion:** @torch.jit.script fused operations (dual-osc + LFO, sine + envelope) minimize GPU memory transfers
- **torch.compile:** CUDA-only JIT compilation (reduce-overhead mode) provides 10-20% additional performance boost
- **Batch Processing:** Auto-tuned batch sizes (CUDA: 32, Metal: 16, CPU: 4) maximize GPU utilization
- **torch.no_grad():** Prevents gradient tracking overhead in inference-only synthesis
- **Cache Management:** Periodic GPU cache clearing (every 100 renders) prevents memory fragmentation

### Results
- **Synthesis Latency:** <35ms per phrase (well under 100ms real-time requirement)
- **Resource Reduction:** 30% decrease in CPU/GPU/memory usage vs. baseline
- **Memory Stability:** <10MB growth over 8+ hour continuous operation
- **Concurrent Capacity:** 10+ simultaneous clients without quality degradation

***

## 🏗️ Installation

### Requirements
- Python 3.10+  
- macOS (M1/M2/M4) or Linux with CUDA/Metal support  
- Node.js 18+ (for client build)

```bash
# Clone repo
git clone https://github.com/yourusername/auralis.git
cd auralis

# Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project in development mode
uv pip install -e ".[dev]"

# Download SoundFont files for realistic instruments
# The server requires FluidR3_GM.sf2 (142MB) in the soundfonts/ directory
#
# Download manually from one of these sources:
# 1. Musical Artifacts: https://musical-artifacts.com/artifacts/738
# 2. Polyphone: https://www.polyphone.io/en/soundfonts/instrument-sets/250-fluidr3-gm
# 3. Or search "FluidR3_GM.sf2 download" for alternative sources
#
# Save the downloaded .sf2 file to: soundfonts/FluidR3_GM.sf2

# Run the development server
uvicorn server.main:app --reload

# Open client
open client/index.html
```

***

## 🧩 Project Structure

```
auralis/
├── server/
│   ├── main.py                # FastAPI entrypoint
│   ├── synthesis_engine.py    # torchsynth synthesis core
│   ├── ring_buffer.py         # Audio buffering
│   └── streaming_server.py    # WebSocket streaming logic
│
├── composition/
│   ├── chord_generator.py     # Markov chord engine
│   ├── melody_generator.py    # Constraint/transformer melody
│   └── percussion_generator.py# Minimal ambient percussion
│
├── client/
│   ├── audio_client.js        # Web Audio playback
│   └── index.html             # Basic control UI
│
├── docs/
│   ├── system_architecture.md
│   ├── implementation_strategies.md
│   └── implementation_plan.md
│
└── README.md
```

***

## 🌈 Usage

### Run the streaming engine
```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### Open the client in your browser
```bash
open client/index.html
```

You’ll hear Auralis generate evolving ambient music in real time.  
Use the on-screen controls to modify **key**, **BPM**, and **intensity**.

***

## 🧪 Development Phases

| Phase | Focus |
|-------|--------|
| **1** | MVP – Markov chords, constraint melodies, basic synthesis |
| **2** | Real-time control, percussion, adaptive buffering |
| **3** | GPU optimization, effects (reverb, delay), monitoring |
| **4** | Production readiness, error handling, deployment |

See [`docs/implementation_plan.md`](docs/implementation_plan.md) for full details.

---

## 🛠️ Tech Stack

- **Python:** FastAPI 0.127+, asyncio, PyTorch 2.5+, torchsynth 1.0.2+, numpy 1.26+
- **Monitoring:** prometheus-client, psutil (CPU affinity, memory tracking)
- **Frontend:** Web Audio API (AudioWorklet), JavaScript (ES6+), HTML5
- **Audio:** 44.1 kHz, 16-bit PCM chunks streamed every 100 ms
- **GPU Acceleration:** torch.jit.script (kernel fusion), torch.compile (CUDA-only)
- **Optimization:** Object pooling, GC tuning, memory pre-allocation, CPU affinity

***

## 📊 Monitoring

Auralis includes comprehensive Prometheus-based monitoring:

### Metrics Exported
- **Synthesis Performance:** Latency histograms, phrase generation rate, render count
- **Memory Health:** RSS usage, GPU memory (allocated/reserved), tracemalloc tracking
- **Streaming Quality:** Chunk delivery jitter (per-client histograms), buffer depth gauges
- **System Health:** GC collection counts (gen0/1/2), memory leak detection (linear regression)

### Access Points
- **Prometheus Metrics:** http://localhost:8000/metrics (auto-updated every 5 seconds)
- **REST API:** http://localhost:8000/api/metrics (human-readable JSON summary)
- **Grafana Dashboards:** Pre-configured dashboards in `docs/grafana/`
  - `smooth-streaming.json` - Real-time jitter and buffer health
  - `resource-usage.json` - CPU/GPU/memory utilization

### Memory Leak Detection
Automated linear regression analysis on 24-hour memory samples with 20MB/hour growth threshold triggers alerts for investigation

***

## 🎨 Roadmap

- [ ] Transformer-conditioned lead melodies  
- [ ] Dynamic percussion textures  
- [ ] Cloud deployment with WebRTC streaming  
- [ ] User presets + MIDI export  
- [ ] Offline render to full-length ambient pieces  

***

## 🧑‍💻 Contributors

- **You!** Pull requests welcome — whether it’s improving synthesis modules, adding new compositional algorithms, or refining the client experience.

***

## 📜 License

MIT License - Copyright 2025
Developed by Andrew Marconi
