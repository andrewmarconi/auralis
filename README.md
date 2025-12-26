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
│  (Markov &   │        │  (torchsynth)   │        │  + Encoder    │
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
| 🎛️ **Synthesis** | Real-time GPU synthesizer using *torchsynth* — multiple oscillators, filters, and ADSR envelopes |
| 🔄 **Streaming** | 100 ms audio chunks streamed over WebSockets via FastAPI |
| 🧠 **Adaptive Buffering** | Client adjusts playback rate to maintain seamless streaming |
| 💻 **Live Controls** | Change key, BPM, or mood intensity from the browser |
| 📊 **Metrics** | Real-time performance monitoring via REST `/api/metrics` |

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

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) enable Apple GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

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

- **Python:** FastAPI, asyncio, PyTorch, torchsynth, numpy  
- **Frontend:** Web Audio API, JavaScript, HTML5  
- **Audio:** 44.1 kHz, 16-bit PCM chunks streamed every 100 ms  
- **Optional:** Opus compression, DistilGPT‑2 melody transformer  

***

## 📊 Monitoring

Every 10 seconds:
- Logs buffer depth, synthesis latency, active clients  
- Reports metrics via `/api/metrics` REST endpoint  
- Future: Prometheus adapter and GPU profiling dashboard  

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
Developed by Andrew MArconi
