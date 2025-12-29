# Auralis Quick Start Guide

Get Auralis running in under 5 minutes.

---

## Prerequisites

- **Python 3.12+**
- **macOS** (Apple Silicon preferred) **or Linux**
- **Git**

---

## Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/auralis.git
cd auralis

# 2. Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment
uv venv

# 4. Activate environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 5. Install dependencies
uv pip install -e ".[dev]"

# 6. Copy configuration template
cp .env.example .env

# 7. (Optional) Edit configuration
nano .env
```

---

## Run Development Server

```bash
# Start FastAPI server
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at `http://localhost:8000`

---

## Open Client

### Option 1: Direct File Access
```bash
# macOS
open client/index.html

# Linux
xdg-open client/index.html

# Windows
start client/index.html
```

### Option 2: Local HTTP Server
```bash
# Python built-in server
cd client
python -m http.server 8080

# Then open http://localhost:8080
```

---

## Verify Installation

### 1. Check Server Health

```bash
curl http://localhost:8000/api/v1/metrics/health
```

Expected response:
```json
{
  "status": "healthy",
  "synthesis_error_rate": 0.0,
  "rtf": 0.0,
  "cpu_percent": 15.2
}
```

### 2. Check Device Detection

```bash
python -c "from auralis.core.device_manager import detect_device; print(detect_device())"
```

Expected output (macOS M4):
```
('mps', 'mps')
```

### 3. Test Synthesis

```python
# test_synthesis.py
from auralis.synthesis.engine_factory import SynthesisEngineFactory

engine = SynthesisEngineFactory.create_engine(sample_rate=44100)

# Render test phrase
audio = engine.render_phrase(
    chords=[(0, 57, "i")],
    melody=[(0, 60, 0.7, 1.0)],
    percussion=[],
    duration_sec=2.0,
)

print(f"Rendered audio shape: {audio.shape}")  # Expected: (2, 88200)
```

---

## Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=auralis --cov=server

# Specific test file
pytest tests/unit/test_chord_generator.py

# Performance benchmarks
pytest tests/performance/ -v
```

---

## Common Issues

### 1. `uv: command not found`

```bash
# Install uv manually
pip install uv

# Or use curl installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. `torch.backends.mps.is_available()` returns `False`

- Ensure macOS 12.3+ with Apple Silicon
- Update PyTorch: `uv pip install --upgrade torch torchaudio`

### 3. WebSocket Connection Fails

- Check firewall settings
- Ensure port 8000 is not in use: `lsof -i :8000`
- Try: `uvicorn server.main:app --host 127.0.0.1 --port 8000`

### 4. Audio Playback Silent

- Check browser console for errors (F12)
- Ensure AudioContext is resumed (Chrome autoplay policy)
- Click "Connect" button on client page

---

## Project Structure Quick Reference

```
auralis/
├── auralis/               # Core library
│   ├── core/              # Config, device management
│   ├── music/             # Music theory
│   ├── composition/       # Generators (chords, melody)
│   ├── synthesis/         # Audio rendering
│   ├── streaming/         # Ring buffer, encoding
│   ├── api/               # Pydantic schemas
│   └── monitoring/        # Metrics, logging
│
├── server/                # FastAPI application
│   ├── main.py            # App entrypoint
│   ├── websocket.py       # Streaming endpoint
│   └── ...
│
├── client/                # Web frontend
│   ├── index.html         # Main UI
│   ├── audio_worklet_processor.js
│   └── audio_client_worklet.js
│
├── tests/                 # Test suite
│   ├── unit/
│   ├── integration/
│   └── performance/
│
└── docs/                  # Documentation
```

---

## Configuration Quick Reference

Key environment variables (`.env`):

```bash
# Device selection
AURALIS_DEVICE=auto  # auto | mps | cuda | cpu

# Server
AURALIS_PORT=8000
AURALIS_MAX_CLIENTS=10

# Generation
AURALIS_DEFAULT_KEY="A minor"
AURALIS_DEFAULT_BPM=70
AURALIS_DEFAULT_INTENSITY=0.5

# Logging
AURALIS_LOG_LEVEL=INFO
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ws/stream` | WebSocket | Audio streaming |
| `/api/v1/control` | POST | Update generation params |
| `/api/v1/metrics` | GET | Get performance metrics |
| `/api/v1/metrics/health` | GET | Health check |

---

## Client Controls

Once connected to `/ws/stream`:

```javascript
// Update generation parameters
client.setControl({
    key: "C minor",
    bpm: 80,
    intensity: 0.7
});

// Get status
const status = client.getStatus();
console.log('Buffer depth:', status.bufferDepthMs);
```

---

## Development Commands

```bash
# Code formatting
black auralis/ tests/

# Linting
ruff check auralis/ tests/

# Type checking
mypy auralis/ --strict

# Run server with auto-reload
uvicorn server.main:app --reload

# Run tests with coverage
pytest --cov=auralis --cov-report=html

# Benchmark synthesis
pytest tests/performance/test_synthesis_benchmark.py -v
```

---

## Docker Deployment

```bash
# Build image
docker build -t auralis -f deployment/Dockerfile .

# Run container
docker run -p 8000:8000 auralis

# Or use docker-compose
cd deployment
docker-compose up
```

---

## Next Steps

1. **Read**: [implementation_plan.md](implementation_plan.md) for phased development
2. **Review**: [system_architecture.md](system_architecture.md) for component overview
3. **Explore**: [project_structure.md](project_structure.md) for module organization
4. **Test**: Run benchmarks to verify hardware performance

---

## Support

- **Issues**: https://github.com/yourusername/auralis/issues
- **Documentation**: See `docs/` directory
- **License**: MIT (see [LICENSE](../LICENSE))

---

*Happy coding! 🎵*
