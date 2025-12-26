# Auralis Quick Start Guide

Get Auralis up and running in 5 minutes.

## Prerequisites

- Python 3.12 or higher
- `uv` package manager installed ([install here](https://github.com/astral-sh/uv))
- Modern web browser (Chrome/Edge)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/auralis.git
cd auralis

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project in development mode
uv pip install -e ".[dev]"
```

## Running the Server

```bash
# Start the FastAPI server
uv run uvicorn server.main:app --reload --host 0.0.0.0 --port 8000

# Alternative: Use the convenience script
uv run python main.py
```

You should see:
```
🎧 Starting Auralis server...
✓ Synthesis engine ready: mps  # or cuda/cpu
✓ Ring buffer initialized: 88200 samples
✓ Server ready for connections
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Accessing the Web Client

1. Open your browser to: **http://localhost:8000**
2. Click **"Start Streaming"**
3. Enjoy real-time generated ambient music!

## Controls

- **Musical Key**: Change the tonal center (A Minor, D Minor, etc.)
- **Tempo**: Adjust BPM (40-120, default: 70)
- **Intensity**: Control note density and range (0.0-1.0)

## API Endpoints

### WebSocket Streaming
```
ws://localhost:8000/ws/stream
```
Receives base64-encoded 16-bit PCM audio chunks (100ms each, 44.1kHz stereo)

### REST API
```bash
# Server status
curl http://localhost:8000/api/status

# Performance metrics
curl http://localhost:8000/api/metrics

# Update parameters
curl -X POST http://localhost:8000/api/control \
  -H "Content-Type: application/json" \
  -d '{"key": "D", "bpm": 80, "intensity": 0.7}'
```

## Troubleshooting

### Port already in use
```bash
# Use a different port
uvicorn server.main:app --port 8001
```

### GPU not detected
Check the server logs. Auralis will automatically fall back to CPU if GPU (Metal/CUDA) is unavailable.

### Audio not playing
1. Ensure browser supports Web Audio API (Chrome/Edge recommended)
2. Check browser console for errors (F12)
3. Verify WebSocket connection status in the UI

### Buffer underruns
If you see "Buffer underflow" warnings:
- Reduce intensity to lower synthesis load
- Check CPU/GPU usage
- Ensure stable network connection

## Development

### Run tests
```bash
pytest
```

### Code quality
```bash
# Format
black server/ composition/ tests/

# Lint
ruff check server/ composition/ tests/

# Type check
mypy server/ composition/ --strict
```

### Monitor performance
```bash
# Watch metrics endpoint
watch -n 1 curl -s http://localhost:8000/api/metrics
```

## Next Steps

- Read [CLAUDE.md](CLAUDE.md) for detailed architecture
- Check [docs/](docs/) for technical specifications
- See [specs/001-phase1-mvp/](specs/001-phase1-mvp/) for implementation details

## Support

- Issues: https://github.com/yourusername/auralis/issues
- Documentation: [README.md](README.md)
