# Auralis - Agent Guidelines

## Project Commands
- **Run:** `uv run python main.py`
- **Install Dependencies:** `uv add fastapi torch torchsynth`
- **Format:** No specific formatter configured
- **Lint:** No specific linter configured  
- **Test:** Integration tests for audio performance via `uv run pytest tests/integration/`

## Code Style Guidelines
- **Python Version:** 3.12+ (managed through uv)
- **UV Management:** All dependencies via `uv add`/`uv run`, no manual venvs
- **Project Type:** Real-time generative ambient music engine
- **Architecture:** FastAPI + PyTorch + WebSockets + Web Audio API
- **Structure:** Modular design with `server/`, `composition/`, and `client/` directories
- **Audio:** 44.1kHz, 16-bit PCM, 100ms chunks over WebSockets
- **Key Libraries:** FastAPI, asyncio, PyTorch, torchsynth, numpy
- **Real-time:** Low latency streaming with adaptive buffering (<100ms latency)
- **Composition:** Markov chains + constraint-based melodies
- **Naming:** Use descriptive names for audio/composition functions
- **GPU:** Prioritize Metal/CUDA acceleration where possible
- **Error Handling:** Focus on seamless audio streaming over exhaustive error recovery