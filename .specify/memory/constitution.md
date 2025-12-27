<!-- Sync Impact Report -->
<!-- Version change: 1.0.0 → 1.1.0 (new principle added) -->
<!-- Modified principles: None -->
<!-- Added sections: VI. Developer Experience & Flow Efficiency -->
<!-- Removed sections: None -->
<!-- Templates requiring updates: ✅ plan-template.md (Constitution Check section updated) -->
<!-- Follow-up TODOs: None -->

# Auralis Constitution

## Core Principles

### I. UV-First Development
All Python development MUST use uv exclusively. No direct `python` commands, no manual venv creation outside uv. All dependencies, scripts, and virtual environments managed through uv. This ensures reproducible builds and consistent dependency management across all environments.

### II. Real-Time Audio Performance
All code changes MUST preserve real-time audio performance. Audio streaming at 44.1kHz, 16-bit PCM in 100ms chunks cannot be interrupted. Any blocking operations, inefficient algorithms, or memory leaks that cause audio glitches are prohibited. Low-latency is the primary constraint.

### III. Modular Architecture
All components MUST follow modular design with clear separation: server/ for FastAPI endpoints, composition/ for generative algorithms, client/ for Web Audio API. Circular dependencies forbidden. Each module should be independently testable and maintainable.

### IV. GPU-Accelerated Synthesis
All audio synthesis MUST prioritize GPU acceleration through Metal/CUDA. CPU-only fallback allowed but not default. PyTorch operations should use Metal on Apple Silicon, CUDA on NVIDIA. This ensures real-time generative music synthesis performance.

### V. WebSocket Streaming Protocol
All client-server communication MUST use WebSockets for real-time audio streaming. No REST for audio data. WebSocket frames must contain base64-encoded PCM chunks with consistent timing metadata. Adaptive client buffering required for seamless playback.

### VI. Developer Experience & Flow Efficiency
All code and implementation work MUST actively address developer experience quality metrics. Specifically:

- **Cyclomatic Complexity**: Keep functions under McCabe complexity of 10. Complex logic MUST be decomposed into smaller, focused functions with clear single responsibilities.
- **Workflow Friction**: Minimize context switching and tool overhead. Development commands MUST be discoverable, consistent, and fast (<5s for common operations).
- **Lead Time for Changes**: Design for rapid iteration. Code changes MUST be implementable and testable within a single focus session without requiring large-scale refactoring.
- **Focus Disruption**: Reduce cognitive load through clear naming, explicit interfaces, and predictable patterns. Avoid deep nesting (>3 levels), magic numbers, and implicit behaviors.

**Rationale**: High-quality developer experience directly impacts code quality, maintainability, and team velocity. By systematically reducing complexity and friction, we enable faster iteration cycles, reduce bugs, and improve long-term sustainability of the codebase.

## Development Standards

### Technology Stack Requirements
- Python 3.12+ managed through uv
- FastAPI for server endpoints
- PyTorch + torchsynth for audio synthesis
- WebSockets for real-time streaming
- Web Audio API for client playback
- Asyncio for concurrent operations

### Code Quality Standards
- All dependencies managed via `uv add`/`uv remove`
- No manual virtual environment creation
- Use `uv run` for all script execution
- Maintain <100ms audio processing latency
- Memory usage optimized for continuous operation
- Error handling must preserve audio stream continuity
- Functions limited to McCabe complexity ≤10
- Maximum nesting depth of 3 levels
- Clear, descriptive naming without abbreviations

## Development Workflow

### Environment Setup
All developers MUST use `uv` for environment setup. No alternative package managers or manual Python installations for development dependencies.

### Code Review Requirements
All PRs MUST verify:
- Real-time audio performance is preserved
- UV dependency management is correct
- No blocking operations in audio pipeline
- Modular architecture boundaries maintained
- GPU acceleration properly utilized
- Cyclomatic complexity within acceptable bounds
- No unnecessary workflow friction introduced
- Changes testable within single focus session

### Testing Strategy
Integration tests MUST verify:
- WebSocket streaming under load
- Audio quality metrics (latency, glitches)
- GPU acceleration effectiveness
- Cross-platform compatibility (Metal/CUDA)

## Governance

This constitution supersedes all other development practices. Amendments require documentation, approval, and migration plan. All development must verify compliance through automated checks and code review.

**Version**: 1.1.0 | **Ratified**: 2025-01-26 | **Last Amended**: 2025-12-27
