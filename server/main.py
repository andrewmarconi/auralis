"""
Auralis FastAPI Server

Main application entrypoint for real-time ambient music streaming.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from server.ring_buffer import RingBuffer
from server.streaming_server import StreamingServer
from server.synthesis_engine import SynthesisEngine


# Configuration Models
class SynthesisParameters(BaseModel):
    """Audio generation configuration settings."""

    key: str = Field(default="A", description="Musical key")
    bpm: int = Field(default=70, ge=40, le=120, description="Tempo in BPM")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="Generation intensity")
    melody_complexity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Controls melodic pattern intricacy"
    )
    chord_progression_variety: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Controls harmonic exploration range"
    )
    harmonic_density: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Controls layered chord richness"
    )


class PerformanceMetrics(BaseModel):
    """Real-time performance monitoring data."""

    timestamp: float
    buffer_depth_ms: float
    active_connections: int
    chunks_generated: int
    synthesis_latency_ms: float
    device_type: str


# Global state
class ApplicationState:
    """Shared application state."""

    def __init__(self):
        self.ring_buffer: Optional[RingBuffer] = None
        self.streaming_server: Optional[StreamingServer] = None
        self.synthesis_engine: Optional[SynthesisEngine] = None
        self.synthesis_task: Optional[asyncio.Task] = None
        self.parameters = SynthesisParameters()
        self.chunks_generated = 0
        self.start_time = time.time()


app_state = ApplicationState()


# Parameter Validation and Adjustment
def validate_parameters(params: SynthesisParameters) -> List[str]:
    """Validate parameter combinations and return warning messages."""
    warnings = []

    # Check for potential conflicts
    if params.melody_complexity > 0.8 and params.intensity < 0.3:
        warnings.append("High melody complexity with low intensity may reduce perceived musicality")

    if params.harmonic_density > 0.8 and params.chord_progression_variety < 0.3:
        warnings.append(
            "High harmonic density with low progression variety may create repetitive harmonies"
        )

    return warnings


def adjust_parameters_for_performance(
    params: SynthesisParameters, target_load: float = 0.8
) -> SynthesisParameters:
    """Automatically adjust parameters if they exceed performance capacity."""
    # Simple scaling based on complexity
    complexity_score = (
        params.melody_complexity + params.chord_progression_variety + params.harmonic_density
    ) / 3.0

    if complexity_score > target_load:
        scale_factor = target_load / complexity_score
        return SynthesisParameters(
            key=params.key,
            bpm=params.bpm,
            intensity=params.intensity,
            melody_complexity=params.melody_complexity * scale_factor,
            chord_progression_variety=params.chord_progression_variety * scale_factor,
            harmonic_density=params.harmonic_density * scale_factor,
        )

    return params


# Lifecycle Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("🎧 Starting Auralis server...")

    # Initialize components
    app_state.ring_buffer = RingBuffer(
        capacity_samples=88200,  # 2 seconds
        sample_rate=44100,
        chunk_size=4410,  # 100ms
    )

    app_state.synthesis_engine = SynthesisEngine(sample_rate=44100)

    app_state.streaming_server = StreamingServer(app_state.ring_buffer)

    # Start background synthesis task
    app_state.synthesis_task = asyncio.create_task(synthesis_loop())

    logger.info(f"✓ Synthesis engine ready: {app_state.synthesis_engine.device.type}")
    logger.info(f"✓ Ring buffer initialized: {app_state.ring_buffer.capacity_samples} samples")
    logger.info("✓ Server ready for connections")

    yield

    # Cleanup
    logger.info("🛑 Shutting down Auralis server...")
    if app_state.synthesis_task:
        app_state.synthesis_task.cancel()
        try:
            await app_state.synthesis_task
        except asyncio.CancelledError:
            logger.info("✓ Synthesis task cancelled")
            raise

    logger.info("✓ Server stopped")


# Create FastAPI app
app = FastAPI(
    title="Auralis",
    description="Real-time Generative Ambient Music Streaming",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # SECURITY: Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Background Synthesis Loop
async def synthesis_loop():
    """
    Background task that continuously generates audio and fills the ring buffer.
    """
    from composition.chord_generator import ChordProgressionGenerator
    from composition.melody_generator import ConstrainedMelodyGenerator

    logger.info("Starting synthesis loop...")

    # Initialize generators
    chord_gen = ChordProgressionGenerator()
    melody_gen = ConstrainedMelodyGenerator()

    phrase_count = 0

    while True:
        try:
            # Generate SHORT phrases (1 second) to avoid buffer overflow
            # This keeps the buffer topped up without overwhelming it
            bpm = app_state.parameters.bpm
            intensity = app_state.parameters.intensity

            # Generate 1-second phrase (fits easily in 2-second buffer)
            duration_sec = 1.0

            # Generate chord progression (for musical context)
            chord_progression = chord_gen.generate_progression(
                length_bars=2, variety=app_state.parameters.chord_progression_variety
            )
            chord_events = chord_progression.to_midi_events(bpm=bpm)

            # Generate melody constrained to chords
            melody_phrase = melody_gen.generate_melody(
                chord_progression=chord_events,
                duration_sec=duration_sec,
                bpm=bpm,
                intensity=intensity,
                complexity=app_state.parameters.melody_complexity,
            )
            melody_events = melody_phrase.to_sample_events()

            # Render audio
            start_time = time.time()
            audio_data = app_state.synthesis_engine.render_phrase(
                chord_events, melody_events, duration_sec
            )
            synthesis_time_ms = (time.time() - start_time) * 1000

            # Write to ring buffer (1 second = 44,100 samples fits in buffer)
            success = app_state.ring_buffer.write(audio_data)

            if not success:
                # Buffer full - wait a bit for it to drain
                logger.debug("Buffer full, waiting for drain...")
                await asyncio.sleep(0.5)
                continue

            phrase_count += 1

            # Log performance periodically
            if phrase_count % 10 == 0:
                buffer_ms = app_state.ring_buffer.get_buffer_depth_ms()
                logger.info(
                    f"Phrase #{phrase_count} | "
                    f"Synthesis: {synthesis_time_ms:.1f}ms | "
                    f"Buffer: {buffer_ms:.0f}ms | "
                    f"BPM: {bpm} | Intensity: {intensity:.2f}"
                )

            # Small wait to prevent tight loop (generate ~10 phrases/sec)
            # This keeps buffer topped up without overwhelming it
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Synthesis loop error: {e}")
            import traceback

            traceback.print_exc()
            await asyncio.sleep(1.0)  # Prevent tight error loop


# WebSocket Endpoint
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.

    Connects clients and streams 100ms audio chunks.
    """
    await app_state.streaming_server.handle_client(websocket)


# REST API Endpoints
@app.get("/api/status")
async def get_status():
    """
    Get server status and health information.

    Returns:
        Server status including device info, buffer depth, active connections
    """
    uptime_sec = time.time() - app_state.start_time

    return {
        "status": "running",
        "uptime_seconds": uptime_sec,
        "synthesis_engine": app_state.synthesis_engine.get_status(),
        "buffer_depth_ms": app_state.ring_buffer.get_buffer_depth_ms(),
        "active_connections": app_state.streaming_server.get_active_client_count(),
        "chunks_generated": app_state.chunks_generated,
        "parameters": app_state.parameters.model_dump(),
    }


@app.get("/api/metrics", response_model=PerformanceMetrics)
async def get_metrics():
    """
    Get real-time performance metrics.

    Returns:
        Performance monitoring data including latency and buffer status
    """
    return PerformanceMetrics(
        timestamp=time.time(),
        buffer_depth_ms=app_state.ring_buffer.get_buffer_depth_ms(),
        active_connections=app_state.streaming_server.get_active_client_count(),
        chunks_generated=app_state.chunks_generated,
        synthesis_latency_ms=app_state.synthesis_engine.get_latency_ms(),
        device_type=app_state.synthesis_engine.device.type,
    )


@app.post("/api/control")
async def update_control(params: SynthesisParameters):
    """
    Update synthesis parameters.

    Args:
        params: Synthesis configuration (key, BPM, intensity, melody_complexity, chord_progression_variety, harmonic_density)

    Returns:
        Confirmation message with updated parameters and any warnings
    """
    # Adjust parameters for performance if needed
    adjusted_params = adjust_parameters_for_performance(params)

    # Validate and get warnings
    warnings = validate_parameters(adjusted_params)

    app_state.parameters = adjusted_params

    if warnings:
        logger.warning(f"Parameter warnings: {warnings}")

    logger.info(f"Parameters updated: {adjusted_params.model_dump()}")

    # Broadcast to all connected clients
    await app_state.streaming_server.broadcast_status(
        {
            "message": "Parameters updated",
            "parameters": adjusted_params.model_dump(),
            "warnings": warnings,
        }
    )

    return {
        "message": "Parameters updated successfully",
        "parameters": adjusted_params.model_dump(),
        "warnings": warnings,
    }


# Static files for client (must be before route definitions)
try:
    from fastapi.staticfiles import StaticFiles
    import os

    client_dir = os.path.abspath("client")
    if os.path.exists(client_dir):
        # Serve static files from /static/
        app.mount("/static", StaticFiles(directory=client_dir), name="static")
        logger.info(f"✓ Static files mounted from {client_dir}")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")


@app.get("/")
async def serve_index():
    """Serve the web client interface."""
    return FileResponse("client/index.html")


@app.get("/{filename}")
async def serve_client_files(filename: str):
    """Serve individual client files (JS, CSS, etc.)."""
    import os
    from fastapi.responses import FileResponse, JSONResponse

    file_path = f"client/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return JSONResponse({"error": "File not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
