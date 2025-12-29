"""FastAPI server entrypoint for Auralis.

Main application with WebSocket streaming endpoint, REST API for status/metrics,
and static file serving for the web client.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from server.config import get_config
from server.device_selector import get_device_info
from server.di_container import cleanup_container, get_container
from server.gc_config import configure_gc
from server.logging_config import setup_logging
from server.presets import list_presets

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Global startup timestamp
_startup_time = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager (startup/shutdown).

    Args:
        app: FastAPI application instance

    Yields:
        Control during application lifetime
    """
    global _startup_time

    # Startup
    logger.info("=" * 60)
    logger.info("Starting Auralis server...")
    _startup_time = time.time()

    # Configure garbage collection
    configure_gc()

    # Get DI container
    container = get_container()
    config = container.get_config()

    logger.info(f"Environment: {config.env}")
    logger.info(f"Host: {config.host}:{config.port}")

    # Detect device
    device_info = get_device_info()
    logger.info(f"Device: {device_info['device']} (FluidSynth: CPU)")

    # Initialize SoundFont manager and load SoundFonts
    try:
        soundfont_manager = container.get_soundfont_manager()
        soundfont_manager.load_all_soundfonts()
        logger.info(
            f"SoundFonts loaded: {soundfont_manager.get_total_memory_mb():.1f}MB"
        )
    except Exception as e:
        logger.error(f"Failed to load SoundFonts: {e}")
        logger.error(
            "Please ensure SoundFonts are downloaded to soundfonts/soundfonts/ directory"
        )
        logger.error("See soundfonts/.env.example for download instructions")
        raise

    # Start memory monitoring
    memory_monitor = container.get_memory_monitor()
    memory_monitor.start_tracking()

    logger.info("Auralis server ready!")
    logger.info("Note: Synthesis engine will start when first client connects")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down Auralis server...")

    # Stop synthesis engine if running
    try:
        synthesis_engine = container.get_synthesis_engine()
        if synthesis_engine.is_running():
            await synthesis_engine.stop_generation_loop()
            logger.info("Synthesis engine stopped")
    except Exception as e:
        logger.warning(f"Error stopping synthesis engine: {e}")

    cleanup_container()
    logger.info("Auralis server stopped")


# Create FastAPI app
app = FastAPI(
    title="Auralis API",
    version="2.0.0",
    description="Real-time generative ambient music streaming engine",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for client
client_dir = Path("client")
if client_dir.exists():
    app.mount("/client", StaticFiles(directory=str(client_dir)), name="client")


# REST API Endpoints


@app.get("/")
async def serve_root() -> HTMLResponse:
    """Serve main HTML client interface.

    Returns:
        HTML response with client application
    """
    index_path = Path("client/index.html")

    if not index_path.exists():
        return HTMLResponse(
            content="<h1>Auralis Server</h1><p>Client not found. Please ensure client/index.html exists.</p>",
            status_code=404,
        )

    with open(index_path) as f:
        return HTMLResponse(content=f.read())


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    """Get system status.

    Returns:
        Dictionary with server status information
    """
    container = get_container()
    streaming_server = container.get_streaming_server()
    ring_buffer = container.get_ring_buffer()
    soundfont_manager = container.get_soundfont_manager()
    device_info = get_device_info()

    return {
        "uptime_sec": time.time() - _startup_time,
        "active_connections": streaming_server.get_active_connections(),
        "buffer_depth": ring_buffer.get_depth(),
        "buffer_capacity": ring_buffer.capacity,
        "device": device_info["device"],
        "soundfont_loaded": soundfont_manager.is_loaded("piano")
        and soundfont_manager.is_loaded("pad"),
        "synthesis_active": True,  # TODO: Track synthesis state
        "timestamp": time.time(),
    }


@app.get("/api/metrics")
async def get_metrics() -> dict[str, Any]:
    """Get performance metrics.

    Returns:
        Dictionary with performance metrics (latency, buffer health, memory)
    """
    container = get_container()
    metrics = container.get_metrics()

    return metrics.get_snapshot()


@app.get("/api/presets")
async def get_presets() -> list[dict[str, str]]:
    """Get available musical presets.

    Returns:
        List of preset definitions
    """
    return list_presets()


# WebSocket Endpoint


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for audio streaming.

    Args:
        websocket: WebSocket connection
    """
    container = get_container()
    streaming_server = container.get_streaming_server()

    await streaming_server.handle_connection(websocket)


# Health check endpoint


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "healthy", "service": "auralis"}
