"""
WebSocket Streaming Server

Handles real-time audio streaming over WebSocket connections with
base64-encoded PCM chunks.
"""

import asyncio
import base64
import json
import time
from typing import Optional, Union

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from numpy.typing import NDArray

from server.ring_buffer import RingBuffer, AdaptiveRingBuffer


# Forward reference for ApplicationState type hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from server.main import ApplicationState
    from server.metrics import PrometheusMetrics


class AudioBufferPool:
    """
    Object pool for audio conversion buffers (T091).

    Reduces GC pressure by reusing pre-allocated numpy arrays for
    float32->int16 conversion and stereo interleaving operations.
    """

    def __init__(self, pool_size: int = 10, chunk_samples: int = 4410):
        """
        Initialize buffer pool.

        Args:
            pool_size: Number of pre-allocated buffer pairs
            chunk_samples: Samples per channel (100ms @ 44.1kHz = 4410)
        """
        self.chunk_samples = chunk_samples
        # Pre-allocate buffer pairs: (int16_buffer, interleaved_buffer)
        self.available_buffers: list[tuple[NDArray[np.int16], NDArray[np.int16]]] = []

        for _ in range(pool_size):
            int16_buf = np.empty((2, chunk_samples), dtype=np.int16)
            interleaved_buf = np.empty(chunk_samples * 2, dtype=np.int16)
            self.available_buffers.append((int16_buf, interleaved_buf))

    def acquire(self) -> tuple[NDArray[np.int16], NDArray[np.int16]] | None:
        """Get a buffer pair from pool, or None if empty."""
        if self.available_buffers:
            return self.available_buffers.pop()
        return None

    def release(self, buffers: tuple[NDArray[np.int16], NDArray[np.int16]]) -> None:
        """Return buffer pair to pool."""
        self.available_buffers.append(buffers)


# Global buffer pool (shared across all streaming instances)
_buffer_pool = AudioBufferPool(pool_size=10)


class AudioChunk:
    """
    100ms segment of audio data for WebSocket transmission.

    Attributes:
        data: Base64-encoded 16-bit PCM audio
        timestamp: Unix timestamp for synchronization
        sequence: Sequential packet number
        sample_rate: Audio sampling rate (44100)
        format: Audio format identifier ("pcm16")
    """

    def __init__(
        self,
        audio_data: NDArray[np.float32],
        sequence: int,
        sample_rate: int = 44100,
    ) -> None:
        """
        Create audio chunk from float32 audio data with object pooling (T091).

        Args:
            audio_data: Stereo audio array, shape (2, num_samples), float32 [-1, 1]
            sequence: Sequential packet number
            sample_rate: Audio sampling rate
        """
        # Try to acquire buffers from pool (T091)
        pooled_buffers = _buffer_pool.acquire()

        if pooled_buffers is not None:
            # Use pooled buffers
            int16_buf, interleaved_buf = pooled_buffers

            # Convert float32 [-1, 1] to int16 [-32768, 32767] in-place
            np.multiply(audio_data, 32767, out=int16_buf, casting='unsafe')

            # Interleave stereo channels: [L, R, L, R, ...] in-place
            interleaved_buf[0::2] = int16_buf[0, :]  # Left channel
            interleaved_buf[1::2] = int16_buf[1, :]  # Right channel

            # Convert to bytes and base64 encode
            pcm_bytes = interleaved_buf.tobytes()
            self.data = base64.b64encode(pcm_bytes).decode("utf-8")

            # Return buffers to pool immediately (encoding is done)
            _buffer_pool.release(pooled_buffers)
        else:
            # Pool exhausted - fall back to dynamic allocation
            audio_int16 = (audio_data * 32767).astype(np.int16)
            interleaved = np.empty(audio_int16.size, dtype=np.int16)
            interleaved[0::2] = audio_int16[0, :]
            interleaved[1::2] = audio_int16[1, :]
            pcm_bytes = interleaved.tobytes()
            self.data = base64.b64encode(pcm_bytes).decode("utf-8")

        self.timestamp = int(time.time() * 1000)  # Milliseconds
        self.sequence = sequence
        self.sample_rate = sample_rate
        self.format = "pcm16"

    def to_json(self) -> str:
        """Serialize chunk to JSON for WebSocket transmission."""
        return json.dumps(
            {
                "type": "audio",
                "data": self.data,
                "timestamp": self.timestamp,
                "sequence": self.sequence,
                "sample_rate": self.sample_rate,
                "format": self.format,
            }
        )


class StreamingServer:
    """
    WebSocket streaming server for real-time audio delivery.

    Manages client connections and streams audio chunks from ring buffer.
    """

    def __init__(
        self,
        ring_buffer: Union[RingBuffer, AdaptiveRingBuffer],
        app_state: Optional["ApplicationState"] = None,
        metrics: Optional["PrometheusMetrics"] = None
    ) -> None:
        """
        Initialize streaming server (T032, T034).

        Args:
            ring_buffer: Shared ring buffer for audio data (RingBuffer or AdaptiveRingBuffer)
            app_state: Application state for parameter updates (optional)
            metrics: Prometheus metrics collector for jitter tracking (optional)
        """
        self.ring_buffer = ring_buffer
        self.app_state = app_state
        self.metrics = metrics
        self.active_connections: set[WebSocket] = set()
        self.sequence_counter = 0
        self.is_adaptive = isinstance(ring_buffer, AdaptiveRingBuffer)

        # Client-specific jitter tracking (T034)
        # Maps client_id -> last chunk send time (ms)
        self._client_last_chunk_time: dict[str, float] = {}

    async def handle_client(self, websocket: WebSocket) -> None:
        """
        Handle individual WebSocket client connection.

        Args:
            websocket: FastAPI WebSocket connection
        """
        await websocket.accept()
        self.active_connections.add(websocket)

        logger.info(f"Client connected. Total active: {len(self.active_connections)}")

        # Client ID for tracking
        client_id = f"client_{id(websocket)}"

        try:
            # Send initial connection confirmation
            await websocket.send_json(
                {
                    "type": "connected",
                    "message": "Audio streaming ready",
                    "sample_rate": self.ring_buffer.sample_rate,
                    "chunk_size_ms": 100,
                }
            )

            # Stream audio chunks
            await self._stream_audio(websocket)

        except WebSocketDisconnect:
            logger.info("Client disconnected gracefully")
        except Exception as e:
            logger.error(f"Error in client handler: {e}")
        finally:
            self.active_connections.discard(websocket)

            # Clean up jitter tracking state (T034)
            if client_id in self._client_last_chunk_time:
                del self._client_last_chunk_time[client_id]

            logger.info(f"Client removed. Total active: {len(self.active_connections)}")

    async def _stream_audio(self, websocket: WebSocket) -> None:
        """
        Stream audio chunks to client in real-time (T032, T034).

        Args:
            websocket: Client WebSocket connection
        """
        # Generate client ID for metrics tracking
        client_id = f"client_{id(websocket)}"

        # Chunk timing control to prevent bursts
        chunk_interval_sec = 0.1  # 100ms per chunk
        next_send_time = time.time()

        while True:
            # Read 100ms chunk from ring buffer
            if self.is_adaptive:
                # Use AdaptiveRingBuffer's read_chunk method (non-blocking)
                audio_data = await asyncio.to_thread(
                    self.ring_buffer.read_chunk  # type: ignore
                )
            else:
                # Use standard RingBuffer's read_blocking method
                audio_data = await asyncio.to_thread(
                    self.ring_buffer.read_blocking,
                    num_samples=self.ring_buffer.chunk_size,
                    timeout=0.5,  # 500ms timeout
                )

            if audio_data is None:
                # Buffer underflow - send silence
                logger.warning("Ring buffer underflow - sending silence")

                # Get chunk size (adaptive buffer uses samples_per_chunk)
                if self.is_adaptive:
                    chunk_size = self.ring_buffer.samples_per_chunk  # type: ignore
                else:
                    chunk_size = self.ring_buffer.chunk_size

                audio_data = np.zeros((2, chunk_size), dtype=np.float32)

                # Notify client of underflow with buffer health (T033)
                warning_data = {
                    "type": "warning",
                    "message": "Buffer underflow - temporary silence",
                }

                if self.is_adaptive:
                    # Add buffer health metrics for adaptive buffer
                    health = self.ring_buffer.get_buffer_health()  # type: ignore
                    warning_data["buffer_health"] = health

                await websocket.send_json(warning_data)

            # Create and send audio chunk
            chunk = AudioChunk(audio_data, self.sequence_counter)
            self.sequence_counter += 1

            # Track chunk delivery timing for jitter metrics (T034)
            now_ms = time.time() * 1000  # Current time in milliseconds

            if self.metrics:
                # Calculate jitter if we have a previous send time
                if client_id in self._client_last_chunk_time:
                    expected_interval_ms = 100.0  # Expected 100ms chunk interval
                    actual_interval_ms = now_ms - self._client_last_chunk_time[client_id]
                    jitter_ms = abs(actual_interval_ms - expected_interval_ms)

                    # Record jitter to Prometheus metrics
                    self.metrics.record_chunk_jitter(client_id, jitter_ms)

                # Update last chunk time
                self._client_last_chunk_time[client_id] = now_ms

                # Report buffer depth per client (T035)
                if self.is_adaptive:
                    # Use AdaptiveRingBuffer's buffer depth
                    health = self.ring_buffer.get_buffer_health()  # type: ignore
                    depth_ms = health.get("depth_ms", 0)
                else:
                    # Use standard RingBuffer's buffer depth
                    depth_ms = self.ring_buffer.get_buffer_depth_ms()

                self.metrics.set_buffer_depth(client_id, depth_ms)

            await websocket.send_text(chunk.to_json())

            # Maintain steady chunk timing to prevent bursts (reduce jitter)
            now = time.time()
            next_send_time += chunk_interval_sec

            # Calculate sleep time to maintain steady 100ms interval
            sleep_time = next_send_time - now
            if sleep_time > 0:
                # Sleep until next chunk should be sent
                await asyncio.sleep(sleep_time)
            else:
                # We're running behind - reset timing and log warning
                if sleep_time < -0.05:  # More than 50ms behind
                    logger.warning(f"Chunk delivery running {-sleep_time*1000:.1f}ms behind schedule")
                next_send_time = now

            # Check for client control messages (non-blocking)
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.001,  # 1ms timeout
                )
                await self._handle_control_message(websocket, message)
            except asyncio.TimeoutError:
                pass  # No control message - continue streaming

    async def _handle_control_message(self, websocket: WebSocket, message: str) -> None:
        """
        Handle control messages from client.

        Args:
            websocket: Client WebSocket connection
            message: JSON control message
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "ping":
                # Respond to ping
                await websocket.send_json({"type": "pong"})

            elif msg_type == "buffer_status":
                # Report buffer depth and health metrics (T033)
                response = {"type": "buffer_status"}

                if self.is_adaptive:
                    # Use AdaptiveRingBuffer's comprehensive health metrics
                    health = self.ring_buffer.get_buffer_health()  # type: ignore
                    response.update(health)
                else:
                    # Basic buffer depth for standard RingBuffer
                    depth_ms = self.ring_buffer.get_buffer_depth_ms()
                    response["depth_ms"] = depth_ms

                await websocket.send_json(response)

            elif msg_type == "control":
                # Handle parameter updates from client
                try:
                    if self.app_state is not None:
                        # Update parameters directly from WebSocket control message
                        updated_fields = []

                        # Update boolean voice enable/disable controls
                        if "enable_pads" in data:
                            self.app_state.parameters.enable_pads = bool(data["enable_pads"])
                            updated_fields.append("enable_pads")
                        if "enable_melody" in data:
                            self.app_state.parameters.enable_melody = bool(data["enable_melody"])
                            updated_fields.append("enable_melody")
                        if "enable_kicks" in data:
                            self.app_state.parameters.enable_kicks = bool(data["enable_kicks"])
                            updated_fields.append("enable_kicks")
                        if "enable_swells" in data:
                            self.app_state.parameters.enable_swells = bool(data["enable_swells"])
                            updated_fields.append("enable_swells")

                        # Update numeric parameters
                        if "key" in data:
                            self.app_state.parameters.key = str(data["key"])
                            updated_fields.append("key")
                        if "bpm" in data:
                            self.app_state.parameters.bpm = int(data["bpm"])
                            updated_fields.append("bpm")
                        if "intensity" in data:
                            self.app_state.parameters.intensity = float(data["intensity"])
                            updated_fields.append("intensity")
                        if "melody_complexity" in data:
                            self.app_state.parameters.melody_complexity = float(data["melody_complexity"])
                            updated_fields.append("melody_complexity")
                        if "chord_progression_variety" in data:
                            self.app_state.parameters.chord_progression_variety = float(data["chord_progression_variety"])
                            updated_fields.append("chord_progression_variety")
                        if "harmonic_density" in data:
                            self.app_state.parameters.harmonic_density = float(data["harmonic_density"])
                            updated_fields.append("harmonic_density")

                        if updated_fields:
                            logger.info(f"Parameters updated via WebSocket: {updated_fields}")

                        await websocket.send_json({
                            "type": "control_ack",
                            "message": f"Parameters updated: {', '.join(updated_fields)}" if updated_fields else "No parameters updated",
                            "updated_fields": updated_fields
                        })
                    else:
                        # Fallback if app_state not available
                        await websocket.send_json(
                            {"type": "control_ack", "message": "Parameters should be updated via REST API"}
                        )
                except Exception as e:
                    logger.error(f"Error updating parameters via WebSocket: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Failed to update parameters: {str(e)}"
                    })

            else:
                logger.warning(f"Unknown control message type: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in control message: {message}")

    def get_active_client_count(self) -> int:
        """Get number of active WebSocket connections."""
        return len(self.active_connections)

    async def broadcast_status(self, status_data: dict) -> None:
        """
        Broadcast status update to all connected clients.

        Args:
            status_data: Status information dictionary
        """
        message = json.dumps({"type": "status", **status_data})

        # Send to all active connections
        disconnected = set()
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send status to client: {e}")
                disconnected.add(websocket)

        # Clean up disconnected clients
        self.active_connections -= disconnected
