"""
WebSocket Streaming Server

Handles real-time audio streaming over WebSocket connections with
base64-encoded PCM chunks.
"""

import asyncio
import base64
import json
import time
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from numpy.typing import NDArray

from server.ring_buffer import RingBuffer


# Forward reference for ApplicationState type hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from server.main import ApplicationState


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
        Create audio chunk from float32 audio data.

        Args:
            audio_data: Stereo audio array, shape (2, num_samples), float32 [-1, 1]
            sequence: Sequential packet number
            sample_rate: Audio sampling rate
        """
        # Convert float32 [-1, 1] to int16 [-32768, 32767]
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Interleave stereo channels: [L, R, L, R, ...]
        interleaved = np.empty(audio_int16.size, dtype=np.int16)
        interleaved[0::2] = audio_int16[0, :]  # Left channel
        interleaved[1::2] = audio_int16[1, :]  # Right channel

        # Convert to bytes and base64 encode
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

    def __init__(self, ring_buffer: RingBuffer, app_state: Optional["ApplicationState"] = None) -> None:
        """
        Initialize streaming server.

        Args:
            ring_buffer: Shared ring buffer for audio data
            app_state: Application state for parameter updates (optional)
        """
        self.ring_buffer = ring_buffer
        self.app_state = app_state
        self.active_connections: set[WebSocket] = set()
        self.sequence_counter = 0

    async def handle_client(self, websocket: WebSocket) -> None:
        """
        Handle individual WebSocket client connection.

        Args:
            websocket: FastAPI WebSocket connection
        """
        await websocket.accept()
        self.active_connections.add(websocket)

        logger.info(f"Client connected. Total active: {len(self.active_connections)}")

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
            logger.info(f"Client removed. Total active: {len(self.active_connections)}")

    async def _stream_audio(self, websocket: WebSocket) -> None:
        """
        Stream audio chunks to client in real-time.

        Args:
            websocket: Client WebSocket connection
        """
        while True:
            # Read 100ms chunk from ring buffer (blocking with timeout)
            audio_data = await asyncio.to_thread(
                self.ring_buffer.read_blocking,
                num_samples=self.ring_buffer.chunk_size,
                timeout=0.5,  # 500ms timeout
            )

            if audio_data is None:
                # Buffer underflow - send silence
                logger.warning("Ring buffer underflow - sending silence")
                audio_data = np.zeros((2, self.ring_buffer.chunk_size), dtype=np.float32)

                # Notify client of underflow
                await websocket.send_json(
                    {
                        "type": "warning",
                        "message": "Buffer underflow - temporary silence",
                    }
                )

            # Create and send audio chunk
            chunk = AudioChunk(audio_data, self.sequence_counter)
            self.sequence_counter += 1

            await websocket.send_text(chunk.to_json())

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
                # Report buffer depth
                depth_ms = self.ring_buffer.get_buffer_depth_ms()
                await websocket.send_json(
                    {
                        "type": "buffer_status",
                        "depth_ms": depth_ms,
                    }
                )

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
