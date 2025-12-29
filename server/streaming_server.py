"""WebSocket streaming server for real-time audio delivery.

Manages client connections and streams audio chunks from the ring buffer
to connected clients.
"""

import asyncio
import logging
import time
from typing import Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect

from server.audio_chunk import AudioChunk
from server.interfaces.buffer import IRingBuffer

logger = logging.getLogger(__name__)


class ClientConnection:
    """Represents a connected WebSocket client."""

    def __init__(self, client_id: str, websocket: WebSocket):
        """Initialize client connection.

        Args:
            client_id: Unique client identifier
            websocket: WebSocket connection instance
        """
        self.client_id = client_id
        self.websocket = websocket
        self.connected_at = time.time()
        self.last_chunk_seq = 0
        self.chunks_sent = 0

    async def send_chunk(self, chunk: AudioChunk) -> bool:
        """Send audio chunk to client.

        Args:
            chunk: Audio chunk to send

        Returns:
            True if sent successfully, False on error
        """
        try:
            await self.websocket.send_json(chunk.to_json())
            self.last_chunk_seq = chunk.seq
            self.chunks_sent += 1
            return True
        except Exception as e:
            logger.error(
                f"Error sending chunk to client {self.client_id}: {e}",
                extra={"client_id": self.client_id, "chunk_seq": chunk.seq},
            )
            return False

    async def send_control_message(self, message: dict) -> bool:
        """Send control message to client.

        Args:
            message: Control message dictionary

        Returns:
            True if sent successfully, False on error
        """
        try:
            await self.websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(
                f"Error sending control message to client {self.client_id}: {e}"
            )
            return False

    def get_connection_duration(self) -> float:
        """Get connection duration in seconds.

        Returns:
            Duration in seconds
        """
        return time.time() - self.connected_at


class StreamingServer:
    """WebSocket streaming server for audio delivery."""

    def __init__(self, ring_buffer: IRingBuffer, synthesis_engine=None):
        """Initialize streaming server.

        Args:
            ring_buffer: Ring buffer for audio chunks
            synthesis_engine: Optional synthesis engine (will be started/stopped with client connections)
        """
        self.ring_buffer = ring_buffer
        self.clients: Dict[str, ClientConnection] = {}
        self.next_client_id = 0
        self.synthesis_engine = synthesis_engine

        logger.info("Streaming server initialized")

    def _generate_client_id(self) -> str:
        """Generate unique client ID.

        Returns:
            Client ID string
        """
        client_id = f"client_{self.next_client_id}"
        self.next_client_id += 1
        return client_id

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle WebSocket connection lifecycle.

        Args:
            websocket: WebSocket connection instance
        """
        # Accept connection
        await websocket.accept()

        # Generate client ID
        client_id = self._generate_client_id()
        client = ClientConnection(client_id, websocket)
        self.clients[client_id] = client

        logger.info(
            f"Client connected: {client_id} (total clients: {len(self.clients)})"
        )

        # Start synthesis engine if this is the first client
        if len(self.clients) == 1 and self.synthesis_engine:
            if not self.synthesis_engine.is_running():
                await self.synthesis_engine.start_generation_loop()
                logger.info("Synthesis engine started (first client connected)")

        try:
            # Send welcome message
            await client.send_control_message(
                {
                    "type": "welcome",
                    "client_id": client_id,
                    "message": "Connected to Auralis streaming server",
                }
            )

            # Start streaming task and control message handler
            streaming_task = asyncio.create_task(self._stream_to_client(client))
            control_task = asyncio.create_task(self._handle_control_messages(client))

            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [streaming_task, control_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining task
            for task in pending:
                task.cancel()

        except WebSocketDisconnect:
            logger.info(
                f"Client {client_id} disconnected normally",
                extra={"client_id": client_id},
            )
        except Exception as e:
            logger.error(
                f"Error handling client {client_id}: {e}", extra={"client_id": client_id}
            )
        finally:
            # Remove client
            if client_id in self.clients:
                del self.clients[client_id]

            logger.info(
                f"Client {client_id} removed "
                f"(duration: {client.get_connection_duration():.1f}s, "
                f"chunks sent: {client.chunks_sent}, "
                f"remaining clients: {len(self.clients)})"
            )

            # Stop synthesis engine if this was the last client
            if len(self.clients) == 0 and self.synthesis_engine:
                if self.synthesis_engine.is_running():
                    await self.synthesis_engine.stop_generation_loop()
                    logger.info("Synthesis engine stopped (last client disconnected)")

    async def _stream_to_client(self, client: ClientConnection) -> None:
        """Stream audio chunks to client from ring buffer.

        Args:
            client: Client connection
        """
        while True:
            # Read chunk from buffer
            chunk = self.ring_buffer.read()

            if chunk is None:
                # Buffer empty, wait a bit
                await asyncio.sleep(0.01)  # 10ms
                continue

            # Send chunk to client
            success = await client.send_chunk(chunk)
            if not success:
                logger.warning(f"Failed to send chunk to {client.client_id}, closing")
                break

            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.001)  # 1ms between chunks

    async def _handle_control_messages(self, client: ClientConnection) -> None:
        """Handle control messages from client.

        Args:
            client: Client connection
        """
        while True:
            try:
                # Receive message from client
                message = await client.websocket.receive_json()

                # Handle different message types
                msg_type = message.get("type")

                if msg_type == "ping":
                    # Respond to ping with pong
                    await client.send_control_message(
                        {"type": "pong", "timestamp": time.time()}
                    )

                elif msg_type == "control":
                    # Handle parameter changes (key, mode, BPM, intensity)
                    logger.info(
                        f"Control message from {client.client_id}: {message}",
                        extra={"client_id": client.client_id},
                    )

                    # Forward to synthesis engine
                    if self.synthesis_engine:
                        self.synthesis_engine.update_parameters(
                            key=message.get("key"),
                            mode=message.get("mode"),
                            bpm=message.get("bpm"),
                            intensity=message.get("intensity")
                        )

                elif msg_type == "parameter_change":
                    # Legacy message type support
                    logger.info(
                        f"Parameter change from {client.client_id}: {message}",
                        extra={"client_id": client.client_id},
                    )

                    # Forward to synthesis engine
                    if self.synthesis_engine:
                        self.synthesis_engine.update_parameters(
                            key=message.get("key"),
                            mode=message.get("mode"),
                            bpm=message.get("bpm"),
                            intensity=message.get("intensity")
                        )

                else:
                    logger.warning(
                        f"Unknown message type from {client.client_id}: {msg_type}"
                    )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error receiving message from {client.client_id}: {e}")
                break

    def get_active_connections(self) -> int:
        """Get number of active client connections.

        Returns:
            Number of connected clients
        """
        return len(self.clients)

    def get_client_stats(self) -> list[dict]:
        """Get statistics for all connected clients.

        Returns:
            List of client stat dictionaries
        """
        return [
            {
                "client_id": client.client_id,
                "connected_at": client.connected_at,
                "duration_sec": client.get_connection_duration(),
                "last_chunk_seq": client.last_chunk_seq,
                "chunks_sent": client.chunks_sent,
            }
            for client in self.clients.values()
        ]
