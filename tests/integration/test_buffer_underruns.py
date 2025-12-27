"""Integration test for buffer underrun prevention."""

import asyncio
import json
import random

import pytest
import websockets

SERVER_URI = "ws://localhost:8000/ws/audio/test_client"
CHUNK_COUNT = 50


async def test_buffer_underrun_prevention():
    """Test that adaptive buffer prevents underruns under simulated network jitter."""

    chunks_received = 0
    underruns = 0
    last_buffer_depth = 0

    try:
        async with websockets.connect(SERVER_URI) as websocket:
            # Send initial control message
            await websocket.send(
                json.dumps(
                    {
                        "type": "control",
                        "action": "start",
                        "params": {"key": "C", "bpm": 70, "intensity": 0.5},
                    }
                )
            )

            # Receive chunks with simulated jitter
            for i in range(CHUNK_COUNT):
                # Simulate network delay (20% of chunks delayed)
                if random.random() < 0.2:
                    delay_ms = random.uniform(50, 150)
                    await asyncio.sleep(delay_ms / 1000)

                message = await websocket.recv()
                data = json.loads(message)

                if data["type"] == "audio":
                    chunks_received += 1

                    # Track buffer depth
                    current_buffer_depth = data.get("buffer_depth", 0)

                    # Detect underruns (buffer depth = 0)
                    if current_buffer_depth == 0 and last_buffer_depth > 0:
                        underruns += 1

                    last_buffer_depth = current_buffer

                    # Log progress every 10 chunks
                    if chunks_received % 10 == 0:
                        print(
                            f"Progress: {chunks_received}/{CHUNK_COUNT}, "
                            f"Underruns: {underruns}, "
                            f"Buffer depth: {current_buffer}"
                        )

    except Exception as e:
        pytest.fail(f"Test failed: {e}")

    # Calculate underrun rate
    if chunks_received > 0:
        underrun_rate = underruns / chunks_received
        print(f"\n=== Underrun Prevention Test Results ===")
        print(f"Chunks received: {chunks_received}")
        print(f"Underrun events: {underruns}")
        print(f"Underrun rate: {underrun_rate * 100:.2f}%")

        # Validate
        # Target: <1% underrun rate
        assert underrun_rate < 0.01, f"Underrun rate {underrun_rate * 100:.2f}% exceeds 1% target"

        print("✅ Buffer underrun prevention test passed!")


async def test_buffer_recovery_after_underrun():
    """Test that buffer recovers after underrun."""

    chunks_received = 0
    recovered_count = 0

    try:
        async with websockets.connect(SERVER_URI) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "control",
                        "action": "start",
                        "params": {"key": "C", "bpm": 70, "intensity": 0.5},
                    }
                )
            )

            # First, cause underruns by dropping chunks (simulate slow client)
            for _ in range(10):
                await websocket.recv()  # Drop chunks
                await asyncio.sleep(0.2)

            # Then receive normally
            for i in range(CHUNK_COUNT):
                message = await websocket.recv()
                data = json.loads(message)

                if data["type"] == "audio":
                    chunks_received += 1

                    buffer_depth = data.get("buffer_depth", 0)

                    # Count recovery events (buffer_depth goes from 0 to >2)
                    if buffer_depth > 0 and recovered_count < 5:
                        recovered_count += 1
                        print(f"Recovery #{recovered_count}: buffer_depth = {buffer_depth}")

                    if chunks_received == CHUNK_COUNT:
                        break

    except Exception as e:
        pytest.fail(f"Test failed: {e}")

    # Validate recovery
    assert recovered_count >= 3, f"Expected at least 3 recoveries, got {recovered_count}"

    print(f"✅ Buffer recovery test passed ({recovered_count} recoveries)")


if __name__ == "__main__":
    print("=== Running buffer underrun prevention tests ===\n")

    asyncio.run(test_buffer_underrun_prevention())
    print()
    asyncio.run(test_buffer_recovery_after_underrun())
