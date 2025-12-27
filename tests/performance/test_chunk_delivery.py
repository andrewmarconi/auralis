"""Performance test for chunk delivery timing."""

import asyncio
import time
from statistics import mean, stdev

import pytest
import websockets
import json

SERVER_URI = "ws://localhost:8000/ws/audio/test_client"
CHUNK_COUNT = 100
EXPECTED_INTERVAL_MS = 100
TARGET_P95_MS = 50


async def test_chunk_delivery_timing():
    """Test that 99% of chunks arrive within target timing."""

    # Test state
    chunks_received = 0
    intervals = []
    start_time = None

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

            # Receive chunks
            while chunks_received < CHUNK_COUNT:
                message = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=1.0,  # Allow up to 1 second per chunk
                )

                data = json.loads(message)

                if data["type"] == "audio":
                    chunks_received += 1

                    # Record interval
                    now = time.time()
                    if start_time is None:
                        start_time = now
                    else:
                        interval_ms = (now - start_time) * 1000
                        intervals.append(interval_ms)

                    start_time = now

    except Exception as e:
        pytest.fail(f"Test failed: {e}")
        return

    # Calculate statistics
    if intervals:
        avg_interval_ms = mean(intervals)
        p50_interval_ms = sorted(intervals)[len(intervals) // 2]
        p95_interval_ms = sorted(intervals)[int(len(intervals) * 0.95)]
        p99_interval_ms = sorted(intervals)[int(len(intervals) * 0.99)]
        std_interval_ms = stdev(intervals) if len(intervals) > 1 else 0
    else:
        avg_interval_ms = p50_interval_ms = p95_interval_ms = p99_interval_ms = std_interval_ms = 0

    # Print results
    print("\n=== Chunk Delivery Timing Test ===")
    print(f"Chunks received: {chunks_received}/{CHUNK_COUNT}")
    print(f"Expected interval: {EXPECTED_INTERVAL_MS}ms")
    print(f"\nStatistics:")
    print(f"  Average interval: {avg_interval_ms:.1f}ms")
    print(f"  P50 interval: {p50_interval_ms:.1f}ms")
    print(f"  P95 interval: {p95_interval_ms:.1f}ms")
    print(f"  P99 interval: {p99_interval_ms:.1f}ms")
    print(f"  Std deviation: {std_interval_ms:.1f}ms")
    print(f"\nTarget: <{TARGET_P95_MS}ms for P95")

    # Assertions
    assert chunks_received == CHUNK_COUNT, f"Expected {CHUNK_COUNT} chunks, got {chunks_received}"

    assert p95_interval_ms < TARGET_P95_MS, (
        f"P95 interval {p95_interval_ms:.1f}ms exceeds {TARGET_P95_MS}ms target"
    )

    print("✅ Chunk delivery timing test passed!")


if __name__ == "__main__":
    test_chunk_delivery_timing()
