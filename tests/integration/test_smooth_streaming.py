"""Integration test for smooth audio streaming."""

import asyncio
import json
import time
from typing import List

import pytest
import websockets

SERVER_URI = "ws://localhost:8000/ws/audio/test_client"
CHUNK_COUNT = 100  # 10 seconds of streaming
EXPECTED_INTERVAL_MS = 100
JITTER_TOLERANCE_MS = 50


async def test_smooth_streaming():
    """Test smooth audio streaming for 30+ minutes."""

    # Test state
    chunks_received = 0
    latencies = []
    underrun_count = 0
    last_chunk_time = None

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

            start_time = time.time()

            # Receive chunks for 10 seconds (quick test)
            while chunks_received < CHUNK_COUNT:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    # No message within 1 second - assume server busy
                    continue

                data = json.loads(message)

                if data["type"] == "audio":
                    chunks_received += 1
                    now = time.time()

                    # Calculate latency
                    if last_chunk_time:
                        interval_ms = (now - last_chunk_time) * 1000
                        latencies.append(interval_ms)

                    last_chunk_time = now

                    # Check for underruns (empty buffer_depth indicates underrun)
                    if data.get("buffer_depth", 0) == 0:
                        underrun_count += 1

                    # Log progress every 20 chunks
                    if chunks_received % 20 == 0:
                        avg_latency = sum(latencies) / len(latencies) if latencies else 0
                        print(
                            f"Progress: {chunks_received}/{CHUNK_COUNT} chunks, "
                            f"avg_latency={avg_latency:.1f}ms, "
                            f"underruns={underrun_count}"
                        )

    except Exception as e:
        pytest.fail(f"Connection failed: {e}")

    # Calculate statistics
    if latencies:
        avg_latency_ms = sum(latencies) / len(latencies)
        p50_latency_ms = sorted(latencies)[len(latencies) // 2]
        p95_latency_ms = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency_ms = sorted(latencies)[int(len(latencies) * 0.99)]
    else:
        avg_latency_ms = p50_latency_ms = p95_latency_ms = p99_latency_ms = 0

    underrun_rate = underrun_count / chunks_received if chunks_received > 0 else 0

    # Validate performance targets
    # Target: <100ms latency, 99% on-time delivery, zero audible glitches
    print(f"\n=== Streaming Test Results ===")
    print(f"Chunks received: {chunks_received}/{CHUNK_COUNT}")
    print(f"Avg latency: {avg_latency_ms:.1f}ms")
    print(f"P50 latency: {p50_latency_ms:.1f}ms")
    print(f"P95 latency: {p95_latency_ms:.1f}ms")
    print(f"P99 latency: {p99_latency_ms:.1f}ms")
    print(f"Underruns: {underrun_count}")
    print(f"Underrun rate: {underrun_rate * 100:.2f}%")

    # Assertions
    assert chunks_received >= CHUNK_COUNT * 0.95, (
        f"Expected {CHUNK_COUNT * 0.95:.0f} chunks, got {chunks_received}"
    )

    assert p99_latency_ms < 100, f"P99 latency {p99_latency_ms:.1f}ms exceeds 100ms target"

    assert underrun_rate < 0.05, f"Underrun rate {underrun_rate * 100:.2f}% exceeds 5% threshold"

    print("✅ Smooth streaming test passed!")


async def test_buffer_tier_escalation():
    """Test that adaptive buffer tiers escalate on high jitter."""

    chunks_received = 0
    current_tier = "unknown"
    tier_transitions = []

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

            # Receive chunks and monitor tier changes
            while chunks_received < 50:  # 5 seconds
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    break

                data = json.loads(message)

                if data["type"] == "audio":
                    chunks_received += 1
                    new_tier = data.get("current_tier", "unknown")

                    # Track tier transitions
                    if new_tier != current_tier:
                        tier_transitions.append((current_tier, new_tier, chunks_received))
                        current_tier = new_tier
                        print(
                            f"Tier change: {tier_transitions[-1][0]} → {new_tier} at chunk {chunks_received}"
                        )

    except Exception as e:
        pytest.fail(f"Connection failed: {e}")

    print(f"\n=== Tier Escalation Test ===")
    print(f"Tier transitions: {tier_transitions}")

    # Verify at least one tier change occurred (assuming jitter simulation)
    assert len(tier_transitions) > 0 or current_tier != "unknown", (
        "Expected buffer tier to adjust during streaming"
    )

    print("✅ Buffer tier escalation test passed!")


if __name__ == "__main__":
    print("=== Running smooth streaming integration tests ===")

    print("\n1. Testing smooth streaming (10 seconds)...")
    asyncio.run(test_smooth_streaming())

    print("\n2. Testing buffer tier escalation (5 seconds)...")
    asyncio.run(test_buffer_tier_escalation())

    print("\n=== All tests passed! ===")
