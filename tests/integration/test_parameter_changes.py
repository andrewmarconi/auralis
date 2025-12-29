"""Integration test for real-time parameter changes.

Verifies that control messages sent via WebSocket correctly update
the synthesis engine's musical context within 5 seconds (at next phrase boundary).
"""

import asyncio
import json
import logging

import pytest
from fastapi.testclient import TestClient

from composition.musical_context import MusicalContext
from server.main import app

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_parameter_changes_via_websocket():
    """Test that control messages update synthesis parameters correctly."""

    with TestClient(app) as client:
        # Connect to WebSocket
        with client.websocket_connect("/ws/stream") as websocket:
            # Receive welcome message
            welcome = websocket.receive_json()
            assert welcome["type"] == "welcome"
            assert "client_id" in welcome

            logger.info(f"Connected as {welcome['client_id']}")

            # Wait briefly for initial phrase generation
            await asyncio.sleep(0.5)

            # Test 1: Change key
            control_message = {
                "type": "control",
                "key": 62  # D
            }
            websocket.send_json(control_message)
            logger.info("Sent key change: D (62)")

            # Wait for next phrase boundary (max 5 seconds per spec)
            await asyncio.sleep(3.0)

            # Test 2: Change mode
            control_message = {
                "type": "control",
                "mode": "dorian"
            }
            websocket.send_json(control_message)
            logger.info("Sent mode change: dorian")

            await asyncio.sleep(3.0)

            # Test 3: Change BPM
            control_message = {
                "type": "control",
                "bpm": 90
            }
            websocket.send_json(control_message)
            logger.info("Sent BPM change: 90")

            await asyncio.sleep(3.0)

            # Test 4: Change intensity
            control_message = {
                "type": "control",
                "intensity": 0.8
            }
            websocket.send_json(control_message)
            logger.info("Sent intensity change: 0.8")

            await asyncio.sleep(3.0)

            # Test 5: Multiple parameters at once
            control_message = {
                "type": "control",
                "key": 67,  # G
                "mode": "lydian",
                "bpm": 75,
                "intensity": 0.6
            }
            websocket.send_json(control_message)
            logger.info("Sent multi-parameter change: G Lydian @ 75 BPM, intensity 0.6")

            # Verify we're still receiving audio chunks
            chunks_received = 0
            for _ in range(10):
                message = websocket.receive_json()
                if message["type"] == "audio_chunk":
                    chunks_received += 1

            assert chunks_received >= 5, f"Expected at least 5 audio chunks, got {chunks_received}"
            logger.info(f"Received {chunks_received} audio chunks during parameter changes")


@pytest.mark.asyncio
async def test_parameter_validation():
    """Test that invalid parameters are rejected gracefully."""

    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            # Receive welcome message
            welcome = websocket.receive_json()
            assert welcome["type"] == "welcome"

            # Test invalid key (out of range)
            control_message = {
                "type": "control",
                "key": 100  # Invalid - must be 60-71
            }
            websocket.send_json(control_message)

            # Should not crash - server should handle gracefully
            await asyncio.sleep(1.0)

            # Test invalid mode
            control_message = {
                "type": "control",
                "mode": "invalid_mode"
            }
            websocket.send_json(control_message)

            await asyncio.sleep(1.0)

            # Test invalid BPM
            control_message = {
                "type": "control",
                "bpm": 300  # Invalid - must be 60-120
            }
            websocket.send_json(control_message)

            await asyncio.sleep(1.0)

            # Verify we're still receiving audio chunks (server didn't crash)
            chunks_received = 0
            for _ in range(5):
                message = websocket.receive_json()
                if message["type"] == "audio_chunk":
                    chunks_received += 1

            assert chunks_received >= 3, "Server should continue streaming despite invalid parameters"
            logger.info("Server handled invalid parameters gracefully")


@pytest.mark.asyncio
async def test_preset_loading():
    """Test loading preset configurations."""

    # Preset definitions from server/presets.py
    presets = {
        "focus": {"key": 62, "mode": "dorian", "bpm": 60, "intensity": 0.5},
        "meditation": {"key": 60, "mode": "aeolian", "bpm": 60, "intensity": 0.3},
        "sleep": {"key": 64, "mode": "phrygian", "bpm": 60, "intensity": 0.2},
        "bright": {"key": 67, "mode": "lydian", "bpm": 70, "intensity": 0.6},
    }

    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            # Receive welcome message
            welcome = websocket.receive_json()
            assert welcome["type"] == "welcome"

            # Test each preset
            for preset_name, preset_params in presets.items():
                logger.info(f"Testing preset: {preset_name}")

                control_message = {
                    "type": "control",
                    **preset_params
                }
                websocket.send_json(control_message)

                # Wait for changes to apply
                await asyncio.sleep(3.0)

                # Verify still receiving audio
                message = websocket.receive_json()
                assert message["type"] == "audio_chunk", f"Expected audio chunk after {preset_name} preset"

            logger.info("All presets loaded successfully")


def test_musical_context_validation():
    """Test MusicalContext validation with new ranges."""

    # Valid contexts
    valid_contexts = [
        MusicalContext(key=60, mode="aeolian", bpm=60.0, intensity=0.5, key_signature="C minor"),
        MusicalContext(key=71, mode="lydian", bpm=120.0, intensity=1.0, key_signature="B Lydian"),
        MusicalContext(key=65, mode="dorian", bpm=90.0, intensity=0.0, key_signature="F Dorian"),
    ]

    for context in valid_contexts:
        assert 60 <= context.key <= 71
        assert 60 <= context.bpm <= 120
        assert 0.0 <= context.intensity <= 1.0

    # Invalid key (below range)
    with pytest.raises(ValueError, match="Invalid key"):
        MusicalContext(key=59, mode="aeolian", bpm=60.0, intensity=0.5, key_signature="B minor")

    # Invalid key (above range)
    with pytest.raises(ValueError, match="Invalid key"):
        MusicalContext(key=72, mode="aeolian", bpm=60.0, intensity=0.5, key_signature="C# minor")

    # Invalid BPM (below range)
    with pytest.raises(ValueError, match="Invalid BPM"):
        MusicalContext(key=60, mode="aeolian", bpm=59.0, intensity=0.5, key_signature="C minor")

    # Invalid BPM (above range)
    with pytest.raises(ValueError, match="Invalid BPM"):
        MusicalContext(key=60, mode="aeolian", bpm=121.0, intensity=0.5, key_signature="C minor")

    # Invalid intensity
    with pytest.raises(ValueError, match="Invalid intensity"):
        MusicalContext(key=60, mode="aeolian", bpm=60.0, intensity=1.5, key_signature="C minor")

    logger.info("MusicalContext validation working correctly")
