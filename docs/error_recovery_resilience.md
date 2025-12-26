# Error Recovery & Resilience Strategy

## Overview

This document defines comprehensive error handling, recovery mechanisms, and graceful degradation strategies for Auralis.

---

## 1. Error Classification

### 1.1 Error Severity Levels

| Level | Description | Action | Examples |
|-------|-------------|--------|----------|
| **DEBUG** | Informational | Log only | Parameter updates, state changes |
| **INFO** | Normal operation | Log only | Client connected, phrase generated |
| **WARNING** | Recoverable issue | Log + auto-recover | Buffer low, temporary synthesis delay |
| **ERROR** | Component failure | Log + fallback | Synthesis failed, invalid control message |
| **CRITICAL** | System failure | Log + alert + shutdown | GPU crashed, out of memory |

### 1.2 Error Categories

**1. Network Errors**
- WebSocket disconnection
- Message timeout
- Invalid message format

**2. Synthesis Errors**
- GPU out of memory
- Torchsynth crash
- Invalid audio output (NaN, Inf)

**3. Generation Errors**
- Invalid chord progression
- Melody generation timeout
- Parameter validation failure

**4. Resource Errors**
- CPU/GPU overload
- Memory exhaustion
- Disk full (logging)

**5. Client Errors**
- Too many connected clients
- Malformed control messages
- Client buffer underrun

---

## 2. Synthesis Error Handling

### 2.1 Resilient Synthesis Worker

```python
# server/synthesis_worker.py
import asyncio
import numpy as np
from loguru import logger
from typing import Optional


class ResilientSynthesisWorker:
    """
    Synthesis worker with automatic error recovery.

    Fallback chain:
    1. Render phrase normally
    2. On error: retry with simpler params
    3. On error: render silence
    """

    def __init__(self, synthesis_engine, max_retries: int = 3):
        self.engine = synthesis_engine
        self.max_retries = max_retries
        self.consecutive_failures = 0
        self.total_failures = 0

    async def render_with_retry(
        self, phrase: dict, duration_sec: float
    ) -> np.ndarray:
        """
        Render phrase with automatic retry and fallback.

        Returns:
            Stereo audio array (2, num_samples), or silence on failure
        """
        for attempt in range(self.max_retries):
            try:
                # Run synthesis in thread pool (CPU-bound)
                audio = await asyncio.to_thread(
                    self.engine.render_phrase,
                    chords=phrase["chords"],
                    melody=phrase["melody"],
                    percussion=phrase["percussion"],
                    duration_sec=duration_sec,
                )

                # Validate output
                if self._validate_audio(audio, duration_sec):
                    self.consecutive_failures = 0  # Reset counter
                    return audio
                else:
                    raise ValueError("Invalid audio output (NaN or incorrect shape)")

            except Exception as e:
                self.consecutive_failures += 1
                self.total_failures += 1

                logger.error(
                    f"Synthesis attempt {attempt + 1}/{self.max_retries} failed: {e}",
                    exc_info=(attempt == self.max_retries - 1),  # Full trace on last attempt
                )

                if attempt < self.max_retries - 1:
                    # Try again with simplified phrase
                    phrase = self._simplify_phrase(phrase)
                    await asyncio.sleep(0.1)  # Brief delay

        # All retries exhausted - return silence
        logger.critical(
            f"Synthesis failed after {self.max_retries} attempts. "
            f"Rendering silence. (Total failures: {self.total_failures})"
        )

        return self._render_silence(duration_sec)

    def _validate_audio(self, audio: np.ndarray, duration_sec: float) -> bool:
        """
        Validate audio output.

        Checks:
        - Correct shape (2, num_samples)
        - No NaN or Inf values
        - Reasonable amplitude range
        """
        sample_rate = 44100
        expected_samples = int(duration_sec * sample_rate)

        # Check shape
        if audio.shape != (2, expected_samples):
            logger.warning(
                f"Incorrect audio shape: {audio.shape}, expected (2, {expected_samples})"
            )
            return False

        # Check for NaN/Inf
        if not np.isfinite(audio).all():
            logger.warning("Audio contains NaN or Inf values")
            return False

        # Check amplitude range (should be roughly [-1, 1])
        if np.abs(audio).max() > 2.0:
            logger.warning(f"Audio amplitude too high: {np.abs(audio).max()}")
            return False

        return True

    def _simplify_phrase(self, phrase: dict) -> dict:
        """
        Create a simplified version of phrase (fewer notes/events).

        Fallback strategy: reduce complexity to avoid synthesis bugs.
        """
        simplified = phrase.copy()

        # Keep only first half of melody notes
        if "melody" in simplified and len(simplified["melody"]) > 0:
            simplified["melody"] = simplified["melody"][: len(simplified["melody"]) // 2]

        # Remove all percussion
        simplified["percussion"] = []

        logger.info("Simplified phrase for retry")
        return simplified

    def _render_silence(self, duration_sec: float) -> np.ndarray:
        """Render silence as last-resort fallback."""
        sample_rate = 44100
        num_samples = int(duration_sec * sample_rate)
        return np.zeros((2, num_samples), dtype=np.float32)

    def get_failure_rate(self) -> float:
        """Calculate synthesis failure rate (for monitoring)."""
        if self.total_failures == 0:
            return 0.0
        # This is a simplified metric; in production, track success count too
        return self.consecutive_failures / max(self.total_failures, 1)
```

### 2.2 GPU Out of Memory Recovery

```python
# auralis/synthesis/gpu_recovery.py
import torch
from loguru import logger


def handle_gpu_oom(device: str):
    """
    Handle GPU out-of-memory errors.

    Recovery steps:
    1. Clear GPU cache
    2. Reduce batch size
    3. Fall back to CPU
    """
    logger.warning(f"GPU OOM on device {device}")

    if device.startswith("cuda"):
        logger.info("Clearing CUDA cache")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    elif device == "mps":
        logger.info("Clearing MPS cache")
        # MPS doesn't have explicit cache clear, but we can trigger GC
        import gc
        gc.collect()

    logger.info("Consider reducing buffer size or falling back to CPU")


def create_synthesis_engine_with_oom_handling(device: str, sample_rate: int):
    """
    Create synthesis engine with OOM recovery.

    Tries GPU first, falls back to CPU on OOM.
    """
    from auralis.synthesis.torchsynth_engine import TorchsynthAmbientEngine

    try:
        engine = TorchsynthAmbientEngine(sample_rate=sample_rate, device=device)
        logger.info(f"Created synthesis engine on {device}")
        return engine

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"GPU OOM during engine initialization: {e}")
            handle_gpu_oom(device)

            # Fall back to CPU
            logger.warning("Falling back to CPU synthesis")
            engine = TorchsynthAmbientEngine(sample_rate=sample_rate, device="cpu")
            return engine
        else:
            raise
```

---

## 3. WebSocket Error Handling

### 3.1 Connection Error Recovery

```python
# server/websocket.py
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger


async def websocket_with_error_handling(websocket: WebSocket):
    """
    WebSocket endpoint with comprehensive error handling.

    Handles:
    - Disconnections
    - Message send failures
    - Client-side errors
    """
    client_id = id(websocket)

    try:
        await websocket.accept()
        logger.info(f"Client {client_id} connected")

        # Heartbeat task
        heartbeat_task = asyncio.create_task(send_heartbeat(websocket, client_id))

        # Streaming task
        streaming_task = asyncio.create_task(stream_audio(websocket, client_id))

        # Control message handler
        control_task = asyncio.create_task(handle_control_messages(websocket, client_id))

        # Wait for any task to complete
        done, pending = await asyncio.wait(
            [heartbeat_task, streaming_task, control_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")

    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}", exc_info=True)

        try:
            await websocket.close(code=1011, reason=f"Server error: {str(e)[:100]}")
        except:
            pass  # Already disconnected

    finally:
        logger.info(f"Cleaning up client {client_id}")
        # Cleanup resources (remove from client manager, etc.)


async def send_heartbeat(websocket: WebSocket, client_id: int):
    """Send periodic heartbeat to keep connection alive."""
    while True:
        try:
            await asyncio.sleep(30)  # Every 30 seconds

            await websocket.send_json({"type": "ping"})
            logger.debug(f"Sent heartbeat to client {client_id}")

        except Exception as e:
            logger.warning(f"Heartbeat failed for client {client_id}: {e}")
            break
```

### 3.2 Message Send Retry

```python
async def send_with_retry(
    websocket: WebSocket,
    message: dict,
    max_retries: int = 3,
    timeout: float = 1.0,
) -> bool:
    """
    Send message with retry logic.

    Returns:
        True if sent successfully, False otherwise
    """
    for attempt in range(max_retries):
        try:
            await asyncio.wait_for(
                websocket.send_json(message),
                timeout=timeout,
            )
            return True

        except asyncio.TimeoutError:
            logger.warning(f"Message send timeout (attempt {attempt + 1})")

        except Exception as e:
            logger.error(f"Message send error: {e}")
            break  # Don't retry on non-timeout errors

        if attempt < max_retries - 1:
            await asyncio.sleep(0.1)  # Brief delay before retry

    return False  # Failed after retries
```

---

## 4. Graceful Degradation

### 4.1 Quality Levels

When system is overloaded, degrade quality instead of failing:

| Load Level | Quality Adjustment | Target RTF |
|------------|-------------------|------------|
| **Normal** (0-60%) | Full quality | >20× |
| **High** (60-80%) | Reduce melody density | >10× |
| **Critical** (80-95%) | Simple chords only, no melody | >5× |
| **Overload** (>95%) | Pre-rendered loops or silence | N/A |

```python
# server/adaptive_quality.py
import psutil
from auralis.composition.engine import CompositionEngine


class AdaptiveQualityManager:
    """
    Adjust generation quality based on system load.

    Monitors CPU/GPU usage and reduces complexity when overloaded.
    """

    def __init__(self, composition_engine: CompositionEngine):
        self.engine = composition_engine
        self.quality_level = "normal"

    async def monitor_and_adjust(self):
        """Periodically check system load and adjust quality."""
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds

            cpu_percent = psutil.cpu_percent(interval=1.0)

            # Determine quality level
            if cpu_percent < 60:
                new_quality = "normal"
            elif cpu_percent < 80:
                new_quality = "high_load"
            elif cpu_percent < 95:
                new_quality = "critical"
            else:
                new_quality = "overload"

            if new_quality != self.quality_level:
                logger.warning(f"Load: {cpu_percent}% - Adjusting quality to {new_quality}")
                self.quality_level = new_quality
                await self._apply_quality_adjustment(new_quality)

    async def _apply_quality_adjustment(self, quality: str):
        """Apply quality adjustments to composition engine."""
        if quality == "normal":
            # Full quality
            await self.engine.update_params({"intensity": 0.6})

        elif quality == "high_load":
            # Reduce melody density
            await self.engine.update_params({"intensity": 0.4})

        elif quality == "critical":
            # Minimal complexity
            await self.engine.update_params({"intensity": 0.2})

        elif quality == "overload":
            # Consider pausing generation or using pre-rendered content
            logger.critical("System overload - consider shedding load")
```

---

## 5. Client Limit Enforcement

```python
# server/client_manager.py
from fastapi import HTTPException, status


class ClientLimitError(Exception):
    """Raised when client limit is reached."""
    pass


async def enforce_client_limit(client_manager, max_clients: int = 10):
    """
    Enforce maximum client limit.

    Raises:
        ClientLimitError if limit exceeded
    """
    if len(client_manager.clients) >= max_clients:
        logger.warning(f"Client limit reached ({max_clients})")
        raise ClientLimitError(f"Server at capacity ({max_clients} clients)")


# In WebSocket endpoint
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    try:
        # Check client limit before accepting
        await enforce_client_limit(client_manager, max_clients=config.max_clients)

        await websocket.accept()
        # ... rest of handler

    except ClientLimitError as e:
        # Reject connection
        await websocket.close(code=1008, reason=str(e))
        logger.info("Rejected connection due to client limit")
```

---

## 6. Logging & Monitoring

### 6.1 Structured Error Logging

```python
# auralis/monitoring/logger.py
from loguru import logger
import sys


def setup_logging(log_level: str = "INFO", log_file: str = "logs/auralis.log"):
    """
    Configure structured logging with loguru.

    Features:
    - JSON formatting for production
    - Rotation & retention
    - Error alerting
    """
    # Remove default handler
    logger.remove()

    # Console handler (human-readable for development)
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )

    # File handler (JSON for production)
    logger.add(
        log_file,
        level="INFO",
        rotation="100 MB",  # Rotate when file reaches 100MB
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress rotated logs
        serialize=True,  # JSON format
        enqueue=True,  # Async logging (thread-safe)
    )

    # Separate error log
    logger.add(
        "logs/errors.log",
        level="ERROR",
        rotation="50 MB",
        retention="90 days",
        serialize=True,
    )

    logger.info(f"Logging initialized at level {log_level}")


# Log structured errors
def log_error_with_context(error: Exception, context: dict):
    """Log error with additional context."""
    logger.error(
        f"{error.__class__.__name__}: {error}",
        extra={
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            **context,
        },
    )


# Example usage
try:
    audio = render_phrase(...)
except Exception as e:
    log_error_with_context(e, {
        "component": "synthesis",
        "phrase_id": phrase.phrase_id,
        "device": device_str,
    })
```

---

## 7. Health Checks

```python
# server/health.py
from fastapi import APIRouter
from pydantic import BaseModel


class HealthStatus(BaseModel):
    status: str  # "healthy", "degraded", "unhealthy"
    synthesis_functional: bool
    composition_functional: bool
    buffer_healthy: bool
    error_rate: float


router = APIRouter()


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Health check endpoint.

    Returns:
        200: System healthy
        503: System degraded or unhealthy
    """
    # Check synthesis
    synthesis_ok = test_synthesis()

    # Check composition
    composition_ok = composition_engine.phrase_queue.qsize() > 0

    # Check buffer
    buffer_ok = not ring_buffer.is_low()

    # Check error rate
    error_rate = synthesis_worker.get_failure_rate()

    # Determine overall status
    if synthesis_ok and composition_ok and buffer_ok and error_rate < 0.1:
        status = "healthy"
        status_code = 200
    elif synthesis_ok or composition_ok:
        status = "degraded"
        status_code = 200
    else:
        status = "unhealthy"
        status_code = 503

    return HealthStatus(
        status=status,
        synthesis_functional=synthesis_ok,
        composition_functional=composition_ok,
        buffer_healthy=buffer_ok,
        error_rate=error_rate,
    )


def test_synthesis() -> bool:
    """Quick synthesis health check."""
    try:
        # Try rendering a tiny test phrase
        test_audio = synthesis_engine.render_phrase(
            chords=[(0, 60, "i")],
            melody=[],
            percussion=[],
            duration_sec=0.1,
        )
        return test_audio is not None and np.isfinite(test_audio).all()
    except:
        return False
```

---

## 8. Error Recovery Checklist

- ✅ Synthesis errors: Retry with simplified params → Render silence
- ✅ GPU OOM: Clear cache → Fall back to CPU
- ✅ WebSocket errors: Retry send → Graceful disconnect
- ✅ Client limit: Reject with clear error message
- ✅ System overload: Adaptive quality degradation
- ✅ Invalid audio: Validation + fallback to silence
- ✅ Health checks: Continuous monitoring + alerts
- ✅ Structured logging: JSON logs with context

---

This comprehensive error recovery strategy ensures Auralis remains stable and provides a degraded service rather than complete failure.
