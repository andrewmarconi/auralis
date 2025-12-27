"""Buffer management components: jitter tracking, rate limiting, adaptive buffering."""

import time
from dataclasses import dataclass, field
from typing import Deque
from collections import deque
import statistics


@dataclass
class ChunkTimestamp:
    """Timestamp for chunk delivery tracking."""

    chunk_id: int
    expected_time: float
    actual_time: float

    @property
    def jitter_ms(self) -> float:
        """Calculate jitter in milliseconds."""
        return abs((self.actual_time - self.expected_time) * 1000)


class JitterTracker:
    """
    EMA-based jitter tracker for adaptive buffer sizing.

    Uses Exponential Moving Average to smooth jitter measurements
    and calculate recommended buffer sizes based on confidence levels.
    """

    TIER_TARGETS = {
        "minimal": 0.5,  # 500ms buffer
        "normal": 1.0,  # 1 second buffer
        "stable": 1.5,  # 1.5 seconds buffer
        "defensive": 2.0,  # 2 seconds buffer
    }

    def __init__(
        self, window_size: int = 50, alpha: float = 0.1, tier_adjustment_interval: int = 50
    ):
        """
        Initialize jitter tracker.

        Args:
            window_size: Number of recent chunks to track
            alpha: EMA smoothing factor (0.1 = slow, 0.5 = fast)
            tier_adjustment_interval: Chunks between tier reevaluations
        """
        self.window_size = window_size
        self.alpha = alpha
        self.tier_adjustment_interval = tier_adjustment_interval

        # EMA state
        self.ema_jitter: float = 0.0
        self.ema_jitter_squared: float = 0.0

        # Recent timestamps
        self.timestamps: Deque[ChunkTimestamp] = deque(maxlen=window_size)

        # Statistics
        self.chunk_count: int = 0
        self.underrun_count: int = 0
        self.current_tier: str = "normal"

    def record_chunk(self, expected_time: float, actual_time: float) -> None:
        """
        Record chunk delivery timing.

        Args:
            expected_time: Expected arrival time (Unix timestamp)
            actual_time: Actual arrival time (Unix timestamp)
        """
        chunk_id = self.chunk_count
        timestamp = ChunkTimestamp(chunk_id, expected_time, actual_time)
        self.timestamps.append(timestamp)

        # Calculate jitter (absolute deviation)
        jitter_ms = timestamp.jitter_ms
        jitter_sec = jitter_ms / 1000.0

        # Update EMA
        if self.ema_jitter == 0.0:
            # First chunk: initialize EMA
            self.ema_jitter = jitter_sec
            self.ema_jitter_squared = jitter_sec**2
        else:
            # EMA update: new = α * x + (1-α) * old
            self.ema_jitter = self.alpha * jitter_sec + (1 - self.alpha) * self.ema_jitter
            self.ema_jitter_squared = (
                self.alpha * jitter_sec**2 + (1 - self.alpha) * self.ema_jitter_squared
            )

        self.chunk_count += 1

        # Check tier adjustment every N chunks
        if self.chunk_count % self.tier_adjustment_interval == 0:
            self._adjust_tier()

    def record_underrun(self) -> None:
        """Record buffer underrun event."""
        self.underrun_count += 1

    def get_current_jitter(self) -> float:
        """
        Get current mean jitter.

        Returns:
            Mean jitter in milliseconds
        """
        return self.ema_jitter * 1000.0

    def get_jitter_std(self) -> float:
        """
        Get jitter standard deviation.

        Returns:
            Standard deviation in milliseconds
        """
        if self.ema_jitter_squared <= self.ema_jitter**2:
            return 0.0
        variance = self.ema_jitter_squared - self.ema_jitter**2
        return max(0.0, variance**0.5) * 1000.0

    def get_underrun_rate(self) -> float:
        """
        Get underrun rate.

        Returns:
            Fraction of chunks that underran (0.0-1.0)
        """
        if self.chunk_count == 0:
            return 0.0
        return self.underrun_count / self.chunk_count

    def get_recommended_buffer_ms(self, confidence: float = 0.95) -> float:
        """
        Calculate recommended buffer size.

        Args:
            confidence: Confidence level (0.95 or 0.99)

        Returns:
            Recommended buffer duration in milliseconds
        """
        mean_jitter_ms = self.get_current_jitter()
        jitter_std_ms = self.get_jitter_std()

        # Calculate buffer based on confidence level
        if confidence >= 0.99:
            # 99% confidence: mean + 3σ
            buffer_ms = mean_jitter_ms + 3 * jitter_std_ms
        elif confidence >= 0.95:
            # 95% confidence: mean + 2σ
            buffer_ms = mean_jitter_ms + 2 * jitter_std_ms
        else:
            # Lower confidence: mean + 1σ
            buffer_ms = mean_jitter_ms + 1 * jitter_std_ms

        return buffer_ms

    def get_tier_buffer_ms(self) -> float:
        """
        Get tier-based buffer duration.

        Returns:
            Buffer duration in milliseconds based on current tier
        """
        tier_multiplier = self.TIER_TARGETS.get(self.current_tier, 1.0)
        return tier_multiplier * 1000.0

    def _adjust_tier(self) -> None:
        """Adjust buffer tier based on jitter statistics."""
        mean_jitter_ms = self.get_current_jitter()
        underrun_rate = self.get_underrun_rate()

        # Tier escalation logic
        if underrun_rate > 0.05 or mean_jitter_ms > 30:
            # High underrun rate or high jitter → escalate
            if self.current_tier == "minimal":
                self.current_tier = "normal"
            elif self.current_tier == "normal":
                self.current_tier = "stable"
            elif self.current_tier == "stable":
                self.current_tier = "defensive"
        elif underrun_rate < 0.01 and mean_jitter_ms < 15:
            # Low underrun rate and low jitter → de-escalate
            if self.current_tier == "defensive":
                self.current_tier = "stable"
            elif self.current_tier == "stable":
                self.current_tier = "normal"
            elif self.current_tier == "normal":
                self.current_tier = "minimal"


class TokenBucket:
    """
    Token bucket rate limiter for WebSocket flow control.

    Tokens replenish at a constant rate (refill_rate) up to capacity.
    Consuming tokens requires available tokens; otherwise, the request is blocked.
    """

    def __init__(self, capacity: int = 10, refill_rate: float = 10.0):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum tokens the bucket can hold (burst capacity)
            refill_rate: Tokens added per second (steady-state rate)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill_time = time.time()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        tokens_to_add = elapsed * self.refill_rate

        self.tokens = min(self.tokens + tokens_to_add, float(self.capacity))
        self.last_refill_time = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if insufficient
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time until tokens available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds (0 if tokens available now)
        """
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        deficit = tokens - self.tokens
        return deficit / self.refill_rate

    def get_token_count(self) -> float:
        """
        Get current token count.

        Returns:
            Number of tokens currently available
        """
        self._refill()
        return self.tokens
