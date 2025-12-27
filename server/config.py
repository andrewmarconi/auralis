"""Configuration module for Auralis performance settings."""

from pydantic import BaseModel, Field
from typing import List


class BufferConfig(BaseModel):
    """Buffer management configuration."""

    enable_adaptive_tiers: bool = True
    enable_jitter_tracking: bool = True
    enable_token_bucket: bool = True
    enable_object_pooling: bool = True

    initial_tier: str = "normal"
    tier_adjustment_interval_chunks: int = 50

    jitter_window_size: int = 50
    jitter_ema_alpha: float = 0.1

    token_bucket_capacity: int = 10
    token_bucket_refill_rate: float = 10.0

    chunk_pool_size: int = 20


class ConcurrencyConfig(BaseModel):
    """WebSocket concurrency configuration."""

    enable_per_client_cursors: bool = True
    enable_broadcast: bool = True
    enable_thread_pool: bool = True

    thread_pool_workers: int = Field(default=2, ge=1, le=8)

    broadcast_interval_ms: int = 100
    max_concurrent_clients: int = 20

    drain_timeout_sec: float = 5.0


class GPUConfig(BaseModel):
    """GPU optimization configuration."""

    enable_memory_prealloc: bool = True
    enable_batch_processing: bool = True
    enable_torch_compile: bool = True
    enable_cuda_streams: bool = True

    max_batch_size: int = 16

    compile_mode: str = "reduce-overhead"
    warmup_iterations: int = 3

    max_phrase_duration_sec: float = 30.0
    max_voice_duration_sec: float = 10.0

    cleanup_interval_renders: int = 100


class MemoryConfig(BaseModel):
    """Memory leak prevention configuration."""

    enable_tracemalloc: bool = True
    enable_gc_tuning: bool = True
    enable_periodic_cleanup: bool = True

    gc_gen0_threshold: int = 50000
    gc_gen1_threshold: int = 500
    gc_gen2_threshold: int = 1000
    auto_gc_enabled: bool = False

    snapshot_interval_sec: int = 300
    leak_detection_threshold_mb_per_hour: float = 20.0


class MonitoringConfig(BaseModel):
    """Performance monitoring configuration."""

    enable_prometheus: bool = True
    enable_grafana: bool = True

    metrics_collection_interval_sec: int = 5
    prometheus_port: int = 9090

    enable_alerts: bool = True
    alert_severity_levels: List[str] = Field(default_factory=lambda: ["critical", "high", "medium"])


class PerformanceConfig(BaseModel):
    """Master performance optimization configuration."""

    buffer: BufferConfig = Field(default_factory=BufferConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    enable_performance_mode: bool = True
    target_latency_ms: float = 100.0
    target_concurrent_users: int = 10
