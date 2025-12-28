"""
GPU Device Selection and Management for Audio Synthesis

Provides intelligent device selection with capability detection for Metal/CUDA/CPU.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger


@dataclass
class DeviceInfo:
    """GPU device information and capabilities."""

    device_type: str  # "mps", "cuda", "cpu"
    device_name: str  # e.g., "Apple M4", "NVIDIA RTX 3090"
    total_memory_mb: float  # Total device memory
    available_memory_mb: float  # Available memory
    compute_capability: Optional[str] = None  # CUDA compute capability
    supports_fp16: bool = False
    supports_bf16: bool = False
    unified_memory: bool = False  # True for Metal


class DeviceSelector:
    """
    Intelligent GPU device selection for audio synthesis.

    Detects available hardware and selects optimal device with fallback logic:
    1. Metal (Apple Silicon M1/M2/M4) - preferred on macOS
    2. CUDA (NVIDIA GPUs) - preferred on Linux/Windows
    3. CPU (fallback)

    Provides device capability queries for optimization decisions.
    """

    @staticmethod
    def get_optimal_device() -> torch.device:
        """
        Detect and return optimal compute device.

        Returns:
            torch.device configured for best available hardware

        Raises:
            RuntimeError: If no compatible device found (should not happen)
        """
        # Priority 1: Apple Metal (MPS)
        if torch.backends.mps.is_available():
            try:
                device = torch.device("mps")
                # Verify device works with test tensor
                _ = torch.zeros(1, device=device)  # Device verification test
                logger.info("✓ Metal (MPS) device detected and verified")
                return device
            except Exception as e:
                logger.warning(f"Metal available but failed verification: {e}")

        # Priority 2: NVIDIA CUDA
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                # Verify device works with test tensor
                _ = torch.zeros(1, device=device)  # Device verification test
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✓ CUDA device detected: {gpu_name}")
                return device
            except Exception as e:
                logger.warning(f"CUDA available but failed verification: {e}")

        # Priority 3: CPU fallback
        logger.warning("⚠ No GPU detected - using CPU (may impact real-time performance)")
        return torch.device("cpu")

    @staticmethod
    def get_device_info(device: Optional[torch.device] = None) -> DeviceInfo:
        """
        Get detailed information about compute device.

        Args:
            device: Device to query (None = auto-detect optimal)

        Returns:
            DeviceInfo with capabilities and memory stats
        """
        if device is None:
            device = DeviceSelector.get_optimal_device()

        if device.type == "mps":
            return DeviceSelector._get_metal_info()
        elif device.type == "cuda":
            return DeviceSelector._get_cuda_info(device)
        else:
            return DeviceSelector._get_cpu_info()

    @staticmethod
    def _get_metal_info() -> DeviceInfo:
        """Get Metal (Apple Silicon) device information."""
        # Metal uses unified memory architecture
        # Query system memory as proxy (exact Metal allocation tracking not exposed)
        try:
            import psutil

            mem = psutil.virtual_memory()
            total_mb = mem.total / (1024**2)
            available_mb = mem.available / (1024**2)
        except ImportError:
            # Fallback if psutil not available
            total_mb = 16384.0  # Assume 16GB
            available_mb = 12000.0

        return DeviceInfo(
            device_type="mps",
            device_name="Apple Metal Performance Shaders",
            total_memory_mb=total_mb,
            available_memory_mb=available_mb,
            unified_memory=True,
            supports_fp16=True,
            supports_bf16=False,  # Metal does not support bfloat16
        )

    @staticmethod
    def _get_cuda_info(device: torch.device) -> DeviceInfo:
        """Get CUDA (NVIDIA) device information."""
        device_idx = device.index or 0

        # Get device name
        device_name = torch.cuda.get_device_name(device_idx)

        # Get memory stats (bytes → MB)
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        total_mb = total_memory / (1024**2)

        # Available = Total - Allocated
        allocated_mb = torch.cuda.memory_allocated(device_idx) / (1024**2)
        available_mb = total_mb - allocated_mb

        # Get compute capability (e.g., "8.6" for RTX 3090)
        props = torch.cuda.get_device_properties(device_idx)
        compute_cap = f"{props.major}.{props.minor}"

        # Check precision support
        supports_fp16 = props.major >= 6  # Pascal and newer
        supports_bf16 = props.major >= 8  # Ampere and newer

        return DeviceInfo(
            device_type="cuda",
            device_name=device_name,
            total_memory_mb=total_mb,
            available_memory_mb=available_mb,
            compute_capability=compute_cap,
            supports_fp16=supports_fp16,
            supports_bf16=supports_bf16,
            unified_memory=False,
        )

    @staticmethod
    def _get_cpu_info() -> DeviceInfo:
        """Get CPU device information."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            total_mb = mem.total / (1024**2)
            available_mb = mem.available / (1024**2)
        except ImportError:
            total_mb = 8192.0  # Assume 8GB
            available_mb = 4096.0

        return DeviceInfo(
            device_type="cpu",
            device_name="CPU",
            total_memory_mb=total_mb,
            available_memory_mb=available_mb,
            unified_memory=True,  # CPU uses system RAM
            supports_fp16=False,
            supports_bf16=False,
        )

    @staticmethod
    def should_use_mixed_precision(device_info: DeviceInfo) -> bool:
        """
        Determine if mixed precision training should be used.

        Args:
            device_info: Device capabilities

        Returns:
            True if device supports and would benefit from FP16
        """
        # Use FP16 on GPUs that support it
        return device_info.device_type in ["mps", "cuda"] and device_info.supports_fp16

    @staticmethod
    def get_recommended_batch_size(device_info: DeviceInfo) -> int:
        """
        Recommend batch size based on available memory.

        Args:
            device_info: Device capabilities

        Returns:
            Recommended max batch size for voice rendering
        """
        available_mb = device_info.available_memory_mb

        # Conservative estimates for voice rendering
        # Each voice requires ~10MB for typical 8-second phrase
        if available_mb > 20000:  # >20GB
            return 32
        elif available_mb > 10000:  # >10GB
            return 16
        elif available_mb > 5000:  # >5GB
            return 8
        else:
            return 4  # Minimum batch size
