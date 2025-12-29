"""GPU/CPU device selection for audio processing.

FluidSynth synthesis is CPU-only (sample-based), but PyTorch effects
can leverage GPU acceleration (Metal/CUDA) in future phases.
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["Metal", "CUDA", "CPU"]


def select_device() -> DeviceType:
    """Select best available device for audio processing.

    Priority:
        1. Metal (Apple Silicon M1/M2/M4)
        2. CUDA (NVIDIA GPUs)
        3. CPU (fallback)

    Returns:
        Device type string: "Metal", "CUDA", or "CPU"

    Note:
        FluidSynth synthesis always runs on CPU (sample-based rendering).
        GPU acceleration applies to future PyTorch effects pipeline.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            logger.info("Metal GPU detected and available for effects processing")
            return "Metal"
        elif torch.cuda.is_available():
            logger.info(
                f"CUDA GPU detected: {torch.cuda.get_device_name(0)} "
                "(available for effects processing)"
            )
            return "CUDA"
        else:
            logger.info("No GPU detected, using CPU for all processing")
            return "CPU"
    except ImportError:
        logger.warning("PyTorch not installed, defaulting to CPU")
        return "CPU"
    except Exception as e:
        logger.warning(f"Error detecting GPU device: {e}, defaulting to CPU")
        return "CPU"


def get_device_info() -> dict[str, str | bool]:
    """Get detailed device information.

    Returns:
        Dictionary with device metadata:
        - device: Device type ("Metal", "CUDA", "CPU")
        - available: Whether GPU is actually usable
        - name: Device name (for CUDA GPUs)
        - fluidsynth_device: Always "CPU" (sample-based synthesis)
    """
    device = select_device()
    info: dict[str, str | bool] = {
        "device": device,
        "available": device != "CPU",
        "fluidsynth_device": "CPU",  # FluidSynth is always CPU
    }

    if device == "CUDA":
        try:
            import torch

            info["name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

    return info
