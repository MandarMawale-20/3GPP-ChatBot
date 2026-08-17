"""Runtime device auto-selection for model backends.

Both the dense/sparse embedding model (BGE-M3) and the cross-encoder
reranker are torch-based and benefit from CUDA when a GPU is present.
Rather than hard-coding GPU-only (or CPU-only) behavior, each provider
queries :func:`get_device` once at construction time:

- ``"cuda"`` — when a usable NVIDIA GPU + CUDA build of torch is available
- ``"cpu"``  — otherwise (no GPU, or a CPU-only torch install)

This keeps the same code path working on GPU dev boxes (e.g. an RTX
1650) and on machines with no GPU at all.
"""

from __future__ import annotations

from loguru import logger


def get_device() -> str:
    """Auto-select the best available compute device.

    Returns ``"cuda"`` when torch reports a usable CUDA GPU, otherwise
    ``"cpu"``. The torch import is deliberately isolated here so that a
    CPU-only environment without torch installed still gets a sensible
    default rather than an import error.
    """
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        # No torch at all ⇒ CPU is the only correct answer.
        device = "cpu"

    logger.debug(f"Auto-selected compute device: {device}")
    return device
