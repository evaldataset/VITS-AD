"""Reproducibility utilities for VITS project.

Provides functions to seed all random number generators and ensure
deterministic behavior across runs.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Seed all random number generators for reproducibility.

    Sets seeds for Python's random module, NumPy, PyTorch (CPU and CUDA),
    and configures PyTorch for deterministic operations.

    Args:
        seed: Random seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("Set random seed to %d for all generators.", seed)


def get_device(device: str | None = None) -> torch.device:
    """Get torch device with sensible defaults.

    Args:
        device: Device string (e.g., 'cuda', 'cuda:0', 'cpu').
            If None, automatically selects CUDA if available.

    Returns:
        torch.device instance.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
