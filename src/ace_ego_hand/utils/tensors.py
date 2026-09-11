"""Tensor plumbing shared across the package: the one torch-to-numpy conversion."""

from __future__ import annotations

import numpy as np
import torch


def to_numpy(value: torch.Tensor) -> np.ndarray:
    """A float32 numpy copy of any tensor, detached and on the host."""

    return value.detach().float().cpu().numpy()
