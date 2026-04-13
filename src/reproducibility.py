"""Reproducibility helpers for scripts and notebooks."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


DEFAULT_RANDOM_SEED = 42


def set_global_seed(seed: int = DEFAULT_RANDOM_SEED) -> int:
    """Set deterministic random seed across common generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def notebook_project_root(start_path: str | Path | None = None) -> Path:
    """
    Resolve project root for notebook usage.
    """
    start = Path(start_path).resolve() if start_path is not None else Path.cwd().resolve()
    if (start / "src").exists():
        return start
    parent = start.parent
    if (parent / "src").exists():
        return parent
    return start
