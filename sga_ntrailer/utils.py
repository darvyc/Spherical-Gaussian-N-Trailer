from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def smooth_clip(x: np.ndarray, limit: float, sharpness: float = 8.0) -> np.ndarray:
    """Smoothly squash values into [-limit, limit]."""
    return limit * np.tanh(sharpness * x / max(limit, 1e-9)) / np.tanh(sharpness)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = ensure_dir(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, dt, axis=0)


def scalar(x: float | np.ndarray) -> float:
    return float(np.asarray(x).reshape(()))


def hypot2(x: float, y: float) -> float:
    return float(math.sqrt(x * x + y * y))
