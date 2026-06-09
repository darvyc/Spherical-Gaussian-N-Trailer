from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class NTrailerConfig:
    """Configuration for an on-axle N-trailer kinematic chain."""

    n_trailers: int = 3
    tractor_length: float = 2.8
    trailer_lengths: Sequence[float] | None = None
    dt: float = 0.08
    reverse_speed: float = -0.85
    max_steer: float = np.deg2rad(33.0)
    max_steer_rate: float = np.deg2rad(70.0)
    max_articulation: float = np.deg2rad(78.0)
    soft_articulation: float = np.deg2rad(58.0)
    cross_track_scale: float = 2.0
    curvature_scale: float = 0.35
    progress_target: float = -12.0
    seed: int = 0
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "cross_track": 3.0,
            "heading": 1.3,
            "articulation": 0.7,
            "steer": 0.04,
            "steer_rate": 0.02,
            "progress": 0.25,
            "jackknife": 100.0,
        }
    )

    def __post_init__(self) -> None:
        if self.n_trailers < 1:
            raise ValueError("n_trailers must be >= 1")
        lengths = self.trailer_lengths
        if lengths is None:
            lengths = tuple(2.4 for _ in range(self.n_trailers))
        if len(lengths) != self.n_trailers:
            raise ValueError("trailer_lengths must have length n_trailers")
        if any(float(length) <= 0 for length in lengths):
            raise ValueError("all trailer lengths must be positive")
        object.__setattr__(self, "trailer_lengths", tuple(float(x) for x in lengths))

    @property
    def state_dim(self) -> int:
        return 3 + self.n_trailers

    @property
    def action_dim(self) -> int:
        return 1

    def clip_steer(self, steer: float) -> float:
        return float(np.clip(steer, -self.max_steer, self.max_steer))
