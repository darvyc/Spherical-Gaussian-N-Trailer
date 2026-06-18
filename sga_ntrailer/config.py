from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


REQUIRED_REWARD_WEIGHTS = frozenset(
    {
        "cross_track",
        "heading",
        "articulation",
        "steer",
        "steer_rate",
        "progress",
        "jackknife",
    }
)


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
        positive_scalars = {
            "tractor_length": self.tractor_length,
            "dt": self.dt,
            "max_steer": self.max_steer,
            "max_steer_rate": self.max_steer_rate,
            "max_articulation": self.max_articulation,
            "soft_articulation": self.soft_articulation,
            "cross_track_scale": self.cross_track_scale,
            "curvature_scale": self.curvature_scale,
        }
        for name, value in positive_scalars.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(self.reverse_speed) or float(self.reverse_speed) == 0.0:
            raise ValueError("reverse_speed must be finite and non-zero")
        if self.soft_articulation > self.max_articulation:
            raise ValueError("soft_articulation must be <= max_articulation")
        lengths = self.trailer_lengths
        if lengths is None:
            lengths = tuple(2.4 for _ in range(self.n_trailers))
        if len(lengths) != self.n_trailers:
            raise ValueError("trailer_lengths must have length n_trailers")
        try:
            normalized_lengths = tuple(float(length) for length in lengths)
        except (TypeError, ValueError) as exc:
            raise ValueError("all trailer lengths must be positive and finite") from exc
        if any(not np.isfinite(length) or length <= 0.0 for length in normalized_lengths):
            raise ValueError("all trailer lengths must be positive and finite")
        object.__setattr__(self, "trailer_lengths", normalized_lengths)
        missing = REQUIRED_REWARD_WEIGHTS.difference(self.reward_weights)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"reward_weights missing required keys: {missing_names}")
        reward_weights = {key: float(value) for key, value in self.reward_weights.items()}
        for key, value in reward_weights.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"reward_weights[{key!r}] must be finite and non-negative")
        object.__setattr__(self, "reward_weights", reward_weights)

    @property
    def state_dim(self) -> int:
        return 3 + self.n_trailers

    @property
    def action_dim(self) -> int:
        return 1

    def clip_steer(self, steer: float) -> float:
        return float(np.clip(steer, -self.max_steer, self.max_steer))
