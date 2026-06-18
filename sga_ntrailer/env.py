from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import NTrailerConfig
from .dynamics import NTrailerState, articulation_barrier, is_jackknifed, step_kinematics, unit_poses
from .features import EncoderInfo, SphericalGaussianEncoder
from .path import Path2D
from .utils import wrap_angle


@dataclass(frozen=True)
class StepInfo:
    cross_track: float
    heading_error: float
    progress: float
    curvature: float
    jackknife: bool
    raw_state: np.ndarray


class NTrailerReverseEnv:
    """Small gym-like environment for reverse path following."""

    def __init__(
        self,
        config: NTrailerConfig,
        path: Path2D,
        encoder: SphericalGaussianEncoder | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.path = path
        self.encoder = encoder or SphericalGaussianEncoder(config, seed=config.seed)
        self.rng = np.random.default_rng(config.seed if seed is None else seed)
        self.state = NTrailerState.zeros(config)
        self.prev_steer = 0.0
        self.steps = 0
        self.last_info: EncoderInfo | None = None

    @property
    def observation_dim(self) -> int:
        return self.encoder.out_dim

    def _initial_state(self, noise: bool = True) -> NTrailerState:
        start, heading, _ = self.path.sample(0.0)
        theta = heading + np.pi
        alpha = np.zeros(self.config.n_trailers, dtype=float)
        x, y = float(start[0]), float(start[1])
        if noise:
            x += float(self.rng.normal(0.0, 0.15))
            y += float(self.rng.normal(0.0, 0.15))
            theta += float(self.rng.normal(0.0, np.deg2rad(4.0)))
            alpha += self.rng.normal(0.0, np.deg2rad(2.0), size=self.config.n_trailers)
        return NTrailerState.from_values(x, y, wrap_angle(theta), alpha)

    def reset(self, state: NTrailerState | np.ndarray | None = None, noise: bool = True) -> np.ndarray:
        if state is None:
            self.state = self._initial_state(noise=noise)
        elif isinstance(state, NTrailerState):
            self.state = state.copy()
        else:
            self.state = NTrailerState(np.asarray(state, dtype=float).copy())
        self.prev_steer = 0.0
        self.steps = 0
        obs, info = self.encoder(self.state.data, self.path)
        self.last_info = info
        return obs

    @staticmethod
    def reverse_heading_error(info: EncoderInfo) -> float:
        return float(wrap_angle(info.heading_error - np.pi))

    def progress_fraction(self, info: EncoderInfo) -> float:
        return float(np.clip(info.progress / max(self.path.total_length, 1e-9), 0.0, 1.0))

    def reward(self, state: NTrailerState, steer: float, prev_steer: float, info: EncoderInfo) -> float:
        w = self.config.reward_weights
        alpha = state.alpha
        reverse_error = self.reverse_heading_error(info)
        progress_bonus = w["progress"] * self.progress_fraction(info)
        cost = 0.0
        cost += w["cross_track"] * (info.cross_track / self.config.cross_track_scale) ** 2
        cost += w["heading"] * (reverse_error / np.pi) ** 2
        cost += w["articulation"] * float(np.mean((alpha / self.config.soft_articulation) ** 2))
        cost += w["steer"] * (steer / self.config.max_steer) ** 2
        cost += w["steer_rate"] * ((steer - prev_steer) / max(self.config.max_steer_rate, 1e-9)) ** 2
        cost += 8.0 * articulation_barrier(alpha, self.config)
        if is_jackknifed(state, self.config):
            cost += w["jackknife"]
        return float(progress_bonus - cost)

    def step(self, action: np.ndarray | list[float] | float) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        steer = float(np.asarray(action).reshape(-1)[0])
        steer = self.config.clip_steer(steer)
        next_state = step_kinematics(self.state, steer, self.config)
        obs, enc_info = self.encoder(next_state.data, self.path)
        r = self.reward(next_state, steer, self.prev_steer, enc_info)
        jackknife = is_jackknifed(next_state, self.config)
        done = bool(jackknife or enc_info.progress >= 0.96 * self.path.total_length)
        self.state = next_state
        self.prev_steer = steer
        self.steps += 1
        self.last_info = enc_info
        info = StepInfo(
            cross_track=enc_info.cross_track,
            heading_error=self.reverse_heading_error(enc_info),
            progress=enc_info.progress,
            curvature=enc_info.curvature,
            jackknife=jackknife,
            raw_state=next_state.data.copy(),
        )
        return obs, r, done, info.__dict__

    def clone(self) -> "NTrailerReverseEnv":
        cloned = NTrailerReverseEnv(self.config, self.path, self.encoder, seed=int(self.rng.integers(0, 2**31 - 1)))
        cloned.state = self.state.copy()
        cloned.prev_steer = self.prev_steer
        cloned.steps = self.steps
        return cloned

    def poses(self) -> np.ndarray:
        return unit_poses(self.state, self.config)
