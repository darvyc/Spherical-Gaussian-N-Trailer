from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import NTrailerConfig
from .utils import wrap_angle


@dataclass(frozen=True)
class NTrailerState:
    """State vector wrapper for an articulated chain."""

    data: np.ndarray

    @classmethod
    def zeros(cls, config: NTrailerConfig) -> "NTrailerState":
        return cls(np.zeros(config.state_dim, dtype=float))

    @classmethod
    def from_values(
        cls,
        x: float,
        y: float,
        theta: float,
        articulations: list[float] | tuple[float, ...] | np.ndarray,
    ) -> "NTrailerState":
        return cls(np.array([x, y, theta, *articulations], dtype=float))

    @property
    def x(self) -> float:
        return float(self.data[0])

    @property
    def y(self) -> float:
        return float(self.data[1])

    @property
    def theta0(self) -> float:
        return float(self.data[2])

    @property
    def alpha(self) -> np.ndarray:
        return self.data[3:]

    def copy(self) -> "NTrailerState":
        return NTrailerState(self.data.copy())


def articulation_to_headings(theta0: float, alpha: np.ndarray) -> np.ndarray:
    """Return headings [theta0, theta1, ..., thetaN]."""
    headings = [float(theta0)]
    theta = float(theta0)
    for a in alpha:
        theta = float(wrap_angle(theta - a))
        headings.append(theta)
    return np.asarray(headings, dtype=float)


def unit_poses(state: NTrailerState | np.ndarray, config: NTrailerConfig) -> np.ndarray:
    """Return [N+1, 3] poses for tractor and trailer centres/rear axle proxies."""
    raw = state.data if isinstance(state, NTrailerState) else np.asarray(state, dtype=float)
    theta = articulation_to_headings(float(raw[2]), raw[3:])
    poses = np.zeros((config.n_trailers + 1, 3), dtype=float)
    poses[0] = [raw[0], raw[1], theta[0]]
    x, y = float(raw[0]), float(raw[1])
    for i, length in enumerate(config.trailer_lengths, start=1):
        x = x - length * np.cos(theta[i])
        y = y - length * np.sin(theta[i])
        poses[i] = [x, y, theta[i]]
    return poses


def state_derivative(raw_state: np.ndarray, steer: float, speed: float, config: NTrailerConfig) -> np.ndarray:
    """Continuous-time standard N-trailer kinematic derivative."""
    x, y, theta0 = map(float, raw_state[:3])
    alpha = np.asarray(raw_state[3:], dtype=float)
    steer = config.clip_steer(steer)
    dx = np.zeros_like(raw_state, dtype=float)
    dx[0] = speed * np.cos(theta0)
    dx[1] = speed * np.sin(theta0)
    yaw_prev = speed / config.tractor_length * np.tan(steer)
    v_prev = speed
    dx[2] = yaw_prev
    for i, (a, length) in enumerate(zip(alpha, config.trailer_lengths), start=3):
        v_i = v_prev * np.cos(a)
        yaw_i = v_i * np.sin(a) / length
        dx[i] = yaw_prev - yaw_i
        v_prev = v_i
        yaw_prev = yaw_i
    return dx


def step_kinematics(
    state: NTrailerState | np.ndarray,
    steer: float,
    config: NTrailerConfig,
    speed: float | None = None,
    dt: float | None = None,
) -> NTrailerState:
    """RK4 integration for one time step."""
    z = state.data if isinstance(state, NTrailerState) else np.asarray(state, dtype=float)
    z = z.astype(float, copy=True)
    v = config.reverse_speed if speed is None else float(speed)
    h = config.dt if dt is None else float(dt)
    steer = config.clip_steer(steer)

    k1 = state_derivative(z, steer, v, config)
    k2 = state_derivative(z + 0.5 * h * k1, steer, v, config)
    k3 = state_derivative(z + 0.5 * h * k2, steer, v, config)
    k4 = state_derivative(z + h * k3, steer, v, config)
    zn = z + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    zn[2:] = wrap_angle(zn[2:])
    return NTrailerState(zn)


def is_jackknifed(state: NTrailerState | np.ndarray, config: NTrailerConfig) -> bool:
    raw = state.data if isinstance(state, NTrailerState) else np.asarray(state, dtype=float)
    return bool(np.any(np.abs(raw[3:]) >= config.max_articulation))


def articulation_barrier(alpha: np.ndarray, config: NTrailerConfig) -> float:
    """Soft barrier that grows sharply near jackknife angles."""
    margin = config.max_articulation - np.abs(alpha)
    soft = config.max_articulation - config.soft_articulation
    danger = np.maximum(0.0, soft - margin) / max(soft, 1e-9)
    return float(np.sum(danger**4))
