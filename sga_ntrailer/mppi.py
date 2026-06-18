from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dynamics import NTrailerState, is_jackknifed, step_kinematics
from .env import NTrailerReverseEnv


@dataclass
class MPPIDiagnostics:
    best_return: float
    mean_return: float
    chosen_action: float
    action_std: float


class MPPIController:
    """Sampling-based model predictive controller used as an expert policy."""

    def __init__(
        self,
        env: NTrailerReverseEnv,
        horizon: int = 24,
        samples: int = 384,
        temperature: float = 4.0,
        noise_std: float | None = None,
        smoothing: float = 0.75,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.config = env.config
        self.horizon = int(horizon)
        self.samples = int(samples)
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.samples < 1:
            raise ValueError("samples must be >= 1")
        self.temperature = float(max(temperature, 1e-6))
        if noise_std is None:
            self.noise_std = float(0.45 * self.config.max_steer)
        else:
            if not np.isfinite(noise_std) or noise_std < 0.0:
                raise ValueError("noise_std must be finite and non-negative")
            self.noise_std = float(noise_std)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.99))
        self.rng = np.random.default_rng(seed)
        self.mean_sequence = np.zeros(self.horizon, dtype=float)
        self.last_diag = MPPIDiagnostics(0.0, 0.0, 0.0, self.noise_std)

    def reset(self) -> None:
        self.mean_sequence[:] = 0.0

    def _smooth_noise(self, noise: np.ndarray) -> np.ndarray:
        if self.smoothing <= 1e-9:
            return noise
        out = np.zeros_like(noise)
        out[:, 0] = noise[:, 0]
        for t in range(1, noise.shape[1]):
            out[:, t] = self.smoothing * out[:, t - 1] + (1.0 - self.smoothing) * noise[:, t]
        return out

    def _rollout_return(self, initial_state: NTrailerState, steer_seq: np.ndarray) -> float:
        state = initial_state.copy()
        prev_steer = self.env.prev_steer
        total = 0.0
        discount = 1.0
        for steer in steer_seq:
            steer = self.config.clip_steer(float(steer))
            state = step_kinematics(state, steer, self.config)
            _, enc_info = self.env.encoder(state.data, self.env.path)
            total += discount * self.env.reward(state, steer, prev_steer, enc_info)
            if is_jackknifed(state, self.config):
                total -= 250.0
                break
            if enc_info.progress >= 0.98 * self.env.path.total_length:
                total += 40.0
                break
            prev_steer = steer
            discount *= 0.985
        return float(total)

    def act(self, obs: np.ndarray | None = None) -> np.ndarray:
        del obs
        noise = self.rng.normal(0.0, self.noise_std, size=(self.samples, self.horizon))
        noise = self._smooth_noise(noise)
        candidates = self.mean_sequence[None, :] + noise
        candidates = np.clip(candidates, -self.config.max_steer, self.config.max_steer)
        returns = np.empty(self.samples, dtype=float)
        initial = self.env.state.copy()
        for k in range(self.samples):
            returns[k] = self._rollout_return(initial, candidates[k])
        best = float(np.max(returns))
        weights = np.exp((returns - best) / self.temperature)
        weights /= np.sum(weights) + 1e-12
        updated = weights @ candidates
        action = float(updated[0])
        self.mean_sequence[:-1] = updated[1:]
        self.mean_sequence[-1] = updated[-1]
        self.last_diag = MPPIDiagnostics(
            best_return=best,
            mean_return=float(np.mean(returns)),
            chosen_action=action,
            action_std=float(np.std(candidates[:, 0])),
        )
        return np.array([self.config.clip_steer(action)], dtype=np.float32)


def rollout_mppi(
    env: NTrailerReverseEnv,
    controller: MPPIController,
    steps: int = 250,
    reset: bool = True,
) -> dict[str, Any]:
    if reset:
        obs = env.reset()
        controller.reset()
    else:
        obs, _ = env.encoder(env.state.data, env.path)
    trace: dict[str, list[Any]] = {
        "state": [],
        "action": [],
        "reward": [],
        "cross_track": [],
        "heading_error": [],
        "progress": [],
        "jackknife": [],
    }
    for _ in range(steps):
        action = controller.act(obs)
        obs, reward, done, info = env.step(action)
        trace["state"].append(info["raw_state"].tolist())
        trace["action"].append(float(action[0]))
        trace["reward"].append(float(reward))
        trace["cross_track"].append(float(info["cross_track"]))
        trace["heading_error"].append(float(info["heading_error"]))
        trace["progress"].append(float(info["progress"]))
        trace["jackknife"].append(bool(info["jackknife"]))
        if done:
            break
    return trace
