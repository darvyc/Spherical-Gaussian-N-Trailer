from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import NTrailerConfig
from .env import NTrailerReverseEnv
from .mppi import MPPIController
from .path import Path2D
from .policy import MLPPolicy, require_torch, save_policy, torch


@dataclass
class Dataset:
    observations: np.ndarray
    actions: np.ndarray


def collect_expert_dataset(
    config: NTrailerConfig,
    path: Path2D,
    episodes: int = 12,
    steps: int = 180,
    horizon: int = 18,
    samples: int = 192,
    seed: int = 0,
) -> Dataset:
    env = NTrailerReverseEnv(config, path, seed=seed)
    expert = MPPIController(env, horizon=horizon, samples=samples, seed=seed)
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    for ep in range(episodes):
        obs = env.reset(noise=True)
        expert.reset()
        for _ in range(steps):
            action = expert.act(obs)
            obs_buf.append(obs.astype(np.float32))
            act_buf.append(action.astype(np.float32))
            obs, _, done, _ = env.step(action)
            if done:
                break
    return Dataset(np.stack(obs_buf, axis=0), np.stack(act_buf, axis=0))


def train_bc_policy(
    dataset: Dataset,
    max_action: float,
    epochs: int = 20,
    batch_size: int = 128,
    hidden: int = 192,
    lr: float = 2e-3,
    seed: int = 0,
) -> tuple[MLPPolicy, list[float]]:
    require_torch()
    torch.manual_seed(seed)
    obs = torch.as_tensor(dataset.observations, dtype=torch.float32)
    actions = torch.as_tensor(dataset.actions, dtype=torch.float32)
    policy = MLPPolicy(obs_dim=obs.shape[1], hidden=hidden, max_action=max_action)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    losses: list[float] = []
    n = obs.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        count = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = policy(obs[idx])
            loss = torch.mean((pred - actions[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
            opt.step()
            total += float(loss.item()) * len(idx)
            count += len(idx)
        losses.append(total / max(count, 1))
    return policy, losses


def train_and_save(
    config: NTrailerConfig,
    path: Path2D,
    out: str | Path,
    episodes: int = 12,
    steps: int = 180,
    epochs: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    dataset = collect_expert_dataset(config, path, episodes=episodes, steps=steps, seed=seed)
    policy, losses = train_bc_policy(
        dataset,
        max_action=config.max_steer,
        epochs=epochs,
        seed=seed,
    )
    meta = {
        "obs_dim": int(dataset.observations.shape[1]),
        "hidden": 192,
        "max_action": float(config.max_steer),
        "n_trailers": config.n_trailers,
        "episodes": episodes,
        "samples": int(dataset.observations.shape[0]),
        "losses": losses,
    }
    save_policy(out, policy, meta)
    return meta
