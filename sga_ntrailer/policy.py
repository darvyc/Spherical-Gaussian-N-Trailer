from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class TorchMissingError(RuntimeError):
    pass


def require_torch() -> None:
    if torch is None or nn is None:
        raise TorchMissingError("PyTorch is required. Install with `pip install -e .[train]`.")


class MLPPolicy(nn.Module if nn is not None else object):
    """Small bounded-steering policy."""

    def __init__(self, obs_dim: int, hidden: int = 192, max_action: float = 0.5) -> None:
        require_torch()
        super().__init__()
        self.max_action = float(max_action)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, obs):  # type: ignore[override]
        return self.max_action * self.net(obs)

    def act(self, obs: np.ndarray) -> np.ndarray:
        require_torch()
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
            action = self(x).cpu().numpy().reshape(-1)
        return action.astype(np.float32)


def save_policy(path: str | Path, policy: MLPPolicy, metadata: dict) -> None:
    require_torch()
    payload = {"state_dict": policy.state_dict(), "metadata": metadata}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_policy(path: str | Path) -> tuple[MLPPolicy, dict]:
    require_torch()
    payload = torch.load(path, map_location="cpu")
    meta = payload.get("metadata", {})
    policy = MLPPolicy(
        obs_dim=int(meta["obs_dim"]),
        hidden=int(meta.get("hidden", 192)),
        max_action=float(meta["max_action"]),
    )
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy, meta
