from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import NTrailerConfig
from .dynamics import NTrailerState, unit_poses
from .path import Path2D
from .utils import read_json


def plot_trace(json_path: str | Path, out: str | Path | None = None) -> None:
    import matplotlib.pyplot as plt

    payload = read_json(json_path)
    path = Path2D.from_waypoints(payload["path"]["waypoints"])
    n = int(payload.get("config", {}).get("n_trailers", len(payload["trace"]["state"][0]) - 3))
    config = NTrailerConfig(n_trailers=n)
    states = np.asarray(payload["trace"]["state"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(path.points[:, 0], path.points[:, 1], linewidth=2, label="path")
    if len(states):
        ax.plot(states[:, 0], states[:, 1], linewidth=1, label="tractor")
        for idx in np.linspace(0, len(states) - 1, min(12, len(states)), dtype=int):
            poses = unit_poses(NTrailerState(states[idx]), config)
            ax.plot(poses[:, 0], poses[:, 1], marker="o", linewidth=1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    ax.set_title("N-trailer reversal trace")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, bbox_inches="tight")
    else:
        plt.show()
