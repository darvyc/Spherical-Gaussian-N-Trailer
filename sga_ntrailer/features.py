from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import NTrailerConfig
from .geometry import sphere_point, tangent_basis, tangent_coordinates
from .path import Path2D
from .utils import wrap_angle


@dataclass(frozen=True)
class EncoderInfo:
    cross_track: float
    heading_error: float
    progress: float
    curvature: float
    sphere_path: np.ndarray
    sphere_vehicle: np.ndarray
    tangent_error: np.ndarray


class GaussianCombinatorialFeatures:
    """Random Fourier and sparse subset Gaussian feature map."""

    def __init__(
        self,
        in_dim: int,
        rff_dim: int = 96,
        subset_dim: int = 64,
        subset_size: int = 3,
        sigma: float = 1.0,
        seed: int = 0,
    ) -> None:
        if in_dim < 1:
            raise ValueError("in_dim must be positive")
        self.in_dim = int(in_dim)
        self.rff_dim = int(rff_dim)
        self.subset_dim = int(subset_dim)
        self.subset_size = int(max(1, min(subset_size, in_dim)))
        self.sigma = float(max(sigma, 1e-6))
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0 / self.sigma, size=(self.rff_dim, self.in_dim))
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=(self.rff_dim,))
        self.subsets = np.stack(
            [rng.choice(self.in_dim, size=self.subset_size, replace=False) for _ in range(self.subset_dim)],
            axis=0,
        )
        self.anchors = rng.normal(0.0, 0.8, size=(self.subset_dim, self.subset_size))
        self.lengths = rng.uniform(0.45, 1.75, size=(self.subset_dim,))
        self.signs = rng.choice([-1.0, 1.0], size=(self.subset_dim, self.subset_size))

    @property
    def out_dim(self) -> int:
        return self.in_dim + self.rff_dim + self.subset_dim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(self.in_dim)
        rff = np.sqrt(2.0 / max(self.rff_dim, 1)) * np.cos(self.W @ x + self.b)
        subset_values = np.empty(self.subset_dim, dtype=float)
        for j, idx in enumerate(self.subsets):
            z = x[idx]
            radial = np.exp(-np.sum((z - self.anchors[j]) ** 2) / (2.0 * self.lengths[j] ** 2))
            signed = np.prod(np.tanh(self.signs[j] * z))
            subset_values[j] = radial * signed
        return np.concatenate([x, rff, subset_values]).astype(np.float32)


class SphericalGaussianEncoder:
    """Encodes state/path geometry into tangent and high-dimensional Gaussian features."""

    def __init__(
        self,
        config: NTrailerConfig,
        rff_dim: int = 96,
        subset_dim: int = 64,
        seed: int = 0,
    ) -> None:
        self.config = config
        # Tangent 2 + cross-track + heading + curvature + progress + articulations + sin/cos articulations
        self.base_dim = 6 + 3 * config.n_trailers
        self.field = GaussianCombinatorialFeatures(
            in_dim=self.base_dim,
            rff_dim=rff_dim,
            subset_dim=subset_dim,
            subset_size=min(4, self.base_dim),
            sigma=1.0,
            seed=seed,
        )

    @property
    def out_dim(self) -> int:
        return self.field.out_dim

    def raw(self, state: np.ndarray, path: Path2D) -> tuple[np.ndarray, EncoderInfo]:
        q = path.nearest(state[:2])
        cross_scaled = q.signed_error / self.config.cross_track_scale
        curv_scaled = q.curvature / self.config.curvature_scale
        sphere_path = sphere_point(q.heading, np.tanh(curv_scaled))
        sphere_vehicle = sphere_point(float(state[2]), np.tanh(cross_scaled))
        tangent_error = tangent_coordinates(sphere_path, sphere_vehicle)
        heading_error = float(wrap_angle(state[2] - q.heading))
        alpha = np.asarray(state[3:], dtype=float)
        progress_norm = 2.0 * (q.progress / max(path.total_length, 1e-9)) - 1.0
        raw = np.concatenate(
            [
                tangent_error,
                np.array([cross_scaled, heading_error / np.pi, np.tanh(curv_scaled), progress_norm]),
                alpha / self.config.max_articulation,
                np.sin(alpha),
                np.cos(alpha),
            ]
        )
        info = EncoderInfo(
            cross_track=float(q.signed_error),
            heading_error=heading_error,
            progress=float(q.progress),
            curvature=float(q.curvature),
            sphere_path=sphere_path,
            sphere_vehicle=sphere_vehicle,
            tangent_error=tangent_error,
        )
        return raw.astype(np.float32), info

    def __call__(self, state: np.ndarray, path: Path2D) -> tuple[np.ndarray, EncoderInfo]:
        raw, info = self.raw(state, path)
        return self.field(raw), info

    def tangent_lines(self, state: np.ndarray, path: Path2D, length: float = 1.0) -> np.ndarray:
        """Return two line segments in R^3 that span the local sphere tangent plane."""
        _, info = self.raw(state, path)
        e1, e2 = tangent_basis(info.sphere_path)
        return np.stack(
            [
                info.sphere_path - length * e1,
                info.sphere_path + length * e1,
                info.sphere_path - length * e2,
                info.sphere_path + length * e2,
            ],
            axis=0,
        )
