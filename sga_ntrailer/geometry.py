from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def sphere_point(heading: float, z_value: float) -> np.ndarray:
    """Lift a heading and bounded scalar into S^2."""
    return normalize(np.array([np.cos(heading), np.sin(heading), z_value], dtype=float))


def tangent_basis(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic orthonormal basis for the tangent plane at q in S^2."""
    q = normalize(q.reshape(3))
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(q, ref))) > 0.92:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = ref - np.dot(ref, q) * q
    e1 = normalize(e1)
    e2 = normalize(np.cross(q, e1))
    return e1, e2


def log_map_s2(base: np.ndarray, point: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Logarithmic map from base to point on the unit sphere."""
    base = normalize(base.reshape(3))
    point = normalize(point.reshape(3))
    cos_omega = float(np.clip(np.dot(base, point), -1.0, 1.0))
    omega = float(np.arccos(cos_omega))
    tangent = point - cos_omega * base
    sin_omega = float(np.sin(omega))
    if abs(sin_omega) < eps:
        return tangent
    return omega * tangent / sin_omega


def exp_map_s2(base: np.ndarray, tangent: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Exponential map on the unit sphere."""
    base = normalize(base.reshape(3))
    tangent = np.asarray(tangent, dtype=float).reshape(3)
    norm_t = float(np.linalg.norm(tangent))
    if norm_t < eps:
        return base
    return normalize(np.cos(norm_t) * base + np.sin(norm_t) * tangent / norm_t)


def tangent_coordinates(base: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Two tangent-plane coordinates of point relative to base."""
    v = log_map_s2(base, point)
    e1, e2 = tangent_basis(base)
    return np.array([float(np.dot(v, e1)), float(np.dot(v, e2))], dtype=float)


def geodesic_arc(base: np.ndarray, point: np.ndarray, n: int = 16) -> np.ndarray:
    """Sample n points on the short geodesic arc between base and point."""
    base = normalize(base.reshape(3))
    point = normalize(point.reshape(3))
    tangent = log_map_s2(base, point)
    return np.stack([exp_map_s2(base, t * tangent) for t in np.linspace(0.0, 1.0, n)], axis=0)
