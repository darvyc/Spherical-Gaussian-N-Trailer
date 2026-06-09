from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .utils import read_json, wrap_angle


@dataclass(frozen=True)
class PathQuery:
    point: np.ndarray
    tangent: np.ndarray
    heading: float
    curvature: float
    signed_error: float
    distance: float
    progress: float


class Path2D:
    """Arc-length polyline path with nearest-segment queries."""

    def __init__(self, waypoints: np.ndarray):
        pts = np.asarray(waypoints, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("waypoints must be an array of shape [M, 2] with M >= 2")
        self.points = pts
        diffs = np.diff(pts, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        if np.any(lengths <= 1e-9):
            raise ValueError("consecutive waypoints must be distinct")
        self.segment_vectors = diffs
        self.segment_lengths = lengths
        self.segment_tangents = diffs / lengths[:, None]
        self.s = np.concatenate([[0.0], np.cumsum(lengths)])
        self.total_length = float(self.s[-1])
        self.segment_headings = np.arctan2(self.segment_tangents[:, 1], self.segment_tangents[:, 0])
        self.segment_curvatures = self._estimate_segment_curvature()

    @classmethod
    def from_waypoints(cls, waypoints: Iterable[Sequence[float]]) -> "Path2D":
        return cls(np.asarray(list(waypoints), dtype=float))

    @classmethod
    def from_json(cls, path: str | Path) -> "Path2D":
        payload = read_json(path)
        return cls.from_waypoints(payload["waypoints"])

    def to_json_dict(self) -> dict[str, list[list[float]]]:
        return {"waypoints": self.points.tolist()}

    def _estimate_segment_curvature(self) -> np.ndarray:
        if len(self.segment_headings) == 1:
            return np.zeros(1, dtype=float)
        headings = np.unwrap(self.segment_headings)
        curv = np.zeros_like(headings)
        for i in range(len(headings)):
            if i == 0:
                ds = 0.5 * (self.segment_lengths[i] + self.segment_lengths[i + 1])
                curv[i] = (headings[i + 1] - headings[i]) / max(ds, 1e-9)
            elif i == len(headings) - 1:
                ds = 0.5 * (self.segment_lengths[i - 1] + self.segment_lengths[i])
                curv[i] = (headings[i] - headings[i - 1]) / max(ds, 1e-9)
            else:
                ds = 0.5 * (self.segment_lengths[i - 1] + self.segment_lengths[i + 1])
                curv[i] = (headings[i + 1] - headings[i - 1]) / max(ds, 1e-9)
        return curv

    def sample(self, progress: float) -> tuple[np.ndarray, float, float]:
        progress = float(np.clip(progress, 0.0, self.total_length))
        idx = int(np.searchsorted(self.s, progress, side="right") - 1)
        idx = min(max(idx, 0), len(self.segment_lengths) - 1)
        local = (progress - self.s[idx]) / self.segment_lengths[idx]
        point = self.points[idx] + local * self.segment_vectors[idx]
        return point, float(self.segment_headings[idx]), float(self.segment_curvatures[idx])

    def nearest(self, position: np.ndarray) -> PathQuery:
        p = np.asarray(position, dtype=float).reshape(2)
        best = None
        for i, (a, d, length, tangent) in enumerate(
            zip(self.points[:-1], self.segment_vectors, self.segment_lengths, self.segment_tangents)
        ):
            t = float(np.clip(np.dot(p - a, d) / (length * length), 0.0, 1.0))
            proj = a + t * d
            delta = p - proj
            dist = float(np.linalg.norm(delta))
            normal = np.array([-tangent[1], tangent[0]])
            signed_error = float(np.dot(delta, normal))
            progress = float(self.s[i] + t * length)
            candidate = (dist, signed_error, progress, proj, tangent, i)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        dist, signed_error, progress, proj, tangent, i = best
        heading = float(np.arctan2(tangent[1], tangent[0]))
        return PathQuery(
            point=proj,
            tangent=tangent,
            heading=heading,
            curvature=float(self.segment_curvatures[i]),
            signed_error=signed_error,
            distance=dist,
            progress=progress,
        )

    def heading_error(self, theta: float, position: np.ndarray) -> float:
        return float(wrap_angle(theta - self.nearest(position).heading))
