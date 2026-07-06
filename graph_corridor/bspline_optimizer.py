"""
Graph-guided path repair and smoothing for M3.

M3 now does two things only:
1. Repair path segments that come within robot_radius of occupied Gaussians,
   by inserting nearby free nodes as detour anchors.
2. Smooth the repaired path into a continuous, curvature-friendly trajectory
   while staying close to the repaired M2 path.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

from utils.planning_repair import compute_segment_min_distance, segment_collides


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    if hasattr(value, 'cpu'):
        try:
            return value.cpu().numpy()
        except Exception:
            pass
    return np.asarray(value)


class GraphSplineOptimizer:
    """M2-guided repair and smoothing optimizer for M3."""

    def __init__(
        self,
        spline_deg: int = 6,
        n_sec: int = 10,
        device: str = 'cpu',
        detour_search_radius: float = 0.12,
        detour_min_clearance: float = 0.01,
        smoothing_iterations: int = 20,
        smooth_weight: float = 0.28,
        curvature_weight: float = 0.10,
        reference_weight: float = 0.55,
        dense_samples: int = 120,
    ):
        self.spline_deg = int(spline_deg)
        self.n_sec = int(n_sec)
        self.device = device
        self.detour_search_radius = float(detour_search_radius)
        self.detour_min_clearance = float(detour_min_clearance)
        self.smoothing_iterations = int(smoothing_iterations)
        self.smooth_weight = float(smooth_weight)
        self.curvature_weight = float(curvature_weight)
        self.reference_weight = float(reference_weight)
        self.dense_samples = int(dense_samples)
        self.coeffs = None
        self._traj = None
        self.last_timing_ms = {}

    def optimize(
        self,
        corridors: List[Tuple[torch.Tensor, torch.Tensor]],
        x0: torch.Tensor,
        xf: torch.Tensor,
        path_coords: Optional[torch.Tensor] = None,
        occupied_positions: Optional[torch.Tensor] = None,
        robot_radius: float = 0.02,
        free_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, bool]:
        """Repair and smooth the M2 path into a collision-aware trajectory."""
        del corridors  # M3 no longer uses corridor QP.

        t_total = time.perf_counter()
        timings = {}

        def _mark(key: str, start_t: float):
            timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start_t) * 1000.0

        t = time.perf_counter()
        path = _to_numpy(path_coords)
        if path is None:
            path = np.stack([
                _to_numpy(x0).reshape(3),
                _to_numpy(xf).reshape(3),
            ], axis=0)
        path = np.asarray(path, dtype=np.float32)
        if path.ndim != 2 or path.shape[0] < 2:
            timings['total_ms'] = (time.perf_counter() - t_total) * 1000.0
            self.last_timing_ms = timings
            return self._strict_failure()
        if path.shape[1] > 3:
            path = path[:, :3]
        _mark('input_parse_ms', t)

        t = time.perf_counter()
        x0_np = np.asarray(_to_numpy(x0), dtype=np.float32).reshape(3)
        xf_np = np.asarray(_to_numpy(xf), dtype=np.float32).reshape(3)
        ref_path = self._drop_duplicates(self._force_endpoints(path.copy(), x0_np, xf_np))
        _mark('reference_path_ms', t)

        t = time.perf_counter()
        occupied = _to_numpy(occupied_positions)
        if occupied is not None:
            occupied = np.asarray(occupied, dtype=np.float32).reshape(-1, 3)
        free_pts = _to_numpy(free_positions)
        if free_pts is not None:
            free_pts = np.asarray(free_pts, dtype=np.float32).reshape(-1, 3)
        _mark('input_tensor_ms', t)

        t = time.perf_counter()
        repaired = self._repair_path_with_free_nodes(ref_path, free_pts, occupied, robot_radius)
        _mark('repair_ms', t)
        if repaired is None or len(repaired) < 2:
            timings['total_ms'] = (time.perf_counter() - t_total) * 1000.0
            self.last_timing_ms = timings
            return self._strict_failure()

        # Smooth the repaired path but keep it anchored to the repaired reference.
        t = time.perf_counter()
        smoothed = self._smooth_path(
            repaired,
            reference_path=repaired,
            occupied=occupied,
            robot_radius=robot_radius,
        )
        _mark('smooth_ms', t)

        candidates = [smoothed, repaired]
        t = time.perf_counter()
        for candidate in candidates:
            traj = self._resample_path(candidate, num_samples=max(self.dense_samples, len(candidate) * 12))
            if occupied is None or len(occupied) == 0 or not self._trajectory_collides(traj, occupied, robot_radius):
                self._traj = traj.astype(np.float32)
                self.coeffs = None
                _mark('candidate_eval_ms', t)
                timings['total_ms'] = (time.perf_counter() - t_total) * 1000.0
                self.last_timing_ms = timings
                return self._traj, True
        _mark('candidate_eval_ms', t)

        # Last pass: slightly tighter smoothing, but still anchored to repaired path.
        t = time.perf_counter()
        conservative = self._smooth_path(
            repaired,
            reference_path=repaired,
            occupied=occupied,
            robot_radius=robot_radius,
            conservative=True,
        )
        _mark('conservative_smooth_ms', t)
        t = time.perf_counter()
        traj = self._resample_path(conservative, num_samples=max(self.dense_samples, len(conservative) * 12))
        if occupied is None or len(occupied) == 0 or not self._trajectory_collides(traj, occupied, robot_radius):
            self._traj = traj.astype(np.float32)
            self.coeffs = None
            _mark('fallback_candidate_eval_ms', t)
            timings['total_ms'] = (time.perf_counter() - t_total) * 1000.0
            self.last_timing_ms = timings
            return self._traj, True
        _mark('fallback_candidate_eval_ms', t)

        timings['total_ms'] = (time.perf_counter() - t_total) * 1000.0
        self.last_timing_ms = timings
        return self._strict_failure()

    def _strict_failure(self) -> Tuple[np.ndarray, bool]:
        self._traj = np.zeros((0, 3), dtype=np.float32)
        return self._traj, False

    def _force_endpoints(self, path: np.ndarray, x0: np.ndarray, xf: np.ndarray) -> np.ndarray:
        pts = np.asarray(path, dtype=np.float32)
        pts[0] = x0
        pts[-1] = xf
        return pts

    def _drop_duplicates(self, path: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        pts = np.asarray(path, dtype=np.float32)
        if len(pts) <= 1:
            return pts
        keep = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - keep[-1]) > eps:
                keep.append(p)
        if len(keep) == 1:
            keep.append(keep[0].copy())
        return np.asarray(keep, dtype=np.float32)

    def _segment_clearance(self, p0, p1, occupied) -> float:
        if occupied is None or len(occupied) == 0:
            return float('inf')
        return float(compute_segment_min_distance(p0, p1, occupied))

    def _segment_needs_repair(self, p0, p1, occupied, robot_radius: float) -> bool:
        # Trigger repair only when a segment is actually inside the robot footprint.
        return self._segment_clearance(p0, p1, occupied) < float(robot_radius)

    def _repair_path_with_free_nodes(
        self,
        path: np.ndarray,
        free_positions: Optional[np.ndarray],
        occupied: Optional[np.ndarray],
        robot_radius: float,
    ) -> Optional[np.ndarray]:
        pts = np.asarray(path, dtype=np.float32)
        if len(pts) < 2:
            return None

        repaired: List[np.ndarray] = [pts[0]]
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            if self._segment_needs_repair(p0, p1, occupied, robot_radius):
                detour = self._choose_detour_free_node(
                    p0, p1, free_positions, occupied, robot_radius
                )
                if detour is not None and np.linalg.norm(detour - repaired[-1]) > 1e-5:
                    repaired.append(detour.astype(np.float32))
            if np.linalg.norm(p1 - repaired[-1]) > 1e-5:
                repaired.append(p1)

        repaired = self._drop_duplicates(np.asarray(repaired, dtype=np.float32))
        if len(repaired) < 2:
            return None
        return repaired

    def _choose_detour_free_node(
        self,
        p0,
        p1,
        free_positions,
        occupied,
        robot_radius: float,
    ) -> Optional[np.ndarray]:
        if free_positions is None or len(free_positions) == 0:
            return None

        p0 = np.asarray(p0, dtype=np.float32).reshape(3)
        p1 = np.asarray(p1, dtype=np.float32).reshape(3)
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            return None
        seg_dir = seg / seg_len
        mid = 0.5 * (p0 + p1)

        free = np.asarray(free_positions, dtype=np.float32).reshape(-1, 3)
        free = free[np.isfinite(free).all(axis=1)]
        if free.shape[0] == 0:
            return None

        # Prefer nodes near the segment and near the midpoint, but only if they
        # remain on the free side of the obstacle field.
        proj = np.dot(free - p0[None, :], seg_dir) / max(seg_len, 1e-6)
        seg_mask = (proj >= -0.15) & (proj <= 1.15)
        if not np.any(seg_mask):
            seg_mask = np.ones(free.shape[0], dtype=bool)

        candidate_pool = free[seg_mask]
        mid_dists = np.linalg.norm(candidate_pool - mid[None, :], axis=1)
        search_radii = [
            max(self.detour_search_radius, 1.5 * robot_radius, 0.25 * seg_len),
            max(1.5 * self.detour_search_radius, 2.0 * robot_radius, 0.35 * seg_len),
            max(2.2 * self.detour_search_radius, 3.0 * robot_radius, 0.50 * seg_len),
        ]

        best = None
        best_score = -np.inf
        for radius in search_radii:
            local = candidate_pool[mid_dists <= radius]
            if local.shape[0] == 0:
                continue
            local_order = np.argsort(np.linalg.norm(local - mid[None, :], axis=1))
            for idx in local_order[:64]:
                cand = local[idx]
                if occupied is not None and len(occupied) > 0:
                    cand_clearance = float(np.min(np.linalg.norm(occupied - cand[None, :], axis=1)))
                else:
                    cand_clearance = float('inf')

                safe_clearance = float(robot_radius + self.detour_min_clearance)
                if cand_clearance < safe_clearance:
                    continue
                if segment_collides(p0, cand, occupied, robot_radius):
                    continue
                if segment_collides(cand, p1, occupied, robot_radius):
                    continue

                t = float(np.dot(cand - p0, seg_dir) / max(seg_len, 1e-6))
                if t < -0.2 or t > 1.2:
                    continue

                score = cand_clearance - 0.7 * float(np.linalg.norm(cand - mid)) - 0.3 * abs(t - 0.5) * seg_len
                if score > best_score:
                    best_score = score
                    best = cand
            if best is not None:
                return best.astype(np.float32)

        # Last resort: search globally for the best free node that still gives a collision-free detour.
        for idx in np.argsort(np.linalg.norm(free - mid[None, :], axis=1))[:128]:
            cand = free[idx]
            if occupied is not None and len(occupied) > 0:
                cand_clearance = float(np.min(np.linalg.norm(occupied - cand[None, :], axis=1)))
            else:
                cand_clearance = float('inf')
            if cand_clearance < float(robot_radius + self.detour_min_clearance):
                continue
            if segment_collides(p0, cand, occupied, robot_radius):
                continue
            if segment_collides(cand, p1, occupied, robot_radius):
                continue
            return cand.astype(np.float32)

        return None

    def _smooth_path(
        self,
        path: np.ndarray,
        occupied: Optional[np.ndarray],
        robot_radius: float,
        reference_path: Optional[np.ndarray] = None,
        conservative: bool = False,
    ) -> np.ndarray:
        pts = np.asarray(path, dtype=np.float32)
        if len(pts) < 3:
            return pts

        smooth_w = self.smooth_weight * (0.7 if conservative else 1.0)
        curve_w = self.curvature_weight * (0.8 if conservative else 1.0)
        ref_w = self.reference_weight * (0.8 if conservative else 1.0)

        ref = pts if reference_path is None else np.asarray(reference_path, dtype=np.float32)
        if len(ref) != len(pts):
            ref = self._resample_path(ref, num_samples=len(pts))
        if len(ref) != len(pts):
            ref = np.repeat(ref[:1], len(pts), axis=0)

        current = pts.copy()
        for _ in range(self.smoothing_iterations):
            updated = current.copy()
            for i in range(1, len(current) - 1):
                p = current[i]
                prev_p = current[i - 1]
                next_p = current[i + 1]
                ref_p = ref[i]

                lap = 0.5 * (prev_p + next_p) - p
                curvature = prev_p - 2.0 * p + next_p
                candidate = p + smooth_w * lap + curve_w * curvature + ref_w * (ref_p - p)

                # Keep motion local so the curve stays tied to M2 instead of spiraling.
                local_scale = max(
                    0.01,
                    0.30 * min(
                        float(np.linalg.norm(p - prev_p)),
                        float(np.linalg.norm(next_p - p)),
                        float(np.linalg.norm(ref_p - p) + 1e-6),
                    ),
                )
                delta = candidate - p
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm > local_scale:
                    candidate = p + delta * (local_scale / max(delta_norm, 1e-8))

                trial = current.copy()
                trial[i] = candidate
                if occupied is not None and len(occupied) > 0:
                    if self._segment_clearance(trial[i - 1], trial[i], occupied) < float(robot_radius):
                        continue
                    if self._segment_clearance(trial[i], trial[i + 1], occupied) < float(robot_radius):
                        continue
                updated[i] = candidate

            updated[0] = current[0]
            updated[-1] = current[-1]
            if np.max(np.linalg.norm(updated - current, axis=1)) < 1e-4:
                current = updated
                break
            current = updated
        return self._drop_duplicates(current)

    def _resample_path(self, path: np.ndarray, num_samples: int = 120) -> np.ndarray:
        pts = np.asarray(path, dtype=np.float32)
        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if len(pts) == 1:
            return np.repeat(pts, num_samples, axis=0)
        if len(pts) == 2:
            ts = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
            return (pts[0][None, :] * (1.0 - ts[:, None]) + pts[1][None, :] * ts[:, None]).astype(np.float32)

        segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(np.sum(segs))
        if total < 1e-8:
            return np.repeat(pts[:1], num_samples, axis=0).astype(np.float32)
        t = np.concatenate([[0.0], np.cumsum(segs) / total])
        ts = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)

        try:
            pchip = PchipInterpolator(t, pts, axis=0)
            dense = np.asarray(pchip(ts), dtype=np.float32)
        except Exception:
            seg_idx = np.searchsorted(t, ts, side='right') - 1
            seg_idx = np.clip(seg_idx, 0, len(t) - 2)
            t0 = t[seg_idx]
            t1 = t[seg_idx + 1]
            alpha = (ts - t0) / np.maximum(t1 - t0, 1e-6)
            dense = pts[seg_idx] * (1.0 - alpha[:, None]) + pts[seg_idx + 1] * alpha[:, None]
        dense[0] = pts[0]
        dense[-1] = pts[-1]
        return dense.astype(np.float32)

    def _trajectory_collides(self, traj_xyz: np.ndarray, occupied_positions, robot_radius: float) -> bool:
        occ = _to_numpy(occupied_positions)
        if occ is None or len(occ) == 0 or traj_xyz.shape[0] < 2:
            return False
        occ = np.asarray(occ, dtype=np.float32).reshape(-1, 3)
        for i in range(traj_xyz.shape[0] - 1):
            if segment_collides(traj_xyz[i], traj_xyz[i + 1], occ, robot_radius):
                return True
        return False

    def get_spline_eval(self) -> Optional[np.ndarray]:
        return self._traj
