"""
GraphVoxelizer: voxelization for graph construction.
"""

import numpy as np
import torch
from typing import Optional, Tuple

from .density_field import DensityFieldEstimator


class GraphVoxelizer:
    def __init__(self, voxel_size: float = 0.1, robot_radius: float = 0.02, device: str = 'cuda',
                 lower_bound: tuple = None, upper_bound: tuple = None):
        self.voxel_size = voxel_size
        self.robot_radius = robot_radius
        self.device = device
        self.density_estimator = DensityFieldEstimator(device=device)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def voxelize(self, gsplat) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        means = torch.as_tensor(gsplat.means, dtype=torch.float32, device=self.device)
        covs_inv = torch.as_tensor(gsplat.covs_inv, dtype=torch.float32, device=self.device)
        covs = torch.as_tensor(gsplat.covs, dtype=torch.float32, device=self.device)
        raw_opacities = getattr(gsplat, 'opacities', None)
        if raw_opacities is None:
            raw_opacities = np.ones(len(gsplat.means), dtype=np.float32)
        opacities = torch.as_tensor(raw_opacities, dtype=torch.float32, device=self.device).reshape(-1)

        bounds, resolution = self._compute_bounds(means)
        self.bounds = bounds
        self.resolution = resolution

        voxel_centers, cell_sizes = self._make_grid_centers(bounds, resolution)
        voxel_centers_flat = voxel_centers.reshape(-1, 3)
        v_total = int(resolution[0] * resolution[1] * resolution[2])

        print(f'[GraphVoxelizer] Grid: {resolution.tolist()}, Voxels: {v_total}, Cell size: {cell_sizes.tolist()}')
        print('[GraphVoxelizer] Estimating density field...')
        density_flat = self.density_estimator.estimate(
            means, covs_inv, covs, opacities, voxel_centers_flat
        )
        density_grid = density_flat.reshape(resolution[0], resolution[1], resolution[2])

        metadata = {
            'bounds': bounds,
            'resolution': resolution,
            'cell_sizes': cell_sizes,
            'voxel_size': self.voxel_size,
            'robot_radius': self.robot_radius,
            'num_voxels': v_total,
        }
        return density_grid, voxel_centers, metadata

    def _compute_bounds(self, means: torch.Tensor, margin: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.lower_bound is not None and self.upper_bound is not None:
            lower = torch.as_tensor(self.lower_bound, dtype=torch.float32)
            upper = torch.as_tensor(self.upper_bound, dtype=torch.float32)
        else:
            if margin is None:
                margin = 2 * self.voxel_size
            lower = means.min(dim=0).values - margin
            upper = means.max(dim=0).values + margin
        extent = upper - lower
        resolution = torch.ceil(extent / self.voxel_size).long()
        resolution = torch.clamp(resolution, min=10)
        bounds = torch.stack([lower, upper], dim=0)
        return bounds, resolution

    def _make_grid_centers(self, bounds: torch.Tensor, resolution: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lower = bounds[0]
        upper = bounds[1]
        cell_sizes = (upper - lower) / resolution

        x = torch.linspace(lower[0] + cell_sizes[0] / 2, upper[0] - cell_sizes[0] / 2, int(resolution[0]), device=self.device)
        y = torch.linspace(lower[1] + cell_sizes[1] / 2, upper[1] - cell_sizes[1] / 2, int(resolution[1]), device=self.device)
        z = torch.linspace(lower[2] + cell_sizes[2] / 2, upper[2] - cell_sizes[2] / 2, int(resolution[2]), device=self.device)

        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        voxel_centers = torch.stack([X, Y, Z], dim=-1)
        return voxel_centers, cell_sizes

    def get_densities_at_points(self, gsplat, points: torch.Tensor) -> torch.Tensor:
        return self.density_estimator.estimate(
            gsplat.means, gsplat.covs_inv, gsplat.covs, gsplat.opacities,
            points, build_hash=False
        )

    def get_gradients_at_points(self, gsplat, points: torch.Tensor) -> torch.Tensor:
        return self.density_estimator.estimate_gradient(
            gsplat.means, gsplat.covs_inv, gsplat.covs, gsplat.opacities,
            points
        )
