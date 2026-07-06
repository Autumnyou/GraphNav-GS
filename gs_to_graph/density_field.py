"""
Density Field Estimation from Gaussian Splatting.
Estimates a continuous 3D density field from discrete Gaussian primitives.

Core formula:
    ρ(p) = Σ_i α_i · N(p; μ_i, Σ_i)
    where α_i = opacity_i, N is multivariate Gaussian PDF

Performance note:
    This implementation uses direct GPU batch computation instead of spatial
    hashing.  For typical 3DGS scenes the Gaussian covariances are very large
    relative to the scene extent, making spatial hashing degenerate (every
    Gaussian maps to all cells).  Batch matmul on GPU is much faster.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List


class DensityFieldEstimator:
    """
    Estimates a continuous 3D density field from Gaussian splatting primitives.
    Uses direct batch GPU computation with chunking to avoid OOM.
    """

    def __init__(self, truncation_sigma: float = 3.0, chunk_size: int = 50000,
                 hash_cell_size: float = 0.3, device: str = 'cuda'):
        """
        Args:
            truncation_sigma: Truncation threshold in standard deviations.
            chunk_size: Number of voxels processed per chunk to avoid OOM.
            hash_cell_size: (unused, kept for API compat) Spatial hash grid cell size.
            device: Compute device.
        """
        self.truncation_sigma = truncation_sigma
        self.chunk_size = chunk_size
        self.hash_cell_size = hash_cell_size
        self.device = device

        # No spatial hash needed — direct batch computation is faster for
        # typical 3DGS data where Gaussian radii are much larger than voxel size.
        self.spatial_hash = None

    def build_spatial_hash(self, means: torch.Tensor, covs: torch.Tensor):
        """No-op kept for API compatibility."""
        pass

    @torch.no_grad()
    def estimate(self, means: torch.Tensor, covs_inv: torch.Tensor,
                 covs: torch.Tensor, opacities: torch.Tensor,
                 voxel_centers: torch.Tensor,
                 build_hash: bool = True) -> torch.Tensor:
        """
        Estimate the density field at the given voxel centers.

        Uses fully-vectorised batch Mahalanobis distance on GPU:
          For each chunk of voxels (V_chunk) and each chunk of Gaussians (N_chunk):
            diff = voxels[:, None, :] - means[None, :, :]   # (V_chunk, N_chunk, 3)
            mahal = einsum('vni,nij,vnj->vn', diff, covs_inv, diff)
            density contribution = sum of valid kernel * weight

        Args:
            means: (N, 3) Gaussian centers.
            covs_inv: (N, 3, 3) Inverse covariance matrices.
            covs: (N, 3, 3) Covariance matrices.
            opacities: (N,) Gaussian opacities.
            voxel_centers: (V, 3) Query voxel center positions.
            build_hash: (unused, kept for API compat).

        Returns:
            density: (V,) Density value at each voxel center.
        """
        means = torch.as_tensor(means, dtype=torch.float32, device=self.device)
        covs_inv = torch.as_tensor(covs_inv, dtype=torch.float32, device=self.device)
        covs = torch.as_tensor(covs, dtype=torch.float32, device=self.device)
        opacities = torch.as_tensor(opacities, dtype=torch.float32, device=self.device).reshape(-1)
        voxel_centers = torch.as_tensor(voxel_centers, dtype=torch.float32, device=self.device)

        N = means.shape[0]
        V = voxel_centers.shape[0]

        # Normalisation constant: sqrt((2π)^3 |Σ|)
        dets = torch.det(covs)  # (N,)
        # Clamp determinants to avoid numerical issues with very small covariances
        dets = dets.clamp(min=1e-12)
        normalizer = torch.sqrt((2 * np.pi) ** 3 * dets)  # (N,)

        # Pre-compute weights = opacity / normalizer for all gaussians
        # Clamp weights to prevent overflow when covariances are very small
        weights = (opacities / normalizer).clamp(max=1e6)  # (N,)

        # Truncation threshold (squared Mahalanobis distance)
        trunc_threshold = self.truncation_sigma ** 2

        density = torch.zeros(V, dtype=torch.float32, device=self.device)

        # Determine chunk sizes to fit in GPU memory.
        # Memory per pair: diff(3 float) + mahal(1 float) ~ 16 bytes
        # Budget: ~1 GB -> ~64M pairs -> chunk_v * N <= 64M
        max_pairs = 64_000_000
        chunk_v = max(1, min(V, max_pairs // max(N, 1)))

        for v_start in range(0, V, chunk_v):
            v_end = min(v_start + chunk_v, V)
            vc = voxel_centers[v_start:v_end]  # (Vc, 3)
            Vc = vc.shape[0]

            # We may also need to chunk over N if N is very large
            chunk_n = max(1, min(N, max_pairs // max(Vc, 1)))

            for n_start in range(0, N, chunk_n):
                n_end = min(n_start + chunk_n, N)

                m = means[n_start:n_end]      # (Nc, 3)
                ci = covs_inv[n_start:n_end]   # (Nc, 3, 3)
                w = weights[n_start:n_end]     # (Nc,)

                # diff: (Vc, Nc, 3)
                diff = vc.unsqueeze(1) - m.unsqueeze(0)

                # Batch Mahalanobis distance: mahal[v,n] = diff[v,n]^T @ ci[n] @ diff[v,n]
                # Use two-step matmul to avoid einsum overhead on large tensors:
                #   tmp = diff @ ci  -> (Vc, Nc, 3)
                #   mahal = (tmp * diff).sum(-1)  -> (Vc, Nc)
                tmp = torch.einsum('vni,nij->vnj', diff, ci)  # (Vc, Nc, 3)
                mahal = (tmp * diff).sum(dim=-1)               # (Vc, Nc)

                # Apply truncation and compute kernel
                valid_mask = mahal < trunc_threshold  # (Vc, Nc)
                # Zero out invalid entries for the kernel computation
                mahal_clamped = torch.where(valid_mask, mahal, torch.zeros_like(mahal))
                kernel = torch.exp(-0.5 * mahal_clamped)  # (Vc, Nc)
                kernel = kernel * valid_mask.float()       # zero out invalid

                # Weighted sum: density += sum_n( weight_n * kernel_vn )
                density[v_start:v_end] += (kernel * w.unsqueeze(0)).sum(dim=1)

        # Clamp final density to prevent numerical overflow
        density = torch.clamp(density, max=1e6)

        return density

    @torch.no_grad()
    def estimate_gradient(self, means: torch.Tensor, covs_inv: torch.Tensor,
                          covs: torch.Tensor, opacities: torch.Tensor,
                          voxel_centers: torch.Tensor) -> torch.Tensor:
        """
        Estimate the gradient of the density field (used for curvature computation).

        Args:
            means: (N, 3) Gaussian centers.
            covs_inv: (N, 3, 3) Inverse covariance matrices.
            covs: (N, 3, 3) Covariance matrices.
            opacities: (N,) Gaussian opacities.
            voxel_centers: (V, 3) Query voxel center positions.

        Returns:
            gradient: (V, 3) Density gradient at each voxel center.
        """
        means = torch.as_tensor(means, dtype=torch.float32, device=self.device)
        covs_inv = torch.as_tensor(covs_inv, dtype=torch.float32, device=self.device)
        covs = torch.as_tensor(covs, dtype=torch.float32, device=self.device)
        opacities = torch.as_tensor(opacities, dtype=torch.float32, device=self.device).reshape(-1)
        voxel_centers = torch.as_tensor(voxel_centers, dtype=torch.float32, device=self.device)

        N = means.shape[0]
        V = voxel_centers.shape[0]

        dets = torch.det(covs)
        normalizer = torch.sqrt((2 * np.pi) ** 3 * dets.clamp(min=1e-30)).to(self.device)
        weights = opacities / normalizer  # (N,)

        trunc_threshold = self.truncation_sigma ** 2

        gradient = torch.zeros(V, 3, dtype=torch.float32, device=self.device)

        # Memory budget for gradient: need diff(V,N,3) + mahal(V,N) + neg_grad(V,N,3) ~ 28 bytes/pair
        max_pairs = 32_000_000
        chunk_v = max(1, min(V, max_pairs // max(N, 1)))

        for v_start in range(0, V, chunk_v):
            v_end = min(v_start + chunk_v, V)
            vc = voxel_centers[v_start:v_end]  # (Vc, 3)
            Vc = vc.shape[0]

            chunk_n = max(1, min(N, max_pairs // max(Vc, 1)))

            for n_start in range(0, N, chunk_n):
                n_end = min(n_start + chunk_n, N)

                m = means[n_start:n_end]
                ci = covs_inv[n_start:n_end]
                w = weights[n_start:n_end]

                # diff: (Vc, Nc, 3)
                diff = vc.unsqueeze(1) - m.unsqueeze(0)

                # Mahalanobis distance
                tmp = torch.einsum('vni,nij->vnj', diff, ci)
                mahal = (tmp * diff).sum(dim=-1)  # (Vc, Nc)

                valid_mask = mahal < trunc_threshold
                mahal_clamped = torch.where(valid_mask, mahal, torch.zeros_like(mahal))
                kernel = torch.exp(-0.5 * mahal_clamped) * valid_mask.float()

                # Gradient: ∇ρ = Σ_i w_i · kernel · (-Σ_i^{-1} · diff_i)
                # neg_grad = -ci @ diff -> (Vc, Nc, 3)   (already computed as -tmp)
                neg_grad = -tmp  # (Vc, Nc, 3)

                # Weighted contribution: (Vc, Nc, 1) * (Vc, Nc, 3) -> sum over Nc
                coeff = (w.unsqueeze(0) * kernel).unsqueeze(-1)  # (Vc, Nc, 1)
                gradient[v_start:v_end] += (coeff * neg_grad).sum(dim=1)  # (Vc, 3)

        return gradient
