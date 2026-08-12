"""Optional hash-grid encoder for future world-space material queries."""
from __future__ import annotations

import torch
import torch.nn as nn


class HashGridEncoder(nn.Module):
    """Lightweight multi-resolution hash features (screen or xyz).

    Not Instant-NGP exact; sufficient as a drop-in positional encoder.
    """

    def __init__(self, in_dim: int = 3, n_levels: int = 4, features: int = 2, table_size: int = 2**14):
        super().__init__()
        self.n_levels = n_levels
        self.features = features
        self.table_size = table_size
        self.tables = nn.ParameterList(
            [nn.Parameter(torch.randn(table_size, features) * 0.01) for _ in range(n_levels)]
        )
        self.resolutions = [16 * (2**i) for i in range(n_levels)]
        self.out_dim = n_levels * features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_dim) in roughly [-1,1]."""
        feats = []
        flat = x.reshape(-1, x.shape[-1])
        for level, table in enumerate(self.tables):
            res = self.resolutions[level]
            scaled = (flat * 0.5 + 0.5) * res
            idx = scaled.long().clamp(min=0)
            # hash mix
            if idx.shape[-1] >= 3:
                h = (idx[..., 0] * 73856093) ^ (idx[..., 1] * 19349663) ^ (idx[..., 2] * 83492791)
            else:
                h = idx[..., 0] * 73856093
            h = h.abs() % self.table_size
            feats.append(table[h])
        out = torch.cat(feats, dim=-1)
        return out.reshape(*x.shape[:-1], self.out_dim)


class HashMaterialMLP(nn.Module):
    """Query material coeffs from encoded 3D/2D coordinates (+ optional image feat)."""

    def __init__(self, enc: HashGridEncoder | None = None, hidden: int = 64):
        super().__init__()
        self.enc = enc or HashGridEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(self.enc.out_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 7),  # albedo3 + spec + rough + metal + abs
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.enc(coords))
