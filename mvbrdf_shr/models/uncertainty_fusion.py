"""Single-image uncertainty fusion between physical and learned estimates."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyGuidedFusion(nn.Module):
    """Predict calibrated confidence, aleatoric uncertainty and a small residual."""

    def __init__(
        self,
        n_domains: int = 5,
        domain_dim: int = 8,
        hidden: int = 48,
        correction_limit: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_domains = n_domains
        self.domain_dim = domain_dim
        self.correction_limit = correction_limit
        self.domain_embedding = nn.Embedding(n_domains, domain_dim)
        # image, physical diffuse, restored diffuse, physical specular, masks
        in_channels = 3 + 3 + 3 + 3 + 1 + 1 + domain_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
        )
        self.gate_head = nn.Conv2d(hidden, 1, 1)
        self.logvar_head = nn.Conv2d(hidden, 1, 1)
        self.correction_head = nn.Conv2d(hidden, 3, 1)
        # Preserve the established restorer at initialization.
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -2.0)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def _domain_map(
        self,
        domain_ids: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = reference.shape
        if domain_ids is None:
            domain_ids = torch.zeros(batch, dtype=torch.long, device=reference.device)
        domain_ids = domain_ids.to(reference.device, dtype=torch.long).flatten()
        if len(domain_ids) == 1 and batch > 1:
            domain_ids = domain_ids.expand(batch)
        if len(domain_ids) != batch:
            raise ValueError("domain_ids must contain one id per image")
        embedding = self.domain_embedding(domain_ids.clamp(0, self.n_domains - 1))
        return embedding[:, :, None, None].expand(-1, -1, height, width)

    def forward(
        self,
        image: torch.Tensor,
        physical_diffuse: torch.Tensor,
        restored: torch.Tensor,
        physical_specular: torch.Tensor,
        physical_mask: torch.Tensor,
        learned_mask: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = torch.cat(
            [
                image,
                physical_diffuse,
                restored,
                physical_specular,
                physical_mask,
                learned_mask,
                self._domain_map(domain_ids, image),
            ],
            dim=1,
        )
        hidden = self.encoder(features)
        gate_logits = self.gate_head(hidden)
        physical_gate = torch.sigmoid(gate_logits)
        base = physical_gate * physical_diffuse + (1 - physical_gate) * restored
        ambiguity = 4 * physical_gate * (1 - physical_gate)
        support = torch.maximum(physical_mask, learned_mask)
        correction = (
            torch.tanh(self.correction_head(hidden))
            * ambiguity
            * support
            * self.correction_limit
        )
        prediction = (base + correction).clamp(0, 1)
        log_variance = self.logvar_head(hidden).clamp(-6, 2)
        uncertainty = F.softplus(log_variance)
        return {
            "pred": prediction,
            "restored_pred": restored,
            "physical_gate": physical_gate,
            "physical_gate_logits": gate_logits,
            "uncertainty": uncertainty,
            "log_variance": log_variance,
            "fusion_correction": correction,
        }
