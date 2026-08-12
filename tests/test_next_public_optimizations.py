from __future__ import annotations

import torch
from torch.utils.data import Dataset

from mvbrdf_shr.data.multidataset import DomainDataset, balanced_concat
from mvbrdf_shr.losses_pro import LossWeights, total_loss
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro
from mvbrdf_shr.models.uncertainty_fusion import UncertaintyGuidedFusion
from mvbrdf_shr.types import Batch


class _TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        image = torch.rand(3, 16, 16)
        clean = image * 0.8
        specular = image - clean
        return Batch(
            image=image,
            image_clean=clean,
            specular_gt=specular,
            mask_gt=(specular.mean(0, keepdim=True) > 0.02).float(),
        )


def test_balanced_dataset_attaches_domain():
    first = DomainDataset(_TinyDataset(), "shiq")
    second = DomainDataset(_TinyDataset(), "psd")
    combined, sampler = balanced_concat([first, second], samples_per_epoch=8)
    assert len(combined) == 4
    assert len(list(sampler)) == 8
    assert first[0].domain_ids.item() == 0
    assert second[0].domain_ids.item() == 3


def test_uncertainty_fusion_shapes_and_bounds():
    module = UncertaintyGuidedFusion()
    rgb = torch.rand(2, 3, 16, 16)
    mask = torch.rand(2, 1, 16, 16)
    result = module(rgb, rgb * 0.8, rgb * 0.7, rgb * 0.2, mask, mask, torch.tensor([0, 3]))
    assert result["pred"].shape == rgb.shape
    assert result["uncertainty"].shape == mask.shape
    assert result["physical_gate"].min() >= 0
    assert result["physical_gate"].max() <= 1
    assert result["fusion_correction"].abs().max() <= 0.05


def test_pipeline_and_extended_losses_backward():
    model = MVBRDFSHRPro(
        resolution=32,
        base_ch=8,
        restorer_base=8,
        use_uncertainty_fusion=True,
    )
    image = torch.rand(1, 3, 32, 32)
    clean = image * 0.8
    specular = image - clean
    mask = (specular.mean(1, keepdim=True) > 0.02).float()
    output = model(image, domain_ids=torch.tensor([1]))
    loss, logs = total_loss(
        output,
        image_clean=clean,
        mask_gt=mask,
        specular_gt=specular,
        weights=LossWeights(),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert "uncertainty_nll" in logs
    assert output["pred"].shape == image.shape
