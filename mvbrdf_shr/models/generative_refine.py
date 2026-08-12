"""Optional generative refinement.

Backends:
  tiny     — TinyRefineUNet (default, no weights)
  sdturbo  — SD-Turbo img2img guided by physics diffuse (needs diffusers)
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyRefineUNet(nn.Module):
    """Lightweight residual refiner conditioned on physics buffers."""

    def __init__(self, in_ch: int = 10, base: int = 32):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.SiLU(),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.out = nn.Sequential(
            nn.Conv2d(base * 2 + base, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, 3, 1),
        )

    def forward(
        self,
        image: torch.Tensor | None = None,
        diffuse: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        normal: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond is None:
            assert image is not None and diffuse is not None
            assert mask is not None and normal is not None
            cond = torch.cat([image, diffuse, mask, normal], dim=1)
        d1 = self.down1(cond)
        d2 = self.down2(d1)
        m = self.mid(d2)
        x = torch.cat([self.up(m), d1], dim=1)
        residual = self.out(x)
        diff = cond[:, 3:6]
        msk = cond[:, 6:7]
        return (diff + msk * residual).clamp(0.0, 1.0)


class SDTurboRefiner(nn.Module):
    """SD-Turbo img2img: start from physics diffuse, lightly denoise toward clean.

    Training: still use TinyRefineUNet as a trainable LoRA-style stand-in when
    full fine-tuning is off; inference can call the frozen pipeline.
    Set MVBRDF_SDTURBO_TRAINABLE=1 to keep a trainable residual on top.
    """

    def __init__(self):
        super().__init__()
        self.trainable_head = TinyRefineUNet()
        self.pipe = None
        self._try_load_pipe()

    def _try_load_pipe(self) -> None:
        try:
            from diffusers import AutoPipelineForImage2Image

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sd-turbo",
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
            )
            if torch.cuda.is_available():
                self.pipe.to("cuda")
            self.pipe.set_progress_bar_config(disable=True)
            print("[GenerativeRefiner] loaded stabilityai/sd-turbo")
        except Exception as e:  # noqa: BLE001 — optional dependency
            print(f"[GenerativeRefiner] SD-Turbo unavailable ({e}); using trainable head only")
            self.pipe = None

    @torch.no_grad()
    def _pipe_refine(
        self,
        image: torch.Tensor,
        diffuse: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.pipe is None:
            return None
        from PIL import Image
        import numpy as np

        B = diffuse.shape[0]
        outs = []
        prompt = "clean diffuse surface photo, no specular highlight, natural texture"
        for b in range(B):
            d = diffuse[b].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            pil = Image.fromarray((d * 255).astype(np.uint8))
            # strength low: keep physics structure
            result = self.pipe(
                prompt=prompt,
                image=pil,
                num_inference_steps=2,
                guidance_scale=0.0,
                strength=0.35,
            ).images[0]
            arr = np.asarray(result).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).to(diffuse.device, diffuse.dtype)
            t = F.interpolate(
                t.unsqueeze(0), size=diffuse.shape[-2:], mode="bilinear", align_corners=False
            )[0]
            # Blend: trust SD only inside highlight mask
            m = mask[b]
            blended = diffuse[b] * (1 - m) + t * m
            outs.append(blended)
        return torch.stack(outs, dim=0)

    def forward(
        self,
        image: torch.Tensor,
        diffuse: torch.Tensor,
        mask: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        cond = torch.cat([image, diffuse, mask, normal], dim=1)
        head_out = self.trainable_head(cond)
        if not self.training and self.pipe is not None:
            piped = self._pipe_refine(image, diffuse, mask)
            if piped is not None:
                # Fuse trainable residual with SD prior
                return (0.5 * head_out + 0.5 * piped).clamp(0, 1)
        return head_out


class GenerativeRefiner(nn.Module):
    """Stage 5: generative fusion on top of physics diffuse prior."""

    def __init__(self, backend: str | None = None):
        super().__init__()
        backend = backend or os.environ.get("MVBRDF_DIFFUSION_BACKEND", "auto")
        if backend == "auto":
            backend = "tiny"
        self.backend = backend
        if backend == "tiny":
            self.model = TinyRefineUNet()
        elif backend == "sdturbo":
            self.model = SDTurboRefiner()
        else:
            raise ValueError(f"Unknown diffusion backend: {backend}")

    def forward(
        self,
        image: torch.Tensor,
        diffuse: torch.Tensor,
        mask: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        if self.backend == "tiny":
            cond = torch.cat([image, diffuse, mask, normal], dim=1)
            return self.model(cond=cond)
        return self.model(image, diffuse, mask, normal)

    def noise_pred_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)
