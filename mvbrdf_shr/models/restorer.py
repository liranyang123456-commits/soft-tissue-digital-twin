"""SOTA-oriented highlight restorer: multi-scale residual UNet with SE attention."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, max(ch // r, 4)),
            nn.SiLU(inplace=True),
            nn.Linear(max(ch // r, 4), ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )
        self.se = SEBlock(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.se(self.conv(x)))


class Down(nn.Module):
    def __init__(self, cin: int, cout: int, n_res: int = 2):
        super().__init__()
        self.proj = nn.Conv2d(cin, cout, 3, stride=2, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(cout) for _ in range(n_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


class Up(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int, n_res: int = 2):
        super().__init__()
        self.proj = nn.Conv2d(cin + skip, cout, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(cout) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.blocks(self.proj(x))


class HighlightRestorer(nn.Module):
    """Physics-conditioned highlight removal UNet.

    Input channels: image(3)+diffuse(3)+mask(1)+normal(3)+albedo(3) = 13
    Output: clean RGB in [0,1], with learned blend against physics diffuse.
    """

    def __init__(self, in_ch: int = 13, base: int = 64, n_res: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.SiLU(inplace=True),
            ResBlock(base),
        )
        self.d1 = Down(base, base * 2, n_res)
        self.d2 = Down(base * 2, base * 4, n_res)
        self.d3 = Down(base * 4, base * 8, n_res)
        self.mid = nn.Sequential(ResBlock(base * 8), ResBlock(base * 8), SEBlock(base * 8))
        self.u2 = Up(base * 8, base * 4, base * 4, n_res)
        self.u1 = Up(base * 4, base * 2, base * 2, n_res)
        self.u0 = Up(base * 2, base, base, n_res)
        self.out_rgb = nn.Conv2d(base, 3, 1)
        self.out_alpha = nn.Conv2d(base, 1, 1)

    def forward(
        self,
        image: torch.Tensor,
        diffuse: torch.Tensor,
        mask: torch.Tensor,
        normal: torch.Tensor,
        albedo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if albedo is None:
            albedo = diffuse
        x = torch.cat([image, diffuse, mask, normal, albedo], dim=1)
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        m = self.mid(s3)
        x = self.u2(m, s2)
        x = self.u1(x, s1)
        x = self.u0(x, s0)
        # Residual learning from input. Physics channels condition the UNet
        # but do NOT blend into the output (avoids physics error bottleneck).
        residual = self.out_rgb(x)
        pred = (image + residual).clamp(0, 1)
        return pred


class SpecularDecomposer(nn.Module):
    """Predict the SHIQ decomposition explicitly.

    SHIQ is constructed so that A = D + S.  The network predicts a bounded
    RGB ratio for S and a separate binary-mask logit T; D_coarse is then
    reconstructed analytically as A - S instead of learned unconstrained.
    Physics specular/mask buffers are inputs, not final answers.
    """

    def __init__(self, base: int = 64, n_res: int = 2):
        super().__init__()
        # input A(3) + rendered specular(3) + rendered mask(1)
        self.stem = nn.Sequential(
            nn.Conv2d(7, base, 3, padding=1),
            nn.SiLU(inplace=True),
            ResBlock(base),
        )
        self.d1 = Down(base, base * 2, n_res)
        self.d2 = Down(base * 2, base * 4, n_res)
        self.d3 = Down(base * 4, base * 8, n_res)
        self.mid = nn.Sequential(
            ResBlock(base * 8), ResBlock(base * 8), SEBlock(base * 8)
        )
        self.u2 = Up(base * 8, base * 4, base * 4, n_res)
        self.u1 = Up(base * 4, base * 2, base * 2, n_res)
        self.u0 = Up(base * 2, base, base, n_res)
        self.spec_ratio = nn.Conv2d(base, 3, 1)
        self.mask_logit = nn.Conv2d(base, 1, 1)
        # Most pixels are non-highlight.  Start close to S=0/T=0 instead of
        # subtracting half the input before the decomposition branch is trained.
        nn.init.zeros_(self.spec_ratio.weight)
        nn.init.constant_(self.spec_ratio.bias, -3.0)
        nn.init.zeros_(self.mask_logit.weight)
        nn.init.constant_(self.mask_logit.bias, -3.0)

    def forward(
        self,
        image: torch.Tensor,
        physics_specular: torch.Tensor,
        physics_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([image, physics_specular, physics_mask], dim=1)
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        x = self.mid(s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        x = self.u0(x, s0)
        # S cannot be negative or exceed the observed radiance A.
        specular = image * torch.sigmoid(self.spec_ratio(x))
        mask_logit = self.mask_logit(x)
        mask = torch.sigmoid(mask_logit)
        coarse = (image - specular).clamp(0, 1)
        return specular, mask, mask_logit, coarse


class PureUNetBaseline(nn.Module):
    """Strong 2D UNet baseline (JSHDR/TSHRNet-capacity proxy) without physics."""

    def __init__(self, base: int = 64):
        super().__init__()
        self.net = HighlightRestorer(in_ch=13, base=base, n_res=2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape
        zeros = torch.zeros(B, 1, H, W, device=image.device, dtype=image.dtype)
        ones_n = torch.zeros(B, 3, H, W, device=image.device, dtype=image.dtype)
        ones_n[:, 2] = 1.0
        return self.net(image, image, zeros, ones_n, image)


class DHANRestorer(nn.Module):
    """Physics-conditioned DHAN restorer (path A: SOTA architecture + physics prior).

    Wraps the DHAN-SHR ``Processor`` (dual-domain Transformer + FFT
    FrequencyProcessor + pixel/channel dual attention — the SHIQ 33.81 SOTA
    backbone, source already vendored in baselines_ext/DHAN-SHR) and injects the
    physics-decomposition prior (albedo/normal/mask) as a ZERO-INITIALIZED side
    branch added to the patch-embed feature.

    Why zero-init side injection (not channel concat): Processor's forward ends
    with ``out = self.output(...) + inp_img``; if inp_channels were changed from
    3 to 10 the residual add would be dimensionally inconsistent, and patch_embed
    weights could not be reused. Keeping inp_channels=3 means the entire DHAN
    pretrained checkpoint loads with strict=False and ~100% reuse; the physics
    side branch (Conv2d(7->dim), zero weights) contributes nothing at init, then
    learns to modulate features during training — so the network starts as pure
    SOTA DHAN and gradually incorporates the physics prior.

    The physics condition is the MVBRDFSHRPro decomposition: albedo(3) +
    normal(3) + highlight mask(1) = 7 channels.
    """

    def __init__(self, dim: int = 36, n_phys_ch: int = 7):
        super().__init__()
        # Import vendored DHAN Processor. baselines_ext is on sys.path via the
        # scripts' PYTHONPATH=.; we import by file-relative path to be robust.
        import importlib.util
        from pathlib import Path

        dhan_path = (
            Path(__file__).resolve().parents[2]
            / "baselines_ext" / "DHAN-SHR" / "models" / "model.py"
        )
        spec = importlib.util.spec_from_file_location("dhan_model", dhan_path)
        dhan_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dhan_mod)
        self.proc = dhan_mod.Processor(inp_channels=3, out_channels=3, dim=dim)
        # Zero-init physics side branch: maps [albedo3+normal3+mask1] -> dim.
        # Zero weights + zero bias => contributes 0 at init, so the network
        # starts identical to pretrained DHAN and learns to use physics.
        self.phys_fuse = nn.Conv2d(n_phys_ch, dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.phys_fuse.weight)
        nn.init.zeros_(self.phys_fuse.bias)

    def forward(
        self,
        image: torch.Tensor,
        diffuse: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        normal: torch.Tensor | None = None,
        albedo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run physics-conditioned DHAN. Mirrors HighlightRestorer's signature.

        Returns the predicted clean image in [0,1] (the restorer predicts a
        residual on the input; DHAN's Processor.forward already adds inp_img).
        """
        p = self.proc
        # DHAN's FrequencyProcessor uses torch.fft; cuFFT in half precision only
        # supports power-of-two sizes and overflows otherwise (NaN at 256,
        # RuntimeError at 200). Force float32 for the whole DHAN forward so it
        # is numerically stable at ANY resolution.
        # DHAN's FrequencyProcessor uses torch.fft; cuFFT in half precision only
        # supports power-of-two sizes and overflows otherwise (NaN at 256,
        # RuntimeError at 200). Force float32 for the whole DHAN forward so it
        # is numerically stable at ANY resolution.
        with torch.autocast(device_type="cuda", enabled=False):
            image = image.float()
            # patch-embed the RGB image (DHAN's native 3-channel input).
            feat = p.patch_embed(image)
            # Inject the physics prior (albedo/normal/mask) via the zero-init side
            # branch. Defaults handle missing priors (e.g. PureUNet-style call).
            B, _, H, W = image.shape
            parts = []
            if albedo is not None:
                parts.append(albedo.float())
            else:
                parts.append(torch.zeros(B, 3, H, W, device=image.device))
            if normal is not None:
                parts.append(normal.float())
            else:
                n_default = torch.zeros(B, 3, H, W, device=image.device)
                n_default[:, 2] = 1.0
                parts.append(n_default)
            if mask is not None:
                parts.append(mask.float())
            else:
                parts.append(torch.zeros(B, 1, H, W, device=image.device))
            physics = torch.cat(parts, dim=1)
            feat = feat + self.phys_fuse(physics)

            # Rest of DHAN's Processor.forward (encoder -> bottleneck -> decoder),
            # operating on the physics-conditioned feature.
            out_enc_level1 = p.encoder_level1(feat)
            out_enc_level2 = p.encoder_level2(p.down1_2(out_enc_level1))
            out_enc_level3 = p.encoder_level3(p.down2_3(out_enc_level2))
            latent = p.bottleneck(out_enc_level3)
            inp_dec_level3 = torch.cat([latent, out_enc_level3], dim=1)
            out_dec_level3 = p.decoder_level3(p.reduce_chan_level3(inp_dec_level3))
            inp_dec_level2 = torch.cat([p.up3_2(out_dec_level3), out_enc_level2], dim=1)
            out_dec_level2 = p.decoder_level2(p.reduce_chan_level2(inp_dec_level2))
            inp_dec_level1 = torch.cat([p.up2_1(out_dec_level2), out_enc_level1], dim=1)
            out_dec_level1 = p.decoder_level1(p.reduce_chan_level1(inp_dec_level1))
            ref_out = p.refinement(out_dec_level1)
            out = p.output(ref_out) + image  # DHAN's internal residual on RGB input
            return out.clamp(0, 1)


def load_dhan_weights(restorer: DHANRestorer, weights_path: str, verbose: bool = False) -> int:
    """Load pretrained DHAN weights into a DHANRestorer's Processor.

    The checkpoint stores ``{'state_dict': {...}}`` with keys prefixed by
    ``sfp.`` (Model.sfp = Processor) and possibly ``module.`` (DataParallel).
    We strip those prefixes and load into ``restorer.proc`` with strict=False:
    every Processor sub-module matches, so ~100% of weights transfer. The
    physics side branch (phys_fuse) is intentionally NOT in the checkpoint and
    stays zero-initialized.

    Returns the number of weights loaded. NOTE: use weights/model.pth (18MB,
    real); weights/DHAN-SHR.pth is a 9-byte "Not Found" placeholder.
    """
    from collections import OrderedDict

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt)
    new_sd = OrderedDict()
    for key, value in raw.items():
        name = key[7:] if key.startswith("module.") else key  # strip DataParallel
        name = name[4:] if name.startswith("sfp.") else name  # strip Model.sfp
        new_sd[name] = value
    loaded = restorer.proc.load_state_dict(new_sd, strict=False)
    n_loaded = len(new_sd) - len(loaded.missing_keys)
    if verbose:
        print(
            f"  DHAN weights: {n_loaded}/{len(new_sd)} loaded, "
            f"{len(loaded.missing_keys)} missing, {len(loaded.unexpected_keys)} unexpected"
        )
    return n_loaded