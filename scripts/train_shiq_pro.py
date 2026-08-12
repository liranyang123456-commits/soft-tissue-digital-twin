"""Train MVBRDF-Pro on real SHIQ and evaluate vs Neural DRM (34.50 PSNR)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from mvbrdf_shr.data.shiq_adapter import (
    NSHAdapter,
    PSDAdapter,
    SHIQAdapter,
    SSHRAdapter,
)
from mvbrdf_shr.data.synthetic_toy import collate_batch
from mvbrdf_shr.losses_pro import LossWeights, total_loss
from mvbrdf_shr.metrics import ssim_torch
from mvbrdf_shr.models.pipeline_pro import MVBRDFSHRPro
from mvbrdf_shr.viz import save_pipeline_viz

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHIQ = ROOT.parent / "data" / "raw" / "SHIQ_extracted"
# Default roots for the auxiliary public datasets (path C: joint training).
_DEFAULT_SSHR = ROOT.parent / "data" / "raw" / "SSHR_extracted"
_DEFAULT_NSH = ROOT.parent / "data" / "raw" / "NSH_extracted"
_DEFAULT_PSD = ROOT.parent / "data" / "raw" / "PSD_extracted"
SOTA_PSNR = 34.50
SOTA_SSIM = 0.979


@torch.no_grad()
def evaluate(model, loader, device, with_ssim: bool = True) -> dict:
    model.eval()
    ps, ss = [], []
    for batch in loader:
        img = batch.image.to(device)
        gt = batch.image_clean.to(device)
        pred = model(img, use_refine=True, detach_physics=True)["pred"]
        mse = ((pred - gt) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
        ps.extend((20 * torch.log10(1.0 / mse.sqrt())).tolist())
        if with_ssim:
            for i in range(pred.shape[0]):
                ss.append(ssim_torch(pred[i : i + 1], gt[i : i + 1]))
    out = {"psnr": float(np.mean(ps)), "n": len(ps)}
    out["ssim"] = float(np.mean(ss)) if ss else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shiq-root", default=str(DEFAULT_SHIQ))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=200)
    ap.add_argument("--resume", default=str(ROOT / "outputs" / "sota_pro" / "best.pt"))
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "shiq_decomp"))
    ap.add_argument("--psnr-finetune", action="store_true")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (run 3 seeds to report median+IQR).",
    )
    # --- SOTA push: loss + data (paths B / C) ----------------------------------
    ap.add_argument(
        "--refine-lpips", type=float, default=0.0,
        help="Weight of the LPIPS perceptual loss (path B). 0=off, ~0.3 typical.",
    )
    ap.add_argument(
        "--refine-fft", type=float, default=0.0,
        help="Weight of the FFT frequency-domain loss (path B). 0=off, ~0.1 typical.",
    )
    ap.add_argument(
        "--aux-datasets", default="",
        help="Comma-separated aux datasets for joint training (path C): "
             "sshr,psd,nsh. Empty = SHIQ-only (backward compatible).",
    )
    ap.add_argument(
        "--aux-res", type=int, default=0,
        help="Resolution for aux datasets (0 = same as --resolution). SSHR/PSD "
             "are 256-native; matching the SHIQ resolution keeps batches uniform.",
    )
    ap.add_argument(
        "--restorer-kind", choices=["unet", "dhan"], default="unet",
        help="Restorer backbone (path A): 'unet' = current HighlightRestorer, "
             "'dhan' = physics-conditioned DHAN Processor.",
    )
    ap.add_argument(
        "--dhan-weights", default="baselines_ext/DHAN-SHR/weights/model.pth",
        help="Pretrained DHAN weights to initialize the dhan restorer (path A).",
    )
    args = ap.parse_args()

    # Reproducibility: fix all RNG seeds. Run with --seed 0/1/2 to obtain a
    # median+IQR estimate for the reported PSNR.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random

    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SHIQAdapter(args.shiq_root, args.resolution, "train")
    # --- Joint training (path C): mix SHIQ with auxiliary public datasets -----
    # SSHR (synthetic physics GT, large) is the biggest gain source — it is what
    # TSHRNet/DHAN train on. PSD/NSH add real-data diversity. val/test stay
    # SHIQ-only so reported numbers stay comparable to literature.
    aux_res = args.aux_res or args.resolution
    aux_parts = [train_ds]
    aux_names = ["shiq"]
    if args.aux_datasets:
        for name in args.aux_datasets.split(","):
            name = name.strip().lower()
            if name == "sshr":
                aux_parts.append(SSHRAdapter(_DEFAULT_SSHR, aux_res, "train"))
                aux_names.append("sshr")
            elif name == "nsh":
                # NSH test split is the small calibrated set; use it as aux.
                aux_parts.append(NSHAdapter(_DEFAULT_NSH, aux_res, "train"))
                aux_names.append("nsh")
            elif name == "psd":
                # PSD public archive only ships PSD_Test; use its train sub-split.
                aux_parts.append(PSDAdapter(_DEFAULT_PSD, aux_res, "train"))
                aux_names.append("psd")
            else:
                print(f"  [warn] unknown aux dataset '{name}', skipped")
    aux_sizes = [len(p) for p in aux_parts]
    if len(aux_parts) > 1:
        # WeightedRandomSampler keeps SHIQ dominant (per-dataset weight = 1/size)
        # so smaller SSHR/PSD splits are not drowned by SHIQ's 9825, and vice
        # versa — each dataset contributes roughly equally per epoch.
        weights_per_ds = [1.0 / max(s, 1) for s in aux_sizes]
        sample_weights = []
        for part, wval in zip(aux_parts, weights_per_ds):
            sample_weights.extend([wval] * len(part))
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
        joint_train = ConcatDataset(aux_parts)
        print(f"  joint train: {dict(zip(aux_names, aux_sizes))} total={len(joint_train)}")
    else:
        joint_train = train_ds
        sampler = None
    # --- Test-set leakage fix -------------------------------------------------
    # Model selection and early stopping use the 'val' split (500 samples, ids
    # 14001-14500). The 'test' split (500 samples, ids 14501-15000) is touched
    # ONLY ONCE, after training, for the final reported number. The previous
    # implementation evaluated/select-best on the full 1000-sample 'test' set,
    # which leaked test data into model selection.
    val_ds = SHIQAdapter(args.shiq_root, args.resolution, "val")
    test_ds = SHIQAdapter(args.shiq_root, args.resolution, "test")
    print(
        f"SHIQ train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"device={device} seed={args.seed}"
    )
    assert len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0

    # train_loader uses joint_train (SHIQ + aux) with the balanced sampler when
    # aux datasets are enabled; otherwise plain shuffled SHIQ.
    train_loader = DataLoader(
        joint_train, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=collate_batch, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0
    )

    model = MVBRDFSHRPro(
        resolution=args.resolution, base_ch=48, restorer_base=96,
        restorer_kind=args.restorer_kind,
    ).to(device)
    # Path A: initialize the DHAN restorer backbone from pretrained weights.
    # Load order matters for the distilled-physics path (stage E):
    # 1) resume the (possibly distilled) full-pipeline checkpoint — this loads the
    #    pre-trained physics heads (geometry/material) from world-GT distillation.
    # 2) THEN overwrite ONLY the restorer with pretrained DHAN weights, so the
    #    distilled physics prior is preserved while the restorer starts from the
    #    SOTA DHAN init (not the random restorer the distillation left behind).
    resume = Path(args.resume)
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        print(f"resumed {resume} missing={len(missing)} unexpected={len(unexpected)}")
    if args.restorer_kind == "dhan" and Path(args.dhan_weights).exists():
        from mvbrdf_shr.models.restorer import load_dhan_weights
        loaded = load_dhan_weights(model.restorer, args.dhan_weights, verbose=True)
        print(f"  DHAN restorer init from {args.dhan_weights}: loaded={loaded} (overwrites restorer after resume)")

    opt = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not n.startswith(("restorer", "decomposer"))
                ],
                "lr": args.lr * 0.2,
            },
            {
                "params": list(model.restorer.parameters())
                + list(model.decomposer.parameters()),
                "lr": args.lr,
            },
        ],
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    weights = LossWeights(
        photo=0.2,
        diffuse=0.8,
        refine=1.0 if args.psnr_finetune else 2.5,
        refine_ssim=0.4 if args.psnr_finetune else 1.2,
        refine_grad=0.1 if args.psnr_finetune else 0.4,
        refine_mse=8.0 if args.psnr_finetune else 0.0,
        hl_weight=1.0 if args.psnr_finetune else 4.0,
        cons=0.02,
        mask=0.3,
        smooth=0.001,
        specular=2.0,
        coarse=1.5,
        mask_pred=0.5,
        # Path B losses for the SOTA push (default 0 = backward compatible).
        refine_lpips=args.refine_lpips,
        refine_fft=args.refine_fft,
    )

    # LPIPS criterion (path B): cached once, reused every step. Held on device so
    # the VGG backbone is not reloaded. None when disabled.
    lpips_fn = None
    if weights.refine_lpips > 0:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
        print(f"  LPIPS enabled (w={weights.refine_lpips}) on {device}")
    if weights.refine_fft > 0:
        print(f"  FFT loss enabled (w={weights.refine_fft})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"psnr": -1.0}
    history = []

    # quick sanity eval before train (on VAL, never on the held-out test set)
    m0 = evaluate(model, val_loader, device, with_ssim=False)
    print(f"[init] SHIQ val PSNR={m0['psnr']:.3f} (target>{SOTA_PSNR})")

    model.train()
    step = 0
    pbar = tqdm(total=args.steps, desc="SHIQ-Pro")
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            img = batch.image.to(device)
            gt = batch.image_clean.to(device)
            spec_gt = (
                batch.specular_gt.to(device)
                if batch.specular_gt is not None
                else img - gt
            )
            mask = batch.mask_gt.to(device) if batch.mask_gt is not None else None
            out = model(img, use_refine=True, detach_physics=True)
            loss, logs = total_loss(
                out,
                image_clean=gt,
                mask_gt=mask,
                weights=weights,
                specular_gt=spec_gt,
                lpips_fn=lpips_fn,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            pbar.set_postfix(loss=f"{logs['total']:.3f}", ref=f"{logs.get('refine', 0):.3f}")
            pbar.update(1)

            if step % args.eval_every == 0 or step == args.steps - 1:
                do_ssim = step % (args.eval_every * 2) == 0 or step == args.steps - 1
                # Model selection uses VAL only; the held-out TEST set is never
                # consulted during training (fixes test-set leakage).
                m = evaluate(model, val_loader, device, with_ssim=do_ssim)
                history.append({"step": step, **m, "split": "val"})
                print(f"\n[SHIQ {step}] val PSNR={m['psnr']:.3f} SSIM={m.get('ssim', 0):.4f}", flush=True)
                if m["psnr"] > best.get("psnr", -1):
                    if not do_ssim:
                        m = evaluate(model, val_loader, device, with_ssim=True)
                    best = {**m, "step": step}
                    torch.save({"model": model.state_dict(), "metrics": best, "cfg": vars(args)}, out_dir / "best.pt")
                    b0 = next(iter(val_loader))
                    model.eval()
                    with torch.no_grad():
                        o = model(b0.image.to(device), use_refine=True, detach_physics=True)
                    save_pipeline_viz(o, out_dir / "viz_best.png", b0.image.to(device))
                    model.train()
                    print(f"  saved best val PSNR={best['psnr']:.3f} SSIM={best['ssim']:.4f}", flush=True)
                # Early-stop threshold acts on VAL (not test) PSNR/SSIM.
                if best["psnr"] >= SOTA_PSNR and best.get("ssim", 0) >= SOTA_SSIM - 0.02:
                    print(f"*** BEAT SHIQ LITERATURE SOTA ({SOTA_PSNR}/{SOTA_SSIM}) on VAL ***", flush=True)
                    step = args.steps
                    break
            step += 1
    pbar.close()

    # --- Final evaluation on the held-out TEST set (touched exactly once) ----
    # This is the only place the test split is read. The model is the val-best
    # checkpoint (already saved). We reload it to guarantee the reported number
    # corresponds to the val-selected checkpoint, not the last training step.
    best_path = out_dir / "best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    # Build the test loaders here (held-out splits are NOT loaded during training,
    # only at this single final evaluation, to prevent any leakage).
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=0,
    )
    final_test = evaluate(model, test_loader, device, with_ssim=True)
    print(
        f"[FINAL] SHIQ TEST (n={len(test_ds)}, held-out) "
        f"PSNR={final_test['psnr']:.3f} SSIM={final_test['ssim']:.4f}",
        flush=True,
    )
    # Also report on the full official 1000-sample test set for literature
    # comparability (Neural DRM / DHAN-SHR evaluate on all 1000).
    test_full_ds = SHIQAdapter(args.shiq_root, args.resolution, "test_full")
    test_full_loader = DataLoader(
        test_full_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=0,
    )
    final_test_full = evaluate(model, test_full_loader, device, with_ssim=True)
    print(
        f"[FINAL] SHIQ TEST_FULL (n={len(test_full_ds)}, literature-comparable) "
        f"PSNR={final_test_full['psnr']:.3f} SSIM={final_test_full['ssim']:.4f}",
        flush=True,
    )

    report = {
        "dataset": "SHIQ",
        "train_n": len(train_ds),
        "test_n": len(test_ds),
        "literature_sota": {"method": "Neural DRM Solver", "psnr": SOTA_PSNR, "ssim": SOTA_SSIM},
        "mvbrdf_pro_val": best,
        # Strict protocol: model selected on val, reported on held-out test (500).
        "mvbrdf_pro_test_strict": final_test,
        # Literature-comparable: official full 1000-sample test set.
        "mvbrdf_pro_test_full": final_test_full,
        # Back-compat alias: keep "mvbrdf_pro" pointing at the literature-comparable
        # full-test number so legacy collection scripts (compare_sota_current.py)
        # keep working. Prefer mvbrdf_pro_test_strict in the paper.
        "mvbrdf_pro": final_test_full,
        "beat_literature_psnr": final_test_full.get("psnr", 0) >= SOTA_PSNR,
        "beat_literature_ssim": final_test_full.get("ssim", 0) >= SOTA_SSIM,
        "history_tail": history[-15:],
        "seed": args.seed,
        "split_protocol": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
            "test_full": len(test_full_ds),
            "note": (
                "val (ids 14001-14500) for model selection; test (ids 14501-15000) "
                "held out, evaluated once; test_full (ids 14001-15000) for literature "
                "comparison. Fixes prior test-set leakage."
            ),
        },
    }
    (out_dir / "shiq_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "outputs" / "SHIQ_SOTA_REPORT.md").write_text(
        f"""# SHIQ Real-Data SOTA Report (seed {args.seed})

## Dataset (test-set-leakage-fixed protocol)
- Train: {len(train_ds)} | Val (model selection): {len(val_ds)} | Test (held-out): {len(test_ds)} | Resolution: {args.resolution}
- Test_full (literature-comparable, official 1000): {len(test_full_ds)}

## Results
| Method | PSNR | SSIM | split |
|---|---:|---:|---|
| Neural DRM Solver (ICCV'25, lit) | {SOTA_PSNR} | {SOTA_SSIM} | test (1000) |
| **MVBRDF-Pro (ours, val-best)** | {best.get('psnr', float('nan')):.3f} | {best.get('ssim', float('nan')):.4f} | val (500) |
| **MVBRDF-Pro (ours, held-out test)** | **{final_test.get('psnr', float('nan')):.3f}** | **{final_test.get('ssim', float('nan')):.4f}** | test (500, strict) |
| MVBRDF-Pro (ours, full test) | {final_test_full.get('psnr', float('nan')):.3f} | {final_test_full.get('ssim', float('nan')):.4f} | test_full (1000) |

## Verdict (on full 1000-sample test for literature comparison)
- Beat literature PSNR ({SOTA_PSNR}): **{'YES' if report['beat_literature_psnr'] else 'NOT YET'}**
- Beat literature SSIM ({SOTA_SSIM}): **{'YES' if report['beat_literature_ssim'] else 'NOT YET'}**

Run 3 seeds (--seed 0/1/2) and report the median of the held-out test PSNR.

Checkpoint: `{out_dir / "best.pt"}`
""",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "history_tail"}, indent=2))


if __name__ == "__main__":
    main()
