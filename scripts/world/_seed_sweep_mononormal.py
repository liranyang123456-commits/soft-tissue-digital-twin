"""Multi-seed robustness sweep: normalreg vs monocular-normal-prior (w0.2).

Trains each config across seeds, evaluates the same-input IR protocol, and
prints a mean +/- std summary so the Normal-MAE claim is not a single run.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENE_ROOT = "data/baseline_protocols/scene0070_rgb_only/ours"
EVAL_ROOT = "data/baseline_protocols/scene0070_rgb_only/evaluation_only"
METRICS = ROOT / "outputs/scene0070_same_protocol/metrics"

# (label, config, base output_dir)
VARIANTS = [
    ("normalreg", "configs/world/benchmark_scene_0070_rgb_only_normalreg.yaml",
     "outputs/scene0070_same_protocol/ours_constant_light_normalreg"),
    ("mononormal_w2", "configs/world/benchmark_scene_0070_rgb_only_mononormal_w2.yaml",
     "outputs/scene0070_same_protocol/ours_constant_light_mononormal_w2"),
]
SEEDS = [7, 1234]  # seed 2026 already run for both


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def evaluate(ckpt_dir: Path, method: str, out_json: Path):
    eval_holdout = ckpt_dir / "eval_holdout"
    run([sys.executable, "-m", "mvbrdf_shr.world.evaluate_ir",
         "--checkpoint", str(ckpt_dir / "last.pt"),
         "--output", str(eval_holdout), "--split", "holdout"])
    run([sys.executable, "scripts/evaluate_scene0070_inverse_baseline.py",
         "--method", method, "--prediction-dir", str(eval_holdout),
         "--scene-root", SCENE_ROOT, "--evaluation-root", EVAL_ROOT,
         "--output", str(out_json)])


def macro(path: Path):
    return json.loads(path.read_text())["macro"]


def main():
    results = {}
    for label, cfg, base_out in VARIANTS:
        results[label] = {}
        # include the existing seed-2026 run
        seed2026 = METRICS / f"{Path(base_out).name}.json"
        if seed2026.exists():
            results[label][2026] = macro(seed2026)
        for seed in SEEDS:
            tag = f"seed{seed}"
            ckpt_dir = ROOT / f"{base_out}_{tag}"
            out_json = METRICS / f"{Path(base_out).name}_{tag}.json"
            if not (ckpt_dir / "last.pt").exists():
                run([sys.executable, "-m", "mvbrdf_shr.world.train",
                     "--config", cfg, "--seed", str(seed), "--tag", tag])
            if not out_json.exists():
                evaluate(ckpt_dir, f"{label}-{tag}", out_json)
            results[label][seed] = macro(out_json)

    # summary
    keys = ["full_psnr", "diffuse_psnr", "albedo_psnr", "normal_mae_deg"]
    print("\n==== MULTI-SEED SUMMARY (mean +/- std over seeds) ====")
    for label in results:
        runs = list(results[label].values())
        line = f"{label:16s} n={len(runs)}  "
        for k in keys:
            vals = [r[k] for r in runs if k in r]
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                line += f"{k}={mean:.2f}+/-{std:.2f}  "
        print(line)
        for seed, m in sorted(results[label].items()):
            print(f"    seed {seed}: NMAE={m['normal_mae_deg']:.2f} "
                  f"RGB={m['full_psnr']:.2f}")


if __name__ == "__main__":
    main()
