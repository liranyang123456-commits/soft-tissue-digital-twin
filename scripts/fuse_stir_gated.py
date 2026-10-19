"""Gate STIR endpoints: keep fused identity unless the local neighbourhood is inconsistent."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_stir_tracking import DATA, OUT, RADII, endpoint_errors, summarize
from scripts.evaluate_stir_tracking_3d import backproject, errors3d, summarize as summarize3d

DEPTH_BAND = 0.05


def neighbourhood(point, tracks):
    uv0, z0, uvT, zT, opacity = tracks["uv0"], tracks["z0"], tracks["uvT"], tracks["zT"], tracks["opacity"]
    usable = (z0 > 0) & (zT > 0) & (opacity > 0.05) & np.isfinite(uvT).all(1)
    distance = np.linalg.norm(uv0 - point, axis=1)
    candidates = np.empty(0, dtype=int)
    radius = RADII[-1]
    for radius in RADII:
        candidates = np.flatnonzero(usable & (distance < radius))
        if len(candidates) >= 3:
            break
    if len(candidates) == 0:
        return None
    depth = z0[candidates]
    front = candidates[depth <= np.percentile(depth, 10) * (1 + DEPTH_BAND)]
    motion = uvT[front] - uv0[front]
    weight = opacity[front] * np.exp(-0.5 * (distance[front] / (radius / 2)) ** 2)
    weight = weight / weight.sum()
    mean = (weight[:, None] * motion).sum(0)
    std = float(np.sqrt((weight * ((motion - mean) ** 2).sum(1)).sum()))
    mag = float(np.linalg.norm(mean) + 1e-6)
    return {
        "end": point + mean,
        "z": float((weight * zT[front]).sum()),
        "n": int(len(front)),
        "std": std,
        "rel": std / mag,
        "lost": float((opacity[candidates] <= 0.05).mean()) if len(candidates) else 1.0,
    }


def prefer_query(stats, start_full, fused_full, query_full, rel_tau=0.45, std_tau=8.0, min_front=3):
    """Use the query track only when the Gaussian neighbourhood is inconsistent
    and the query did not explode relative to that neighbourhood."""
    if stats is None:
        return True
    inconsistent = stats["n"] < min_front or stats["rel"] > rel_tau or stats["std"] > std_tau or stats["lost"] > 0.5
    if not inconsistent:
        return False
    fused_mag = float(np.linalg.norm(fused_full - start_full))
    query_mag = float(np.linalg.norm(query_full - start_full))
    return query_mag <= 3.0 * max(fused_mag, 8.0)


def main():
    dest = OUT / "present_fused_v6"
    dest.mkdir(parents=True, exist_ok=True)
    index = json.loads((DATA / "index.json").read_text())
    gt3 = json.loads((OUT / "gt3d.json").read_text())
    pooled2 = {k: [] for k in ["fused", "query", "gated", "raft"]}
    pooled3 = {k: [] for k in ["fused_v5", "gated", "raft"]}
    switched = 0
    total = 0
    hard = []
    for entry in index:
        name = entry["name"]
        meta = json.loads((DATA / name / "stir_meta.json").read_text())
        shift = np.asarray(meta["shift_half_res"])
        start_full = np.asarray(meta["start_points_full_res"], dtype=np.float64)
        end_full = np.asarray(meta["end_points_full_res"], dtype=np.float64)
        start_half = start_full / 2.0 + shift
        fused = dict(np.load(OUT / "present_fused_v4" / name / "tracks.npz"))
        fused5 = dict(np.load(OUT / "present_fused_v5" / name / "tracks.npz"))
        query = np.load(OUT / "query_refresh_v1" / name / "points.npz")["end_full_res"]
        query_tr = dict(np.load(OUT / "query_refresh_v1" / name / "tracks.npz"))
        raft = np.load(OUT / "raft" / name / "points.npz")["end_full_res"]
        pred_f, pred_g, z_g, flags = [], [], [], []
        for i, point in enumerate(start_half):
            st = neighbourhood(point, fused)
            st5 = neighbourhood(point, fused5)
            q_full = query[i]
            fused_full = (st["end"] - shift) * 2.0 if st is not None else start_full[i]
            use_q = prefer_query(st, start_full[i], fused_full, q_full)
            if use_q:
                pred_g.append(q_full)
                z_g.append(float(query_tr["zT"][i]))
                flags.append(1)
            else:
                end_h = st5["end"] if st5 is not None else st["end"]
                pred_g.append((end_h - shift) * 2.0)
                z_g.append(float((st5 or st)["z"]))
                flags.append(0)
            pred_f.append((st["end"] - shift) * 2.0 if st is not None else start_full[i])
        pred_f = np.asarray(pred_f)
        pred_g = np.asarray(pred_g)
        z_g = np.asarray(z_g)
        flags = np.asarray(flags)
        switched += int(flags.sum())
        total += len(flags)
        pooled2["fused"].append(endpoint_errors(pred_f, end_full))
        pooled2["query"].append(endpoint_errors(query, end_full))
        pooled2["gated"].append(endpoint_errors(pred_g, end_full))
        pooled2["raft"].append(endpoint_errors(raft, end_full))
        out = dest / name
        out.mkdir(parents=True, exist_ok=True)
        z0 = np.asarray(query_tr["z0"], dtype=np.float64)
        z0 = np.where(z0 > 0.01, z0, np.maximum(z_g, 1.0))
        np.savez_compressed(
            out / "tracks.npz",
            uv0=start_half,
            z0=z0,
            uvT=pred_g / 2.0 + shift,
            zT=z_g,
            opacity=np.ones(len(pred_g)),
            gated_query=flags.astype(np.uint8),
        )
        np.savez_compressed(out / "points.npz", end_full_res=pred_g)
        src_run = OUT / "present_fused_v4" / name / "run.json"
        if src_run.exists():
            shutil.copy(src_run, out / "run.json")
        if name in gt3:
            labels = gt3[name]
            K = np.asarray(labels["K_full"])
            start_3d = np.asarray(labels["start_3d"])
            end_3d = np.asarray(labels["end_3d"])
            start2 = np.asarray(labels["start_2d"], dtype=np.float64)
            pred3, z3, end5, z5 = [], [], [], []
            for point_full in start2:
                point = point_full / 2.0 + shift
                st = neighbourhood(point, fused)
                st5 = neighbourhood(point, fused5)
                j = int(np.argmin(np.linalg.norm(start_full - point_full, axis=1)))
                fused_full = ((st5 or st)["end"] - shift) * 2.0 if (st5 or st) is not None else point_full
                if prefer_query(st, point_full, fused_full, query[j]):
                    pred3.append(query[j])
                    z3.append(float(query_tr["zT"][j]))
                else:
                    end_h = (st5 or st)["end"]
                    pred3.append((end_h - shift) * 2.0)
                    z3.append(float((st5 or st)["z"]))
                if st5 is None:
                    end5.append(point)
                    z5.append(np.nan)
                else:
                    end5.append(st5["end"])
                    z5.append(st5["z"])
            Xg = backproject(np.asarray(pred3), np.asarray(z3), K)
            X5 = backproject((np.asarray(end5) - shift) * 2.0, np.asarray(z5), K)
            raft3 = np.load(OUT / "raft3d" / name / "points3d.npz")["raft_sgm"]
            pooled3["fused_v5"].append(errors3d(X5, end_3d, start_3d))
            pooled3["gated"].append(errors3d(Xg, end_3d, start_3d))
            pooled3["raft"].append(errors3d(raft3, end_3d, start_3d))
        if name in ("clip010", "clip020", "clip031", "clip035", "clip055"):
            hard.append((
                name, int(flags.sum()), len(flags),
                round(float(np.median(endpoint_errors(pred_f, end_full))), 2),
                round(float(np.median(endpoint_errors(pred_g, end_full))), 2),
                round(float(np.median(endpoint_errors(query, end_full))), 2),
                round(float(np.median(endpoint_errors(raft, end_full))), 2),
            ))
        print(name, "switch", int(flags.sum()), "/", len(flags), flush=True)

    print("switched", switched, "/", total)
    print("=== 2D ===")
    summary2 = {k: summarize(np.concatenate(v)) for k, v in pooled2.items()}
    for k, s in summary2.items():
        print(f"{k:10s} acc {s['acc_avg']:.3f}  med {s['epe_median']:.2f}  mean {s['epe_mean']:.2f}")
    print("=== 3D ===")
    summary3 = {k: summarize3d(np.concatenate(v)) for k, v in pooled3.items()}
    for k, s in summary3.items():
        print(f"{k:10s} acc {s['acc_avg']:.3f}  med {s['error_median_mm']:.2f}  mean {s['error_mean_mm']:.2f}")
    print("hard name n_switch n fused gated query raft")
    for row in hard:
        print(row)
    (OUT / "summary_gated_v6.json").write_text(json.dumps({"2d": summary2, "3d": summary3, "switched": switched, "total": total}, indent=2))


if __name__ == "__main__":
    main()