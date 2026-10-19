# Result summaries

Small summaries behind the numbers reported in the manuscript. Each file was
written by the script named below (scripts are in `../scripts/`). Raw data,
meshes, network weights, and the hospital CT data are not included, and
absolute local paths have been removed.

| Folder | Produced by | Contents |
|---|---|---|
| `stir_tracking/` | `run_stir_tracking.py`, `run_stir_query_refresh.py`, `fuse_stir_gated.py`, `run_stir_raft.py`, `run_stir_raft3d.py`, `evaluate_stir_tracking.py`, `evaluate_stir_tracking_3d.py` | 2D tracking on 60 STIR clips (`summary_v7_2d.json`, `per_clip_v7_2d.csv`: fused tracking with reference refresh, control, and chained RAFT) and 3D tracking on 59 clips (`summary3d_v7_3d.json`, `per_clip3d_v7_3d.csv`: including RAFT with SGM depth; `summary3d_all.json`: including RAFT-Stereo). Files named `*_all60*` hold tracking without and with flow initialization; `*_subset20*` hold the EndoGaussian subset. |
| `fem_validation_60/` | `run_fem60_campaign.py` | Two-region inversion on the 60-scenario finite-element benchmark. |
| `fem_bg_only40_60/` | `fem_bg_only40_expand.py`, `fem_holdout_per_load_60.py` | Homogeneous estimate, training-residual selection, and unseen-load errors. |
| `fem_model_mismatch/` | `run_fem_model_mismatch.py`, `run_fem_mismatch_holdout.py` | Mooney-Rivlin and refined-mesh data generators. |
| `force_propagation/` | `run_force_propagation.py` | Video-force error model and force-scale sweep. |
| `ets_phantom/` | `run_ets_audit_demo.py` | Force-free stiffness contrast on the ETS indentation phantom. |
| `depoll_registration/` | `run_depoll_registration.py`, `depoll_shape_ambiguity.py` | Same-animal DePoLL registration. |
| `endonerf_common/` | `aggregate_endonerf_common.py` | Repeated canonical fits on EndoNeRF. |
| `scared_tier1/` | `run_scared_tier1.py` | Canonical reconstruction on SCARED keyframes. |
