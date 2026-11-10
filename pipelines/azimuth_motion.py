r"""azimuth_motion.py -- azimuth-only motion metrics per minute (actigraphy), from the raw power cube.

Adapted from the validated `ml_apnea/rearing/motion_track.py`, restricted to the AZIMUTH (angle) axis --
the finding that carried the genotype signal ("ang_path_per_min", d ~ -0.85: sicker mice travel less in
azimuth).  Method (whole recording, decimated, streaming the FULL cube):

  * BACKGROUND SUBTRACTION = per-voxel WINDOWED VARIANCE of the magnitude (static clutter -> ~0 variance;
    the animal, breathing OR moving, -> high variance).  An MTI-style mover map over all voxels.
  * AZIMUTH POSITION per frame = variance-weighted CENTROID of the top-N highest-variance voxels, projected
    onto the ANGLE axis (robust vs argmax teleport).  A continuous azimuth trajectory, present even where no
    clean breathing is resolvable (the animal is moving).
  * MOTION = path length of that azimuth trajectory (median-smoothed, per-step capped to reject residual
    jumps), converted to degrees using the header's angle spacing; plus azimuth span and fraction moving.

Reads the raw experiment directly (no segmentation substrate needed).  Decimated to ~15 Hz: azimuth
position is slow, so this keeps the full-cube read fast.
"""
from __future__ import annotations
import os, sys, json, csv, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for _p in (os.path.join(_PKG, "engine"), _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import voxel_store as VS

EPS = 1e-12
WIN_S, HOP_S, DEC_HZ = 1.0, 0.5, 15.0
TOPN = 8                     # centroid over the N highest-variance voxels per frame
BATCH_S = 240.0
STEP_CAP_IDX = 8.0          # cap per-step azimuth motion (angle-index units) to reject residual jumps
MOVE_THR_IDX = 0.5         # azimuth step (angle-index units) above which a frame counts as "moving"


def azimuth_trajectory(power, meta, fs, max_min=None, log=print):
    """Return (t, ai_centroid, energy): the variance-weighted azimuth (angle-index) centroid per frame."""
    nb = int(meta["n_bin"]); na = int(meta["n_angle"]); nv = na * nb
    N = int(power.shape[0]) if max_min is None else min(int(power.shape[0]), int(max_min * 60 * fs))
    q = max(1, int(round(fs / DEC_HZ))); fs2 = fs / q; N2 = N // q
    nps = max(2, int(round(WIN_S * fs2))); hop = max(1, int(round(HOP_S * fs2)))
    starts = np.arange(0, max(1, N2 - nps + 1), hop)
    t = (starts + nps / 2.0) / fs2; nt = starts.size
    ai_c = np.zeros(nt, np.float32); en = np.zeros(nt, np.float32)
    ii = np.arange(nv); ang_of = (ii // nb).astype(np.float32)
    batch2 = max(nps, int(round(BATCH_S * fs2)))
    t0 = time.time()
    for b0 in range(0, N2, batch2):
        fi0 = int(np.searchsorted(starts, b0, "left"))
        fi1 = int(np.searchsorted(starts, min(N2, b0 + batch2) - nps + 1, "left"))
        if fi1 <= fi0:
            continue
        lo = int(starts[fi0]); hi = int(starts[fi1 - 1] + nps)
        stripe = np.sqrt(np.maximum(np.asarray(power[lo * q:hi * q:q, :], np.float64), 0.0))   # [L, nv]
        csp = np.zeros((stripe.shape[0] + 1, nv)); cs2 = np.zeros((stripe.shape[0] + 1, nv))
        np.cumsum(stripe, axis=0, out=csp[1:]); np.cumsum(stripe * stripe, axis=0, out=cs2[1:])
        s = starts[fi0:fi1] - lo
        mean = (csp[s + nps] - csp[s]) / nps
        var = np.maximum((cs2[s + nps] - cs2[s]) / nps - mean * mean, 0.0)                     # [nf, nv]
        for k in range(var.shape[0]):
            v = var[k]
            top = np.argpartition(v, -TOPN)[-TOPN:]
            w = v[top]; sw = w.sum() + EPS
            ai_c[fi0 + k] = (w * ang_of[top]).sum() / sw
            en[fi0 + k] = float(w.sum())
        del stripe, csp, cs2, var
        log(f"    [azimuth] {hi/fs2:.0f}s ({time.time()-t0:.0f}s)")
    return t, ai_c, en, fs2


def azimuth_metrics(t, ai_c, meta, dur_min):
    from scipy.ndimage import median_filter
    angles = np.asarray(meta["angles_deg"], float)
    deg_per_idx = float(np.median(np.abs(np.diff(angles)))) if angles.size > 1 else 1.0
    hop_s = float(np.median(np.diff(t))) if t.size > 1 else HOP_S
    k = max(1, int(round(1.5 / hop_s)) | 1)               # ~1.5 s median smoother
    ais = median_filter(ai_c, k)
    step = np.abs(np.diff(ais))
    step = np.minimum(step, STEP_CAP_IDX)                  # cap per-step (angle-index units)
    path_idx_per_min = float(step.sum() / dur_min) if dur_min > 0 else float("nan")
    visited = np.round(ais).astype(int)
    return dict(
        azimuth_path_deg_per_min=round(path_idx_per_min * deg_per_idx, 2),
        azimuth_path_idx_per_min=round(path_idx_per_min, 3),
        azimuth_span_deg=round(float(ais.max() - ais.min()) * deg_per_idx, 2),
        n_angles_visited=int(len(np.unique(visited))),
        frac_moving=round(float(np.mean(step > MOVE_THR_IDX)), 4),
        azimuth_speed_p90_deg_s=round(float(np.percentile(step, 90)) * deg_per_idx / hop_s, 3),
        total_min=round(dur_min, 2),
    )


# ---- pipeline entry point -------------------------------------------------------------------------
NAME = "motion"
KIND = "raw"          # consumes the raw experiment directory, not a segmentation substrate


def run(exp_dir, out_dir=None, max_min=None, save_trajectory=True, log=print):
    """Azimuth motion metrics for one raw experiment.  Writes <out>/<rec>.motion.csv (+ .json, + an
    optional trajectory .npz) and returns the flat metric dict."""
    power, meta, cfg = VS.load_experiment(exp_dir)
    rec = VS.rec_name(exp_dir)
    fs = float(meta["fs"])
    N = int(power.shape[0]) if max_min is None else min(int(power.shape[0]), int(max_min * 60 * fs))
    dur_min = N / fs / 60.0
    t, ai_c, en, fs2 = azimuth_trajectory(power, meta, fs, max_min=max_min, log=log)
    m = dict(recording=rec, **azimuth_metrics(t, ai_c, meta, dur_min))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, rec + ".motion.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(m.keys()); w.writerow(m.values())
        json.dump(m, open(os.path.join(out_dir, rec + ".motion.json"), "w"), indent=1, default=float)
        if save_trajectory:
            np.savez_compressed(os.path.join(out_dir, rec + ".motion.npz"),
                                t=t.astype(np.float32), ai=ai_c, energy=en.astype(np.float32))
    log(f"[motion]   {rec:20s} azimuth_path={m['azimuth_path_deg_per_min']} deg/min "
        f"span={m['azimuth_span_deg']} deg frac_moving={m['frac_moving']}")
    return m


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="azimuth-only motion metrics for one raw experiment")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-min", type=float, default=None)
    a = ap.parse_args()
    run(a.exp, out_dir=a.out, max_min=a.max_min, log=lambda *x: print(*x, flush=True))
