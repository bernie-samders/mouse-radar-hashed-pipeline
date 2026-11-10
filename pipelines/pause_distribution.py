r"""pause_distribution.py -- breath-PAUSE length distribution from the two-template segmentation.

Every respiratory cycle from the DP segmentation is a breath-MOTION segment (kind 0) followed by an
end-expiratory PAUSE segment (kind 1).  The pause length is where respiratory arrhythmia / near-apnea
burden lives.  This pipeline pools the per-cycle pause lengths over the FULL substrate (NOT AC-gated --
the arousal gate removes exactly the long-pause episodes that carry the burden) and reports:

  * a histogram  -> <out>/<rec>.pause_hist.png  (linear + log-x panels)  and the bin counts as a CSV
  * summary stats -> the returned flat dict (+ <out>/<rec>.pauses.csv): median / p90 / p95 / p99 / max,
    the fraction of cycles with a pause exceeding 0.5 / 1 / 2 s, and the fraction right-censored at the
    2.5 s DP ceiling.

Pause is right-censored at bmaxP = 2.5 s, so summarise the TAIL (p90, frac>tau), never the floor-dominated
mean.  Runs off the cached substrate only -- no raw-data access.
"""
from __future__ import annotations
import os, sys, json, csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for _p in (os.path.join(_PKG, "engine"), _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import substrate as SUB

PAUSE_CENSOR_S = 2.49          # bmaxP = 2.5 s -> pauses at/above this are right-censored
HIST_MAX_S = 2.6               # x-limit for the linear histogram (just past the 2.5 s ceiling)
N_BINS = 60


def pause_stats(tpause):
    tp = np.asarray(tpause, float)
    if tp.size == 0:
        return dict(n_cycles=0)
    return dict(
        n_cycles=int(tp.size),
        pause_med_ms=round(float(np.median(tp)) * 1000, 1),
        pause_mean_ms=round(float(np.mean(tp)) * 1000, 1),
        pause_p90_ms=round(float(np.percentile(tp, 90)) * 1000, 1),
        pause_p95_ms=round(float(np.percentile(tp, 95)) * 1000, 1),
        pause_p99_ms=round(float(np.percentile(tp, 99)) * 1000, 1),
        pause_max_ms=round(float(tp.max()) * 1000, 1),
        frac_pause_gt_0p5s=round(float(np.mean(tp > 0.5)), 4),
        frac_pause_gt_1s=round(float(np.mean(tp > 1.0)), 4),
        frac_pause_gt_2s=round(float(np.mean(tp > 2.0)), 4),
        censored_2p5_frac=round(float(np.mean(tp >= PAUSE_CENSOR_S)), 4),
    )


def _write_hist_csv(tpause, path, n_bins=N_BINS, hi=HIST_MAX_S):
    counts, edges = np.histogram(np.clip(tpause, 0, hi), bins=n_bins, range=(0, hi))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["bin_lo_s", "bin_hi_s", "count"])
        for i in range(n_bins):
            w.writerow([round(float(edges[i]), 4), round(float(edges[i + 1]), 4), int(counts[i])])


def _plot(tpause, rec, path, hi=HIST_MAX_S, n_bins=N_BINS):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[pauses] matplotlib unavailable, skipping PNG ({e})")
        return False
    tp = np.asarray(tpause, float)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    ax0.hist(np.clip(tp, 0, hi), bins=n_bins, range=(0, hi), color="#0072B2", alpha=0.85)
    ax0.axvline(2.5, color="#D55E00", ls="--", lw=1, label="2.5 s DP ceiling")
    ax0.set_xlabel("end-expiratory pause length (s)"); ax0.set_ylabel("cycle count")
    ax0.set_title(f"{rec}  (n={tp.size} cycles)"); ax0.legend(fontsize=8)
    pos = tp[tp > 0]
    if pos.size:
        ax1.hist(pos, bins=np.logspace(np.log10(max(pos.min(), 1e-3)), np.log10(hi), n_bins),
                 color="#0072B2", alpha=0.85, density=True)
        ax1.set_xscale("log")
    ax1.set_xlabel("pause length (s, log)"); ax1.set_ylabel("density")
    ax1.set_title("log-x (tail shape)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return True


# ---- pipeline entry point -------------------------------------------------------------------------
NAME = "pauses"
KIND = "segments"


def run(rec, sub_dir=None, out_dir=None, log=print):
    """Pause-length distribution for one recording.  Writes a histogram PNG + bins CSV + stats CSV, and
    returns the flat stats dict."""
    if sub_dir is not None:
        SUB.SUB_DIR = sub_dir
    info = SUB.rec_info(rec)
    cycles = SUB.build_cycle_series(rec)["cycles"]           # FULL substrate (burden view; not AC-gated)
    tpause = np.array([c["tpause"] for c in cycles]) if cycles else np.array([])
    stats = dict(recording=rec, cohort=info["cohort"], genotype=info["genotype"], **pause_stats(tpause))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, rec + ".pauses.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(stats.keys()); w.writerow(stats.values())
        if tpause.size:
            _write_hist_csv(tpause, os.path.join(out_dir, rec + ".pause_hist.csv"))
            _plot(tpause, rec, os.path.join(out_dir, rec + ".pause_hist.png"))
        json.dump(stats, open(os.path.join(out_dir, rec + ".pauses.json"), "w"), indent=1, default=float)
    log(f"[pauses]   {rec:20s} n={stats.get('n_cycles')} med={stats.get('pause_med_ms')}ms "
        f"p90={stats.get('pause_p90_ms')}ms frac>1s={stats.get('frac_pause_gt_1s')}")
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="pause-length distribution for one segmented recording")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--sub-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.rec, sub_dir=a.sub_dir, out_dir=a.out)
