r"""rate.py -- RATE-LEVEL radar heart rate: beats where b2b tracked them, else the rate tracker (new in 1.8; RATE.md).

The paper reports heart rate at two levels. The beat level is the `b2b` step (B2B.md): individual heartbeats, only where
the heart is strong. The rate level is this step: a 2 s heart-rate value wherever EITHER b2b tracked OR the rate tracker
gives a trustworthy value, so it covers more of the clean breathing (and all of the beat level).

Runs AFTER `b2b`. What it does for one recording (no ECG anywhere):
  1. `hr` on every still, regular-breathing period >= 6 s (not b2b's 20 s floor)   -> <out>/rate_tracker/<rec>.hr.*
  2. `hr_curve --band continuity` on those periods (pipelines/hr_curve.py, the pll -> <out>/rate_tracker/<rec>.hr_curve.*
     particle-filter tracker of packages 1.4-1.6, engine/pll_tracker/, HR_CURVE.md)
  3. per 2 s bin (the b2b grid: centres 1, 3, 5 ... s):
       b2b value where b2b reported one (<rec>.b2b.npz; 60 / median beat interval within +-3 s, >= 5 intervals);
       else the mean of the 25 Hz curve within +-3 s, if the bin centre lies in a tracker period that is trusted:
       heart-SNR grade ok / strong (q1 > 0.5, i.e. above -3 dB) and period curve mean <= 750 bpm (the 1.7 guard value:
       the tracker also grades the non-cardiac ~780-830 bpm line `strong`).
Steps 1-2 are exactly `run.py hr --min-period-s 6` + `run.py hr_curve --band continuity` of package 1.6, the recipe
behind the paper's rate-level numbers (output/mecp2_hr_rerun/, output/hr_ratecov/); step 3 is those analyses' rule
(output/paper_figs/ecg_validation/common.py pkg_bins / b2b_bins). The tracker step's own outputs stay in
<out>/rate_tracker/ so they cannot be mistaken for the default `hr` outputs (>= 20 s periods) that b2b reads.

Outputs (<out> = metrics dir): <rec>.rate.{csv,json} (one row), <rec>.rate_bins.csv (the 2 s values and their source),
<rec>.rate_periods.csv (one row per >= 6 s period: used or why not).
"""
from __future__ import annotations
import os, sys, json, csv, glob, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for _p in (os.path.join(_PKG, "engine"), _HERE, _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import voxel_store as VS
import hr_bands as HB
import hr as HR
import hr_curve as HC
import b2b as B2B

NAME = "rate"
KIND = "segments"
NEEDS_RAW = True              # run.py passes exp_dir= (the raw folder); falls back to the substrate's session_dir
PARALLEL = True               # run.py passes workers=
DEJUMP = True                 # run.py hands it the de-jumped cube copy of a July-2026 capture (DEJUMP.md)

TRACKER_SUBDIR = "rate_tracker"
MIN_PERIOD_S = 6.0            # tracker periods [chosen 2026-11-11 after 529 / 530 of 11-10; b2b keeps hr's 20 s floor]
BAND = "continuity"           # band centre by Viterbi across periods [ng511 recipe, 2026-10-16]
TRUST = ("ok", "strong")      # heart-SNR grades used (hr_curve.SNR_MIN_Q1 = 0.5: above -3 dB)
HR_MAX_BPM = B2B.HR_MAX_BPM   # 750: a period whose curve mean is above it is dropped [ECG max 612; line ~780-830]
BIN_S, WIN_S = B2B.BIN_S, B2B.WIN_S   # 2 s grid, +-3 s window (the b2b values' grid)
CLEAN_FRAC = 0.5              # a bin is clean breathing when >= half of it lies inside a clean run

PERIOD_COLS = ["recording", "period", "t0_s", "t1_s", "start_min", "dur_s", "breaths_per_min", "band_centre_bpm",
               "tracked", "heart_snr_db", "snr_grade", "curve_mean_bpm", "used", "reason", "n_bins_curve", "n_bins_beats"]


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def tracker_periods(trk, rec):
    """Every tracker period with its 25 Hz curve ([] samples if not tracked) and why it is used or not."""
    npz = os.path.join(trk, rec + ".hr_curve.npz")
    z = np.load(npz) if os.path.exists(npz) else None
    out = []
    for c in _rows(os.path.join(trk, rec + ".hr_curve_periods.csv")):
        per = int(c["period"])
        s = (z["period"] == per) if z is not None else np.zeros(0, bool)
        t = z["t_s"][s] if z is not None else np.zeros(0)
        hr = z["hr_bpm"][s] if z is not None else np.zeros(0)
        grade = c.get("snr_grade") or "none"
        mean = float(np.mean(hr)) if hr.size else None
        reason = ("" if grade in TRUST and hr.size and mean <= HR_MAX_BPM else
                  "curve_above_max" if grade in TRUST and hr.size else
                  "breathing_halved" if grade == "invalid" else "low_heart_snr" if grade == "low" else "not_tracked")
        out.append(dict(c, period=per, t0=float(c["t0_s"]), t1=float(c["t1_s"]), grade=grade, t=t, hr=hr,
                        curve_mean=mean, reason=reason))
    return out


def curve_bins(c, P):
    """Mean of the curve within +-3 s of each bin centre inside a used period; NaN elsewhere. Also the period per bin."""
    out, per = np.full(c.size, np.nan), np.zeros(c.size, int)
    for p in P:
        if p["reason"]:
            continue
        for i in np.flatnonzero((c >= p["t0"]) & (c < p["t1"])):
            k = np.abs(p["t"] - c[i]) <= WIN_S / 2
            if k.any():
                out[i] = p["hr"][k].mean(); per[i] = p["period"]
    return out, per


def beat_bins(c, out_dir, rec, rec_s):
    """The b2b step's 2 s values on the same grid (recomputed from its beats at full precision); NaN where none."""
    out, per = np.full(c.size, np.nan), np.zeros(c.size, int)
    npz = os.path.join(out_dir, rec + ".b2b.npz")
    if not os.path.exists(npz):
        return out, per, False
    beats = {str(k): np.asarray(v, float) for k, v in np.load(npz).items()}
    P = [dict(period=int(r["period"]), t0=float(r["t0_s"]), t1=float(r["t1_s"]))
         for r in _rows(os.path.join(out_dir, rec + ".b2b_periods.csv")) if r.get("tracked") == "yes"]
    for t, k, h in B2B.hr_bins(P, beats, rec_s):
        i = int(round((t - BIN_S / 2) / BIN_S))
        out[i] = h; per[i] = int(k)
    return out, per, True


def _overlap(c, runs):
    f = np.zeros(c.size)
    for a, b in runs:
        f += np.clip(np.minimum(c + BIN_S / 2, b) - np.maximum(c - BIN_S / 2, a), 0, None) / BIN_S
    return f


def analyze(rec, sub_dir, out_dir, exp_dir=None, workers=None, log=print, reuse_tracker=False):
    """Everything for one recording. Returns (summary metrics, period rows, bin rows, info)."""
    T0 = time.time()
    seg_base = os.path.join(sub_dir, rec)
    seg_json = json.load(open(seg_base + ".json"))
    trk = os.path.join(out_dir, TRACKER_SUBDIR)
    os.makedirs(trk, exist_ok=True)
    have = os.path.exists(os.path.join(trk, rec + ".hr.json")) and os.path.exists(os.path.join(trk, rec + ".hr_curve.json"))
    if not (reuse_tracker and have):
        log(f"[rate] {rec}: rate tracker = hr on periods >= {MIN_PERIOD_S:g} s, then hr_curve --band {BAND} "
            f"(outputs in {trk})")
        HR.run(rec, sub_dir=sub_dir, out_dir=trk, log=log, exp_dir=exp_dir, workers=workers, min_period_s=MIN_PERIOD_S)
        HC.run(rec, sub_dir=sub_dir, out_dir=trk, log=log, exp_dir=exp_dir, workers=workers, band=BAND)
    hm = json.load(open(os.path.join(trk, rec + ".hr.json")))["metrics"]
    cm = json.load(open(os.path.join(trk, rec + ".hr_curve.json")))["metrics"]

    exp_dir = exp_dir or seg_json.get("session_dir")
    if exp_dir and glob.glob(os.path.join(exp_dir, "*_voxels.json")):
        power, vmeta, _ = VS.load_experiment(exp_dir)
        rec_s = power.shape[0] / float(vmeta["fs"])
    else:                                                     # the bins only need the recording length
        rec_s = float(seg_json["rec_min"]) * 60
    c = np.arange(BIN_S / 2, rec_s, BIN_S)
    P = tracker_periods(trk, rec)
    cur, cper = curve_bins(c, P)
    bt, bper, have_b2b = beat_bins(c, out_dir, rec, rec_s)
    use_b = np.isfinite(bt)
    rate = np.where(use_b, bt, cur)
    ok = np.isfinite(rate)
    _, _, runs = HB.breathing(seg_base, 0.0)
    clean = _overlap(c, runs) >= CLEAN_FRAC

    notes = []
    if not have_b2b:
        notes.append("the b2b step has not run on this output folder: the rate level here is the rate tracker only -- "
                     "run `python run.py b2b` first, then this step again")
    frozen = dict(hr=bool(hm.get("frozen_code_ok")), tracker=bool(cm.get("frozen_code_ok")),
                  b2b=B2B.frozen_ok() if have_b2b else None)
    if not all(v for v in frozen.values() if v is not None):
        notes.append("a frozen heart-rate file differs from the validated one (" +
                     ", ".join(k for k, v in frozen.items() if v is False) + ") -- results are not the validated method")
    if cm.get("breathing_halved"):
        notes.append("breathing halved (engine/breath_check.py): the tracker's grades are `invalid`, so only b2b values "
                     "are left, and those are not trustworthy either -- re-segment")
    n_cap = sum(p["reason"] == "curve_above_max" for p in P)
    if n_cap:
        notes.append(f"{n_cap} tracker period(s) dropped by the {HR_MAX_BPM:.0f} bpm guard (curve mean above it: the "
                     f"non-cardiac ~800 bpm line, not a mouse heartbeat on our ECG sessions)")
    if cm.get("snr_verdict") == "LOW":
        notes.append("the tracker graded the recording LOW heart SNR: fewer than half of its still minutes clear -3 dB "
                     "(only the ok / strong periods are used here)")

    prow = []
    for p in P:
        s = (c >= p["t0"]) & (c < p["t1"])
        prow.append(dict(recording=rec, period=p["period"], t0_s=p["t0_s"], t1_s=p["t1_s"], start_min=p.get("start_min"),
                         dur_s=p.get("dur_s"), breaths_per_min=p.get("breaths_per_min"),
                         band_centre_bpm=p.get("band_centre_bpm"), tracked=p.get("tracked"),
                         heart_snr_db=p.get("heart_snr_db"), snr_grade=p["grade"],
                         curve_mean_bpm=None if p["curve_mean"] is None else round(p["curve_mean"], 2),
                         used="no" if p["reason"] else "yes", reason=p["reason"],
                         n_bins_curve=int((s & ok & ~use_b).sum()), n_bins_beats=int((s & use_b).sum())))
    brow = [(float(c[i]), float(rate[i]), "beats" if use_b[i] else "curve", int(bper[i] if use_b[i] else cper[i]),
             bool(clean[i])) for i in np.flatnonzero(ok)]
    q = lambda p: round(float(np.percentile(rate[ok], p)), 1) if ok.any() else None
    mins = lambda m: round(float(m.sum()) * BIN_S / 60, 2)
    m = dict(recording=rec, cohort=seg_json.get("cohort"), age_weeks=seg_json.get("age_weeks"),
             hr_median_bpm=q(50), hr_p25_bpm=q(25), hr_p75_bpm=q(75),
             covered_min=mins(ok), beats_min=mins(use_b), curve_min=mins(ok & ~use_b),
             clean_breathing_min=mins(clean), covered_clean_min=mins(ok & clean),
             covered_clean_frac=round(float((ok & clean).sum() / clean.sum()), 3) if clean.any() else None,
             n_periods=len(P), n_used=sum(not p["reason"] for p in P),
             n_not_tracked=sum(p["reason"] == "not_tracked" for p in P),
             n_low_snr=sum(p["reason"] == "low_heart_snr" for p in P), n_guard_hr=n_cap,
             tracker_snr_verdict=cm.get("snr_verdict"), min_period_s=MIN_PERIOD_S, band_mode=BAND,
             hr_max_guard_bpm=HR_MAX_BPM, b2b_merged=have_b2b,
             frozen_code_ok=all(v for v in frozen.values() if v is not None),
             breathing_halved=cm.get("breathing_halved"), cube_dejumped=cm.get("cube_dejumped"),
             runtime_s=round(time.time() - T0, 1))
    m["notes"] = " | ".join(notes)
    info = dict(exp_dir=exp_dir, segments=seg_base, tracker_dir=trk, frozen=frozen,
                rule=f"per {BIN_S:g} s bin: b2b value, else mean of the 25 Hz curve within +-{WIN_S / 2:g} s inside a "
                     f"tracker period graded {' / '.join(TRUST)} whose curve mean is <= {HR_MAX_BPM:g} bpm")
    return m, prow, brow, info


def run(rec, sub_dir=None, out_dir=None, log=print, exp_dir=None, workers=None, reuse_tracker=False):
    """Rate-level heart rate for one recording whose b2b step has run. Returns the flat summary dict."""
    import substrate as SUB
    sub_dir = sub_dir or SUB.SUB_DIR
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(sub_dir)), "metrics")
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, glob.escape(rec) + ".rate*")):
        os.remove(f)
    m, prow, brow, info = analyze(rec, sub_dir, out_dir, exp_dir=exp_dir, workers=workers, log=log,
                                  reuse_tracker=reuse_tracker)
    with open(os.path.join(out_dir, rec + ".rate.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(m.keys()); w.writerow(m.values())
    with open(os.path.join(out_dir, rec + ".rate_periods.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, PERIOD_COLS, extrasaction="ignore"); w.writeheader(); w.writerows(prow)
    with open(os.path.join(out_dir, rec + ".rate_bins.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["t_s", "hr_bpm", "source", "period", "clean_breathing"])
        w.writerows([f"{t:.1f}", f"{h:.2f}", s, k, int(cl)] for t, h, s, k, cl in brow)
    json.dump(dict(metrics=m, periods=prow, info=info), open(os.path.join(out_dir, rec + ".rate.json"), "w"), indent=1,
              default=float)
    med = "n/a" if m["hr_median_bpm"] is None else f"{m['hr_median_bpm']:.0f} bpm"
    cov = "n/a" if m["covered_clean_frac"] is None else f"{100 * m['covered_clean_frac']:.0f} %"
    log(f"[rate] {rec:20s} median {med} over {m['covered_min']} min of 2 s values ({m['beats_min']} from beats, "
        f"{m['curve_min']} from the rate tracker) | {cov} of {m['clean_breathing_min']} min clean breathing | tracker "
        f"periods used {m['n_used']}/{m['n_periods']} (low SNR {m['n_low_snr']}, no signal {m['n_not_tracked']}, "
        f"750 guard {m['n_guard_hr']})  [{m['runtime_s']:.0f}s]" + (f"\n           NOTE: {m['notes']}" if m["notes"] else ""))
    return m


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="rate-level radar heart rate: b2b beats, else the rate tracker (run after b2b)")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--sub-dir", default=None)
    ap.add_argument("--exp", default=None, help="raw experiment folder (default: the substrate's session_dir)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--reuse-tracker", action="store_true", help="keep an existing rate_tracker/ run; redo only the merge")
    a = ap.parse_args()
    run(a.rec, sub_dir=a.sub_dir, out_dir=a.out, exp_dir=a.exp, workers=a.workers, reuse_tracker=a.reuse_tracker)
