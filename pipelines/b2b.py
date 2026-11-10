r"""b2b.py -- BEAT-TO-BEAT radar heart rate inside each still, regular-breathing period (new in 1.7; B2B.md).

The paper's headline heart-rate method. Runs AFTER the `hr` step and reuses its periods, chest patch and period heart
rate (<rec>.hr.json). Per period (no ECG anywhere):
  1. breath-order residual Y of the chest-patch acceleration (engine/hr_final_pipeline.py, the frozen hr code; the same
     Y the frozen hr code uses);
  2. engine/b2b_beats.estimate(Y, seed, t0): space-time whitened matched filter + second-order beat Viterbi, run from
     11 starts, agreement medoid; abstains unless the kept run's median matched-filter output at its beats >= 2.0 noise
     SD. seed = the hr step's period heart rate;
  3. the same estimate at seed x .998 / .999 / 1.001 / 1.002 (family_agree: a stability flag; single draws are
     lock-chaotic). Only the x1 draw is reported.
High-HR guard (new in 1.7, set before the confirmatory data): a period whose seed or whose b2b heart rate is above
HR_MAX_BPM (750) is not reported. The hr step's top band (centre 780 bpm) can lock onto a non-cardiac line at ~780-830
bpm, and b2b seeded there once read 1058 bpm (NAS Animal_B_16_6wks). On the 9 ECG sessions every seed above 650 was
>= 823 and research b2b never reported above 612, so the guard changes no ECG-validated number; on the 83 NAS cubes 750
and 800 report the same periods, and 750 keeps a margin below that line.
Recording heart rate (the paper's statistic, = output/nas_batch/run_b2b.py): on a 2 s grid inside the tracked periods,
60 / median of the beat intervals whose midpoints lie within +-3 s (>= 5 intervals); hr_median_bpm = median of those
values, hr_covered_min = their count x 2 s. hr_per_breath_median = median over tracked periods of period HR / period
breathing rate.
engine/b2b_beats.py is byte-identical to the validated release (sha256 checked on every run).

Outputs (<out> = metrics dir): <rec>.b2b.{csv,json} (one row), <rec>.b2b_periods.csv (one row per hr period),
<rec>.b2b_beats.csv (one row per beat), <rec>.b2b_bins.csv (the 2 s heart-rate values), <rec>.b2b.npz ({period: beat
times in s}, the research est.npz layout).
Research provenance: output/ecg_0928/hr_overnight/short_runs/b2b/elegant/release/ (method) + output/508_1102/hr/b2b.py
(driver) + output/paper_figs/ecg_validation/ (ECG scores).
"""
from __future__ import annotations
import os, sys, json, csv, glob, time, shutil, hashlib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for _p in (os.path.join(_PKG, "engine"), _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import voxel_store as VS
import hr_bands as HB

NAME = "b2b"
KIND = "segments"
NEEDS_RAW = True              # run.py passes exp_dir= (the raw folder); falls back to the hr step's / substrate's folder
PARALLEL = True               # run.py passes workers=
DEJUMP = True                 # run.py hands it the de-jumped cube copy of a July-2026 capture (DEJUMP.md)

FROZEN_FILE = os.path.join(_PKG, "engine", "b2b_beats.py")
FROZEN_SHA256 = "a3ffffc38659b2e6c6672b4eac2a1ae4e0e75251cc968e3e072d4a53406d2960"   # release 2026-10-14, as run for Fig 2
DRAWS = (1.0, 0.998, 0.999, 1.001, 1.002)   # seed family; x1 = the estimate
HR_MAX_BPM = 750.0            # high-HR guard on seed and result [ECG max 612 bpm; non-cardiac line ~780-830 bpm]
BIN_S, WIN_S, MIN_IBI = 2.0, 6.0, 5   # recording HR: 2 s grid, +-3 s window, >= 5 beat intervals (nas_batch/run_b2b.py)
_BLAS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
_W = {}

PERIOD_COLS = ["recording", "period", "t0_s", "t1_s", "start_min", "dur_s", "breaths_per_min", "seed_hr_bpm",
               "tracked", "abstain_reason", "snr", "n_beats", "hr_bpm", "hr_unguarded_bpm", "family_abstain",
               "family_agree"]


def default_workers():
    return max(1, min(12, (os.cpu_count() or 2) - 2))


def frozen_ok():
    return hashlib.sha256(open(FROZEN_FILE, "rb").read()).hexdigest() == FROZEN_SHA256


def _init(cfg):
    import harness as H
    H.register(cfg["rec"], cfg["cube"], cfg["shape"], cfg["cols"], cfg["on"], cfg["tt"])
    _W.update(cfg)


def _build_y(p):
    """Breath-order residual Y of one period (float32 npy, as the research driver cached it)."""
    import harness as H, hr_final_pipeline as FP
    yp = os.path.join(_W["work"], f"Y_{p['t0']:.3f}.npy")
    acc = os.path.join(_W["work"], f"A_{p['t0']:.3f}.npy")
    HB.build_accel(((_W["cube"], _W["shape"]), _W["cols"], p["t0"], p["t1"], acc))
    X = np.asarray(H.patch_accel(_W["rec"], dict(accel_patch=acc)), float)
    Y = FP.residual(X, FP.breath_phase(_W["rec"], dict(t0=p["t0"], t1=p["t1"]), X.shape[0], p["rr"]), p["rr"])
    np.save(yp, Y.astype(np.float32))
    os.remove(acc)
    return yp


def _est(args):
    pid, yp, seed, t0, f = args
    import b2b_beats as BB
    beats, snr, info = BB.estimate(np.load(yp), seed * f, t0)
    return pid, f, (None if beats is None else np.asarray(beats, float)), float(snr), dict(info)


def _agree(a, b, tol=0.010):
    """Share of a's beats with a b beat within tol after one constant lag (median nearest offset)."""
    if a is None or b is None or a.size < 3 or b.size < 3:
        return np.nan
    j = np.clip(np.searchsorted(b, a), 1, b.size - 1)
    d = np.where(np.abs(b[j] - a) < np.abs(b[j - 1] - a), b[j] - a, b[j - 1] - a)
    return float(np.mean(np.abs(d - np.median(d)) <= tol))


def _hr(beats):
    return None if beats is None or beats.size < 3 else float(60.0 / np.median(np.diff(beats)))


def hr_bins(P, beats, rec_s):
    """[(t_s, period, hr_bpm)] on the 2 s grid inside the tracked periods (the paper's 2 s heart-rate values)."""
    c = np.arange(BIN_S / 2, rec_s, BIN_S); out = []
    for r in P:
        b = beats.get(str(r["period"]))
        if b is None or b.size < 3:
            continue
        mid, ibi = (b[1:] + b[:-1]) / 2, np.diff(b)
        for i in np.flatnonzero((c >= r["t0"]) & (c < r["t1"])):
            k = np.abs(mid - c[i]) <= WIN_S / 2
            if k.sum() >= MIN_IBI:
                out.append((float(c[i]), r["period"], float(60.0 / np.median(ibi[k]))))
    return out


def _summary(rec, seg_json, rows, beats, notes, extra, bins=()):
    tr = [r for r in rows if r["tracked"] == "yes"]
    hr = np.array([h for _, _, h in bins], float)
    q = lambda x, p: round(float(np.percentile(x, p)), 1) if x.size else None
    reason = lambda k: sum(r["abstain_reason"] == k for r in rows)
    ratio = [r["hr_bpm"] / r["breaths_per_min"] for r in tr if r["hr_bpm"] and r["breaths_per_min"]]
    m = dict(recording=rec, cohort=seg_json.get("cohort"), age_weeks=seg_json.get("age_weeks"),
             n_periods=len(rows), n_tracked=len(tr), tracked_min=round(sum(r["dur_s"] for r in tr) / 60, 2),
             n_beats=int(sum(b.size for b in beats.values())),
             hr_median_bpm=q(hr, 50), hr_p25_bpm=q(hr, 25), hr_p75_bpm=q(hr, 75),
             hr_covered_min=round(hr.size * BIN_S / 60, 2),
             hr_per_breath_median=round(float(np.median(ratio)), 3) if ratio else None,
             n_abstain_snr=reason("snr_below_gate"), n_guard_seed=reason("seed_above_max"),
             n_guard_hr=reason("hr_above_max"), hr_max_guard_bpm=HR_MAX_BPM)
    m.update(extra)
    m["notes"] = " | ".join(notes)
    return m


def analyze(rec, sub_dir, out_dir, exp_dir=None, workers=None, keep_work=False, log=print):
    """Everything for one recording. Returns (summary metrics, period rows, {period: beats}, 2 s HR values, info)."""
    from concurrent.futures import ProcessPoolExecutor
    T0 = time.time()
    seg_base = os.path.join(sub_dir, rec)
    seg_json = json.load(open(seg_base + ".json"))
    hr_json = os.path.join(out_dir, rec + ".hr.json")
    if not os.path.exists(hr_json):
        raise FileNotFoundError(f"{hr_json} not found -- b2b reuses the hr step's periods and chest patch: run "
                                f"`python run.py hr --rec {rec}` first")
    HRJ = json.load(open(hr_json))
    hm, hrows = HRJ["metrics"], HRJ["periods"]
    ok = frozen_ok()
    notes = []
    extra = dict(frozen_code_ok=ok, b2b_sha256=FROZEN_SHA256, breathing_halved=hm.get("breathing_halved"))
    if not ok:
        notes.append("engine/b2b_beats.py differs from the validated file -- results are not the validated method")
    if hm.get("breathing_halved"):
        notes.append("breathing halved (engine/breath_check.py): a breathing harmonic can pass for a heartbeat -- do "
                     "not use these beats")
        log(f"[b2b] WARNING {rec}: {notes[-1]}")
    if not hrows:
        notes.append("the hr step found no still period -- no beats")
        return _summary(rec, seg_json, [], {}, notes, extra), [], {}, [], {}

    exp_dir = exp_dir or HRJ.get("info", {}).get("exp_dir") or seg_json.get("session_dir")
    if not (exp_dir and glob.glob(os.path.join(exp_dir, "*_voxels.json"))):
        raise FileNotFoundError(f"raw experiment folder not found ({exp_dir!r}); b2b re-reads the voxels -- pass "
                                f"--exp <folder with *_voxels.f32>")
    power, vmeta, _ = VS.load_experiment(exp_dir)
    cube_path, shape = str(power.filename), tuple(power.shape)
    extra["cube_dejumped"] = bool(vmeta.get("dejump"))
    if hm.get("cube_dejumped") is not None and bool(hm["cube_dejumped"]) != extra["cube_dejumped"]:
        notes.append("the hr step ran on a " + ("de-jumped" if hm["cube_dejumped"] else "raw") + " cube but this step on a "
                     + ("de-jumped" if extra["cube_dejumped"] else "raw") + " one -- re-run both with the same --dejump")
        log(f"[b2b] WARNING {rec}: {notes[-1]}")
    if seg_json.get("rec_min") is not None and abs(float(seg_json["rec_min"]) - shape[0] / float(vmeta["fs"]) / 60) > 0.05:
        raise ValueError(f"{exp_dir!r} does not hold the {seg_json['rec_min']} min that {rec} was segmented from -- "
                         f"wrong --exp folder?")

    on, tt, runs = HB.breathing(seg_base, 0.0)
    lo, hi = (int(x) for x in str(hm["patch_beams"]).split("-"))
    cols = HB.patch_cols(vmeta, list(range(lo, hi + 1)))
    P = []
    for r in hrows:
        t0, t1 = float(r["t0_s"]), float(r["t1_s"])
        x = min(runs, key=lambda u: abs(u[0] - t0) + abs(u[1] - t1))     # exact edges (outputs round to 1 ms)
        if abs(x[0] - t0) + abs(x[1] - t1) >= 0.01:
            raise ValueError(f"hr period {r['period']} (t0 {t0} s) is not a clean-breathing run of the current segments "
                             f"-- were the segments re-made after the hr step? Re-run hr.")
        t0, t1 = x
        P.append(dict(period=int(r["period"]), t0=t0, t1=t1, seed=float(r["hr_bpm"]), rr=HB.run_rr(on, tt, t0, t1)))
    todo = [p for p in P if p["seed"] <= HR_MAX_BPM and p["rr"] is not None]
    log(f"[b2b] {rec}: {len(P)} still periods from the hr step ({sum(p['t1'] - p['t0'] for p in P) / 60:.1f} min), "
        f"chest patch beams {lo}-{hi}; {len(P) - len(todo)} seeded above {HR_MAX_BPM:.0f} bpm (or without a breathing "
        f"rate) are not run")

    work = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "b2b_work", rec)
    os.makedirs(work, exist_ok=True)
    workers = int(workers or default_workers())
    saved = {k: os.environ.get(k) for k in _BLAS}
    for k in _BLAS:
        os.environ[k] = "1"                       # workers: 1 BLAS thread each (the research driver ran OMP = 1)
    t = time.time(); res = {}
    try:
        cfg = dict(rec=rec, cube=cube_path, shape=shape, cols=cols, on=on, tt=tt, work=work)
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(cfg,)) as pool:
            ys = list(pool.map(_build_y, todo))
            log(f"[b2b] breath-order residuals built [{time.time() - t:.0f} s]; beat tracking {len(todo)} periods x "
                f"{len(DRAWS)} seed draws with {workers} workers ...")
            jobs = [(p["period"], y, p["seed"], p["t0"], f) for p, y in
                    sorted(zip(todo, ys), key=lambda u: u[0]["t0"] - u[0]["t1"]) for f in DRAWS]   # longest first
            for pid, f, beats, snr, info in pool.map(_est, jobs):
                res[(pid, f)] = (beats, snr, info)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
            try:
                os.rmdir(os.path.dirname(work))
            except OSError:
                pass

    rows, beats_out = [], {}
    for p in P:
        row = dict(recording=rec, period=p["period"], t0_s=round(p["t0"], 3), t1_s=round(p["t1"], 3),
                   start_min=round(p["t0"] / 60, 2), dur_s=round(p["t1"] - p["t0"], 2),
                   breaths_per_min=None if p["rr"] is None else round(float(p["rr"]), 2), seed_hr_bpm=round(p["seed"], 2),
                   tracked="no", abstain_reason="", snr=None, n_beats=0, hr_bpm=None, hr_unguarded_bpm=None,
                   family_abstain=None, family_agree=None)
        if p["seed"] > HR_MAX_BPM:
            row["abstain_reason"] = "seed_above_max"
        elif p["rr"] is None:
            row["abstain_reason"] = "no_breathing_rate"
        else:
            b1, snr, _ = res[(p["period"], 1.0)]
            fam = [res[(p["period"], f)][0] for f in DRAWS[1:]]
            ag = [_agree(b1, b) for b in fam]
            hr = _hr(b1)
            row.update(snr=round(snr, 4), family_abstain=int(sum(b is None for b in fam)),
                       family_agree=round(float(np.nanmean(ag)), 4) if np.isfinite(ag).any() else None)
            if b1 is None:
                row["abstain_reason"] = "snr_below_gate"
            elif hr is not None and hr > HR_MAX_BPM:
                row.update(abstain_reason="hr_above_max", hr_unguarded_bpm=round(hr, 2))
            else:
                row.update(tracked="yes", n_beats=int(b1.size), hr_bpm=None if hr is None else round(hr, 2))
                beats_out[str(p["period"])] = b1
        rows.append(row)
    n_g = sum(r["abstain_reason"] in ("seed_above_max", "hr_above_max") for r in rows)
    if n_g:
        notes.append(f"{n_g} period(s) not reported by the high-HR guard (seed or b2b heart rate > {HR_MAX_BPM:.0f} bpm: "
                     f"not a mouse heartbeat on our ECG sessions)")
    extra.update(workers=workers, runtime_s=round(time.time() - T0, 1))
    info = dict(exp_dir=exp_dir, cube=cube_path, segments=seg_base, hr_json=hr_json, patch_beams=f"{lo}-{hi}",
                draws=DRAWS, timing_s=dict(tracking=round(time.time() - t, 1), total=round(time.time() - T0, 1)))
    bins = hr_bins(P, beats_out, shape[0] / float(vmeta["fs"]))
    return _summary(rec, seg_json, rows, beats_out, notes, extra, bins), rows, beats_out, bins, info


def run(rec, sub_dir=None, out_dir=None, log=print, exp_dir=None, workers=None, keep_work=False):
    """Beat-to-beat heart rate for one recording whose hr step has run. Returns the flat summary dict."""
    import substrate as SUB
    sub_dir = sub_dir or SUB.SUB_DIR
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(sub_dir)), "metrics")
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, glob.escape(rec) + ".b2b*")):
        os.remove(f)
    m, rows, beats, bins, info = analyze(rec, sub_dir, out_dir, exp_dir=exp_dir, workers=workers, keep_work=keep_work,
                                         log=log)
    with open(os.path.join(out_dir, rec + ".b2b.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(m.keys()); w.writerow(m.values())
    with open(os.path.join(out_dir, rec + ".b2b_periods.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, PERIOD_COLS); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, rec + ".b2b_beats.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["period", "beat_s", "ibi_s", "hr_bpm"])
        for k, b in beats.items():
            for i, x in enumerate(b):
                d = None if i == 0 else b[i] - b[i - 1]
                w.writerow([k, f"{x:.4f}", "" if d is None else f"{d:.4f}", "" if d is None else f"{60.0 / d:.2f}"])
    with open(os.path.join(out_dir, rec + ".b2b_bins.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["t_s", "period", "hr_bpm"])
        w.writerows([f"{t:.1f}", k, f"{h:.2f}"] for t, k, h in bins)
    np.savez(os.path.join(out_dir, rec + ".b2b.npz"), **beats)
    json.dump(dict(metrics=m, periods=rows, info=info), open(os.path.join(out_dir, rec + ".b2b.json"), "w"),
              indent=1, default=float)
    med = "n/a" if m["hr_median_bpm"] is None else f"{m['hr_median_bpm']:.0f} bpm"
    log(f"[b2b] {rec:20s} {m['n_tracked']}/{m['n_periods']} periods tracked ({m['tracked_min']} min, {m['n_beats']} beats)"
        f" | median {med} over {m['hr_covered_min']} min of 2 s values | beats/breath {m['hr_per_breath_median']} | abstained: SNR {m['n_abstain_snr']}, high-HR guard {m['n_guard_seed']} seed + "
        f"{m['n_guard_hr']} result  [{m.get('runtime_s', 0):.0f}s]" + (f"\n           NOTE: {m['notes']}" if m["notes"] else ""))
    return m


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="beat-to-beat radar heart rate within each still period (run after hr)")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--sub-dir", default=None)
    ap.add_argument("--exp", default=None, help="raw experiment folder (default: the one the hr step used)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    run(a.rec, sub_dir=a.sub_dir, out_dir=a.out, exp_dir=a.exp, workers=a.workers, keep_work=a.keep_work)
