r"""hr_curve.py -- radar HEART-RATE CURVE inside each still, regular-breathing period: the `pll` tracker (2026-10-17).

The best ECG-free method found for following heart rate WITHIN a period. Beat-to-beat score ve1 (share of the ECG's
within-period R-R variation it explains): 508 0.77, 507 day 2 0.55, ng511 0.07. Read HR_CURVE.md before citing
numbers: some constants were tuned on those same three ECG mice.

Runs AFTER the `hr` step and reuses its periods, chest patch, heartbeat band and 20 beamed voxels (<rec>.hr.json,
<rec>.hr_templates.npz). Per period (no ECG anywhere):
  1. breath-order residual of the chest-patch acceleration (engine/hr_final_pipeline.py, the frozen hr code);
  2. band centre: the hr step's band (`--band bspec`, default) or `--band continuity` (a Viterbi over all periods in
     time order; the recipe used on the fast breather ng511);
  3. one multi-voxel beam per heart harmonic k = 1..5 around k x centre (k = 1: centre / 1.45 .. centre x 1.45, lower
     edge kept >= 10 % above the breathing rate);
  4. particle filter over beat phase and log heart rate (1024 particles, 1 s fixed-lag smoother, forward + reversed
     pass, 2 seeds); harmonics k >= 2 join only where they are unambiguous (EM z-score > 16);
  5. posterior-mean heart rate at 25 Hz. A period whose k = 1 beam shows no heart signal (q1 <= 0) is NOT tracked: the
     tracker would return a flat line at the band centre, so it gets no curve.
  6. heart-SNR gate: q1 in dB per period (snr_grade none / low / ok / strong). Below -3 dB the curve did not follow the
     ECG at all; a recording with < half its still minutes above -3 dB is flagged LOW (warning, QC line, notes): use
     the curve only in its ok / strong periods. Halved breathing (engine/breath_check.py) makes every grade `invalid`
     and the verdict INVALID: a leftover breathing harmonic then passes for a strong heart signal (ng511, default
     breath bounds: +110 bpm at "strong" SNR).
engine/pll_tracker/ holds the tracker byte-identical to the validated research files (sha256 checked on every run).

Outputs (<out> = metrics dir): <rec>.hr_curve.{csv,json} (one row), <rec>.hr_curve_periods.csv (one row per period),
<rec>.hr_curve_track.csv + <rec>.hr_curve.npz (curve of the tracked periods over [t0 + 1 s, t1 - 1 s)), <rec>.hr_curve.png.
Research provenance: output/ng511_hr/b2b/pll/ (RESULT.md: method, every constant's provenance) + output/hr_best/.
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

NAME = "hr_curve"
KIND = "segments"
NEEDS_RAW = True              # run.py passes exp_dir= (the raw folder); falls back to the hr step's / substrate's folder
PARALLEL = True               # run.py passes workers=
DEJUMP = True                 # run.py hands it the de-jumped cube copy of a July-2026 capture (engine/dejump.py, DEJUMP.md)

TRACKER_DIR = os.path.join(_PKG, "engine", "pll_tracker")
FROZEN_SHA256 = {             # output/ng511_hr/b2b/{beams,pll}/ as frozen 2026-10-17 (est_blind/repro_check.txt)
    "beams.py": "cd2d190d49f05d33fab350dcfd5e31ee33a9ee1b55514652782a425ac9b02e26",
    "variants.py": "ec120946ca3de567e7417fd574423b064c95f21e5879c4fd75ba36512d5e40da",
    "beamx.py": "02d33b783037273e00c7fa7ada8a9fc541a96a5fa05fe59401d21535cd8e1ab8",
    "pf.py": "6539457d7cc8a4d153c20f4b4899426d6d1f5d354c3864aef5f27f4c9bb2f6a8",
    "pll.py": "ed0c1e99c8372961ca9dbc8d672936f462150cd71d7a29d742f830599f474ff6",
    "pll_variants.py": "f4557ee80312e7a835fd82d1164bbc63678d2733a55851856fb083271b37be59"}
VARIANT = "pfh_g45_z16_bd_r2"  # pll_variants.FROZEN_NAME
BANDS = ("bspec", "continuity")
K = 5                         # heart harmonics beamed
CENTRES = np.asarray(HB.CENTRES, float)
# continuity band: Viterbi emission beta * z-scored B_spec, transition Gaussian in log centre with
# sd = SIG * sqrt(gap, min) + SIG_MIN [a priori, set before ng511 was scored; output/ng511_hr/track_probe.py]
CONT_SIG, CONT_SIG_MIN, CONT_BETA = 0.05, 0.02, 1.0
EDGE_S = 1.0                  # outputs cover [t0 + 1, t1 - 1): filter edges; the ECG scoring ignored them too
# Heart-SNR gate on q1 (cross-fitted k = 1 heart contrast P_in / P_ring - 1 of the beam, the tracker's own SNR input;
# dB = 10 log10 q1). Calibrated in-sample on the 114 ECG-scored periods of 508 / 507d2 / ng511
# (output/hr_curve_package/snr_gate.py): pooled ve1 -0.01 at q1 <= 0.5 (n 19, all three mice) vs 0.63 above, 0.75 above 2.
SNR_MIN_Q1 = 0.5              # -3 dB: below it the curve does not follow the ECG
SNR_STRONG_Q1 = 2.0           # +3 dB
REC_MIN_FRAC = 0.5            # LOW recording: < this share of still minutes above -3 dB (508 0.93, 507d2 0.92, ng511 0.44)
OUT_FS = 25.0                 # tracker readout rate (pf: 50 Hz state, every 2nd sample)
# Recording summary over the trustworthy periods only (new in 1.5; the rules of the July-2026 cohort comparison,
# output/july_dejump/README.md): grade ok / strong, and period HR <= 650 bpm (the SNR gate also grades a constant
# ~780-820 bpm line 'strong', seen on several recordings; no ECG-checked heart was above 612). A value needs >= 3 such
# periods and >= 2 min, the spread >= 5 periods; else None.
OK_HR_MAX_BPM, OK_MIN_PERIODS, OK_MIN_MIN, OK_SPREAD_MIN_PERIODS = 650.0, 3, 2.0, 5
_BLAS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")

PERIOD_COLS = ["recording", "period", "t0_s", "t1_s", "start_min", "dur_s", "breaths_per_min", "period_hr_bpm",
               "bspec_centre_bpm", "band_centre_bpm", "tracked", "heart_contrast_q1", "heart_snr_db", "snr_grade",
               "k1_band_lo_bpm",
               "k1_band_hi_bpm", "harmonics_used", "curve_mean_bpm", "curve_sd_bpm", "curve_min_bpm", "curve_max_bpm",
               "n_samples"]
NPZ_README = """<rec>.hr_curve.npz -- the heart-rate curve of every tracked period (hr_curve pipeline, pll tracker).
  period, t_s, hr_bpm, ibi_s   [N]  25 Hz samples over [t0 + 1 s, t1 - 1 s) of each tracked period; ibi_s = 60 / hr_bpm
  p_period, p_t0_s, p_t1_s     [P]  every period of the hr step (= hr_curve_periods.csv)
  p_centre_bpm, p_tracked, p_q1 [P] band centre the tracker used, tracked flag, k = 1 heart contrast (<= 0: not tracked)
  p_snr_grade                  [P]  none (not tracked) / low (<= -3 dB: curve unreliable) / ok / strong (> +3 dB) /
                                    invalid (breathing halved: not a heart rate)
  snr_grade                    [N]  the grade of the period each sample belongs to
Full float64 precision (the CSV rounds to 0.01 bpm)."""


def default_workers():
    return max(1, min(12, (os.cpu_count() or 2) - 2))


def frozen_ok():
    """(all tracker files match the validated sha256, [names that differ])."""
    bad = [f for f, h in FROZEN_SHA256.items()
           if hashlib.sha256(open(os.path.join(TRACKER_DIR, f), "rb").read()).hexdigest() != h]
    return not bad, bad


def _tracker():
    """Import the frozen tracker; the bh shim first, so beams.py binds `bh` to it."""
    if TRACKER_DIR not in sys.path:
        sys.path.insert(0, TRACKER_DIR)
    import bh                                                 # noqa: F401
    import beamx, pf, pll_variants
    return beamx, pf, pll_variants


def _prep_worker(args):
    """One period: breath-order residual Y (float32 npy, as the research cache stored it) + z-scored B_spec band
    scores of that Y (continuity only; = research track_probe.bscores)."""
    rec, cube_path, shape, cols, onset, ttot, a, b, rr, acc_path, y_path, want_z = args
    import harness as H
    import hr_final_pipeline as FP
    t = time.time()
    HB.build_accel(((cube_path, shape), cols, a, b, acc_path))
    H.register(rec, cube_path, shape, cols, onset, ttot)
    X = np.asarray(H.patch_accel(rec, dict(accel_patch=acc_path)), float)     # = process_run's X
    Y = FP.residual(X, FP.breath_phase(rec, dict(t0=a, t1=b), X.shape[0], rr), rr).astype(np.float32)
    np.save(y_path, Y)
    del X
    try:
        os.remove(acc_path)
    except OSError:
        pass
    z = None
    if want_z:
        f, L = FP.lam_spectrogram(np.asarray(Y, float))
        _, B = FP.choose_bspec(f, FP.whiten(f, L), CENTRES)
        z = (B - B.mean()) / (B.std() + 1e-12)
    return dict(z=z, elapsed_s=time.time() - t)


def _track_worker(args):
    """The frozen tracker on one period (= research pll/estimate.py's per-period loop, FROZEN variant)."""
    pid, y_path, picks, centre, rr, t0, t1 = args
    beamx, PF, VA = _tracker()
    t = time.time()
    cfg = dict(VA.ALL[VA.FROZEN_NAME]); cfg.pop("engine", "pf"); guard = cfg.pop("guard", None)
    p = dict(id=pid, t0=t0, t1=t1, rr=rr, centre=centre, Y=np.load(y_path), picks=np.asarray(picks))
    c2 = dict(cfg)
    Z, q, (lo, hi) = beamx.beams_guard(p, p["centre"], K, None, cfg["half1"], guard)
    c2["lo1"] = lo
    q1 = float(np.max(q[:, 0]))
    Zc = Z.astype(np.complex64).astype(complex)                # research cached the beams as complex64
    e, r = PF.curve(Zc, p["centre"], q1, p["t0"], c2)
    return dict(t=np.asarray(e["t"], float), ibi=np.asarray(e["ibi"], float), q1=q1, lo1=float(lo), hi1=float(hi),
                mu=float(r["mu"]), A=[float(v) for v in np.abs(np.atleast_2d(r["A"])).mean(-1)],
                elapsed_s=time.time() - t)


def continuity_bands(P, Z):
    """Band index per period, periods in time order (research track_probe.viterbi at the a priori setting)."""
    C = CENTRES; lc = np.log(C); n = len(P)
    D = np.zeros((n, C.size)); bp = np.zeros((n, C.size), int)
    D[0] = CONT_BETA * Z[0]
    for i in range(1, n):
        gap = max(0.0, P[i]["t0"] - P[i - 1]["t1"]) / 60.0
        sd = CONT_SIG * np.sqrt(gap) + CONT_SIG_MIN
        T = -0.5 * ((lc[None, :] - lc[:, None]) / sd) ** 2         # [from, to]
        M = D[i - 1][:, None] + T
        bp[i] = np.argmax(M, 0); D[i] = M[bp[i], np.arange(C.size)] + CONT_BETA * Z[i]
    path = [int(np.argmax(D[-1]))]
    for i in range(n - 1, 0, -1):
        path.append(int(bp[i][path[-1]]))
    return path[::-1]


def snr_grade(q1):
    """none (not tracked) / low (<= -3 dB) / ok / strong (> +3 dB)."""
    if not np.isfinite(q1) or q1 <= 0:
        return "none"
    return "low" if q1 <= SNR_MIN_Q1 else ("strong" if q1 > SNR_STRONG_Q1 else "ok")


def snr_db(q1):
    return round(10 * float(np.log10(q1)), 1) if np.isfinite(q1) and q1 > 0 else None


def _readout(res, t0, t1):
    t, ibi = res["t"], res["ibi"]
    m = (t >= t0 + EDGE_S) & (t < t1 - EDGE_S)
    return t[m], ibi[m]


def _write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def _period_row(rec, q):
    r = q.get("res")
    row = dict(recording=rec, period=q["period"], t0_s=round(q["t0"], 3), t1_s=round(q["t1"], 3),
               start_min=round(q["t0"] / 60, 2), dur_s=round(q["t1"] - q["t0"], 1),
               breaths_per_min=round(q["rr"], 1), period_hr_bpm=q["period_hr"], bspec_centre_bpm=int(q["bspec_centre"]),
               band_centre_bpm=int(q["centre"]), tracked="no")
    if r is None:
        return row
    row.update(heart_contrast_q1=round(r["q1"], 4), heart_snr_db=snr_db(r["q1"]), snr_grade=snr_grade(r["q1"]),
               k1_band_lo_bpm=round(q["centre"] * np.exp(-r["lo1"]), 1),
               k1_band_hi_bpm=round(q["centre"] * np.exp(r["hi1"]), 1))
    if r["q1"] > 0:
        _, ibi = _readout(r, q["t0"], q["t1"]); hr = 60.0 / ibi
        row.update(tracked="yes", harmonics_used="+".join(str(k + 1) for k, a in enumerate(r["A"]) if a > 0),
                   curve_mean_bpm=round(float(hr.mean()), 2), curve_sd_bpm=round(float(hr.std()), 2),
                   curve_min_bpm=round(float(hr.min()), 2), curve_max_bpm=round(float(hr.max()), 2), n_samples=int(hr.size))
    return row


def _figure(path, rec, P, rows, hr_track_csv, verdict=None, band="bspec"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = [(q, r) for q, r in zip(P, rows) if r["tracked"] == "yes"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw=dict(height_ratios=[1.4, 1]))
    for q, r in zip(P, rows):
        ax[0].plot([q["t0"] / 60, q["t1"] / 60], [r["period_hr_bpm"]] * 2, color="#0b0b0b" if r["tracked"] == "yes"
                   else "#b5b3ad", lw=2.2, solid_capstyle="butt", zorder=3)
    for q, r in tr:
        t, ibi = _readout(q["res"], q["t0"], q["t1"])
        ax[0].plot(t / 60, 60 / ibi, color={"low": "#e8743b", "invalid": "#c0392b"}.get(r["snr_grade"], "#2a78d6"),
                   lw=0.7)
    hrs = np.array([r["period_hr_bpm"] for r in rows], float)
    ax[0].set_ylim(max(200, np.nanmin(hrs) - 150), np.nanmax(hrs) + 150)
    ax[0].set_xlabel("time (min)"); ax[0].set_ylabel("heart rate (bpm)")
    ax[0].set_title(f"{rec}: heart-rate curve per still period\nblue = curve, orange = curve below the -3 dB heart-SNR "
                    f"minimum, black = hr period value{' (its own band)' if band != 'bspec' else ''}, grey = not tracked",
                    loc="left", fontsize=10)
    banner = {"LOW": "LOW HEART SNR: within-period changes are usable only in the blue (ok / strong) periods",
              "INVALID": "BREATHING HALVED: these curves are not a heart rate -- re-segment (HR_CURVE.md)"}.get(verdict)
    if banner:
        ax[0].text(0.5, 0.93, banner, transform=ax[0].transAxes, ha="center", va="top", fontsize=12, fontweight="bold",
                   color="#c0392b", bbox=dict(facecolor="white", edgecolor="#c0392b", boxstyle="round,pad=0.3"))
    good = [x for x in tr if x[1]["snr_grade"] in ("ok", "strong")] or tr
    if good:
        q, r = max(good, key=lambda x: x[0]["t1"] - x[0]["t0"])
        t, ibi = _readout(q["res"], q["t0"], q["t1"])
        if hr_track_csv and os.path.exists(hr_track_csv):
            with open(hr_track_csv, newline="") as fh:
                ft = [(float(d["t_s"]), float(d["hr_bpm"])) for d in csv.DictReader(fh) if int(d["period"]) == q["period"]]
            if ft:
                ax[1].plot([x for x, _ in ft], [y for _, y in ft], color="#8a8884", lw=0.9, label="hr step's 10 Hz track")
        ax[1].plot(t, 60 / ibi, color="#2a78d6", lw=1.2, label="hr_curve (pll)")
        ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("heart rate (bpm)")
        sub = {"low": "-- ALL periods are below the SNR minimum",
               "invalid": "-- breathing halved: NOT a heart rate"}.get(r["snr_grade"])
        ax[1].set_title(f"longest tracked period ({q['t0'] / 60:.1f} min, {q['t1'] - q['t0']:.0f} s) {sub}" if sub else
                        f"longest tracked period above the SNR minimum ({q['t0'] / 60:.1f} min, {q['t1'] - q['t0']:.0f} s, "
                        f"heart SNR {r['heart_snr_db']:+.1f} dB)", loc="left", fontsize=10)
        ax[1].legend(frameon=False, fontsize=9)
    else:
        ax[1].set_axis_off()
    for a in ax:
        a.grid(color="#e4e3df", lw=0.6); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def snr_summary(P, rows, halved=False):
    """Recording-level heart SNR: share of still minutes above -3 dB, duration-weighted median dB, verdict.
    halved (breath check): verdict INVALID whatever the SNR -- a leftover breathing harmonic reads as a strong heart."""
    if not rows:
        return dict(snr_verdict=None, snr_ok_frac=None, heart_snr_median_db=None, n_snr_ok=0, snr_ok_min=0.0)
    L = np.array([q["t1"] - q["t0"] for q in P]); q1 = np.array([q["res"]["q1"] for q in P])
    ok = np.isfinite(q1) & (q1 > SNR_MIN_Q1)
    o = np.argsort(q1); wmed = float(q1[o][np.searchsorted(np.cumsum(L[o]) / L.sum(), 0.5)])
    frac = float(L[ok].sum() / L.sum())
    verdict = "INVALID" if halved else ("OK" if frac >= REC_MIN_FRAC else "LOW")
    return dict(snr_verdict=verdict, snr_ok_frac=round(frac, 3),
                heart_snr_median_db=snr_db(wmed), n_snr_ok=int(ok.sum()), snr_ok_min=round(float(L[ok].sum()) / 60, 2))


def ok_summary(rows):
    """Heart rate, breathing rate and their ratio / product over the ok / strong periods (OK_* rules above).
    hr_x_rr is exploratory (post hoc in the cohort comparison): HR x breaths/min / 1000."""
    v = [r for r in rows if r.get("snr_grade") in ("ok", "strong") and r.get("breaths_per_min")
         and r["period_hr_bpm"] <= OK_HR_MAX_BPM]
    mins = sum(r["dur_s"] for r in v) / 60
    out = dict(ok_n_periods=len(v), ok_min=round(mins, 2), ok_hr_median_bpm=None, ok_breaths_per_min_median=None,
               ok_hr_over_rr_median=None, ok_hr_x_rr_median_k=None, ok_hr_iqr_bpm=None)
    if len(v) >= OK_MIN_PERIODS and mins >= OK_MIN_MIN:
        hr = np.array([r["period_hr_bpm"] for r in v], float); rr = np.array([r["breaths_per_min"] for r in v], float)
        out.update(ok_hr_median_bpm=round(float(np.median(hr)), 2), ok_breaths_per_min_median=round(float(np.median(rr)), 2),
                   ok_hr_over_rr_median=round(float(np.median(hr / rr)), 3),
                   ok_hr_x_rr_median_k=round(float(np.median(hr * rr)) / 1000, 3))
        if len(v) >= OK_SPREAD_MIN_PERIODS:
            q1, q3 = np.percentile(hr, [25, 75]); out["ok_hr_iqr_bpm"] = round(float(q3 - q1), 2)
    return out


def _summary(rec, seg_json, P, rows, notes, extra):
    """Curve statistics use only periods at or above the SNR minimum (grade ok / strong)."""
    tr = [q for q, r in zip(P, rows) if r["tracked"] == "yes"]
    gd = [q for q, r in zip(P, rows) if r.get("snr_grade") in ("ok", "strong")]
    hr = np.concatenate([60.0 / _readout(q["res"], q["t0"], q["t1"])[1] for q in gd]) if gd else np.zeros(0)
    sd = np.array([r["curve_sd_bpm"] for r in rows if r.get("snr_grade") in ("ok", "strong")], float)
    q_ = lambda x, p: round(float(np.percentile(x, p)), 1) if x.size else None
    m = dict(recording=rec, cohort=seg_json.get("cohort"), age_weeks=seg_json.get("age_weeks"),
             n_periods=len(rows), n_tracked=len(tr), tracked_min=round(sum(q["t1"] - q["t0"] for q in tr) / 60, 2),
             **snr_summary(P, rows, extra.get("breathing_halved")),
             curve_hr_median_bpm=q_(hr, 50), curve_hr_p25_bpm=q_(hr, 25), curve_hr_p75_bpm=q_(hr, 75),
             within_period_sd_median_bpm=round(float(np.median(sd)), 2) if sd.size else None,
             n_with_harmonics=sum(r.get("snr_grade") in ("ok", "strong") and r.get("harmonics_used", "1") != "1"
                                  for r in rows),
             **ok_summary(rows))
    m.update(extra)
    m["notes"] = " | ".join(notes)
    return m


def analyze(rec, sub_dir, out_dir, exp_dir=None, workers=None, band="bspec", keep_work=False, log=print):
    """Everything for one recording. Returns (summary metrics, period rows, periods with results, info)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    T0 = time.time()
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, not {band!r}")
    seg_base = os.path.join(sub_dir, rec)
    seg_json = json.load(open(seg_base + ".json"))
    hr_json = os.path.join(out_dir, rec + ".hr.json")
    tpl_npz = os.path.join(out_dir, rec + ".hr_templates.npz")
    if not os.path.exists(hr_json):
        raise FileNotFoundError(f"{hr_json} not found -- the heart-rate curve reuses the hr step's periods and voxels: "
                                f"run `python run.py hr --rec {rec}` first")
    HRJ = json.load(open(hr_json))
    hm, hrows = HRJ["metrics"], HRJ["periods"]
    ok, bad = frozen_ok()
    notes = []
    extra = dict(band_mode=band, variant=VARIANT, frozen_code_ok=ok, breath_check_ratio=None, breathing_halved=None,
                 min_period_s=float(hm.get("min_period_s") or HB.MIN_RUN_S))
    if not ok:
        notes.append(f"engine/pll_tracker/{', '.join(bad)} differ(s) from the validated files -- results are not the "
                     f"validated method")
    try:
        import breath_check as BC
        chk = BC.rate_ratio(seg_base)
        if np.isfinite(chk["ratio"]):
            extra["breath_check_ratio"] = round(chk["ratio"], 3)
        extra["breathing_halved"] = BC.is_halved(chk)
        if extra["breathing_halved"]:                        # a breathing harmonic passes for a strong heart signal
            notes.append(BC.halved_note(chk, "heart-rate curves and heart-SNR grades"))
            log("[hr_curve] " + "!" * 100); log(f"[hr_curve] WARNING: {notes[-1]}."); log("[hr_curve] " + "!" * 100)
    except Exception as ex:                                   # the check is advisory; never lose the curve over it
        notes.append(f"breath check skipped ({type(ex).__name__}: {ex})")
    if not hrows:
        notes.append("the hr step found no still period -- no heart-rate curve")
        return _summary(rec, seg_json, [], [], notes, extra), [], [], {}
    if not os.path.exists(tpl_npz):
        raise FileNotFoundError(f"{tpl_npz} not found -- re-run the hr step (v1.3+ writes it)")

    exp_dir = exp_dir or HRJ.get("info", {}).get("exp_dir") or seg_json.get("session_dir")
    if not (exp_dir and glob.glob(os.path.join(exp_dir, "*_voxels.json"))):
        raise FileNotFoundError(f"raw experiment folder not found ({exp_dir!r}); the heart-rate curve re-reads the "
                                f"voxels -- pass --exp <folder with *_voxels.f32>")
    power, vmeta, _ = VS.load_experiment(exp_dir)
    cube_path, shape = str(power.filename), tuple(power.shape)
    extra["cube_dejumped"] = bool(vmeta.get("dejump"))
    if hm.get("cube_dejumped") is not None and bool(hm["cube_dejumped"]) != extra["cube_dejumped"]:
        notes.append("the hr step ran on a " + ("de-jumped" if hm["cube_dejumped"] else "raw") + " cube but this step on a "
                     + ("de-jumped" if extra["cube_dejumped"] else "raw") + " one -- re-run both with the same --dejump")
        log(f"[hr_curve] WARNING {rec}: {notes[-1]}")
    if seg_json.get("rec_min") is not None and abs(float(seg_json["rec_min"]) - shape[0] / float(vmeta["fs"]) / 60) > 0.05:
        raise ValueError(f"{exp_dir!r} does not hold the {seg_json['rec_min']} min that {rec} was segmented from -- "
                         f"wrong --exp folder?")

    on, tt, runs = HB.breathing(seg_base, extra["min_period_s"])
    lo, hi = (int(x) for x in str(hm["patch_beams"]).split("-"))
    cols = HB.patch_cols(vmeta, list(range(lo, hi + 1)))
    pos = {int(c): i for i, c in enumerate(cols)}
    tpl = np.load(tpl_npz)
    P = []
    for r in hrows:
        i = [k for k, (a, b) in enumerate(runs) if abs(a - float(r["t0_s"])) < 0.01]
        if len(i) != 1:
            raise ValueError(f"hr period {r['period']} (t0 {r['t0_s']} s) is not a clean-breathing run of the current "
                             f"segments -- were the segments re-made after the hr step? Re-run hr.")
        a, b = runs[i[0]]
        j = np.flatnonzero(tpl["period"] == int(r["period"]))
        picks = [pos[int(c)] for c in tpl["voxel_col"][j[0]] if int(c) >= 0] if j.size else []
        rr = HB.run_rr(on, tt, a, b)
        if rr is None or len(picks) == 0:
            notes.append(f"period {r['period']}: no breathing rate / beamed voxels -- skipped")
            continue
        P.append(dict(period=int(r["period"]), t0=a, t1=b, rr=rr, period_hr=r["hr_bpm"],
                      bspec_centre=float(r["band_centre_bpm"]), centre=float(r["band_centre_bpm"]), picks=picks))
    if any(P[k + 1]["t0"] < P[k]["t0"] for k in range(len(P) - 1)):
        raise ValueError("hr periods are not in time order")
    log(f"[hr_curve] {rec}: {len(P)} still periods from the hr step ({sum(q['t1'] - q['t0'] for q in P) / 60:.1f} min), "
        f"band = {band}, chest patch beams {lo}-{hi}")

    work = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "hr_curve_work", rec)
    os.makedirs(work, exist_ok=True)
    for q in P:
        q["y_path"] = os.path.join(work, f"Y_{q['t0']:.2f}.npy")
    workers = int(workers or default_workers())
    order = sorted(range(len(P)), key=lambda i: -(P[i]["t1"] - P[i]["t0"]))      # longest first: load balance
    saved = {k: os.environ.get(k) for k in _BLAS}
    for k in _BLAS:
        os.environ[k] = "1"                       # workers: 1 BLAS thread each (no oversubscription)
    t = time.time()
    log(f"[hr_curve] tracking {len(P)} periods with {workers} workers ...")
    try:
        with ProcessPoolExecutor(workers) as pool:
            track = {}

            def submit(i):
                q = P[i]
                track[pool.submit(_track_worker, (f"{rec}_{q['t0']:.2f}", q["y_path"], q["picks"], q["centre"], q["rr"],
                                                  q["t0"], q["t1"]))] = i
            prep = {pool.submit(_prep_worker, (rec, cube_path, shape, cols, on, tt, P[i]["t0"], P[i]["t1"], P[i]["rr"],
                                               os.path.join(work, f"A_{P[i]['t0']:.2f}.npy"), P[i]["y_path"],
                                               band == "continuity")): i for i in order}
            for f in as_completed(prep):
                i = prep[f]; P[i]["z"] = f.result()["z"]
                if band == "bspec":
                    submit(i)
            if band == "continuity":
                for q, j in zip(P, continuity_bands(P, [q["z"] for q in P])):
                    q["centre"] = float(CENTRES[j])
                n_moved = sum(q["centre"] != q["bspec_centre"] for q in P)
                log(f"[hr_curve] continuity band differs from the hr step's band in {n_moved}/{len(P)} periods")
                for i in order:
                    submit(i)
            for n, f in enumerate(as_completed(track), 1):
                i = track[f]; q = P[i]; q["res"] = x = f.result()
                cur = 60 / _readout(x, q["t0"], q["t1"])[1]
                log(f"[hr_curve]   {n:3d}/{len(P)}  period {q['period']:3d} at {q['t0'] / 60:6.1f} min "
                    f"({q['t1'] - q['t0']:5.1f} s): band {q['centre']:.0f}, q1 {x['q1']:+.2f}, "
                    + (f"curve {cur.mean():5.0f} +- {cur.std():4.1f} bpm" if x["q1"] > 0 else "not tracked (no heart signal)")
                    + f"  [{x['elapsed_s']:.0f} s]")
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
    rows = [_period_row(rec, q) for q in P]
    if extra["breathing_halved"]:
        for r in rows:
            if r.get("snr_grade") not in (None, "none"):
                r["snr_grade"] = "invalid"
    n_un = sum(r["tracked"] == "no" for r in rows)
    if n_un:
        notes.append(f"{n_un} of {len(rows)} period(s) not tracked (no heart signal in the k = 1 beam)")
    sn = snr_summary(P, rows, extra["breathing_halved"])
    n_low = sum(r.get("snr_grade") == "low" for r in rows)
    period_src = ("curve_mean_bpm in hr_curve_periods.csv (the continuity band; hr_periods.csv uses the hr step's own "
                  "band, which continuity overrides; on ng511 the average error vs ECG was 49 vs 31 bpm in the ok / "
                  "strong periods)"
                  if band == "continuity" else "hr_periods.csv")
    if sn["snr_verdict"] == "LOW":
        notes.append(f"LOW HEART SNR: only {100 * sn['snr_ok_frac']:.0f} % of the still minutes reach the -3 dB minimum "
                     + (f"(median {sn['heart_snr_median_db']} dB). " if sn["heart_snr_median_db"] is not None else
                        "(most still minutes show no heart signal at all). ")
                     + (f"Trust heart rate only in the {sn['n_snr_ok']} period(s) graded ok / strong ({sn['snr_ok_min']} "
                        f"min): their curve and its period mean (curve_mean_bpm). In the other periods" if sn["n_snr_ok"]
                        else "No period clears it, so no heart rate here is trustworthy: in such periods")
                     + " neither the curve nor the period values were reliable on our low-SNR ECG mouse (ng511: ~64 bpm "
                       "average error)")
        log("[hr_curve] " + "!" * 100)
        log(f"[hr_curve] WARNING {rec}: {notes[-1]}.")
        log("[hr_curve] " + "!" * 100)
    elif sn["snr_verdict"] == "OK":
        if n_low:
            notes.append(f"{n_low} tracked period(s) below the -3 dB heart-SNR minimum (snr_grade = low): ignore their "
                         f"within-period changes")
        if band == "continuity":
            notes.append(f"band = continuity: period heart rate = {period_src}")
    extra.update(workers=workers, runtime_s=round(time.time() - T0, 1))
    info = dict(exp_dir=exp_dir, cube=cube_path, segments=seg_base, hr_json=hr_json, patch_beams=f"{lo}-{hi}",
                tracker_sha256=FROZEN_SHA256, timing_s=dict(tracking=round(time.time() - t, 1),
                                                            total=round(time.time() - T0, 1)),
                period_diag={q["period"]: dict(q1=q["res"]["q1"], lo1=q["res"]["lo1"], mu=q["res"]["mu"],
                                               A=q["res"]["A"], elapsed_s=round(q["res"]["elapsed_s"], 1)) for q in P})
    return _summary(rec, seg_json, P, rows, notes, extra), rows, P, info


def run(rec, sub_dir=None, out_dir=None, log=print, exp_dir=None, workers=None, band="bspec", keep_work=False,
        figure=True):
    """Heart-rate curve for one recording whose hr step has run. Returns the flat summary dict."""
    import substrate as SUB
    sub_dir = sub_dir or SUB.SUB_DIR
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(sub_dir)), "metrics")
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, glob.escape(rec) + ".hr_curve*")):
        os.remove(f)
    m, rows, P, info = analyze(rec, sub_dir, out_dir, exp_dir=exp_dir, workers=workers, band=band, keep_work=keep_work,
                               log=log)
    with open(os.path.join(out_dir, rec + ".hr_curve.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(m.keys()); w.writerow(m.values())
    _write_csv(os.path.join(out_dir, rec + ".hr_curve_periods.csv"), PERIOD_COLS, rows)
    per, ts, ib, gr = [], [], [], []
    for q, r in zip(P, rows):
        if r["tracked"] == "yes":
            t, ibi = _readout(q["res"], q["t0"], q["t1"])
            per.append(np.full(t.size, q["period"], np.int32)); ts.append(t); ib.append(ibi)
            gr.append(np.full(t.size, r["snr_grade"]))
    cat = lambda z, dt: np.concatenate(z).astype(dt) if z else np.zeros(0, dt)
    per, ts, ib, gr = cat(per, np.int32), cat(ts, float), cat(ib, float), cat(gr, "U8")
    with open(os.path.join(out_dir, rec + ".hr_curve_track.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["period", "t_s", "hr_bpm", "snr_grade"])
        for a, b, c, g in zip(per, ts, ib, gr):
            w.writerow([int(a), f"{b:.2f}", f"{60.0 / c:.2f}", g])
    if P:
        np.savez_compressed(os.path.join(out_dir, rec + ".hr_curve.npz"), README=np.array(NPZ_README),
                            period=per, t_s=ts, hr_bpm=60.0 / ib, ibi_s=ib,
                            p_period=np.array([q["period"] for q in P], np.int32), p_t0_s=np.array([q["t0"] for q in P]),
                            p_t1_s=np.array([q["t1"] for q in P]), p_centre_bpm=np.array([q["centre"] for q in P]),
                            p_tracked=np.array([r["tracked"] == "yes" for r in rows]),
                            p_q1=np.array([q["res"]["q1"] for q in P]),
                            p_snr_grade=np.array([r.get("snr_grade") or "none" for r in rows]), snr_grade=gr)
    json.dump(dict(metrics=m, periods=rows, info=info), open(os.path.join(out_dir, rec + ".hr_curve.json"), "w"),
              indent=1, default=float)
    if figure and P:
        try:
            _figure(os.path.join(out_dir, rec + ".hr_curve.png"), rec, P, rows,
                    os.path.join(out_dir, rec + ".hr_track.csv"), verdict=m["snr_verdict"], band=band)
        except Exception as ex:                       # a figure failure must never lose the numbers
            log(f"[hr_curve] figure skipped ({ex})")
    med = "n/a" if m["curve_hr_median_bpm"] is None else f"{m['curve_hr_median_bpm']:.0f} bpm"
    sd = "n/a" if m["within_period_sd_median_bpm"] is None else f"{m['within_period_sd_median_bpm']} bpm"
    log(f"[hr_curve] {rec:20s} {m['n_tracked']}/{m['n_periods']} periods tracked ({m['tracked_min']} min) | heart SNR "
        f"{m['snr_verdict']} ({m['n_snr_ok']} periods / {m['snr_ok_min']} min above -3 dB, median "
        f"{'no heart signal' if m['heart_snr_median_db'] is None else str(m['heart_snr_median_db']) + ' dB'}) | "
        f"curve median {med} | within-period SD median {sd} | band {band}  "
        f"[{m.get('runtime_s', 0):.0f}s]" + (f"\n           NOTE: {m['notes']}" if m["notes"] else ""))
    return m


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="radar heart-rate curve within each still period (run after hr)")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--sub-dir", default=None)
    ap.add_argument("--exp", default=None, help="raw experiment folder (default: the one the hr step used)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--band", default="bspec", choices=BANDS)
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    run(a.rec, sub_dir=a.sub_dir, out_dir=a.out, exp_dir=a.exp, workers=a.workers, band=a.band, keep_work=a.keep_work)
