r"""qc.py -- automatic quality control for one processed recording.

WHY THIS EXISTS
---------------
The live capture monitor writes a `*_summary.json` at acquisition time with quick metrics
(`rr_med_bpm`, `coverage_pct`, ...). Those live numbers are *approximate and have drifted as the
capture code evolved*: the live breathing rate is prone to OCTAVE-DOUBLING (a strong 2nd harmonic
wins, so a 180 bpm mouse reads ~360 bpm), and live "coverage" is really the CALM fraction -- the
share of time the animal was breathing *regularly*. A sick or aroused mouse breathes the whole time
but irregularly, so its live "coverage" looks tiny even though the recording is perfect. Passing or
failing an experiment on those live numbers therefore throws away exactly the sickest, most valuable
recordings (see the B9 mutant: live coverage 7.6%, but the animal is actually breathing 96% of the
recording).

WHAT THIS DOES
--------------
This QC step runs AFTER the pipeline has re-segmented the recording from the STORED VOXELS with the
current algorithm. It judges data quality on the RECOMPUTED signal, not the live summary, and it
distinguishes two very different things:

  * DATA-QUALITY failures (a real problem): no breathing recovered at all, the animal was out of the
    beam, the wrong voxel file, a truncated/short capture, a nonstandard sampling rate. -> FAIL / REVIEW.
  * PHYSIOLOGY (not a problem, and the whole point of the study): irregular breathing, apneas, long
    pauses, low calm fraction, low movement. A sick mouse looks like this. -> PASS (with a note).

When a `*_summary.json` is found next to the raw data, QC also prints a LIVE-vs-RECOMPUTED comparison
so you can see *why* the live pass/fail was wrong (e.g. the live rate was octave-doubled, or "coverage"
was just the calm fraction).

The verdict is one of PASS / REVIEW / FAIL. It biases toward PASS: it only fails a recording when the
recomputed signal itself is missing or implausible, never because the animal was active or sick.

Output: <out>/<rec>.qc.json (full report) + <out>/<rec>.qc.csv (one flat row), and a printed banner.
Runs off the cached substrate; the total-duration / live-summary extras are best-effort (they need the
raw experiment folder still at the substrate's `session_dir`, which is the normal case right after a run).
"""
from __future__ import annotations
import os, sys, json, csv, glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for _p in (os.path.join(_PKG, "engine"), _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import substrate as SUB

# ---- thresholds (all documented; tune here, there is no hidden config) ----------------------------
FS_EXPECTED = 250.0        # the validated mouse capture sampling rate
RR_LO_BPM, RR_HI_BPM = 90.0, 320.0   # plausible recomputed breathing rate (calm mouse ~130-210 bpm;
                                     # sustained faster is almost always a harmonic -> ~300 bpm guard)
CLEAN_MIN_FAIL = 0.5       # < this many MINUTES of breathing recovered -> genuinely no usable signal
CLEAN_MIN_REVIEW = 3.0     # < this many minutes -> thin data, worth a human glance (not a fail)
N_CYCLES_FAIL = 40         # fewer than this many paired breath/pause cycles -> no usable signal
REG_CYCLES_INFO = 200      # fewer calm cycles than this -> the arrhythmia metric (ttot_cv) may be blank
                           #   (expected for short / very active recordings -- informational, NEVER a fail)
PRESENCE_ABSENT = 0.03     # breathing present < 3% of the recording (with little clean data) -> animal absent
OCTAVE_LO, OCTAVE_HI = 1.7, 2.3      # live/recomputed rate ratio in here -> the LIVE rate was octave-doubled

_LEVELS = {"ok": 0, "review": 1, "fail": 2}
_VERDICT = {0: "PASS", 1: "REVIEW", 2: "FAIL"}


# ---- best-effort context from the raw experiment folder (still at substrate.session_dir) ----------
def _raw_total_min(meta):
    """Total recording length in minutes. Prefer the substrate field `total_min` (written by newer
    segmentation runs); else read it from the raw `*_voxels.json` header at `session_dir`; else None."""
    tm = meta.get("total_min")
    if tm:
        return float(tm)
    sdir = meta.get("session_dir")
    if sdir and os.path.isdir(sdir):
        vj = glob.glob(os.path.join(sdir, "*_voxels.json"))
        if vj:
            try:
                h = json.load(open(vj[0]))
                return float(h["n_time"]) / float(h["fs"]) / 60.0
            except Exception:
                pass
    return None


def _live_summary(meta):
    """The live-capture `*_summary.json` next to the raw data, if present (best-effort). Returns the
    handful of fields the old live QC keyed on, or {}."""
    sdir = meta.get("session_dir")
    if not (sdir and os.path.isdir(sdir)):
        return {}
    sj = glob.glob(os.path.join(sdir, "*_summary.json"))
    if not sj:
        return {}
    try:
        s = json.load(open(sj[0]))
    except Exception:
        return {}
    out = {}
    for k in ("rr_med_bpm", "coverage_pct", "clean_time_s", "duration_s", "n_episodes"):
        if s.get(k) is not None:
            out[k] = s[k]
    return out


def _load_sibling(out_dir, rec, suffix):
    """Fold in a sibling pipeline's json (stettner/pauses/motion) if it was already written."""
    if not out_dir:
        return {}
    p = os.path.join(out_dir, rec + suffix)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    return {}


# ---- the QC assessment ----------------------------------------------------------------------------
def assess(rec, out_dir=None):
    """Recompute the QC signals for one recording from its cached substrate and return a report dict."""
    meta = SUB.load_meta(rec)
    info = SUB.rec_info(rec, meta)
    fs = info["fs"]
    clean_min = float(info["clean_min"] or 0.0)
    n_episodes = int(info["n_episodes"] or 0)
    med_bpm = info["med_bpm"]

    eps = meta.get("episodes", []) or []
    acs = [e.get("ac") for e in eps if e.get("ac") is not None]
    med_ac = float(np.median(acs)) if acs else None

    # cycle counts (paired breath+pause cycles; and the calm/arousal-matched subset for arrhythmia)
    n_cycles = n_reg = None
    try:
        cyc = SUB.build_cycle_series(rec, meta=meta)["cycles"]
        n_cycles = len(cyc)
        n_reg = len(SUB.regularity_cycles(cyc))
    except Exception:
        n_cycles = int(meta.get("n_ibi") or 0) or None      # fall back to a rough proxy

    total_min = _raw_total_min(meta)
    present_pct = (clean_min / total_min) if (total_min and total_min > 0) else None
    live = _live_summary(meta)

    # optional richer context from sibling pipelines (never affects the verdict)
    stett = _load_sibling(out_dir, rec, ".stettner.json")
    stett_flat = {}
    if stett:
        for grp in ("regularity", "burden", "morphology"):
            stett_flat.update(stett.get(grp, {}) or {})
    calm_frac = (n_reg / n_cycles) if (n_reg is not None and n_cycles) else None

    checks = []

    def add(name, level, message):
        checks.append(dict(name=name, level=level, message=message))

    # 1) IS THERE A BREATHING SIGNAL AT ALL?  (the only true "bad data" gate)
    if n_episodes == 0 or clean_min < CLEAN_MIN_FAIL or (n_cycles is not None and n_cycles < N_CYCLES_FAIL):
        add("signal_present", "fail",
            f"Almost no breathing was recovered from the stored voxels "
            f"({clean_min:.1f} min, {n_episodes} episodes, {n_cycles if n_cycles is not None else '?'} cycles). "
            f"This points to a capture problem -- the animal out of the beam, a truncated recording, or the "
            f"wrong voxel file -- NOT a physiology finding.")
    elif clean_min < CLEAN_MIN_REVIEW:
        add("signal_present", "review",
            f"Only {clean_min:.1f} min of breathing was recovered. The recording is usable but thin; "
            f"rate-and-burden numbers will be noisier than usual.")
    else:
        add("signal_present", "ok",
            f"Breathing recovered: {clean_min:.1f} min across {n_episodes} episodes"
            + (f", {n_cycles} cycles" if n_cycles is not None else "") + ".")

    # 2) IS THE RECOMPUTED BREATHING RATE PHYSIOLOGICALLY PLAUSIBLE?
    if med_bpm is None:
        add("breathing_rate", "review", "No breathing rate could be estimated from the recovered signal.")
    elif RR_LO_BPM <= med_bpm <= RR_HI_BPM:
        msg = f"Recomputed breathing rate {med_bpm:.0f} bpm (normal mouse range)."
        live_rr = live.get("rr_med_bpm")
        if live_rr:
            ratio = live_rr / med_bpm
            if OCTAVE_LO <= ratio <= OCTAVE_HI:
                msg += (f"  NOTE: the live capture reported {live_rr:.0f} bpm -- about 2x this -- a classic "
                        f"octave-doubling error in the live rate. The recomputed value is the correct one.")
            elif ratio > 1.25 or ratio < 0.8:
                msg += (f"  NOTE: the live capture reported {live_rr:.0f} bpm; they differ, so trust the "
                        f"recomputed rate, not the live one.")
        add("breathing_rate", "ok", msg)
    else:
        add("breathing_rate", "review",
            f"Recomputed breathing rate {med_bpm:.0f} bpm is outside the usual {RR_LO_BPM:.0f}-{RR_HI_BPM:.0f} bpm "
            f"range. Inspect the trace -- this can be a residual rate/harmonic anomaly or an unusual animal.")

    # 3) SAMPLING RATE (validated setup is 250 Hz)
    if abs(fs - FS_EXPECTED) < 1e-6:
        add("sampling_rate", "ok", f"Sampling rate {fs:.0f} Hz (the validated setup).")
    else:
        add("sampling_rate", "review",
            f"Sampling rate is {fs:.0f} Hz, not the validated {FS_EXPECTED:.0f} Hz. Metrics still compute, "
            f"but confirm the capture config -- rate-dependent numbers are only calibrated at 250 Hz.")

    # 4) PRESENCE (breathing present vs the whole recording) -- the reframe of live "coverage"
    if present_pct is not None:
        if present_pct < PRESENCE_ABSENT and clean_min < CLEAN_MIN_REVIEW:
            add("presence", "review",
                f"Breathing was present for only {present_pct*100:.0f}% of the {total_min:.0f} min recording. "
                f"The animal may have been out of the beam for most of it -- worth a look.")
        else:
            add("presence", "ok",
                f"Breathing present {present_pct*100:.0f}% of the {total_min:.0f} min recording. "
                f"(Time without recovered breathing = the animal moving/active, which is normal.)")

    # 5) ENOUGH CALM DATA FOR THE ARRHYTHMIA METRIC?  (informational -- NEVER a fail)
    if n_reg is not None:
        if n_reg >= REG_CYCLES_INFO:
            add("arrhythmia_data", "ok",
                f"Enough calm breathing for the arrhythmia metric (ttot_cv): {n_reg} calm cycles.")
        else:
            add("arrhythmia_data", "ok",
                f"Only {n_reg} calm cycles, so the arrhythmia metric (ttot_cv) may come out blank. That is "
                f"EXPECTED for short or very active/aroused recordings and is NOT a data-quality failure -- the "
                f"other metrics are still valid.")

    # 6) AROUSAL REFRAME (informational) -- the thing the old QC mistook for bad data
    if calm_frac is not None or med_ac is not None:
        bits = []
        if calm_frac is not None:
            bits.append(f"the animal was calm/regular for ~{calm_frac*100:.0f}% of its breathing")
        if med_ac is not None:
            bits.append(f"median episode regularity (AC) {med_ac:.2f}")
        add("arousal", "ok",
            ("; ".join(bits).capitalize() + ". A low calm fraction means an active or sick (irregular) "
             "animal -- that is a phenotype we want to keep, NOT bad data."))

    # 7) DID THE SEGMENTER COUNT EVERY BREATH?  A breather faster than ~250/min is HALVED by the default breath bounds,
    #    and every breathing metric and the heart rate then run on a half-rate clock; one just above their reach is
    #    CLIPPED (breaths squeezed against the 0.20 s minimum, ng511 10-21) (engine/breath_check.py).
    bchk = None
    try:
        import breath_check as BC
        bchk = BC.rate_ratio(os.path.join(SUB.SUB_DIR, rec))
        bchk["halved"] = BC.is_halved(bchk)
        bchk["clipped"] = BC.is_clipped(bchk)
    except Exception:
        pass
    reseg = bool((meta.get("breath_check") or {}).get("resegmented"))
    if bchk is not None and bchk["halved"]:
        add("breath_count", "review",
            f"The segmentation counted every other breath (spectral {bchk['spec_bpm']:.0f}/min vs segmenter "
            f"{bchk['dp_bpm']:.0f}/min). Breathing rate, cycle timing, apneas and heart rate are all built on a "
            f"half-rate clock. Re-segment with --breath-bounds 0.12,0.21 (run.py segment / all do this "
            f"automatically) and re-run.")
    elif bchk is not None and bchk["clipped"]:
        add("breath_count", "review",
            f"The segmentation squeezed the breaths against its minimum breath length ({100 * bchk['pinned_frac']:.0f}% "
            f"of breath motions at {bchk['bmin_b']:g} s; spectral {bchk['spec_bpm']:.0f}/min vs segmenter "
            f"{bchk['dp_bpm']:.0f}/min): the mouse breathes too fast for these bounds. Breath timing, cycle variability "
            f"and heart rate are distorted. Re-segment with --breath-bounds 0.12,0.21 (run.py segment / all do this "
            f"automatically) and re-run.")
    elif bchk is not None and np.isfinite(bchk["ratio"]):
        add("breath_count", "ok",
            f"Every breath counted (spectral {bchk['spec_bpm']:.0f}/min vs segmenter {bchk['dp_bpm']:.0f}/min)"
            + (", after automatic re-segmentation with the fast breath bounds (a breather too fast for the default "
               "bounds)." if reseg else "."))

    level = max((_LEVELS[c["level"]] for c in checks), default=0)
    verdict = _VERDICT[level]

    signals = dict(
        clean_min=round(clean_min, 1), total_min=round(total_min, 1) if total_min else None,
        present_pct=round(present_pct, 3) if present_pct is not None else None,
        n_episodes=n_episodes, n_cycles=n_cycles, n_reg_cycles=n_reg,
        med_bpm=med_bpm, med_ac=round(med_ac, 3) if med_ac is not None else None,
        calm_frac=round(calm_frac, 3) if calm_frac is not None else None, fs=fs,
        breath_check_ratio=round(bchk["ratio"], 3) if bchk and np.isfinite(bchk["ratio"]) else None,
        segmenter_bpm=round(bchk["dp_bpm"], 1) if bchk and np.isfinite(bchk["dp_bpm"]) else None,
        breathing_halved=bchk["halved"] if bchk else None, breathing_clipped=bchk["clipped"] if bchk else None,
        pinned_frac=round(bchk["pinned_frac"], 3) if bchk and np.isfinite(bchk["pinned_frac"]) else None,
        resegmented_fast=reseg,
    )
    for k in ("ttot_cv", "apnea_rate_per_min", "tpause_p90_ms"):
        if k in stett_flat:
            signals[k] = stett_flat[k]
    apn = (_load_sibling(out_dir, rec, ".apnea.json") or {}).get("metrics", {})     # informational, never a verdict
    for k in ("apnea_3x_n", "apnea_3x_per_clean_min", "apnea_gt1s_n", "apnea_gt1s_per_clean_min"):
        if k in apn:
            signals[k] = apn[k]
    if "rate_reliable" in apn:
        signals["apnea_rate_reliable"] = apn["rate_reliable"]
    hrm = (_load_sibling(out_dir, rec, ".hr.json") or {}).get("metrics", {})        # informational, never a verdict
    for k in ("hr_median_bpm", "hr_p25_bpm", "hr_p75_bpm", "n_periods", "scored_min", "n_fallback", "breathing_halved"):
        if k in hrm:
            signals["hr_" + k if not k.startswith("hr_") else k] = hrm[k]
    bbm = (_load_sibling(out_dir, rec, ".b2b.json") or {}).get("metrics", {})       # informational, never a verdict
    for k in ("n_periods", "n_tracked", "tracked_min", "n_beats", "hr_median_bpm", "hr_covered_min",
              "hr_per_breath_median", "n_guard_seed", "n_guard_hr", "frozen_code_ok"):
        if k in bbm:
            signals["b2b_" + k] = bbm[k]

    return dict(recording=rec, cohort=info["cohort"], genotype=info["genotype"], age_weeks=info["age_weeks"],
                verdict=verdict, checks=checks, signals=signals, live=live)


# ---- rendering ------------------------------------------------------------------------------------
_MARK = {"PASS": "[ PASS ]", "REVIEW": "[REVIEW]", "FAIL": "[ FAIL ]"}
_BAR = "=" * 72


def render_banner(rep):
    """A plain-English banner for a nontechnical reader (ASCII only, safe in any terminal)."""
    s = rep["signals"]; live = rep.get("live") or {}
    L = [_BAR, f"  QC VERDICT:  {_MARK[rep['verdict']]}    {rep['recording']}", _BAR]

    if rep["verdict"] == "PASS":
        L.append("  Analyzable. Recomputed from the stored voxels, this recording looks")
        L.append("  physiologically reasonable.")
    elif rep["verdict"] == "REVIEW":
        L.append("  Usable, but worth a human glance before you trust the numbers (see below).")
    else:
        L.append("  This recording does NOT look analyzable -- the recomputed signal itself is")
        L.append("  missing or implausible. This is a data-capture problem, not a finding.")
    L.append("")

    def row(label, val):
        L.append(f"    {label:22s}: {val}")

    pres = f"{s['present_pct']*100:.0f}% of {s['total_min']:.0f} min" if s.get("present_pct") is not None else "n/a"
    row("Recovered breathing", f"{s['clean_min']:.1f} min  (present {pres})")
    seg_bpm, med = s.get("segmenter_bpm"), s.get("med_bpm")
    if seg_bpm and med and not 0.77 <= seg_bpm / med <= 1.3:    # fast breather: the per-episode estimate halves it
        row("Breathing rate", f"{seg_bpm:.0f} bpm (two-template segmentation; the per-episode estimate reads {med:.0f})")
    else:
        row("Breathing rate", f"{med:.0f} bpm" if med is not None else "not estimated")
    if s.get("breathing_halved"):
        row("", "BREATHING HALVED: every other breath counted -- re-segment (see the flag below)")
    elif s.get("breathing_clipped"):
        row("", "BREATHING CLIPPED: breaths squeezed against the minimum breath length -- re-segment (see below)")
    row("Cycles for metrics", f"{s.get('n_cycles')} total, {s.get('n_reg_cycles')} calm")
    row("Sampling rate", f"{s['fs']:.0f} Hz")
    if "ttot_cv" in s:
        row("Arrhythmia (ttot_cv)", s["ttot_cv"] if s["ttot_cv"] is not None else "blank (too little calm data)")
    if "apnea_3x_n" in s:
        na = lambda v: "n/a" if v is None else v
        row("Apneas (see APNEA.md)", f"{s['apnea_3x_n']} > 3x cycle ({na(s['apnea_3x_per_clean_min'])}/clean-min), "
                                     f"{s['apnea_gt1s_n']} > 1 s ({na(s['apnea_gt1s_per_clean_min'])}/clean-min)"
                                     + ("" if s.get("apnea_rate_reliable", True) else "  [< 3 clean min: don't pool]"))
    if "hr_n_periods" in s:
        row("Heart rate", (f"{s['hr_median_bpm']:.0f} bpm median (IQR {s['hr_p25_bpm']:.0f}-{s['hr_p75_bpm']:.0f}) "
                           f"over {s['hr_n_periods']} still periods, {s['hr_scored_min']} min")
                          if s.get("hr_median_bpm") is not None else "none (no still period >= 20 s)")
        if s.get("hr_breathing_halved"):
            row("", "BREATHING HALVED -> NOT a heart rate (a breathing harmonic); re-segment and re-run")
    if "b2b_n_tracked" in s:
        row("Beat-to-beat HR (B2B.md)",
            (f"{s['b2b_hr_median_bpm']:.0f} bpm median over {s['b2b_hr_covered_min']} min of 2 s values; "
             f"{s['b2b_n_tracked']} of {s['b2b_n_periods']} periods tracked, {s['b2b_n_beats']} beats"
             if s.get("b2b_hr_median_bpm") is not None else
             f"none: abstained on all {s['b2b_n_periods']} periods (heart too weak to time beats)"))
        if s.get("b2b_n_guard_seed") or s.get("b2b_n_guard_hr"):
            row("", f"{s.get('b2b_n_guard_seed', 0) + s.get('b2b_n_guard_hr', 0)} period(s) dropped by the 750 bpm guard "
                    f"(non-cardiac ~800 bpm line)")

    # the reasons, worst first
    ordered = sorted(rep["checks"], key=lambda c: -_LEVELS[c["level"]])
    flagged = [c for c in ordered if c["level"] != "ok"]
    if flagged:
        L.append(""); L.append("  Flags:")
        for c in flagged:
            L.append(f"    - ({c['level'].upper()}) {c['message']}")

    # live-vs-recomputed contrast: why the old pass/fail was misleading
    if live:
        L.append(""); L.append("  Live capture summary (what the OLD pass/fail used):")
        if live.get("rr_med_bpm") is not None:
            L.append(f"    rate {live['rr_med_bpm']:.0f} bpm   (recomputed: "
                     f"{s['med_bpm']:.0f} bpm)" if s.get("med_bpm") is not None else
                     f"    rate {live['rr_med_bpm']:.0f} bpm")
        if live.get("coverage_pct") is not None:
            extra = (f"   (recomputed breathing present: {s['present_pct']*100:.0f}%)"
                     if s.get("present_pct") is not None else "")
            L.append(f"    \"coverage\" {live['coverage_pct']:.1f}%{extra}")
        L.append("    -> live \"coverage\" is the CALM fraction, not data quality; a low value means an")
        L.append("       active/sick animal, not a bad recording. Trust the recomputed numbers above.")
    L.append(_BAR)
    return "\n".join(L)


# ---- pipeline entry point (called by run.py) ------------------------------------------------------
NAME = "qc"
KIND = "segments"     # runs off the cached substrate (no raw access required for the verdict itself)


def run(rec, sub_dir=None, out_dir=None, log=print):
    """Assess one recording, write <out>/<rec>.qc.{json,csv}, print the banner, and return a flat dict."""
    if sub_dir is not None:
        SUB.SUB_DIR = sub_dir
    rep = assess(rec, out_dir=out_dir)
    flat = dict(recording=rec, cohort=rep["cohort"], genotype=rep["genotype"], age_weeks=rep["age_weeks"],
                verdict=rep["verdict"],
                n_flags=sum(1 for c in rep["checks"] if c["level"] != "ok"),
                reasons=" | ".join(c["message"] for c in rep["checks"] if c["level"] != "ok"),
                **rep["signals"])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json.dump(rep, open(os.path.join(out_dir, rec + ".qc.json"), "w"), indent=1, default=float)
        with open(os.path.join(out_dir, rec + ".qc.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(flat.keys()); w.writerow(flat.values())
    log(render_banner(rep))
    return flat


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="quality-control check for one segmented recording")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--sub-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.rec, sub_dir=a.sub_dir, out_dir=a.out)
