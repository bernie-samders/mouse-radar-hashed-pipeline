r"""stettner.py -- Stettner (2007) breathing metrics from the two-template segmentation.

Per-experiment metric bundle, computed from one segmentation substrate (<rec>.json + <rec>.npz produced
by engine/segmentation.py).  Ported verbatim from the validated `output/stettner_two_template/extract.py`
so the numbers are identical; only the cohort-statistics / genotype-testing layer was dropped (this returns
one experiment's metrics, not a WT-vs-mut contrast).

REGULARITY / COUPLING / MORPHOLOGY are computed on the AROUSAL-MATCHED substrate (per-episode AC >= 0.45
AND calm Ttot in [0.25,1.0] s), cessation-gated.  BURDEN (pause tail, apnea rate) uses the FULL substrate
(no AC gate -- the gate removes exactly the arrhythmic long-pause episodes that carry the burden).  See the
project README / the original SPEC for the physiology and the gotchas.

Metric groups returned (flat dict):
  regularity : ttot_cv (respiratory arrhythmia; the established genotype separator), ttot_var_ms2,
               ti_te_ratio_cv, and DP-cycle / pause twins.
  coupling   : plateau<->Ttot, pause<->Ttot, breath<->Ttot correlations (radar-limited; see README caveats).
  burden     : tpause median/p90/p95, frac pause > 0.5/1/2 s, apnea rate/min, apnea fraction of time.
  morphology : Ti/Te fractions, Tpi plateau (flagged POLARITY-ARTIFACT -- do NOT report as a genotype
               effect; its *_flip twins expose the artifact), median cycle length, rate.
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
import morphology as MO
import stettner_metrics as SM        # _windows, _cv, _var_ms2, _wmed, _wz, MIN_WIN
import phase_lib as P                # cessation_slices, pearson, WIN_CYCLES
from scipy.ndimage import median_filter

CALM_BAND = SUB.CALM_BAND
AC_FLOOR = SUB.AC_FLOOR
WIN = P.WIN_CYCLES                    # 60-cycle windows (paper)
MIN_WIN = SM.MIN_WIN                  # >=20 cycles -> a variance/coupling point
APNEA_MULT = 2.0                      # apnea = pause >= 2 x local-median cycle
PAUSE_CENSOR_S = 2.49                 # bmaxP=2.5 s -> pauses at/above this are right-censored


# ---- small helpers --------------------------------------------------------------------------------
def _med_ms(x):
    x = np.asarray(x, float)
    return round(float(np.median(x)) * 1000.0, 1) if x.size else None


def _pct_ms(x, q):
    x = np.asarray(x, float)
    return round(float(np.percentile(x, q)) * 1000.0, 1) if x.size else None


def _r2(v, nd=4):
    return round(float(v), nd) if v is not None and np.isfinite(v) else None


def _group_by_episode(cycles):
    from collections import defaultdict
    g = defaultdict(list)
    for c in cycles:
        g[c["ep"]].append(c)
    return {ep: sorted(cs, key=lambda c: c["t_onset"]) for ep, cs in g.items()}


# ---- REGULARITY / COUPLING on the DP-cycle series (arousal-matched, cessation-gated) --------------
def _windowed_dp(dp_by_ep, ac_floor=AC_FLOOR, calm=CALM_BAND):
    var_recs, cpl_recs = [], []
    for ep, cs in dp_by_ep.items():
        ac = cs[0].get("ac", np.nan)
        if not np.isfinite(ac) or ac < ac_floor:
            continue
        ttot = np.array([c["ttot"] for c in cs])
        calm_mask = (ttot >= calm[0]) & (ttot <= calm[1]) if calm else np.ones(ttot.size, bool)
        if calm_mask.sum() < max(8, MIN_WIN // 2):
            continue
        idx = np.where(calm_mask)[0]; sub = [cs[i] for i in idx]; t = ttot[idx]
        for (a, b) in P.cessation_slices(t):
            for (w0, w1) in SM._windows(a, b, w=WIN, min_win=MIN_WIN):
                win = sub[w0:w1]; n = w1 - w0
                tdp = np.array([c["ttot"] for c in win]); tpa = np.array([c["tpause"] for c in win])
                tbr = np.array([c["tbreath"] for c in win])
                var_recs.append(dict(n=int(n), ttot_cv_dp=SM._cv(tdp), ttot_var_dp=SM._var_ms2(tdp),
                                     tpause_cv=SM._cv(tpa), tpause_var=SM._var_ms2(tpa)))
                if n >= MIN_WIN:
                    cpl_recs.append(dict(n=int(n), r_pause_ttot=P.pearson(tpa, tdp),
                                         r_tbreath_ttot=P.pearson(tbr, tdp)))
    return var_recs, cpl_recs


# ---- BURDEN (full substrate, no gate) -------------------------------------------------------------
def _apnea_mask(ttot, tpause, mult=APNEA_MULT, win=WIN):
    ttot = np.asarray(ttot, float)
    if ttot.size < 3:
        return np.zeros(ttot.size, bool)
    loc = median_filter(ttot, size=max(3, (win | 1)), mode="nearest")
    return np.asarray(tpause, float) >= mult * loc


def _burden(cycles):
    if not cycles:
        return {}
    tpause = np.array([c["tpause"] for c in cycles])
    ttot = np.array([c["ttot"] for c in cycles])
    eps = _group_by_episode(cycles)
    apnea_pause = []
    n_apnea = 0
    for cs in eps.values():
        tt = np.array([c["ttot"] for c in cs]); tp = np.array([c["tpause"] for c in cs])
        a = _apnea_mask(tt, tp)
        n_apnea += int(a.sum())
        apnea_pause += list(tp[a])
    apnea_pause = np.asarray(apnea_pause, float)
    analyzed_min = float(ttot.sum()) / 60.0
    return dict(
        tpause_med_ms=_med_ms(tpause), tpause_p90_ms=_pct_ms(tpause, 90), tpause_p95_ms=_pct_ms(tpause, 95),
        frac_pause_gt_0p5s=round(float(np.mean(tpause > 0.5)), 4),
        frac_pause_gt_1s=round(float(np.mean(tpause > 1.0)), 4),
        frac_pause_gt_2s=round(float(np.mean(tpause > 2.0)), 4),
        apnea_rate_per_min=round(n_apnea / analyzed_min, 3) if analyzed_min > 0 else None,
        apnea_frac_time=round(float(apnea_pause.sum() / ttot.sum()), 4) if ttot.sum() > 0 else None,
        apnea_dur_med_s=round(float(np.median(apnea_pause)), 3) if apnea_pause.size else None,
        censored_2p5_frac=round(float(np.mean(tpause >= PAUSE_CENSOR_S)), 4),
        n_apnea=n_apnea, analyzed_min=round(analyzed_min, 1),
    )


# ---- MORPHOLOGY + windowed regularity/coupling on the phase_lib trace series ----------------------
def _morphology_and_windows(rec, meta, npz, fs, ac_floor=AC_FLOOR, calm=CALM_BAND):
    eps = meta.get("episodes", []) or []
    POOL_KEYS = ("ti_frac", "te_frac", "plateau_frac", "ti_te_ratio", "skew",
                 "plateau_frac_flip", "plateau_frac_peak", "plateau_frac_trough", "eip", "eip_flip", "ttot_pl")
    pooled = {k: [] for k in POOL_KEYS}
    pol_votes = []
    var_recs, cpl_recs = [], []
    for ep, e in enumerate(eps):
        ac = e.get("ac")
        if ac is None or ac < ac_floor:
            continue
        tr = np.asarray(npz["trace"][ep], float)
        s = MO.per_cycle(tr, float(fs), e.get("med_bpm") or 96.0)
        ttot = s["ttot"]
        if ttot.size == 0:
            continue
        pol_votes.append(s["polarity"])
        res = s["resolvable"]
        calm_mask = (ttot >= calm[0]) & (ttot <= calm[1]) if calm else np.ones(ttot.size, bool)
        keep = res & calm_mask
        if keep.sum() < 5:
            continue
        for k in ("ti_frac", "te_frac", "plateau_frac", "ti_te_ratio", "skew",
                  "plateau_frac_flip", "plateau_frac_peak", "plateau_frac_trough"):
            pooled[k] += list(np.asarray(s[k])[keep])
        pooled["eip"] += list(s["eip"][keep]); pooled["eip_flip"] += list(np.asarray(s["eip_flip"])[keep])
        pooled["ttot_pl"] += list(ttot[keep])
        idx = np.where(keep)[0]
        ttk = ttot[idx]; eipk = s["eip"][idx]; ratk = np.asarray(s["ti_te_ratio"])[idx]
        for (a, b) in P.cessation_slices(ttk):
            for (w0, w1) in SM._windows(a, b, w=WIN, min_win=MIN_WIN):
                n = w1 - w0
                tt = ttk[w0:w1]; ei = eipk[w0:w1]; ra = ratk[w0:w1]
                ra_f = ra[np.isfinite(ra)]
                var_recs.append(dict(n=int(n), ttot_cv=SM._cv(tt), ttot_var=SM._var_ms2(tt),
                                     ti_te_ratio_cv=SM._cv(ra_f) if ra_f.size >= 2 else float("nan")))
                if n >= MIN_WIN:
                    mf = np.isfinite(ei) & np.isfinite(tt)
                    cpl_recs.append(dict(n=int(n),
                                         r_plateau_ttot=P.pearson(ei[mf], tt[mf]) if mf.sum() >= 4 else float("nan")))
    pol, mixed = MO.session_polarity(pol_votes)
    out = dict(polarity_choice=pol, polarity_mixed_frac=mixed)
    if pooled["ttot_pl"]:
        out.update(
            ttot_med_ms=_med_ms(pooled["ttot_pl"]),
            ti_frac_med=_r2(np.median(pooled["ti_frac"]), 3), te_frac_med=_r2(np.median(pooled["te_frac"]), 3),
            ti_te_ratio_med=_r2(np.median(pooled["ti_te_ratio"]), 3), skew_med=_r2(np.median(pooled["skew"]), 3),
            plateau_frac_med=_r2(np.median(pooled["plateau_frac"]), 4),          # chosen orientation (biased)
            plateau_frac_med_flip=_r2(np.median(pooled["plateau_frac_flip"]), 4),
            plateau_frac_peak_med=_r2(np.median(pooled["plateau_frac_peak"]), 4),
            plateau_frac_trough_med=_r2(np.median(pooled["plateau_frac_trough"]), 4),
            tpi_ms=_med_ms(pooled["eip"]), tpi_ms_flip=_med_ms(pooled["eip_flip"]),
            n_morph_cycles=len(pooled["ttot_pl"]),
        )
    else:
        out.update({k: None for k in ("ttot_med_ms", "ti_frac_med", "te_frac_med", "ti_te_ratio_med",
                    "skew_med", "plateau_frac_med", "plateau_frac_med_flip", "plateau_frac_peak_med",
                    "plateau_frac_trough_med", "tpi_ms", "tpi_ms_flip")})
        out["n_morph_cycles"] = 0
    return out, var_recs, cpl_recs


# ---- top-level per-experiment analysis ------------------------------------------------------------
def analyze_recording(rec, ac_floor=AC_FLOOR, calm=CALM_BAND):
    """Full Stettner metric bundle for one recording.  Returns a nested dict plus a flat `_flat` dict."""
    meta = SUB.load_meta(rec)
    npz = np.load(os.path.join(SUB.SUB_DIR, rec + ".npz"), allow_pickle=True)
    info = SUB.rec_info(rec, meta)
    cs = SUB.build_cycle_series(rec, meta=meta, npz=npz)
    cycles = cs["cycles"]

    morph, var_pl, cpl_pl = _morphology_and_windows(rec, meta, npz, info["fs"], ac_floor=ac_floor, calm=calm)
    var_dp, cpl_dp = _windowed_dp(_group_by_episode(cycles), ac_floor=ac_floor, calm=calm)

    regularity = dict(
        ttot_cv=_r2(SM._wmed(var_pl, "ttot_cv")) if var_pl else None,                # PRIMARY (arrhythmia; Fig 2B)
        ttot_var_ms2=_r2(SM._wmed(var_pl, "ttot_var"), 1) if var_pl else None,
        ti_te_ratio_cv=_r2(SM._wmed(var_pl, "ti_te_ratio_cv")) if var_pl else None,
        ttot_cv_dp=_r2(SM._wmed(var_dp, "ttot_cv_dp")) if var_dp else None,           # DP-cycle (pause-floor confounded)
        ttot_var_ms2_dp=_r2(SM._wmed(var_dp, "ttot_var_dp"), 1) if var_dp else None,
        tpause_cv=_r2(SM._wmed(var_dp, "tpause_cv")) if var_dp else None,             # floor-dominated -> report-only
        tpause_var_ms2=_r2(SM._wmed(var_dp, "tpause_var"), 1) if var_dp else None,
    )
    coupling = dict(
        coupling_plateau_ttot_r=_r2(SM._wz(cpl_pl, "r_plateau_ttot"), 3) if cpl_pl else None,
        coupling_pause_ttot_r=_r2(SM._wz(cpl_dp, "r_pause_ttot"), 3) if cpl_dp else None,
        coupling_tbreath_ttot_r=_r2(SM._wz(cpl_dp, "r_tbreath_ttot"), 3) if cpl_dp else None,
        n_cpl_windows=len(cpl_pl),
    )
    burden = _burden(SUB.burden_cycles(cycles))

    n_reg = sum(r["n"] for r in var_pl)
    result = dict(
        recording=rec, cohort=info["cohort"], genotype=info["genotype"], genotype_legacy=info["genotype_legacy"],
        age_weeks=info["age_weeks"], severity_ref=info["severity_ref"], is_matched=info["is_matched"],
        fs=info["fs"], med_bpm=info["med_bpm"], clean_min=info["clean_min"],
        n_cycles=len(cycles), n_reg_cycles=int(n_reg), n_windows=len(var_pl),
        regularity=regularity, coupling=coupling, burden=burden, morphology=morph,
    )
    flat = dict(recording=rec, cohort=info["cohort"], genotype=info["genotype"],
                age_weeks=info["age_weeks"], is_matched=info["is_matched"],
                med_bpm=info["med_bpm"], clean_min=info["clean_min"],
                n_cycles=len(cycles), n_reg_cycles=int(n_reg))
    for group in (regularity, coupling, burden, morph):
        for k, v in group.items():
            if isinstance(v, (int, float)) or v is None:
                flat[k] = v
    result["_flat"] = flat
    return result


# ---- pipeline entry point (called by run.py) ------------------------------------------------------
NAME = "stettner"
KIND = "segments"     # consumes the segmentation substrate


def run(rec, sub_dir=None, out_dir=None, log=print):
    """Compute + save Stettner metrics for one recording.  Writes <out_dir>/<rec>.stettner.json and
    returns the flat metric dict."""
    if sub_dir is not None:
        SUB.SUB_DIR = sub_dir
    res = analyze_recording(rec)
    flat = res["_flat"]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json.dump({k: v for k, v in res.items() if k != "_flat"},
                  open(os.path.join(out_dir, rec + ".stettner.json"), "w"), indent=1, default=float)
        with open(os.path.join(out_dir, rec + ".stettner.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(flat.keys()); w.writerow(flat.values())
    log(f"[stettner] {rec:20s} ttot_cv={flat.get('ttot_cv')} tpause_p90_ms={flat.get('tpause_p90_ms')} "
        f"apnea/min={flat.get('apnea_rate_per_min')} (n_reg_cycles={flat.get('n_reg_cycles')})")
    return flat


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Stettner metrics for one segmented recording")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--sub-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.rec, sub_dir=a.sub_dir, out_dir=a.out)
