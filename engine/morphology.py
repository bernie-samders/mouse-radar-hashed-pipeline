r"""morphology.py -- within-cycle PHASE morphology (Ti / Te / Tpi-plateau) from the breathing trace.

The two-template DP (substrate.py) gives faithful CYCLE TIMING (Ttot, end-expiratory pause) but its
breath-MOTION segment is ceiling-clipped (0.20-0.35 s), so it CANNOT measure the within-cycle phase
split or the high-volume post-inspiratory plateau.  Those come from the breathing-trace SHAPE instead
-- exactly the user's instinct ("use the template breaths, not each individual breath").

Two shape read-outs, both on the cached conditioned `trace` (fs=250, finer than the 125 Hz DP signal
and the substrate phase_lib was validated on):

  (1) PER-CYCLE decomposition (phase_lib.per_cycle_series, reused unedited): trough->peak->trough gives
      rise (~Ti), fall (~Te+early-pause), and the end-inspiratory dwell EIP (~Tpi, the high-volume
      plateau).  These per-cycle values -- smoothed by phase_lib's period/8 filter -- drive the
      windowed CV / coupling metrics.
  (2) A cycle-triggered TEMPLATE breath per analysis window: amplitude-normalised, length-warped
      average of the window's cycles.  Region-adaptive (re-learned per window), noise-robust, and the
      natural home for a denoised plateau read-out + the WT-vs-mut morphology figure.

EMPIRICAL FINDING (validated on the 23-rec cohort, see RESULTS.md): the plateau fraction
(EIP / Ttot ~ Tpi prolongation) separates mutant > WT with matched-8wk Cohen d ~ +2.4 -- Stettner's
central "prolonged post-inspiration" result, recovered on radar and confound-robust.  The paper's
Tpi<->Ttot COUPLING does NOT reproduce on radar (EIP is 4-8 ms quantised and cycle-length variance is
end-expiratory-pause-dominated, not plateau-dominated) -- reported but flagged, never the headline.

POLARITY: radar magnitude has no absolute inspiration/expiration sign, so end-inspiration is chosen as
the extreme with the longer high-volume dwell (phase_lib.episode_polarity).  Every polarity-dependent
scalar is emitted under the chosen AND flipped hypothesis (`*_flip`).
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import phase_lib as P
import eqradio_hr as eq                     # eq.warp (length-normalising resample) for the template

MORPH_FS = 250.0                            # the conditioned `trace` channel is stored at fs (=250)
TPL_M = 48                                  # template length (samples) for the cycle-triggered average


# ==================================================================================================
# (1) per-cycle phase decomposition (reuse phase_lib on the conditioned trace)
# ==================================================================================================
def per_cycle(trace, fs, med_bpm):
    """Per-cycle within-breath morphology for one episode's conditioned trace.  Returns a dict of
    aligned float arrays (seconds / fractions), one row per interior cycle, plus polarity meta:
        ttot, rise, fall, ti, te, eip(=plateau dwell), phase_asym, resolvable, polarity,
        ti_frac(=ti/ttot), te_frac(=te/ttot), plateau_frac(=eip/ttot), skew(=te_frac-ti_frac),
        ti_te_ratio(=ti/te).
    Empty arrays if the trace is unusable (delegates to phase_lib.per_cycle_series)."""
    s = P.per_cycle_series(dict(trace=np.asarray(trace, float), fs=float(fs), med_bpm=float(med_bpm)))
    ttot = s["ttot"]
    pol = s["polarity"]
    # POLARITY IS AMBIGUOUS (radar has no absolute inspiration sign; episode_polarity picks the
    # LONGER-dwell extreme -> selection-on-outcome).  So every polarity-dependent scalar is emitted under
    # the CHOSEN orientation AND the FLIPPED one (`*_flip`), and the plateau under a FIXED peak/trough,
    # so downstream can test whether an effect survives removing the polarity choice (it does not -- the
    # plateau "Tpi" separation is a polarity-selection artifact; see RESULTS.md).
    eip_pk, eip_tr = s["eip_peak"], s["eip_trough"]
    eip_flip = eip_tr if pol == "peak" else eip_pk                             # dwell at the OTHER extreme
    out = dict(ttot=ttot, rise=s["rise"], fall=s["fall"], ti=s["ti"], te=s["te"], eip=s["eip"],
               eip_peak=eip_pk, eip_trough=eip_tr, eip_flip=eip_flip,
               phase_asym=s["phase_asym"], resolvable=s["resolvable"], polarity=pol)
    keys = ("ti_frac", "te_frac", "plateau_frac", "skew", "ti_te_ratio",
            "ti_frac_flip", "te_frac_flip", "plateau_frac_flip", "ti_te_ratio_flip",
            "plateau_frac_peak", "plateau_frac_trough")
    if ttot.size == 0:
        for k in keys:
            out[k] = np.array([], float)
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ti_frac"] = np.where(ttot > 0, s["ti"] / ttot, np.nan)
        out["te_frac"] = np.where(ttot > 0, s["te"] / ttot, np.nan)
        out["plateau_frac"] = np.where(ttot > 0, s["eip"] / ttot, np.nan)      # chosen-orientation Tpi/Ttot
        out["ti_te_ratio"] = np.where(s["te"] > 0, s["ti"] / s["te"], np.nan)
        # flipped-orientation twins (inspiration<->expiration swapped)
        out["ti_frac_flip"] = out["te_frac"]; out["te_frac_flip"] = out["ti_frac"]
        out["ti_te_ratio_flip"] = np.where(s["ti"] > 0, s["te"] / s["ti"], np.nan)
        out["plateau_frac_flip"] = np.where(ttot > 0, eip_flip / ttot, np.nan)
        # fixed-orientation (polarity-choice-free) plateaus, for the artifact test
        out["plateau_frac_peak"] = np.where(ttot > 0, eip_pk / ttot, np.nan)
        out["plateau_frac_trough"] = np.where(ttot > 0, eip_tr / ttot, np.nan)
    out["skew"] = out["te_frac"] - out["ti_frac"]                              # +ve = expiration-dominant
    return out


# ==================================================================================================
# (2) cycle-triggered TEMPLATE breath (amplitude-normalised warped average) -- the "template breaths"
# ==================================================================================================
def build_template(trace, fs, med_bpm, m=TPL_M):
    """Amplitude-normalised, length-warped average breath over the trough->peak->trough cycles of one
    episode/window, oriented so end-inspiration is the POSITIVE extreme.  Returns dict:
        tpl (m,), peak_pos in [0,1], plateau_frac, ti_frac, te_frac, polarity, n_cycles
    or None if too few cycles.  This is the region-adaptive 'template breath' morphology carrier.

    USE: the SHAPE vector (WT-vs-mut figure) and the Ti:Te split.  NOTE (empirical): warping cycles of
    unequal length smears the peak, so the TEMPLATE plateau_frac is NOISIER than the per-cycle EIP
    median -- the plateau (Tpi) METRIC OF RECORD is per_cycle()['plateau_frac'], not this.  Keep the
    template build over a short, rate-homogeneous window (calm cycles) for a clean shape."""
    trace = np.asarray(trace, float)
    det = P.detect_cycles(trace, float(fs), float(med_bpm))
    if det is None:
        return None
    sm, pk, tr = det["sm"], det["peaks"], det["troughs"]
    pol, _, _ = P.episode_polarity(det)
    sign = 1.0 if pol == "peak" else -1.0                     # orient end-inspiration up
    acc = np.zeros(m); wsum = 0.0; nc = 0
    for k in range(len(tr) - 1):                              # cycle = trough[k] -> trough[k+1]
        a, b = tr[k], tr[k + 1]
        if b - a < 6:
            continue
        seg = sign * (sm[a:b] - sm[a:b].mean())
        w = eq.warp(seg, m)
        nrm = float(np.linalg.norm(w)) + 1e-12
        acc += (b - a) * (w / nrm); wsum += (b - a); nc += 1
    if wsum <= 0 or nc < 3:
        return None
    tpl = acc / wsum
    tpl = tpl - tpl.mean()
    # peak position + high-volume plateau dwell (within EIP_FRAC of the max) as a cycle fraction
    peak = int(np.argmax(tpl))
    amp = float(tpl.max() - tpl.min()) + 1e-12
    thr = tpl[peak] - P.EIP_FRAC * amp
    lo = peak
    while lo > 0 and tpl[lo - 1] >= thr:
        lo -= 1
    hi = peak
    while hi < m - 1 and tpl[hi + 1] >= thr:
        hi += 1
    return dict(tpl=tpl.astype(np.float32), peak_pos=peak / (m - 1),
                plateau_frac=float((hi - lo) / (m - 1)),
                ti_frac=float(peak / (m - 1)), te_frac=float(1.0 - peak / (m - 1)),
                polarity=pol, n_cycles=nc)


# ==================================================================================================
# episode-polarity vote (session-level label, mirrors stettner_metrics)
# ==================================================================================================
def session_polarity(votes):
    """Majority polarity across per-episode votes; returns (choice, mixed_frac)."""
    from collections import Counter
    if not votes:
        return "peak", 0.0
    c = Counter(votes); pol = c.most_common(1)[0][0]
    return pol, round(1.0 - c[pol] / len(votes), 3)


if __name__ == "__main__":
    # quick visual/quant check: WT vs mut template breath + plateau
    import argparse
    sys.path.insert(0, _HERE)
    import substrate as SUB
    ap = argparse.ArgumentParser()
    ap.add_argument("--recs", nargs="+", default=["Match5Mouse1", "Animal_B_9_8wk"])
    args = ap.parse_args()
    for rec in args.recs:
        m = SUB.load_meta(rec); npz = np.load(os.path.join(SUB.SUB_DIR, rec + ".npz"), allow_pickle=True)
        info = SUB.rec_info(rec, m)
        pfs = []
        for ep, e in enumerate(m.get("episodes", [])):
            ac = e.get("ac")
            if ac is None or ac < 0.45:
                continue
            tr = np.asarray(npz["trace"][ep], float)
            t = build_template(tr, 250.0, e.get("med_bpm") or 96.0)
            if t:
                pfs.append(t["plateau_frac"])
        print(f"{rec:20s} {info['genotype']:10s} n_win_tpl={len(pfs):3d} "
              f"plateau_frac(template) med={np.median(pfs) if pfs else float('nan'):.3f}")
