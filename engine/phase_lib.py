"""ml_apnea/cohort/phase_lib.py -- per-breath PHASE decomposition of the radar displacement trace.

The reusable, physics-honest crux for the Stettner-et-al-2007 metrics (phase durations + arrhythmia).
It is to the phase/arrhythmia metrics what apnea_lib is to the apnea detectors: pure functions, no I/O.

WHAT THE PAPER MEASURED vs WHAT RADAR CAN SEE
---------------------------------------------
Stettner (J Physiol 2007) decomposed each breath, from phrenic + vagal NERVE recordings, into
inspiration (Ti), postinspiration (Tpi) and late expiration (Te2), and reported (i) the phase
DURATIONS (KO: prolonged Tpi, shortened Ti/Te2) and (ii) their VARIANCE + the breath-by-breath
Tpi<->Ttot COUPLING (KO r=0.97 vs WT 0.66).  Radar sees only chest-wall DISPLACEMENT, so:

  * Ttot and the timing statistics (variance, CV, coupling) port faithfully -- they are properties
    of the mechanical cycle.
  * Ti/Te can only be recovered as the two half-cycle limbs (trough->peak "rise", peak->trough
    "fall").  Radar magnitude has NO absolute inspiration/expiration polarity (clean_substrate even
    selects negative-going cells), so the physiological Ti/Te LABEL requires a per-episode polarity
    assumption -- made explicit here and sensitivity-tested downstream (polarity_flip_* fields).
  * Postinspiration (Tpi) has NO direct radar correlate (it is a vagal/laryngeal motor event).  We
    compute an END-INSPIRATORY-PAUSE (EIP) PROXY: the dwell at high lung volume just after the
    displacement peak.  It is labelled a PROXY everywhere; it conflates a true post-I pause with any
    breath-hold at high volume.

LANDMARKS
---------
clean_substrate stores `breath_idx` from adaptive_breaths, but those land on a filtered signal and do
NOT sit on the stored condition_trace extrema (verified: ~1/25 on a peak).  So we RE-DETECT peaks and
troughs directly on the (lightly smoothed, morphology-preserving) condition_trace, with a
period-tied minimum distance and prominence.  Validated: re-detected peak count / breath_idx count =
1.00-1.06 across all Match1/3/4/5 sessions -- i.e. one detected cycle per real breath.

A cycle is trough->peak->trough (one breath, containing one peak):
    ttot = tr_right - tr_left ;  rise = peak - tr_left ;  fall = tr_right - peak .
`rise`/`fall`/`phase_asym` are label-free.  `ti`/`te`/`eip` require the episode polarity.
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import core   # _regular_runs, _wmedian, PAUSE_MULT, MIN_RUN

# --- detection / resolvability constants (validated on the clean_sub substrate) -------------------
SMOOTH_FRAC   = 1.0 / 8.0    # smoothing window = period/8 (kills fast noise, keeps rise/fall shape)
DIST_FRAC     = 0.55         # min peak spacing = 0.55 x expected cycle (no over-segmentation)
PROM_FRAC     = 0.25         # peak prominence >= 0.25 x P90(|smoothed|)
EIP_FRAC      = 0.10         # EIP band = within 10% of cycle amplitude of the extreme
MIN_SAMP_CYC  = 12           # a cycle needs >= 12 samples (>= ~6/limb) to be phase-resolvable
MIN_AMP_MAD   = 3.0          # a cycle amplitude must exceed 3 x local MAD of the trace
WIN_CYCLES    = 60           # paper's "60 consecutive breathing cycles" window for variance/coupling


def _smooth(x, n):
    return uniform_filter1d(np.asarray(x, float), max(3, int(n)))


def run_slices(ttot, mult=core.PAUSE_MULT, med_win=core.RUN_MED_WIN, min_run=core.MIN_RUN):
    """Index ranges (a, b) of contiguous REGULAR-breathing runs in a per-cycle ttot series, breaking
    at every pause (ttot > mult x local running median) -- the multi-array analogue of
    core._regular_runs (which returns values, not indices, so cannot slice the aligned phase arrays)."""
    from scipy.ndimage import median_filter
    x = np.asarray(ttot, float)
    if x.size < 3:
        return []
    loc = median_filter(x, size=max(3, med_win | 1), mode="nearest")
    is_pause = x > mult * loc
    out, start = [], 0
    for i, p in enumerate(is_pause):
        if p:
            if i - start >= min_run:
                out.append((start, i))
            start = i + 1
    if x.size - start >= min_run:
        out.append((start, x.size))
    return out


CESS_MULT = 3.0        # ttot > this x local median = a real cessation (apnea / skipped breath)
CESS_ABS_S = 1.5       # ... or an absolute gap this long (a mouse at 100-300 bpm: >=5 skipped breaths)


def cessation_slices(ttot, mult=CESS_MULT, abs_gate_s=CESS_ABS_S, med_win=core.RUN_MED_WIN, min_run=20):
    """Index ranges (a, b) of consecutive breathing, breaking ONLY at true cessations (apneas), so the
    naturally-varying (arrhythmic) breaths are KEPT.  This is the substrate for the paper's VARIANCE
    and phase<->cycle COUPLING (Fig 2B/3), which are computed over ~60 consecutive breaths and whose
    whole KO signal IS the breath-to-breath variation.  (Contrast run_slices, which removes every
    >1.5x-median breath for beat-to-beat regularity -- that erases the coupling signal.)"""
    from scipy.ndimage import median_filter
    x = np.asarray(ttot, float)
    if x.size < 3:
        return []
    loc = median_filter(x, size=max(3, med_win | 1), mode="nearest")
    cess = (x > mult * loc) | (x > abs_gate_s)
    out, start = [], 0
    for i, c in enumerate(cess):
        if c:
            if i - start >= min_run:
                out.append((start, i))
            start = i + 1
    if x.size - start >= min_run:
        out.append((start, x.size))
    return out


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 4 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def detect_cycles(trace, fs, med_bpm):
    """Re-detect breath peaks + interior troughs on the condition_trace.

    Returns dict(sm, peaks, troughs, amp90, fs) or None if < 4 peaks (no interior cycles).
    `troughs[k]` is the trough between peaks[k] and peaks[k+1] (len = len(peaks)-1)."""
    x = np.asarray(trace, float)
    x = x - np.median(x)
    per = fs * 60.0 / max(float(med_bpm), 1.0)          # expected samples per cycle
    sm = _smooth(x, round(per * SMOOTH_FRAC))
    amp90 = float(np.percentile(np.abs(sm), 90))
    if amp90 <= 0:
        return None
    dist = max(2, int(round(per * DIST_FRAC)))
    pk, _ = find_peaks(sm, distance=dist, prominence=PROM_FRAC * amp90)
    if pk.size < 4:
        return None
    tr = np.array([a + int(np.argmin(sm[a:b + 1])) for a, b in zip(pk[:-1], pk[1:])], int)
    return dict(sm=sm, x=x, peaks=pk, troughs=tr, amp90=amp90, fs=float(fs))


def _dwell_forward(sm, i0, i_stop, thr, drop):
    """Contiguous forward dwell (samples) from i0 until sm crosses `thr` (drop=True: sm<thr)."""
    j = i0
    n = min(i_stop, len(sm) - 1)
    while j < n and not (sm[j] < thr if drop else sm[j] > thr):
        j += 1
    return j - i0


def _band_dwell(sm, idx, i_lo, i_hi, amp, top):
    """Contiguous dwell (samples) around idx within EIP_FRAC*amp of the extreme value at idx,
    bounded to [i_lo, i_hi].  Used symmetrically for the polarity decision."""
    val = sm[idx]
    thr = val - EIP_FRAC * amp if top else val + EIP_FRAC * amp
    lo = idx
    while lo > i_lo and (sm[lo - 1] >= thr if top else sm[lo - 1] <= thr):
        lo -= 1
    hi = idx
    while hi < i_hi and (sm[hi + 1] >= thr if top else sm[hi + 1] <= thr):
        hi += 1
    return hi - lo


def episode_polarity(det):
    """Decide once per episode which extreme is 'end-inspiration' (high lung volume).

    The post-inspiratory plateau sits at high volume, so end-inspiration is the extreme with the
    systematically LONGER adjacent within-band dwell.  Returns ('peak'|'trough', dwell_peak, dwell_tr)."""
    sm, pk, tr, amp = det["sm"], det["peaks"], det["troughs"], det["amp90"]
    dp, dt = [], []
    for k in range(1, len(pk) - 1):
        lo, hi = tr[k - 1], tr[k]
        dp.append(_band_dwell(sm, pk[k], lo, hi, amp, top=True))
        # trough between this peak and the next
        dt.append(_band_dwell(sm, tr[k], pk[k], pk[k + 1], amp, top=False))
    dpm = float(np.median(dp)) if dp else 0.0
    dtm = float(np.median(dt)) if dt else 0.0
    return ("peak" if dpm >= dtm else "trough"), dpm, dtm


def per_cycle_series(ep):
    """Per-breath phase table for one clean episode dict (from apnea_lib.iter_episodes).

    Returns dict of aligned float arrays (seconds), one row per interior peak, plus polarity meta:
      ttot, rise, fall, phase_asym, amp, resolvable(bool),
      eip_peak (dwell after peak into fall), eip_trough (dwell after trough into rise),
      ti, te, eip  -- under the chosen polarity (ti=rise & eip=eip_peak if polarity=='peak'),
      polarity ('peak'|'trough').
    Empty arrays if the trace is unusable."""
    fs = float(ep["fs"])
    det = detect_cycles(ep["trace"], fs, ep["med_bpm"])
    empty = {k: np.array([], float) for k in
             ("ttot", "rise", "fall", "phase_asym", "amp", "eip_peak", "eip_trough", "ti", "te", "eip")}
    empty["resolvable"] = np.array([], bool)
    empty["polarity"] = "peak"
    if det is None:
        return empty
    sm, pk, tr, amp90 = det["sm"], det["peaks"], det["troughs"], det["amp90"]
    pol, _, _ = episode_polarity(det)
    mad = float(np.median(np.abs(sm - np.median(sm)))) + 1e-12

    ttot, rise, fall, asym, amp, eip_pk, eip_tr, res = [], [], [], [], [], [], [], []
    for k in range(1, len(pk) - 1):
        trL, trR, p = tr[k - 1], tr[k], pk[k]        # trough-left, trough-right, peak
        r = (p - trL) / fs
        f = (trR - p) / fs
        tt = (trR - trL) / fs
        a = float(sm[p] - min(sm[trL], sm[trR]))
        n_samp = trR - trL
        ok = (n_samp >= MIN_SAMP_CYC) and (a >= MIN_AMP_MAD * mad) and (r > 0) and (f > 0)
        # EIP proxies (forward dwell out of the extreme, in samples -> s)
        e_pk = _dwell_forward(sm, p, trR, sm[p] - EIP_FRAC * a, drop=True) / fs      # after peak
        e_tr = _dwell_forward(sm, trR, pk[k + 1], sm[trR] + EIP_FRAC * a, drop=False) / fs  # after trough
        ttot.append(tt); rise.append(r); fall.append(f)
        asym.append(min(r, f) / max(r, f) if max(r, f) > 0 else np.nan)
        amp.append(a); eip_pk.append(e_pk); eip_tr.append(e_tr); res.append(ok)

    out = dict(ttot=np.array(ttot), rise=np.array(rise), fall=np.array(fall),
               phase_asym=np.array(asym), amp=np.array(amp),
               eip_peak=np.array(eip_pk), eip_trough=np.array(eip_tr),
               resolvable=np.array(res, bool), polarity=pol)
    # labelled (polarity-dependent): peak=end-inspiration -> inspiration is the rise, EIP after peak
    if pol == "peak":
        out["ti"], out["te"], out["eip"] = out["rise"], out["fall"], out["eip_peak"]
    else:
        out["ti"], out["te"], out["eip"] = out["fall"], out["rise"], out["eip_trough"]
    return out
