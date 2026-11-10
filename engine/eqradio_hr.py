"""ml_apnea/cohort/eqradio_hr.py -- EQ-Radio heartbeat SEGMENTATION for FMCW voxel-magnitude traces.

Faithful port of the EQ-Radio beat extractor (Zhao, Adib, Katabi, "Emotion Recognition using
Wireless Signals", MobiCom 2016, Section 5), adapted to MOUSE cardiac physiology (fs~250 Hz, HR
~300-780 bpm, IBI ~0.08-0.20 s).  Unlike hr_carrier.py -- which is a *frequency-domain* HR estimator
(carrier projection + harmonic product) -- this is a *time-domain* MORPHOLOGY extractor: it learns the
shape of one heartbeat and finds the beat boundaries by joint template + segmentation optimisation.
That yields beat-to-beat IBIs (hence real time-domain HRV) rather than a windowed HR scalar.

Why EQ-Radio beats a band-pass + peak-pick for heartbeats behind breathing:

  1. ACCELERATION, NOT BAND-PASS (Sec 5.1).  Chest breathing DISPLACEMENT dwarfs the heartbeat's
     (~orders of magnitude), but breathing ACCELERATION is tiny (slow, smooth) while a heartbeat is a
     fast localised muscle contraction with large acceleration.  Operating on the 2nd derivative
     therefore SUPPRESSES breathing and EMPHASISES beats -- WITHOUT a narrowband cardiac filter that
     would also distort the beat morphology we are trying to learn.  We use the Holoborodko
     noise-robust 2nd-derivative differentiator (paper Eqn 1), a symmetric 7-tap FIR, then only a MILD
     low-pass (~3x max HR) to tame the HF noise the 2nd derivative amplifies.
  2. TEMPLATE + SEGMENTATION AS ONE OBJECTIVE (Def 5.1, Eqns 2/4).  A segmentation S partitions x into
     contiguous beats; Var(S) = min_mu sum_i ||s_i - Gamma(mu,|s_i|)||^2 where Gamma linearly WARPS a
     single learned template mu (cubic spline) to each beat's length.  Minimising Var jointly over the
     cut points AND the template shape locks onto whatever morphology genuinely REPEATS -- the beat --
     and is robust to the residual breathing baseline the acceleration step leaves behind.
  3. ALTERNATING MINIMISATION (Algorithm 1).  Given mu, UpdateSegmentation is an exact O(n*(bmax-bmin))
     DYNAMIC PROGRAM (Eqn 6) -- global optimum of the cut points.  Given S, UpdateTemplate is the
     closed-form length-weighted mean of the warped beats (Eqn 8).  Iterate to convergence (~8 iters).
  4. CONFIDENCE = beat-shape consistency in [0,1]: 1 - (template residual)/(flat-baseline residual).
     High => the segmentation locked onto a real repeating heartbeat morphology, not noise.  This is
     how we separate a genuine cardiac lock from a spurious one.

Mouse vs the paper (human, 24 GHz): beat bounds default to [0.08, 0.22] s (750..273 bpm) instead of
human [0.5, 1.3] s; template length m and the bounds can be tightened by an optional hr_hint_bpm.  The
substrate is the beamformed-POWER voxel magnitude (sqrt of the .f32), same as hr_carrier.py -- no phase.

Paper's zero-init note: mu^0 = zeros makes the FIRST DP degenerate (with a zero template every
segmentation costs exactly ||x||^2, so all cut points tie and the template averages phase-random
windows to ~zero -> it never locks).  We keep zero-init as a selectable code path (init='zero') for
fidelity but DEFAULT to init='auto', which seeds mu^0 from a fixed-period (autocorr- or hint-derived)
bootstrap template.  See _seed_template / the self-test's zero_init report.
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline, interp1d

# self-contained (numpy/scipy only); keep the engine dir importable for sibling modules.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# --- mouse-tuned defaults --------------------------------------------------------------------------
ACCEL_LP_HZ = 45.0       # mild low-pass after the 2nd derivative (~3x max HR of 13 Hz); NOT a cardiac bandpass
BMIN_S, BMAX_S = 0.08, 0.22    # beat-length bounds (s): 750 bpm .. 273 bpm
IBI_CLIP_S = (0.055, 0.28)     # physiologic IBI window for the HRV summary
PNN_MS = 4.0             # successive-IBI-difference threshold (ms) for pNN (mouse-scaled; human uses 50 ms)


# ==================================================================================================
# (1) Acceleration preprocessing -- Holoborodko noise-robust 2nd derivative (paper Eqn 1)
# ==================================================================================================
def accel_holoborodko(x, fs):
    """2nd derivative via the noise-robust 7-tap symmetric differentiator (paper Eqn 1, Holoborodko):
        f''_0 = [ 4 f_0 + (f_1+f_-1) - 2(f_2+f_-2) - (f_3+f_-3) ] / (16 h^2),   h = 1/fs
    i.e. taps [-1,-2,+1,+4,+1,-2,-1]/(16 h^2) at offsets [-3..+3].  Then a MILD Butterworth low-pass
    (~3x max HR) tames the HF noise the 2nd derivative amplifies.  Breathing (slow -> tiny accel) is
    suppressed; heartbeats (fast contraction -> large accel) are emphasised.  Same length as x."""
    x = np.asarray(x, float)
    x = x - np.mean(x)
    h = 1.0 / fs
    # Holoborodko noise-robust 2nd-derivative taps, physically signed (+f'').  The paper's Eqn 1 as
    # printed is the negation; a global sign flip is irrelevant to the morphology-based segmentation
    # (template mu just inverts; IBIs/HR/HRV/confidence unchanged), but we return a true +f'' so the
    # acceleration reads correctly if ever plotted or interpreted directly.
    taps = np.array([1.0, 2.0, -1.0, -4.0, -1.0, 2.0, 1.0]) / (16.0 * h * h)  # symmetric -> conv == corr
    a = np.convolve(x, taps, mode="same")
    # mild low-pass (guard against being asked for >Nyquist)
    fc = min(ACCEL_LP_HZ, 0.45 * fs)
    b, aa = butter(4, fc / (fs / 2.0), "low")
    return filtfilt(b, aa, a)


# ==================================================================================================
# (2) Gamma -- linear time-warp of a template to a target length (cubic spline)
# ==================================================================================================
def warp(mu, L):
    """Gamma(mu, L): resample template mu (length m) onto L linearly-spaced positions spanning mu's
    support [0, m-1] -- cubic-spline linear warp.  Used both ways: template->beat-length (in the DP)
    and beat->template-length (in the template update)."""
    mu = np.asarray(mu, float)
    m = mu.size
    L = int(L)
    if L <= 0:
        return np.zeros(0)
    if m == 0:
        return np.zeros(L)
    if m == 1:
        return np.full(L, mu[0])
    if L == 1:
        return np.array([mu[0]])
    src = np.arange(m, dtype=float)
    tgt = np.linspace(0.0, m - 1.0, L)
    if m >= 4:
        return CubicSpline(src, mu)(tgt)
    # too few knots for a cubic -> quadratic/linear fallback
    kind = "quadratic" if m == 3 else "linear"
    return interp1d(src, mu, kind=kind)(tgt)


# ==================================================================================================
# (4a) UpdateSegmentation -- exact dynamic program (paper Eqn 6)
# ==================================================================================================
def _seg_cost_table(x, warped, bmin, bmax):
    """C[li, t] = ||x[t-L : t] - warp(mu,L)||^2 for L = bmin+li, expanded as
        sum x^2  - 2*<x, w>  + sum w^2
    and evaluated for ALL end-points t at once (prefix-sum of x^2 + valid cross-correlation).  inf
    where the segment would run off the front."""
    x = np.asarray(x, float)
    n = x.size
    numL = bmax - bmin + 1
    xsq = np.concatenate([[0.0], np.cumsum(x * x)])           # xsq[t] = sum x[:t]^2
    C = np.full((numL, n + 1), np.inf)
    for li, L in enumerate(range(bmin, bmax + 1)):
        if L > n:
            continue
        w = warped[L]
        wsq = float(np.dot(w, w))
        cross = np.correlate(x, w, mode="valid")              # cross[tau] = <x[tau:tau+L], w>, tau=0..n-L
        taus = np.arange(0, n - L + 1)
        ts = taus + L
        winsq = xsq[ts] - xsq[taus]
        C[li, ts] = winsq - 2.0 * cross + wsq
    return C


def update_segmentation(x, mu, bmin, bmax):
    """DP of paper Eqn 6.  D_t = min_{L in [bmin,bmax]} D_{t-L} + ||x[t-L:t] - Gamma(mu,L)||^2, D_0 = 0.
    Global optimum of the cut points for the given template mu.  Returns the boundary indices INCLUDING
    the endpoints 0 and n, i.e. segments are (b[i], b[i+1])."""
    x = np.asarray(x, float)
    n = x.size
    bmin = int(bmin); bmax = int(bmax)
    bmax = min(bmax, n)
    if bmin > bmax:
        return [0, n]
    warped = {L: warp(mu, L) for L in range(bmin, bmax + 1)}
    C = _seg_cost_table(x, warped, bmin, bmax)
    Ls = np.arange(bmin, bmax + 1)
    D = np.full(n + 1, np.inf); D[0] = 0.0
    back = np.full(n + 1, -1, int)
    for t in range(bmin, n + 1):
        taus = t - Ls
        ok = taus >= 0
        if not ok.any():
            continue
        lis = np.where(ok)[0]
        cand = D[taus[ok]] + C[lis, t]
        j = int(np.argmin(cand))
        if np.isfinite(cand[j]):
            D[t] = cand[j]
            back[t] = int(Ls[ok][j])
    # backtrack from the largest reachable end-point (n itself is reachable for any sane bounds, where
    # bmax/bmin >= 2).  If a pathological narrow [bmin,bmax] leaves n unreachable, stop at the last
    # reachable end and DROP the sub-bmin tail rather than emit a runt segment that would violate the
    # length bound (Eqn 3/4).
    end = n
    while end > 0 and not np.isfinite(D[end]):
        end -= 1
    bounds = [end]
    t = end
    while t > 0 and back[t] > 0:
        t -= back[t]
        bounds.append(t)
    bounds = sorted(set(int(b) for b in bounds))
    if bounds[0] != 0:
        bounds = [0] + bounds
    return bounds


# ==================================================================================================
# (4b) UpdateTemplate -- closed form (paper Eqn 8)
# ==================================================================================================
def update_template(x, boundaries, m):
    """Eqn 8: mu = ( sum_i |s_i| Gamma(s_i, m) ) / ( sum_i |s_i| ).  Warp EACH beat s_i TO the template
    length m (note the warp direction is opposite to the DP), weight by its own length, average."""
    x = np.asarray(x, float)
    m = int(m)
    acc = np.zeros(m); wsum = 0.0
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        seg = x[a:b]
        L = seg.size
        if L < 1:
            continue
        acc += L * warp(seg, m)
        wsum += L
    if wsum <= 0:
        return np.zeros(m)
    return acc / wsum


# ==================================================================================================
# helpers: cost, confidence, seeding
# ==================================================================================================
def _var_cost(x, boundaries, mu):
    """Var(S) for the current (S, mu): sum_i ||s_i - Gamma(mu, |s_i|)||^2."""
    x = np.asarray(x, float); tot = 0.0
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        seg = x[a:b]
        if seg.size < 1:
            continue
        tot += float(np.sum((seg - warp(mu, seg.size)) ** 2))
    return tot


def _confidence(x, boundaries, mu, m):
    """Repeating-morphology SNR in [0,1] = 1 - (residual vs the learned template)/(variance vs a flat
    scalar mean), measured on the beats warped to the common length m.  Reads as: "what fraction of the
    warped-beat energy is explained by a CONSISTENT NON-FLAT heartbeat shape".  ~1 => the beats collapse
    onto one repeating morphology (a real cardiac lock); ~0 => no consistent shape, the template is no
    better than a flat mean (noise).  Cardiac-free nulls score ~0; genuine locks score ~0.4-0.95.

    NB: the baseline is deliberately a flat SCALAR mean, NOT the per-sample mean shape -- the per-sample
    mean shape essentially equals the length-weighted template mu, which would force this ratio to ~0 and
    destroy the discrimination.  This is a shape-repeatability index, not an accuracy proxy: a correct
    beat lock on a noisier substrate (e.g. the acceleration path) legitimately reads lower than on a
    pre-band-passed trace even when the recovered IBIs are identical."""
    x = np.asarray(x, float)
    G = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        seg = x[a:b]
        if seg.size >= 1:
            G.append(warp(seg, m))
    if len(G) < 2:
        return 0.0
    G = np.array(G)
    num = float(np.sum((G - mu[None, :]) ** 2))        # residual vs the learned template
    den = float(np.sum((G - G.mean()) ** 2))           # variance vs a flat scalar baseline
    if den <= 1e-30:
        return 0.0
    return float(np.clip(1.0 - num / den, 0.0, 1.0))


def _seed_template(x, fs, bmin, bmax, m, hr_hint_bpm):
    """Bootstrap mu^0 for init='auto' (the paper's zero-init is degenerate -- see module docstring).
    Pick a period L0: from hr_hint if given, else the autocorrelation peak of x within [bmin,bmax].
    Lay down uniform beats of length L0 and take the length-weighted warped mean (Eqn 8) as the seed --
    a phase-blurred but non-zero morphology the alternation then sharpens."""
    x = np.asarray(x, float)
    n = x.size
    if hr_hint_bpm:
        L0 = int(round(60.0 / hr_hint_bpm * fs))
    else:
        xz = x - x.mean()
        ac = np.correlate(xz, xz, mode="full")[n - 1:]
        lo, hi = bmin, min(bmax, ac.size - 1)
        L0 = int(lo + np.argmax(ac[lo:hi + 1])) if hi > lo else (bmin + bmax) // 2
    L0 = int(np.clip(L0, bmin, bmax))
    ubounds = list(range(0, n + 1, L0))
    if ubounds[-1] != n:
        ubounds.append(n)
    return update_template(x, ubounds, m)


# ==================================================================================================
# (4) Alternating minimisation driver (paper Algorithm 1)
# ==================================================================================================
def segment_signal(x, fs, bmin_s=BMIN_S, bmax_s=BMAX_S, m=None, hr_hint_bpm=None, max_iter=16,
                   init="auto"):
    """Alternate UpdateSegmentation / UpdateTemplate to convergence (Algorithm 1).  hr_hint_bpm tightens
    the bounds to [0.65,1.5]*T_hint and sets m = T_hint samples.  init='auto' seeds a bootstrap template
    (default, robust); init='zero' is the paper's mu^0=zeros (kept for fidelity, usually degenerate).
    Returns dict(boundaries, ibi_s, mu, var_final, confidence, n_iter, bmin, bmax, m)."""
    x = np.asarray(x, float); x = x - np.mean(x)
    n = x.size
    if hr_hint_bpm:
        T = 60.0 / hr_hint_bpm
        bmin = int(round(0.65 * T * fs)); bmax = int(round(1.5 * T * fs))
        if m is None:
            m = int(round(fs * 60.0 / hr_hint_bpm))
    else:
        bmin = int(round(bmin_s * fs)); bmax = int(round(bmax_s * fs))
        if m is None:
            m = int(round(0.5 * (bmin + bmax)))
    bmin = max(3, bmin); bmax = max(bmin + 1, min(bmax, n))
    m = max(4, int(m))

    mu = np.zeros(m) if init == "zero" else _seed_template(x, fs, bmin, bmax, m, hr_hint_bpm)

    prev_bounds, prev_cost, n_iter = None, np.inf, 0
    bounds = update_segmentation(x, mu, bmin, bmax)
    for it in range(max_iter):
        n_iter = it + 1
        mu = update_template(x, bounds, m)
        new_bounds = update_segmentation(x, mu, bmin, bmax)
        cost = _var_cost(x, new_bounds, mu)
        stop = (new_bounds == bounds) or (cost > prev_cost - 1e-9 * max(prev_cost, 1.0))
        bounds = new_bounds
        prev_cost = cost
        if stop:
            break

    mu = update_template(x, bounds, m)
    ibi_s = np.diff(bounds) / float(fs)
    return dict(boundaries=bounds, ibi_s=ibi_s, mu=mu, var_final=_var_cost(x, bounds, mu),
                confidence=_confidence(x, bounds, mu, m), n_iter=n_iter, bmin=bmin, bmax=bmax, m=m)


# ==================================================================================================
# (5) Public HR/HRV wrapper
# ==================================================================================================
def _hrv_summary(ibi_s):
    """Time-domain HRV from beat IBIs (clipped to the physiologic window).  Returns
    (hr_bpm, dict(sdnn_ms, rmssd_ms, cv, pnn), n_beats)."""
    ibi = np.asarray(ibi_s, float)
    ci = ibi[(ibi >= IBI_CLIP_S[0]) & (ibi <= IBI_CLIP_S[1])]
    if ci.size < 2:
        return float("nan"), dict(sdnn_ms=float("nan"), rmssd_ms=float("nan"),
                                  cv=float("nan"), pnn=float("nan")), int(ci.size)
    mean_ibi = float(np.mean(ci))
    d = np.diff(ci)
    hrv = dict(
        sdnn_ms=float(np.std(ci) * 1000.0),
        rmssd_ms=float(np.sqrt(np.mean(d * d)) * 1000.0),
        cv=float(np.std(ci) / mean_ibi),
        pnn=float(100.0 * np.mean(np.abs(d) * 1000.0 > PNN_MS)),
    )
    return 60.0 / mean_ibi, hrv, int(ci.size)


def eqradio_hr(trace, fs, hr_hint_bpm=None, use_accel=True):
    """End-to-end EQ-Radio heartbeat extraction on a 1D voxel-magnitude trace.
      use_accel=True  -> accel_holoborodko first (breathing suppression), then segment.  DEFAULT.
      use_accel=False -> segment the trace as-is (e.g. an already band-passed 'wideband' cardiac trace).
    Returns dict(hr_bpm, ibi_s, boundaries, confidence, hrv={sdnn_ms,rmssd_ms,cv,pnn}, template, n_beats)."""
    trace = np.asarray(trace, float)
    sig = accel_holoborodko(trace, fs) if use_accel else (trace - np.mean(trace))
    res = segment_signal(sig, fs, hr_hint_bpm=hr_hint_bpm)
    hr_bpm, hrv, n_beats = _hrv_summary(res["ibi_s"])
    return dict(hr_bpm=hr_bpm, ibi_s=res["ibi_s"], boundaries=res["boundaries"],
                confidence=res["confidence"], hrv=hrv, template=res["mu"], n_beats=n_beats,
                n_iter=res["n_iter"], var_final=res["var_final"])


# ==================================================================================================
# SELF-TEST -- synthetic mouse cardiac trace with heavy breathing interference
# ==================================================================================================
def _dog(L):
    """Sharp biphasic 'heartbeat' morphology (derivative-of-Gaussian), one-IBI wide, unit peak."""
    t = np.linspace(-1.0, 1.0, L)
    y = -t * np.exp(-((t / 0.35) ** 2))
    return y / (np.max(np.abs(y)) + 1e-12)


def _local_ibi_at(gt, p):
    """Ground-truth IBI (samples) of the beat interval containing sample position p."""
    i = np.searchsorted(gt, p, side="right") - 1
    i = int(np.clip(i, 0, len(gt) - 2))
    return gt[i + 1] - gt[i]


def _median_ibi_err_ms(gt_beats, boundaries, fs):
    """Phase-offset-free IBI error: for each recovered beat interval, compare its length to the GT
    interval covering its centre.  Median absolute error in ms."""
    gt = np.asarray(sorted(gt_beats), float)
    bnd = np.asarray(boundaries, float)
    errs = []
    for a, b in zip(bnd[:-1], bnd[1:]):
        rec = b - a
        cm = 0.5 * (a + b)
        if cm < gt[0] or cm > gt[-1]:
            continue
        errs.append(abs(rec - _local_ibi_at(gt, cm)))
    return float(np.median(errs)) / fs * 1000.0 if errs else float("nan")


def _localization(gt_beats, boundaries, fs):
    """PHASE-LOCK quality of the recovered cut points -- the check the interval-length metric can't do
    (a uniform grid nails interval lengths yet never LOCATES a beat).  For each interior recovered
    boundary, signed distance (samples) to the nearest GT beat.  A good lock puts every cut at a roughly
    CONSTANT phase of the beat cycle -> small offset SPREAD; a non-lock (or a free-running grid) drifts
    across all phases -> spread ~0.29*IBI.  Returns (offset_ms, jitter_ms) = (median signed offset,
    robust spread 1.4826*MAD after removing that offset), in ms.  Jitter -- not offset -- is what makes
    beat-to-beat HRV trustworthy; a constant offset is just where in the cycle the cut falls."""
    gt = np.asarray(sorted(gt_beats), float)
    bnd = np.asarray(list(boundaries)[1:-1], float)          # interior cut points only
    if bnd.size < 3 or gt.size < 2:
        return float("nan"), float("nan")
    d = []
    for p in bnd:
        j = int(np.clip(np.searchsorted(gt, p), 1, gt.size - 1))
        near = gt[j - 1] if abs(p - gt[j - 1]) <= abs(p - gt[j]) else gt[j]
        d.append(p - near)
    d = np.asarray(d, float)
    off = float(np.median(d))
    jit = float(1.4826 * np.median(np.abs(d - off)))
    return off / fs * 1000.0, jit / fs * 1000.0


def _make_synth(fs, dur, breath_cardiac_ratio=30.0, with_cardiac=True, seed=0):
    """Synthetic mouse trace: jittered ~461 bpm cardiac (biphasic beats warped to the local IBI) buried
    under a `breath_cardiac_ratio`x 2 Hz breathing tone + slow drift + noise.  with_cardiac=False builds
    the CARDIAC-FREE NULL (breathing+drift+noise only) used to prove confidence/localization discriminate
    signal from interference."""
    n = int(fs * dur); rng = np.random.default_rng(seed); A = 1.0
    beat_t = [0.6]
    while beat_t[-1] < dur - 0.7:
        ibi = 0.13 + rng.normal(0.0, 0.008)
        beat_t.append(beat_t[-1] + float(np.clip(ibi, 0.095, 0.185)))
    gt = np.round(np.array(beat_t) * fs).astype(int)
    gt = gt[(gt >= 0) & (gt < n)]
    card = np.zeros(n)
    if with_cardiac:
        for a, b in zip(gt[:-1], gt[1:]):
            card[a:b] += A * _dog(b - a)
    t = np.arange(n) / fs
    breathing = breath_cardiac_ratio * A * np.sin(2 * np.pi * 2.0 * t)
    drift = 10.0 * A * np.sin(2 * np.pi * 0.05 * t + 0.7)
    noise = 0.1 * A * rng.standard_normal(n)
    return card + breathing + drift + noise, gt


def _run_selftest():
    fs = 250; dur = 40.0
    trace, gt = _make_synth(fs, dur, with_cardiac=True)
    hr_true = 60.0 / (np.mean(np.diff(gt)) / fs)
    med_ibi_ms = 1000.0 * np.median(np.diff(gt)) / fs
    jit_lock_max = 0.15 * med_ibi_ms          # a real lock: cut-phase spread < 15% of an IBI
    print(f"[synthetic] fs={fs} dur={dur}s  beats={len(gt)}  HR_true={hr_true:.1f} bpm  IBI~{med_ibi_ms:.0f} ms"
          f"  breathing/cardiac amp = 30x  (lock jitter threshold = {jit_lock_max:.0f} ms)")

    def _report(tag, r):
        err = _median_ibi_err_ms(gt, r["boundaries"], fs)
        dhr = abs(r["hr_bpm"] - hr_true) if np.isfinite(r["hr_bpm"]) else float("nan")
        off, jit = _localization(gt, r["boundaries"], fs)
        print(f"[{tag:9s}] n={r['n_beats']:3d}  HR={r['hr_bpm']:.1f} |dHR|={dhr:.1f}  "
              f"IBIerr={err:.1f}ms  loc_offset={off:+.0f}ms loc_jitter={jit:.1f}ms  "
              f"conf={r['confidence']:.3f}  it={r['n_iter']}")
        return err, dhr, jit

    # --- primary: ACCELERATION path (the paper's approach on the raw magnitude trace) ---------------
    r = eqradio_hr(trace, fs, hr_hint_bpm=460, use_accel=True)
    err_a, dhr_a, jit_a = _report("accel", r)

    # --- secondary: SEGMENTER ALONE on a pre-band-passed 'wideband' cardiac trace -------------------
    bb, aa = butter(4, [4.5 / (fs / 2), 42.0 / (fs / 2)], "bandpass")
    wb = filtfilt(bb, aa, trace - trace.mean())
    r2 = eqradio_hr(wb, fs, hr_hint_bpm=460, use_accel=False)
    err_w, dhr_w, jit_w = _report("no-accel", r2)

    # --- CARDIAC-FREE NULL: confidence + localization must COLLAPSE (proves they discriminate) -------
    null_trace, _ = _make_synth(fs, dur, with_cardiac=False, seed=1)
    rn = eqradio_hr(null_trace, fs, hr_hint_bpm=460, use_accel=True)
    _, jit_n = _localization(gt, rn["boundaries"], fs)     # vs the (unrelated) beat grid -> should be ~random
    print(f"[null     ] conf={rn['confidence']:.3f}  loc_jitter={jit_n:.1f}ms   "
          f"(cardiac-free: conf must be ~0)")

    # --- UNIFORM-GRID BASELINE: nails IBI length, must FAIL localization (jitter ~0.29*IBI) ----------
    n = trace.size; L0 = int(round(fs * 60.0 / hr_true))
    grid = list(range(0, n + 1, L0));  grid = grid if grid[-1] == n else grid + [n]
    _, jit_grid = _localization(gt, grid, fs)
    err_grid = _median_ibi_err_ms(gt, grid, fs)
    print(f"[grid-base] IBIerr={err_grid:.1f}ms (trivially low)  loc_jitter={jit_grid:.1f}ms "
          f"(free-running -> large: the localization check is what separates real locks)")

    # --- zero-init convergence (paper mu^0 = zeros) -------------------------------------------------
    rz = segment_signal(accel_holoborodko(trace, fs), fs, hr_hint_bpm=460, init="zero")
    errz = _median_ibi_err_ms(gt, rz["boundaries"], fs)
    zero_ok = np.isfinite(errz) and errz < 8.0
    print(f"[zero-init] IBIerr={errz:.1f}ms conf={rz['confidence']:.3f} -> "
          f"{'converged' if zero_ok else 'DEGENERATE (needs seed; default init=auto)'}")

    # --- PASS GATE: the SEGMENTER must genuinely locate beats (no-accel path is the clean-substrate
    #     reference), the null must collapse, and the grid must fail localization while the segmenter
    #     passes it.  Whether the ACCEL path also locks phase for mice is REPORTED (it is the scientific
    #     question), not required -- mouse breathing (2-4 Hz) sits close to HR (7-13 Hz) so the (f_hr/
    #     f_resp)^2 acceleration boost may leave residual breathing the human 24 GHz case did not have.
    seg_locates = (err_w < 8.0) and (dhr_w < 15.0) and (jit_w < jit_lock_max)
    null_collapses = rn["confidence"] < 0.15
    grid_fails_loc = jit_grid > 3.0 * jit_w
    passed = seg_locates and null_collapses and grid_fails_loc
    print(f"\nSEGMENTER locates beats (no-accel): {seg_locates}  | null collapses: {null_collapses}  "
          f"| grid fails localization: {grid_fails_loc}")
    print(f"ACCEL path phase-locks for mice: {jit_a < jit_lock_max}  "
          f"(loc_jitter {jit_a:.1f}ms vs {jit_lock_max:.0f}ms threshold) -- REPORTED, informs data run")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}  (segmenter localizes + null collapses + grid fails)")
    return dict(passed=bool(passed), hr_true=hr_true, hr_accel=r["hr_bpm"], hr_noaccel=r2["hr_bpm"],
                ibierr_accel=err_a, ibierr_noaccel=err_w, jit_accel=jit_a, jit_noaccel=jit_w,
                jit_null=jit_n, jit_grid=jit_grid, conf_accel=r["confidence"],
                conf_noaccel=r2["confidence"], conf_null=rn["confidence"],
                accel_phase_locks=bool(jit_a < jit_lock_max), zero_ok=bool(zero_ok))


if __name__ == "__main__":
    _run_selftest()
