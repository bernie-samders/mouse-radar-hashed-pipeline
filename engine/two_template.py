"""two_template.py -- the two-template (breath-MOTION + variable PAUSE) EQ-Radio segmenter.

The novel extension over eqradio_hr.py: a second, FIXED-FLAT pause template and a 2-state dynamic
program so the segmentation alternates breath-motion (B) and exhale-pause (P) segments.

Model (Stage 0 confirmed): a cycle = a short ~fixed breath motion + a variable-length exhale pause;
arrhythmia = variation in the pause length; a long pause = an apnea (one continuum).  Decoupling the
two lets the motion template stay a clean repeating shape while all timing variation goes into the
pause length -- which a single warped template cannot do (it smears across both and loses the lock).

Key adaptations vs the cardiac eqradio_hr:
  - AMPLITUDE-INVARIANT breath cost  rB = ||x||^2 - <x,w>^2/||w||^2  (fit SHAPE, free amplitude).
    Crucial for B9: faint-but-correctly-shaped breaths get rB~0 (detected), not a big residual that
    would push them into the pause class.  The raw ||x - w||^2 penalises amplitude mismatch.
  - FLAT pause template mu_P == 0  ->  rP = ||x||^2  (computed for free from the prefix-sum).
  - Per-segment penalty lam_seg (parsimony) + optional pause bias_P (MDL), calibrated to a noise floor.

DP state D[t][k], k in {B=0, P=1}:  adjacency = a pause only between breaths (into P only from B,
tau>0; answer ends on B), and B may follow B (fast burst, no pause), P, or the start.

Run:  python output/eqradio_resp/two_template.py            # synthetic correctness self-test
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import eqradio_hr as eq  # noqa: E402
import common as C  # noqa: E402
import resp_segment as R  # noqa: E402


# ==================================================================================================
# cost tables
# ==================================================================================================
def _prefix_sq(x):
    return np.concatenate([[0.0], np.cumsum(np.asarray(x, float) ** 2)])


def breath_cost_table(x, muB, bminB, bmaxB, amp_invariant=True, rms_floor=0.0):
    """C_B[li, t] = breath-fit residual for a segment of length L=bminB+li ending at sample t.
    amp_invariant: ||x||^2 - <x,w>^2/||w||^2  (shape-only, amplitude free).
    else (paper): ||x - w||^2 = ||x||^2 - 2<x,w> + ||w||^2.
    A breath is INADMISSIBLE (cost=inf) where the segment RMS < rms_floor -- a breath must carry
    breath-scale amplitude, so low-amplitude noise regions fall to the pause class instead of being
    'explained' as faint breaths by the amplitude-invariant fit.  inf also where the segment runs
    off the front."""
    x = np.asarray(x, float)
    n = x.size
    xsq = _prefix_sq(x)
    numL = bmaxB - bminB + 1
    C_ = np.full((numL, n + 1), np.inf)
    floor_e = rms_floor * rms_floor
    for li, L in enumerate(range(bminB, bmaxB + 1)):
        if L > n:
            continue
        w = eq.warp(muB, L)
        wsq = float(w @ w) + 1e-12
        cross = np.correlate(x, w, mode="valid")     # <x[tau:tau+L], w>
        taus = np.arange(0, n - L + 1)
        ts = taus + L
        winsq = xsq[ts] - xsq[taus]
        if amp_invariant:
            cost = winsq - (cross * cross) / wsq
        else:
            cost = winsq - 2.0 * cross + wsq
        if floor_e > 0.0:
            cost = np.where(winsq / L >= floor_e, cost, np.inf)   # RMS admissibility gate
        C_[li, ts] = cost
    return C_


# ==================================================================================================
# the two-state dynamic program
# ==================================================================================================
def two_template_dp(x, muB, bminB, bmaxB, bminP, bmaxP, lam_seg, bias_P, amp_invariant=True,
                    rms_floor=0.0):
    """Exact 2-state DP with STRICT ALTERNATION breath -> pause -> breath -> pause ...

    Every inhale-exhale MOTION (B) is followed by a PAUSE (P), however short -- a short exhale-hold and
    a long apnea are the SAME structure (a P segment, different lengths), so the per-cycle pause length
    is the arrhythmia read-out.  Transitions: into B only from P (or the start); into P only from B (or
    the start).  There is NO B->B, so a short pause can never be mislabelled as a second breath.  The
    first segment may be B or P (arbitrary trace cut), and the trace may end on either.

    A long cessation (up to bmaxP) is one P between the flanking breaths; a region with NO admissible
    breath (amp floor) cannot satisfy the alternation, so it is left unsegmented -> 0 breaths (correctly
    'absent').  Returns segs = list of (a, b, kind), kind 0=breath, 1=pause."""
    x = np.asarray(x, float)
    n = x.size
    xsq = _prefix_sq(x)
    bmaxB = min(bmaxB, n)
    bmaxP = min(bmaxP, n)
    CB = breath_cost_table(x, muB, bminB, bmaxB, amp_invariant, rms_floor)
    LsB = np.arange(bminB, bmaxB + 1)
    LsP = np.arange(bminP, bmaxP + 1)

    INF = np.inf
    D = np.full((n + 1, 2), INF)                     # D[t,0]=end at t with a BREATH; D[t,1]=with a PAUSE
    backL = np.zeros((n + 1, 2), int)
    backS = np.full((n + 1, 2), -1, int)
    D[0, 0] = 0.0                                     # START: the first segment may be a breath ...
    D[0, 1] = 0.0                                     # ... or a pause (mid-pause trace start)

    for t in range(1, n + 1):
        # BREATH ending at t: predecessor must be a PAUSE (D[.,1]); D[0,1]=0 => breath-from-start
        taus = t - LsB
        ok = taus >= 0
        if ok.any():
            tt = taus[ok]
            cand = D[tt, 1] + CB[LsB[ok] - bminB, t] + lam_seg
            j = int(np.argmin(cand))
            if np.isfinite(cand[j]):
                D[t, 0] = cand[j]; backL[t, 0] = int(LsB[ok][j]); backS[t, 0] = 1
        # PAUSE ending at t: predecessor must be a BREATH (D[.,0]); D[0,0]=0 => pause-from-start.
        # No lam_seg on a pause: under strict alternation a pause is the structural companion of every
        # breath (one per cycle), so penalising pause count would double-count the breath penalty and
        # bias toward MERGING a faint real breath into its neighbouring pause (the missed-breath bug).
        # Breath COUNT parsimony is carried entirely by lam_seg on the breath.
        tausP = t - LsP
        okP = tausP >= 0
        if okP.any():
            tt = tausP[okP]
            cand = D[tt, 0] + (xsq[t] - xsq[tt]) + bias_P
            j = int(np.argmin(cand))
            if np.isfinite(cand[j]):
                D[t, 1] = cand[j]; backL[t, 1] = int(LsP[okP][j]); backS[t, 1] = 0

    # end on whichever state is finite/cheaper; if the tail is unsegmentable (absent), stop at the last
    # reachable end and drop the tail (it carries no breaths)
    end = n
    while end > 0 and not (np.isfinite(D[end, 0]) or np.isfinite(D[end, 1])):
        end -= 1
    if end == 0:
        return [], D
    s = 0 if D[end, 0] <= D[end, 1] else 1
    segs = []
    t = end
    while t > 0:
        L = backL[t, s]
        if L <= 0:
            break
        segs.append((t - L, t, s))
        ns = backS[t, s]
        t -= L
        s = ns if ns >= 0 else s
    segs.reverse()
    return segs, D


# ==================================================================================================
# template update (amplitude-normalised shape) + alternation driver
# ==================================================================================================
def update_shape_template(x, segs, m):
    """Length-weighted mean of the AMPLITUDE-NORMALISED breath segments, warped to length m; zero-mean
    and unit-norm.  Amplitude-normalising each segment keeps mu_B a pure SHAPE (loud bursts don't
    dominate), matching the amplitude-invariant cost."""
    x = np.asarray(x, float)
    acc = np.zeros(m); wsum = 0.0
    for (a, b, k) in segs:
        if k != 0 or b - a < 2:
            continue
        w = eq.warp(x[a:b], m)
        nrm = float(np.linalg.norm(w)) + 1e-12
        acc += (b - a) * (w / nrm)
        wsum += (b - a)
    if wsum <= 0:
        return None
    mu = acc / wsum
    mu = mu - mu.mean()
    return mu / (float(np.linalg.norm(mu)) + 1e-12)


def estimate_sigma2(x, fs, win_s=0.1):
    """Noise-floor variance: the low-decile of short-window mean energy (the quietest stretches)."""
    x = np.asarray(x, float)
    w = max(2, int(round(win_s * fs)))
    xsq = _prefix_sq(x)
    ts = np.arange(w, x.size + 1)
    e = (xsq[ts] - xsq[ts - w]) / w
    return float(np.percentile(e, 10)) if e.size else float(np.var(x))


def active_rms_ref(x, fs, win_s=0.2, pct=70):
    """Robust breathing-amplitude reference: a high percentile of short-window RMS (active breathing
    sets the upper RMS percentiles).  Used to place the breath-admissibility amplitude floor."""
    x = np.asarray(x, float)
    w = max(2, int(round(win_s * fs)))
    xsq = _prefix_sq(x)
    ts = np.arange(w, x.size + 1)
    rms = np.sqrt(np.maximum(0.0, (xsq[ts] - xsq[ts - w]) / w))
    return float(np.percentile(rms, pct)) if rms.size else float(np.std(x))


def segment_two(x, fs, bminB_s=R.BMIN_B_S, bmaxB_s=R.BMAX_B_S, bminP_s=R.BMIN_P_S, bmaxP_s=R.BMAX_P_S,
                m=None, lam_seg=None, c_seg=0.5, bias_P=0.0, sigma2=None, amp_frac=0.40, rms_floor=None,
                fit_update=0.5, max_iter=12, muB0=None, amp_invariant=True):
    """Alternating minimisation for the two-template model.  Returns a result dict.
    amp_frac: breath-admissibility floor as a fraction of the LOCAL BREATH amplitude (see below).
    fit_update: only breath segments with shape-fit >= this refine mu_B (robust update).

    Admissibility floor is RELATIVE TO THE BREATH AMPLITUDE, not the noise floor: the false breaths
    we must reject are the HEARTBEAT continuing through an apnea -- structured signal well above noise
    (so an SNR-vs-noise gate keeps them), but only ~0.2x the breath amplitude, whereas a faint-but-real
    breath is ~0.5x.  So we gate at amp_frac x (median RMS of confident breaths), refined each
    iteration (bootstrapped from the active-RMS reference)."""
    x = np.asarray(x, float); x = x - np.mean(x)
    n = x.size
    bminB = max(3, int(round(bminB_s * fs))); bmaxB = max(bminB + 1, int(round(bmaxB_s * fs)))
    bminP = max(2, int(round(bminP_s * fs))); bmaxP = max(bminP + 1, int(round(bmaxP_s * fs)))
    if m is None:
        m = max(6, int(round(0.5 * (bminB + bmaxB))))
    if sigma2 is None:
        sigma2 = estimate_sigma2(x, fs)
    if lam_seg is None:
        lam_seg = c_seg * sigma2 * m                 # parsimony ~ residual energy of a noise seg
    if rms_floor is None:
        rms_floor = amp_frac * active_rms_ref(x, fs)
    # seed mu_B: a normalised bootstrap breath shape (autocorr period), refined by the alternation
    if muB0 is None:
        muB0 = eq._seed_template(x, fs, bminB, bmaxB, m, None)
        muB0 = muB0 - muB0.mean()
        nrm = float(np.linalg.norm(muB0))
        muB = muB0 / nrm if nrm > 1e-9 else _default_shape(m)
    else:
        muB = muB0

    segs = None
    for it in range(max_iter):
        new_segs, _ = two_template_dp(x, muB, bminB, bmaxB, bminP, bmaxP, lam_seg, bias_P,
                                      amp_invariant, rms_floor)
        # ROBUST template update: only well-fitting breath segments refine mu_B, so noise/quiet
        # segments the DP happened to label 'breath' cannot contaminate the shape (which would
        # otherwise raise their fit and lock in a degenerate all-breath solution).
        pm = breath_metrics(x, new_segs, muB)["per"]
        good_pm = [p for p in pm if p["fit"] >= fit_update]
        if len(good_pm) < 3:                     # too few confident breaths -> keep all (bootstrap)
            good_pm = pm
        # relative-amplitude floor: refine from the median RMS of the confident breaths (a low-amplitude
        # cardiac/noise wiggle sits far below it -> pause; a faint-but-real breath stays above -> breath)
        if len(good_pm) >= 3:
            rms_floor = amp_frac * float(np.median([p["rms"] for p in good_pm]))
        new_mu = update_shape_template(x, [(p["a"], p["b"], 0) for p in good_pm], m)
        stop = (new_segs == segs) or (new_mu is None)
        segs = new_segs
        if new_mu is not None:
            muB = new_mu
        if stop:
            break

    metrics = breath_metrics(x, segs, muB)
    return dict(segments=segs, muB=muB, m=m, fs=fs, lam_seg=lam_seg, sigma2=sigma2, bias_P=bias_P,
                rms_floor=rms_floor, bminB=bminB, bmaxB=bmaxB, bminP=bminP, bmaxP=bmaxP,
                n_iter=it + 1, **metrics)


def _default_shape(m=13):
    """A generic skewed breath pulse (fast rise, slower fall) as a last-resort seed."""
    t = np.linspace(0, 1, m)
    y = np.sin(np.pi * t) * np.exp(-2.0 * t)
    y = y - y.mean()
    return y / (np.linalg.norm(y) + 1e-12)


def _fit_tol(x, a, b, muB, shift=2):
    """Phase-tolerant squared shape-correlation: max cos^2 over small boundary shifts.  A breath is
    only ~12-28 samples, so a 1-sample DP boundary error tanks the raw cos^2; the decision should key
    on 'is there a breath-shaped feature here', not sub-sample phase."""
    L = b - a
    w = eq.warp(muB, L)
    wsq = float(w @ w) + 1e-12
    best = 0.0
    for s in range(-shift, shift + 1):
        aa, bb = a + s, b + s
        if aa < 0 or bb > x.size:
            continue
        seg = x[aa:bb]
        winsq = float(seg @ seg) + 1e-12
        cross = float(seg @ w)
        best = max(best, (cross * cross) / (wsq * winsq))
    return best


def breath_metrics(x, segs, muB):
    """Per-breath-segment fit (squared shape correlation, in [0,1]) + amplitude, and summary coverage.
    `fit` is aligned cos^2 (used for the robust template update); `fit_tol` is phase-tolerant (used by
    the decision rule)."""
    x = np.asarray(x, float)
    per = []
    for (a, b, k) in segs:
        if k != 0 or b - a < 2:
            continue
        seg = x[a:b]; w = eq.warp(muB, b - a)
        wsq = float(w @ w) + 1e-12; cross = float(seg @ w); winsq = float(seg @ seg) + 1e-12
        fit = (cross * cross) / (wsq * winsq)
        alpha = cross / wsq
        per.append(dict(a=int(a), b=int(b), L=int(b - a), fit=float(fit),
                        fit_tol=float(_fit_tol(x, a, b, muB)),
                        amp=float(abs(alpha) * np.sqrt(wsq)), rms=float(np.sqrt(winsq / (b - a)))))
    n = int(segs[-1][1]) if segs else 0
    b_samp = sum(p["b"] - p["a"] for p in per)
    p_samp = sum(b - a for (a, b, k) in segs if k == 1)
    return dict(per=per, n_breaths=len(per),
                coverage_B=float(b_samp / max(1, n)), coverage_P=float(p_samp / max(1, n)),
                mean_fit=float(np.mean([p["fit"] for p in per])) if per else 0.0)


# ==================================================================================================
# SELF-TEST -- synthetic motion + variable pauses + long apnea + noise-only region
# ==================================================================================================
def _synth(fs=50.0, seed=0):
    rng = np.random.default_rng(seed)
    shape_m = 12
    tt = np.linspace(0, 1, shape_m)
    pulse = np.sin(np.pi * tt) * np.exp(-1.8 * tt)      # skewed breath motion (~0.24 s)
    pulse = pulse - pulse.mean(); pulse /= np.linalg.norm(pulse)
    Lm = int(round(0.24 * fs))
    x = []
    breaths = []            # sample index of each placed breath (segment start)
    # 1) normal-amplitude breathing with SHORT-to-MEDIUM variable pauses (the arrhythmia)
    for i in range(18):
        amp = 1.0 + 0.2 * rng.standard_normal()
        breaths.append(len(x))
        x += list(amp * eq.warp(pulse, Lm))
        pause = int(round(max(0.02, 0.20 + 0.18 * rng.standard_normal()) * fs))
        x += [0.0] * pause
    # 2) a LONG apnea, but with the HEARTBEAT continuing through it: low-amplitude (0.2x) breath-SHAPED
    #    wiggles well above the noise floor.  These must stay PAUSE (the relative-amplitude floor, not an
    #    SNR-vs-noise gate, is what rejects them) -- the top-tile false-breath failure mode.
    a0 = len(x)
    x += [0.0] * int(round(0.8 * fs))
    for i in range(5):
        x += list(0.2 * eq.warp(pulse, Lm))               # cardiac masquerade (breath shape, 0.2x amp)
        x += [0.0] * int(round(0.18 * fs))
    x += [0.0] * int(round(0.8 * fs))
    apnea_span = (a0, len(x))
    # 3) MODERATELY-FAINT but clearly-present breathing (0.5x, correct shape) -- must still be BREATHS
    #    (the bottom-tile missed-breath failure mode: reduced amplitude, same morphology).
    for i in range(10):
        breaths.append(len(x))
        x += list(0.5 * eq.warp(pulse, Lm))
        x += [0.0] * int(round(0.30 * fs))
    # 4) QUIET/ABSENT region (0.08 noise, below breath scale) -- must be pause, no fabricated breaths
    noise0 = len(x); x += list(0.08 * rng.standard_normal(int(round(6.0 * fs)))); noise_span = (noise0, len(x))
    x = np.asarray(x, float) + 0.04 * rng.standard_normal(len(x))
    return x, np.asarray(breaths), apnea_span, noise_span, pulse


def breathing_segments(res, fit_min=0.35):
    """The breath segments that pass the decision fit gate (phase-tolerant shape match)."""
    return [p for p in res["per"] if p["fit_tol"] >= fit_min]


def presence_mask(res, n, fit_min=0.35):
    """Per-sample breathing-present indicator: covered by a gated breath segment."""
    mask = np.zeros(n, bool)
    for p in breathing_segments(res, fit_min):
        mask[p["a"]:p["b"]] = True
    return mask


def _coverage_breath(segs, span, per=None, fit_min=None):
    """Fraction of `span` covered by breath segments.  If per+fit_min given, only fit-gated breaths."""
    a0, a1 = span
    if per is not None and fit_min is not None:
        gated = {(p["a"], p["b"]) for p in per if p["fit_tol"] >= fit_min}
        segs = [(a, b, 0) for (a, b, k) in segs if k == 0 and (a, b) in gated]
    tot = 0
    for (a, b, k) in segs:
        if k == 0:
            tot += max(0, min(b, a1) - max(a, a0))
    return tot / max(1, a1 - a0)


def run_selftest(fs=100.0, fit_min=0.35):
    x, breaths, apnea, noise, pulse = _synth(fs)
    res = segment_two(x, fs, amp_invariant=True)
    segs = res["segments"]; per = res["per"]
    # breath recall vs placed breaths (a placed breath is "found" if a B-segment start is within 0.15 s)
    bstarts = np.array([a for (a, b, k) in segs if k == 0], float)
    tol = 0.15 * fs
    found = sum(1 for g in breaths if bstarts.size and np.min(np.abs(bstarts - g)) <= tol)
    recall = found / breaths.size
    # DECISION-gated breath coverage (only fit_tol>=fit_min breaths count as breathing)
    apnea_bcov = _coverage_breath(segs, apnea, per, fit_min)
    noise_bcov = _coverage_breath(segs, noise, per, fit_min)
    breath_bcov = _coverage_breath(segs, (0, apnea[0]), per, fit_min)

    # contrast: single-template eqradio must FABRICATE breaths through the apnea
    single = eq.segment_signal(x, fs, bmin_s=R.BMIN_CYCLE_S, bmax_s=R.BMAX_CYCLE_S, init="auto")
    sb = np.asarray(single["boundaries"][1:-1], float)
    single_apnea_breaths = int(np.sum((sb > apnea[0]) & (sb < apnea[1])))

    # null: pure noise -> gated breath coverage must be low
    xn = 0.12 * np.random.default_rng(9).standard_normal(int(20 * fs))
    rn = segment_two(xn, fs, amp_invariant=True)
    null_bcov = _coverage_breath(rn["segments"], (0, xn.size), rn["per"], fit_min)

    print("=" * 74)
    print(f"TWO-TEMPLATE SELF-TEST @ {fs:.0f} Hz (motion + variable pause + apnea + noise)")
    print("=" * 74)
    print(f"breath recall (real breaths found)        : {recall:.3f}   (want > 0.90)")
    print(f"BREATHING region gated-coverage (high)    : {breath_bcov:.3f}  (want > 0.40)")
    print(f"apnea gated breath-coverage (~0)          : {apnea_bcov:.3f}  (want < 0.15)")
    print(f"QUIET region gated breath-coverage (low)  : {noise_bcov:.3f}  (want < 0.30)")
    print(f"pure-noise gated coverage (null)          : {null_bcov:.3f}  (want < 0.30)")
    print(f"single-template breaths INSIDE apnea      : {single_apnea_breaths}  (fabricated; two-templ=0)")
    print(f"mean breath fit (aligned / tolerant)      : {res['mean_fit']:.3f} / "
          f"{np.mean([p['fit_tol'] for p in per]):.3f}")
    ok = (recall > 0.90 and breath_bcov > 0.40 and apnea_bcov < 0.15 and noise_bcov < 0.30
          and null_bcov < 0.30 and single_apnea_breaths >= 2)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    _fig_selftest(x, fs, segs, apnea, noise, single)
    return ok


def _fig_selftest(x, fs, segs, apnea, noise, single):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    t = np.arange(x.size) / fs
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax[0].plot(t, x, lw=0.6, color="0.4")
    for (a, b, k) in segs:
        ax[0].axvspan(a / fs, b / fs, color=("C2" if k == 0 else "C3"), alpha=0.18)
    ax[0].axvspan(apnea[0] / fs, apnea[1] / fs, color="none", ec="k", lw=1.2, ls="--")
    ax[0].set_title("TWO-template: green=breath, red=pause (dashed box = true apnea)")
    ax[1].plot(t, x, lw=0.6, color="0.4")
    for bd in np.asarray(single["boundaries"][1:-1], int):
        ax[1].axvline(bd / fs, color="C0", lw=0.7, alpha=0.7)
    ax[1].axvspan(apnea[0] / fs, apnea[1] / fs, color="none", ec="k", lw=1.2, ls="--")
    ax[1].set_title("SINGLE-template: vertical = cycle bounds (note fabrication through the apnea)")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout(); os.makedirs(C.FIGS, exist_ok=True)
    fig.savefig(os.path.join(C.FIGS, "stage2_selftest.png"), dpi=120)
    plt.close(fig)
    print(f"figure -> {os.path.join(C.FIGS, 'stage2_selftest.png')}")


if __name__ == "__main__":
    run_selftest()
