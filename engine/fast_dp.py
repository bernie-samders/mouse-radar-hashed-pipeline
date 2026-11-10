"""fast_dp.py -- a numba-JIT, BIT-IDENTICAL drop-in for the two-template DP hot loop.

Milestone 1 of the wide-voxel plan.  The only expensive part of `two_template.segment_two` is the
sequential recurrence in `two_template.two_template_dp` (a pure-Python `for t in range(n)` loop at
~85 us/sample).  Everything around it -- the cost tables (`breath_cost_table`, via a vectorised
np.correlate + the prefix-sum `xsq`), the template seed/update, `breath_metrics` -- is already
vectorised numpy and cheap.  So we re-implement ONLY the loop in `@njit` and reuse the reference
module for everything else.

Bit-identical by construction:
  - the cost tables (`CB`, `xsq`) are the SAME arrays the reference builds (we call the reference's
    `breath_cost_table` / `_prefix_sq` unchanged), so the per-candidate costs are the same floats;
  - the loop visits candidates in ascending-L order and keeps the first strict minimum (`c < best`),
    which is exactly `np.argmin`'s first-occurrence tie-break on the reference's ascending-L `cand`
    vector, and the scalar addition order `(D + cost) + penalty` matches the reference's vector add;
  - the backtrace is copied verbatim.
The self-test (`python output/wide_resp/fast_dp.py`) asserts identical segments AND identical D vs the
reference on real-shaped signals, and reports the speedup.

Reuse (do NOT edit): `output/eqradio_resp/two_template.py` (breath_cost_table, _prefix_sq, segment_two,
breath_metrics) and `ml_apnea/cohort/eqradio_hr.py` (warp, seed).  `segment_two_fast` runs the exact
reference `segment_two` with `two_template_dp` transiently redirected to the fast core (try/finally),
so the alternating-minimisation wrapper cannot silently diverge from the reference.
"""
from __future__ import annotations
import os, sys
import numpy as np
from numba import njit

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import two_template as TT   # the reference (breath_cost_table, _prefix_sq, segment_two, breath_metrics)
import eqradio_hr as eq      # the reference warp (CubicSpline template time-warp)


# ==================================================================================================
# memoised warp -- BIT-IDENTICAL to eq.warp, but cached by (template bytes, L).
# Profiling shows ~88% of segment_two's time is scipy CubicSpline inside eq.warp, called thousands of
# times with REPEATING (muB, L) pairs: breath_cost_table warps muB to each length L; breath_metrics
# and _fit_tol re-warp muB to every segment's length (many segments share a length).  Caching the
# CubicSpline OUTPUT (not approximating it) collapses those to one spline build per unique (muB, L),
# changing no number.  Cache is cleared per segment_two_fast call to bound memory.
# ==================================================================================================
from scipy.interpolate import CubicSpline

_warp_orig = eq.warp
_WARP_CACHE: dict = {}


def _warp_memo(mu, L):
    """BIT-identical eq.warp, cached by (template bytes, L)."""
    mu = np.asarray(mu, float)
    key = (mu.tobytes(), int(L))
    v = _WARP_CACHE.get(key)
    if v is None:
        v = _warp_orig(mu, L)
        _WARP_CACHE[key] = v
    return v                      # callers never mutate the warp result in place (verified)


# ---- matrix warp: warp(mu,L) is LINEAR in mu, so it equals A_{m,L} @ mu ---------------------------
# For m>=4 eq.warp is CubicSpline(arange(m), mu)(linspace(0,m-1,L)) -- a linear map of mu.  Build the
# L x m matrix once per (m,L) by warping the m basis vectors (same spline, so A@mu == warp exactly in
# exact arithmetic; in fp they differ by ~1e-14, far below any DP decision margin -- the self-test
# confirms segments stay identical).  There are only a couple dozen distinct (m,L) in the whole run,
# so this amortises CubicSpline away entirely.
_WARP_MAT: dict = {}


def _warp_matrix_for(m, L):
    A = _WARP_MAT.get((m, L))
    if A is None:
        src = np.arange(m, dtype=float)
        tgt = np.linspace(0.0, m - 1.0, L)
        A = np.empty((L, m))
        e = np.zeros(m)
        for i in range(m):
            e[i] = 1.0
            A[:, i] = CubicSpline(src, e)(tgt)
            e[i] = 0.0
        _WARP_MAT[(m, L)] = A
    return A


def _warp_matrix(mu, L):
    mu = np.asarray(mu, float)
    m = mu.size
    L = int(L)
    if m < 4 or L <= 1:                      # degenerate/small -> exact reference (rare; keeps fidelity)
        return _warp_orig(mu, L)
    return _warp_matrix_for(m, L) @ mu


# ==================================================================================================
# the JIT'd 2-state recurrence -- exact port of two_template.two_template_dp's inner loop
# ==================================================================================================
@njit(cache=True, fastmath=False)
def _dp_core(xsq, CB, bminB, bmaxB, bminP, bmaxP, lam_seg, bias_P):
    """Forward 2-state DP.  Returns (D, backL, backS).  bmaxB/bmaxP must already be clamped to n
    (and CB built with the same clamped bmaxB).  D[t,0]=end-on-BREATH cost, D[t,1]=end-on-PAUSE.
    Strict-`<` first-minimum matches np.argmin over the reference's ascending-L candidate vector."""
    n = xsq.shape[0] - 1
    INF = np.inf
    D = np.full((n + 1, 2), INF)
    backL = np.zeros((n + 1, 2), np.int64)
    backS = np.full((n + 1, 2), -1, np.int64)
    D[0, 0] = 0.0                                   # START may be a breath ...
    D[0, 1] = 0.0                                   # ... or a mid-pause trace start
    for t in range(1, n + 1):
        # BREATH ending at t: predecessor is a PAUSE (D[.,1]); breath length L in [bminB, bmaxB], L<=t.
        best = INF
        bestL = -1
        Lmax = bmaxB if bmaxB < t else t
        for L in range(bminB, Lmax + 1):
            tau = t - L
            c = D[tau, 1] + CB[L - bminB, t] + lam_seg
            if c < best:
                best = c
                bestL = L
        if bestL > 0:
            D[t, 0] = best
            backL[t, 0] = bestL
            backS[t, 0] = 1
        # PAUSE ending at t: predecessor is a BREATH (D[.,0]); flat pause cost = xsq[t]-xsq[tau]+bias_P.
        # No lam_seg on a pause (structural companion of each breath -- see reference docstring).
        best = INF
        bestL = -1
        Lmax = bmaxP if bmaxP < t else t
        for L in range(bminP, Lmax + 1):
            tau = t - L
            c = D[tau, 0] + (xsq[t] - xsq[tau]) + bias_P
            if c < best:
                best = c
                bestL = L
        if bestL > 0:
            D[t, 1] = best
            backL[t, 1] = bestL
            backS[t, 1] = 0
    return D, backL, backS


def two_template_dp_fast(x, muB, bminB, bmaxB, bminP, bmaxP, lam_seg, bias_P, amp_invariant=True,
                         rms_floor=0.0):
    """Drop-in replacement for two_template.two_template_dp: identical signature and return
    (segs, D).  Builds the SAME CB/xsq as the reference, runs the JIT core, then the reference's
    exact backtrace."""
    x = np.asarray(x, float)
    n = x.size
    xsq = TT._prefix_sq(x)
    bmaxB = min(bmaxB, n)
    bmaxP = min(bmaxP, n)
    CB = np.ascontiguousarray(TT.breath_cost_table(x, muB, bminB, bmaxB, amp_invariant, rms_floor))
    D, backL, backS = _dp_core(np.ascontiguousarray(xsq, dtype=np.float64), CB,
                               int(bminB), int(bmaxB), int(bminP), int(bmaxP),
                               float(lam_seg), float(bias_P))
    # backtrace -- verbatim from the reference
    end = n
    while end > 0 and not (np.isfinite(D[end, 0]) or np.isfinite(D[end, 1])):
        end -= 1
    if end == 0:
        return [], D
    s = 0 if D[end, 0] <= D[end, 1] else 1
    segs = []
    t = end
    while t > 0:
        L = int(backL[t, s])
        if L <= 0:
            break
        segs.append((t - L, t, s))
        ns = int(backS[t, s])
        t -= L
        s = ns if ns >= 0 else s
    segs.reverse()
    return segs, D


def segment_two_fast(x, fs, exact_warp=False, **kw):
    """Exact `two_template.segment_two`, with the DP transiently redirected to the JIT core AND
    eq.warp to a fast equivalent, so the alternating-minimisation wrapper is byte-for-byte the
    reference (no divergence risk).
      exact_warp=False (default): matrix warp -- numerically identical (~1e-14), fastest.
      exact_warp=True           : memoised CubicSpline -- BIT-identical, for regression checks."""
    orig_dp = TT.two_template_dp
    orig_warp = eq.warp
    TT.two_template_dp = two_template_dp_fast
    eq.warp = _warp_memo if exact_warp else _warp_matrix
    _WARP_CACHE.clear()
    try:
        return TT.segment_two(x, fs, **kw)
    finally:
        TT.two_template_dp = orig_dp
        eq.warp = orig_warp
        _WARP_CACHE.clear()


# ==================================================================================================
# correctness + speed self-test
# ==================================================================================================
def _dp_params_like_segment_two(x, fs, muB=None):
    """Reproduce segment_two's default bounds/penalties so we can compare the RAW DP directly."""
    import resp_segment as R
    x = np.asarray(x, float); x = x - x.mean()
    bminB = max(3, int(round(R.BMIN_B_S * fs))); bmaxB = max(bminB + 1, int(round(R.BMAX_B_S * fs)))
    bminP = max(2, int(round(R.BMIN_P_S * fs))); bmaxP = max(bminP + 1, int(round(R.BMAX_P_S * fs)))
    m = max(6, int(round(0.5 * (bminB + bmaxB))))
    sigma2 = TT.estimate_sigma2(x, fs)
    lam_seg = 0.5 * sigma2 * m
    rms_floor = 0.40 * TT.active_rms_ref(x, fs)
    if muB is None:
        import eqradio_hr as eq
        muB0 = eq._seed_template(x, fs, bminB, bmaxB, m, None); muB0 = muB0 - muB0.mean()
        nrm = float(np.linalg.norm(muB0)); muB = muB0 / nrm if nrm > 1e-9 else TT._default_shape(m)
    return dict(x=x, muB=muB, bminB=bminB, bmaxB=bmaxB, bminP=bminP, bmaxP=bmaxP,
                lam_seg=lam_seg, bias_P=0.0, rms_floor=rms_floor)


def run_selftest():
    import time
    fs = 125.0
    rng = np.random.default_rng(0)
    ok_all = True

    # ---- (A) raw-DP bit-identity on several signals (synthetic + noise + real-shaped) ----
    print("=" * 74)
    print("FAST-DP CORRECTNESS  (raw two_template_dp_fast vs reference two_template_dp)")
    print("=" * 74)
    sigs = []
    xs, *_ = TT._synth(fs=fs, seed=1); sigs.append(("synth@125", xs))
    sigs.append(("noise", 0.1 * rng.standard_normal(int(12 * fs))))
    sigs.append(("faint+apnea", TT._synth(fs=fs, seed=7)[0]))
    for name, x in sigs:
        p = _dp_params_like_segment_two(x, fs)
        ref_segs, ref_D = TT.two_template_dp(p["x"], p["muB"], p["bminB"], p["bmaxB"], p["bminP"],
                                             p["bmaxP"], p["lam_seg"], p["bias_P"], True, p["rms_floor"])
        fast_segs, fast_D = two_template_dp_fast(p["x"], p["muB"], p["bminB"], p["bmaxB"], p["bminP"],
                                                 p["bmaxP"], p["lam_seg"], p["bias_P"], True, p["rms_floor"])
        segs_match = (ref_segs == fast_segs)
        # D: identical up to inf==inf; compare finite mask + values
        fin = np.isfinite(ref_D)
        D_match = np.array_equal(fin, np.isfinite(fast_D)) and np.allclose(ref_D[fin], fast_D[fin],
                                                                           rtol=0, atol=0)
        print(f"  {name:14s} n={p['x'].size:5d}  segs={len(ref_segs):3d}  "
              f"segments {'IDENTICAL' if segs_match else 'DIFFER'}  |  D {'IDENTICAL' if D_match else 'DIFFER'}")
        ok_all = ok_all and segs_match and D_match

    # ---- (B) end-to-end segment_two vs segment_two_fast (full alternating minimisation) ----
    #   exact_warp=True must be BIT-identical; matrix warp must give IDENTICAL segments (~1e-14 noise).
    print("\nEND-TO-END  (segment_two_fast vs segment_two)")
    for name, x in sigs:
        ref = TT.segment_two(x, fs, amp_invariant=True)
        f_exact = segment_two_fast(x, fs, amp_invariant=True, exact_warp=True)
        f_matrix = segment_two_fast(x, fs, amp_invariant=True, exact_warp=False)
        m_exact = (ref["segments"] == f_exact["segments"])
        m_matrix = (ref["segments"] == f_matrix["segments"])
        print(f"  {name:14s}  ref n_breaths={ref['n_breaths']:3d}  exact-warp "
              f"{'IDENTICAL' if m_exact else 'DIFFER'}  |  matrix-warp "
              f"{'IDENTICAL' if m_matrix else 'DIFFER'}")
        ok_all = ok_all and m_exact and m_matrix

    # ---- (C) speed: averaged over several distinct 8 s windows (reference is noisy run-to-run) ----
    print("\nSPEED  (8 s windows @ 125 Hz, multivox SEG_KW: max_iter=8, bmaxP=2.5; mean over 6 windows)")
    SEG_KW = dict(max_iter=8, bmaxP_s=2.5, amp_invariant=True)
    wins = [TT._synth(fs=fs, seed=sd)[0][:int(8 * fs)] for sd in range(6)]
    TT.segment_two(wins[0], fs, **SEG_KW); segment_two_fast(wins[0], fs, **SEG_KW)  # warmup (+JIT compile)
    def _mean_ms(fn):
        vs = []
        for w in wins:
            t0 = time.perf_counter()
            for _ in range(8):
                fn(w)
            vs.append((time.perf_counter() - t0) / 8)
        return float(np.mean(vs))
    t_ref = _mean_ms(lambda w: TT.segment_two(w, fs, **SEG_KW))
    t_fast = _mean_ms(lambda w: segment_two_fast(w, fs, **SEG_KW))
    print(f"  reference   : {t_ref*1000:8.2f} ms/call   ({t_ref/8*1000:6.2f} ms/signal-second)")
    print(f"  fast (numba): {t_fast*1000:8.2f} ms/call   ({t_fast/8*1000:6.2f} ms/signal-second)")
    print(f"  SPEEDUP     : {t_ref/t_fast:6.1f}x  (vs original 24 ms/sig-sec anchor: "
          f"{24.0/(t_fast/8*1000):.1f}x)")

    print(f"\nRESULT: {'PASS' if ok_all else 'FAIL'}  (bit-identity {'held' if ok_all else 'BROKEN'})")
    return ok_all


if __name__ == "__main__":
    raise SystemExit(0 if run_selftest() else 1)
