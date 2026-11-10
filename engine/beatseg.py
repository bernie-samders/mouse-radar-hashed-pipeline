"""beatseg.py -- EQ-Radio heartbeat segmentation (paper §5, Algorithm 1): fast, multi-piece, multi-start.

Same algorithm as engine eqradio_hr.segment_signal: DP over beat lengths (Eqn 6) alternating with the closed-form
length-weighted template (Eqn 8); stop when the boundaries repeat or the cost stops falling. Differences:
  - numba DP core + matrix warps (fast_dp._warp_matrix_for) -> same boundaries as the reference (self-test);
  - the bounds passed are always used (the reference's hr_hint path silently replaces them with [0.65, 1.5] T);
  - input is a list of valid pieces (the breath residual is undefined outside fit-ok cycles): the DP runs per piece,
    template and cost are pooled; each piece is mean-removed (= the reference's whole-signal mean for one piece);
  - several starts, lowest pooled cost wins: 'zero' (paper mu0 = 0; DP ties -> effectively a grid seed), 'acf'
    (engine _seed_template, ACF pooled over pieces via FFT), 'p<bpm>' (uniform grid at that rate).
The first and last segment of each piece are forced to the piece edges, so HR/IBI/onsets use interior cuts only.

    python beatseg.py      -> self-test: identity vs eq.segment_signal, synthetic HR recovery without a hint
"""
from __future__ import annotations
import os, sys
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:                 # engine modules import each other flatly
    sys.path.insert(0, HERE)
import eqradio_hr as eq
import fast_dp as FD

BPM_LO, BPM_HI = 350.0, 800.0           # calm mouse HR 400-600 (508 ECG); excludes half/double beats there
INITS = ("zero", "acf", "p400", "p475", "p575", "p700")


def bounds_samples(fs, bpm_lo=BPM_LO, bpm_hi=BPM_HI):
    bmin = max(3, int(round(60.0 / bpm_hi * fs))); bmax = max(bmin + 1, int(round(60.0 / bpm_lo * fs)))
    return bmin, bmax, max(4, int(round(0.5 * (bmin + bmax))))


def warp(v, L):
    v = np.asarray(v, float); m = v.size; L = int(L)
    if m < 4 or L <= 1:
        return eq.warp(v, L)
    return FD._warp_matrix_for(m, L) @ v


def _prefix(x):
    return np.concatenate([[0.0], np.cumsum(x * x)])


def cost_table(x, xsq, mu, bmin, bmax):
    """C[li, t] = ||x[t-L:t] - warp(mu, L)||^2, L = bmin + li (eq._seg_cost_table, same op order)."""
    n = x.size
    C = np.full((bmax - bmin + 1, n + 1), np.inf)
    for li, L in enumerate(range(bmin, bmax + 1)):
        if L > n:
            continue
        w = warp(mu, L)
        wsq = float(np.dot(w, w))
        cross = np.correlate(x, w, mode="valid")
        taus = np.arange(0, n - L + 1); ts = taus + L
        C[li, ts] = (xsq[ts] - xsq[taus]) - 2.0 * cross + wsq
    return C


@njit(cache=True)
def _dp1(C, bmin, bmax):
    """D_t = min_L D_{t-L} + C[L, t]; strict-< over ascending L = np.argmin's first minimum (reference)."""
    n = C.shape[1] - 1
    D = np.full(n + 1, np.inf); D[0] = 0.0
    back = np.full(n + 1, -1, np.int64)
    for t in range(bmin, n + 1):
        best = np.inf; bestL = -1
        Lmax = bmax if bmax < t else t
        for L in range(bmin, Lmax + 1):
            c = D[t - L] + C[L - bmin, t]
            if c < best:
                best = c; bestL = L
        if bestL > 0:
            D[t] = best; back[t] = bestL
    return D, back


def update_segmentation(x, xsq, mu, bmin, bmax):
    """eq.update_segmentation with the JIT DP; backtrace verbatim (unreachable tail dropped)."""
    n = x.size
    bmax = min(bmax, n)
    if bmin > bmax:
        return [0, n]
    D, back = _dp1(cost_table(x, xsq, mu, bmin, bmax), bmin, bmax)
    end = n
    while end > 0 and not np.isfinite(D[end]):
        end -= 1
    bounds = [end]; t = end
    while t > 0 and back[t] > 0:
        t -= back[t]; bounds.append(int(t))
    bounds = sorted(set(int(b) for b in bounds))
    if bounds[0] != 0:
        bounds = [0] + bounds
    return bounds


def _segs(pieces, blist):
    for x, b in zip(pieces, blist):
        for a, c in zip(b[:-1], b[1:]):
            if c > a:
                yield x[a:c]


def update_template(pieces, blist, m):
    acc = np.zeros(m); wsum = 0.0
    for s in _segs(pieces, blist):
        acc += s.size * warp(s, m); wsum += s.size
    return acc / wsum if wsum > 0 else np.zeros(m)


def var_cost(pieces, blist, mu):
    return float(sum(np.sum((s - warp(mu, s.size)) ** 2) for s in _segs(pieces, blist)))


def confidence(pieces, blist, mu, m):
    """eq._confidence pooled over pieces: 1 - residual vs template / variance vs a flat scalar mean."""
    G = np.array([warp(s, m) for s in _segs(pieces, blist)])
    if len(G) < 2:
        return 0.0
    num = float(np.sum((G - mu[None, :]) ** 2)); den = float(np.sum((G - G.mean()) ** 2))
    return float(np.clip(1.0 - num / den, 0.0, 1.0)) if den > 1e-30 else 0.0


def harmonic_fraction(mu):
    """1 - |c1|^2 / sum_k>=1 |c_k|^2 of the template over one beat: 0 for a sine, -> 1 for a sharp beat."""
    p = np.abs(np.fft.rfft(np.asarray(mu, float) - np.mean(mu))) ** 2
    return float(1.0 - p[1] / p[1:].sum()) if p[1:].sum() > 0 else np.nan


def _acf_L0(pieces, bmin, bmax):
    ac = np.zeros(bmax + 1)
    for x in pieces:
        n = x.size; nf = 1 << int(np.ceil(np.log2(2 * n)))
        F = np.fft.rfft(x, nf)
        a = np.fft.irfft(F * np.conj(F), nf)[:min(bmax + 1, n)]
        ac[:a.size] += a
    return int(bmin + np.argmax(ac[bmin:bmax + 1]))


def _grid_template(pieces, L0, m):
    bl = []
    for x in pieces:
        b = list(range(0, x.size + 1, L0))
        bl.append(b if b[-1] == x.size else b + [x.size])
    return update_template(pieces, bl, m)


def init_template(pieces, init, fs, bmin, bmax, m):
    if init == "zero":
        return np.zeros(m)
    L0 = _acf_L0(pieces, bmin, bmax) if init == "acf" else int(round(60.0 / float(init[1:]) * fs))
    return _grid_template(pieces, int(np.clip(L0, bmin, bmax)), m)


def segment(pieces, fs, bmin, bmax, m, init="acf", max_iter=16):
    """Algorithm 1 on a list of pieces. Returns dict(blist, mu, cost, energy, n_iter, init)."""
    pieces = [np.asarray(p, float) - np.mean(p) for p in pieces]
    xsqs = [_prefix(x) for x in pieces]
    mu = init_template(pieces, init, fs, bmin, bmax, m)
    blist = [update_segmentation(x, q, mu, bmin, bmax) for x, q in zip(pieces, xsqs)]
    prev, n_iter = np.inf, 0
    for it in range(max_iter):
        n_iter = it + 1
        mu = update_template(pieces, blist, m)
        new = [update_segmentation(x, q, mu, bmin, bmax) for x, q in zip(pieces, xsqs)]
        cost = var_cost(pieces, new, mu)
        stop = (new == blist) or (cost > prev - 1e-9 * max(prev, 1.0))
        blist = new; prev = cost
        if stop:
            break
    mu = update_template(pieces, blist, m)
    return dict(blist=blist, mu=mu, cost=var_cost(pieces, blist, mu), energy=float(sum(np.sum(x * x) for x in pieces)),
                n_iter=n_iter, init=init, conf=confidence(pieces, blist, mu, m), pieces=pieces)


def beats(res, offsets, fs):
    """Interior cut times (s) and the IBIs (s) between consecutive interior cuts of the same piece."""
    on, ibi, ibi_t = [], [], []
    for b, o in zip(res["blist"], offsets):
        c = (np.asarray(b[1:-1], float) + o) / fs
        on.append(c)
        if c.size > 1:
            ibi.append(np.diff(c)); ibi_t.append(c[1:])
    cat = lambda z: np.concatenate(z) if z else np.zeros(0)
    return cat(on), cat(ibi), cat(ibi_t)


def multistart(pieces, fs, bmin, bmax, m, inits=INITS, max_iter=16):
    runs = [segment(pieces, fs, bmin, bmax, m, init=i, max_iter=max_iter) for i in inits]
    k = int(np.argmin([r["cost"] for r in runs]))
    return runs[k], runs


# ---- self-test ----------------------------------------------------------------------------------------------------

def _selftest():
    fs = 250
    trace, gt = eq._make_synth(fs, 40.0, with_cardiac=True)
    hr_true = 60.0 / (np.mean(np.diff(gt)) / fs)
    acc = eq.accel_holoborodko(trace, fs)
    bb = eq.butter(4, [4.5 / (fs / 2), 42.0 / (fs / 2)], "bandpass"); wb = eq.filtfilt(*bb, trace - trace.mean())
    rng = np.random.default_rng(3)
    ok = True
    print("identity vs eq.segment_signal (no hint, engine bounds 0.08-0.22 s, init acf/auto):")
    for name, x in (("accel synth", acc), ("wideband synth", wb), ("noise", rng.standard_normal(5000))):
        ref = eq.segment_signal(x, fs, bmin_s=eq.BMIN_S, bmax_s=eq.BMAX_S, init="auto")
        bmin, bmax, m = ref["bmin"], ref["bmax"], ref["m"]
        r = segment([x], fs, bmin, bmax, m, init="acf")
        same = list(map(int, ref["boundaries"])) == list(map(int, r["blist"][0]))
        print(f"  {name:15s} beats {len(ref['boundaries']) - 1:4d}  boundaries {'IDENTICAL' if same else 'DIFFER'}  "
              f"conf ref {ref['confidence']:.4f} fast {r['conf']:.4f}")
        ok &= same
    bmin, bmax, m = bounds_samples(fs)
    print(f"\nsynthetic recovery, our bounds {BPM_LO:.0f}-{BPM_HI:.0f} bpm ({bmin}-{bmax} samples, m {m}), no hint, "
          f"HR true {hr_true:.1f}:")
    for name, x in (("accel", acc), ("wideband", wb)):
        best, runs = multistart([x], fs, bmin, bmax, m)
        on, ibi, _ = beats(best, [0], fs)
        off, jit = eq._localization(gt, best["blist"][0], fs)
        hr = 60.0 / np.median(ibi)
        print(f"  {name:9s} best init {best['init']:5s} HR {hr:6.1f}  jitter {jit:4.1f} ms  conf {best['conf']:.3f}  "
              f"H {harmonic_fraction(best['mu']):.2f}  | per init HR: "
              + " ".join(f"{r['init']}:{60 / np.median(beats(r, [0], fs)[1]):.0f}" for r in runs))
        ok &= abs(hr - hr_true) < 10
    t = np.arange(int(20 * fs)) / fs
    sine = np.sin(2 * np.pi * 8.0 * t) + 0.05 * rng.standard_normal(t.size)
    rs = segment([sine], fs, bmin, bmax, m, init="acf")
    print(f"\nreference H for a pure 480 bpm sine: {harmonic_fraction(rs['mu']):.3f}; for the synthetic biphasic beat: "
          f"{harmonic_fraction(segment([wb], fs, bmin, bmax, m)['mu']):.3f}")
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
