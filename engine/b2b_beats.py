"""b2b_beats.py -- beat-to-beat radar heart tracking for one clean-breathing period (numpy / scipy only, ECG-free).

    beats, snr, info = estimate(Y, hr, t0)   # Y [T, N] breath-order residual of the chest patch at FS (bh.Y);
                                             # hr = seed period HR (bpm); beats = abs s, or None = abstain
Method (space-time whitened matched filter, STMF):
  prep     high-pass at 0.55 HR, 2 Hz running-median spectral whitening, unit MAD per voxel. Keeps heartbeat harmonics
           2-5, which carry the beat timing.
  start    analytic-phase 2 pi crossings of a periodic spatial beam (GEV of the lag-I0 vs lag-0 covariance in
           [HR/1.5, 1.5 HR], read out in [HR/1.2, 1.2 HR]).
  track    10 iterations of: per fold (4 contiguous time folds), template and template-removed noise covariance from
           the beats / samples OUTSIDE the fold (no beat matches its own template) -> soft whitened matched filter over
           all voxels (every generalised-eigen direction of (T'T, R) weighted max(0, 1 - edge / lambda), edge = the
           Marchenko-Pastur edge of a template averaged over n_b beats; no rank constant) -> d(t) in noise SDs ->
           local maxima -> Gaussian LLR -> second-order beat Viterbi (log-IBI priors: ratio to the previous IBI,
           SD 0.08; ratio to I0, SD 0.25; IBI within the I0 prior's +-2 SD).
  select   the tracker has several near-tied locks (a wrong one merges beats and has a slightly HIGHER SNR), so it is
           run from 11 cheap starts (beams 0-2 x {plain, un-whitened first covariance fit}, half-cycle-shifted
           crossings of beams 0-2, beam 0 at seed x0.97 / x1.03). Among the runs whose median d at their beats >= 2.0,
           keep the MEDOID: the run whose beats agree best with the other runs (share within 10 ms after one constant
           lag). A slip is specific to one start, so the majority outvotes it. Abstain if no run passes.
No per-mouse constant. Bit-identical to elegant/r2_selector ALL_AGREE_A (v_ALL_AGREE_A.py).
"""
import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import median_filter
from scipy.signal import butter, hilbert, sosfiltfilt

FS = 250.0
HP_FRAC, WHITEN_HZ = 0.55, 2.0                # prep: high-pass at 0.55 HR, 2 Hz running-median spectral whitening
BAND_GEV, BAND_READ = 1.5, 1.2                # start: periodic-beam band, crossing read-out band (x / HR)
NFOLD, N_ITER, R_EVERY, LOAD = 4, 10, 3, 0.1  # time folds, iterations, covariance refit period, diagonal loading
HALF_FRAC, S_R, S_0 = 0.5, 0.08, 0.25         # template half-width / I0; Viterbi IBI-ratio and I0 priors (log SD)
IBI_RANGE = (np.exp(-2 * S_0), np.exp(2 * S_0))   # hard IBI range / I0 = the I0 prior's +-2 SD
GATE, AGREE_S = 2.0, 0.010                    # keep a run if median d at its beats >= GATE; beat agreement tolerance (s)
# start = (periodic beam, crossing phase / 2 pi, seed factor of the start's bands, loading of the first covariance fit)
STARTS = ((0, 0.0, 1.0, None), (0, 0.0, 1.0, 1.0), (1, 0.0, 1.0, None), (1, 0.0, 1.0, 1.0),
          (2, 0.0, 1.0, None), (2, 0.0, 1.0, 1.0), (0, 0.5, 1.0, None), (1, 0.5, 1.0, None),
          (2, 0.5, 1.0, None), (0, 0.0, 0.97, None), (0, 0.0, 1.03, None))


def prep(Y, hr):
    """High-pass (Butterworth-4, zero-phase), divide the FFT by its running-median magnitude, high-pass again, unit MAD."""
    Y = np.asarray(Y, float)
    sos = butter(4, HP_FRAC * (hr / 60.0), "highpass", fs=FS, output="sos")
    Y = sosfiltfilt(sos, Y, axis=0)
    n = Y.shape[0]
    F = np.fft.rfft(Y, axis=0)
    w = max(3, int(round(WHITEN_HZ / (FS / n))) | 1)
    S = np.stack([median_filter(a, size=w, mode="nearest") for a in np.abs(F).T], axis=1)
    Y = sosfiltfilt(sos, np.fft.irfft(F / (S + 1e-12 * S.max()), n=n, axis=0), axis=0)
    return Y / (1.4826 * np.median(np.abs(Y - np.median(Y, 0)), 0) + 1e-30)


def start_markers(Yp, hr, comp, phase=0.0, factor=1.0):
    """Analytic-phase crossings of 2 pi (k + phase) (samples) of periodic beam `comp` (0 = most periodic)."""
    hr = hr * factor if factor != 1.0 else hr
    bp = lambda x, f: sosfiltfilt(butter(4, [hr / 60 / f, hr / 60 * f], "bandpass", fs=FS, output="sos"), x, axis=0)
    Yb = bp(Yp, BAND_GEV)
    tau = int(round(FS * 60.0 / hr))
    C1, C0 = Yb[tau:].T @ Yb[:-tau], Yb.T @ Yb
    V = eigh(C1 + C1.T, (1 - LOAD) * C0 + LOAD * np.trace(C0) / C0.shape[0] * np.eye(C0.shape[0]))[1][:, ::-1]
    th = np.unwrap(np.angle(hilbert(bp((Yb @ V[:, :comp + 1])[:, comp], BAND_READ))))
    if phase:
        th = th - 2 * np.pi * phase
    k = np.floor(th / (2 * np.pi))
    i = np.flatnonzero(np.diff(k) > 0)
    return i + ((k[i] + 1) * 2 * np.pi - th[i]) / (th[i + 1] - th[i])


def fits(marks, half, T):
    m = np.round(marks).astype(int)
    return m[(m - half >= 0) & (m + half <= T)]


def template(Y, marks, half):
    """Per-voxel mean of Y over [m - half, m + half) at the markers whose window fits -> [2 half, N]."""
    acc, n = np.zeros((2 * half, Y.shape[1])), 0
    for m in fits(marks, half, Y.shape[0]):
        acc += Y[m - half:m + half]; n += 1                       # sequential sum (frozen rounding)
    return acc / max(n, 1)


def residual(Y, marks, tm, half):
    """Y minus the template overlap-added at the markers (so the heart is not counted as noise)."""
    Yr = Y.copy()
    for m in fits(marks, half, Y.shape[0]):
        Yr[m - half:m + half] -= tm
    return Yr


def whitened(tm, R, nu):
    """M = sum_i w_i (T v_i) v_i' over the GEV directions of (T'T, R) (V'RV = I), w_i = max(0, 1 - edge / lambda_i),
    edge = nu (1 + sqrt(2h / N))^2 (Marchenko-Pastur edge; nu = N_eff / n_b) -> M, MF noise variance."""
    lam, V = eigh(tm.T @ tm, R)
    edge = nu * (1 + np.sqrt(tm.shape[0] / R.shape[0])) ** 2
    M = ((tm @ V) * np.clip(1 - edge / np.maximum(lam, 1e-300), 0, None)) @ V.T
    return M, float(np.sum((M @ R) * M))


def mf(Y, M, half):
    """Centred multichannel matched filter d[t] = sum_s M[s] . Y[t - half + s]; NaN where the window does not fit."""
    T, L = Y.shape[0], M.shape[0]
    Z = Y @ M.T
    c = np.zeros(T - L + 1)
    for s in range(L):                                            # lag-sequential sum (frozen rounding)
        c += Z[s:s + T - L + 1, s]
    d = np.full(T, np.nan)
    d[half:half + c.size] = c
    return d


def mad_sd(d):
    d = d[np.isfinite(d)]
    return 1.4826 * np.median(np.abs(d - np.median(d)))


def local_max(d):
    """Local maxima of d (strict left, weak right) with parabolic refinement -> (times in samples, heights)."""
    f = np.isfinite(d)
    i = np.where((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:]) & f[1:-1] & f[:-2] & f[2:])[0] + 1
    y0, y1, y2 = d[i - 1], d[i], d[i + 1]
    den = y0 - 2 * y1 + y2
    off = np.clip(np.where(den != 0, 0.5 * (y0 - y2) / np.where(den != 0, den, 1), 0), -0.5, 0.5)
    return i + off, y1 - 0.25 * (y0 - y2) * off


def viterbi(tc, e, I0):
    """Best beat chain over sorted candidates tc: sum e - (log IBI/I0)^2 / 2 S_0^2 - (log IBI/IBI_prev)^2 / 2 S_R^2,
    IBI in IBI_RANGE x I0, first / last beat within the longest IBI of the first / last candidate -> indices."""
    n = tc.size
    if n < 3:
        return np.arange(n)
    lo, hi = IBI_RANGE[0] * I0, IBI_RANGE[1] * I0
    P = int((np.arange(n) - np.searchsorted(tc, tc - hi)).max()) + 2
    pre = np.arange(n)[:, None] - P + np.arange(P)                # state (j, c) = transition pre[j, c] -> j
    D = tc[:, None] - tc[np.maximum(pre, 0)]
    ok = (pre >= 0) & (D >= lo) & (D <= hi)
    D1 = np.where(ok, D, 1.0)
    p0 = -np.log(D1 / I0) ** 2 / (2 * S_0 ** 2)
    S, Bk = np.full((n, P), -np.inf), np.full((n, P), -1)
    first = np.where(tc <= tc[0] + hi, e, -np.inf)
    j0 = 0
    while j0 < n:                                                 # block of rows that cannot precede each other
        j1 = j0 + 1
        while j1 < n and tc[j1] - tc[j0] < lo:
            j1 += 1
        jr, c = np.nonzero(ok[j0:j1])
        jr += j0
        i = pre[jr, c]
        sc = S[i] - np.log(D[jr, c][:, None] / D1[i]) ** 2 / (2 * S_R ** 2)
        k = np.argmax(sc, axis=1)                                 # first max (tie order)
        sk = sc[np.arange(i.size), k]
        take = sk > first[i]
        best = np.where(take, sk, first[i])
        f = np.isfinite(best)
        S[jr[f], c[f]] = best[f] + e[jr[f]] + p0[jr[f], c[f]]
        Bk[jr[f], c[f]] = np.where(take, pre[i, k], -1)[f]
        j0 = j1
    end = np.flatnonzero(tc >= tc[-1] - hi)
    last = S[end].ravel()
    if last.size == 0 or not np.isfinite(last.max()):
        return np.array([], int)
    q = int(np.argmax(last))
    path = [int(end[q // P]), int(pre[end[q // P], q % P])]
    while Bk[path[-2], path[-1] - path[-2] + P] >= 0:
        path.append(int(Bk[path[-2], path[-1] - path[-2] + P]))
    return np.array(path[::-1], int)


def track(Yp, hr, marks, load0=None):
    """Iterate fold-cross-fitted template + covariance -> soft-whitened MF -> LLR -> Viterbi from marks (samples);
    load0 = loading of the first covariance fit (None = LOAD). -> (marks in samples, SNR = median d / noise SD)."""
    T, N = Yp.shape
    I0 = FS * 60.0 / hr
    half = int(round(HALF_FRAC * I0))
    edges = np.linspace(0, T, NFOLD + 1).round().astype(int)
    marks, Rs, nus, d, sig = np.asarray(marks, float), [None] * NFOLD, [None] * NFOLD, None, 1.0
    for it in range(N_ITER):
        if marks.size == 0:
            break
        ld = load0 if (it < R_EVERY and load0 is not None) else LOAD
        fold = np.clip(np.searchsorted(edges, np.round(marks), side="right") - 1, 0, NFOLD - 1)
        d = np.full(T, np.nan)
        for b in range(NFOLD):                                    # template / R from beats and samples outside fold b
            a, z = edges[b], edges[b + 1]
            mk = marks[fold != b]
            tm = template(Yp, mk, half)
            if Rs[b] is None or it % R_EVERY == 0:
                out = np.ones(T, bool); out[a:z] = False
                Zr = residual(Yp, mk, tm, half)[out]
                R = Zr.T @ Zr / Zr.shape[0]
                Rs[b] = (1 - ld) * R + ld * np.trace(R) / N * np.eye(N)
                nus[b] = float(np.trace(np.linalg.solve(Rs[b], R)))   # N_eff: noise dimensions left after loading
            M, var = whitened(tm, Rs[b], nus[b] / max(fits(mk, half, T).size, 1))
            d[a:z] = (mf(Yp, M, half) / np.sqrt(max(var, 1e-30)))[a:z]
        sig = mad_sd(d)
        tc, dc = local_max(d)
        mu = max(np.median(np.interp(marks, np.arange(T), np.where(np.isfinite(d), d, 0))), sig)   # beat amplitude
        marks = tc[viterbi(tc, (mu * dc - mu ** 2 / 2) / sig ** 2, I0)]                         # Gaussian LLR
    if marks.size == 0 or d is None:
        return np.array([]), 0.0
    return marks, float(np.median(np.interp(marks, np.arange(T), np.where(np.isfinite(d), d, 0)) / sig))


def agreement(a, b, tol=AGREE_S):
    """Share of the beats of a and b within tol of each other, after the constant lag (+-half b's median IBI, 1 ms grid)
    that maximises the number of a's beats within tol of b's; counted inside the common span."""
    if a.size < 2 or b.size < 3:
        return 0.0
    near = lambda x, y: np.min(np.abs(x[:, None] - y[np.clip(np.searchsorted(y, x), 1, y.size - 1)[:, None] - [1, 0]]),
                               axis=1) <= tol
    I = float(np.median(np.diff(b)))
    L = np.arange(-I / 2, I / 2, 0.001)
    O = a[None, :] - L[:, None]
    k = np.clip(np.searchsorted(b, O), 1, b.size - 1)
    o = a - float(L[int(np.argmax((np.minimum(np.abs(O - b[k - 1]), np.abs(O - b[k])) <= tol).sum(1)))])
    lo, hi = max(o[0], b[0]) - tol, min(o[-1], b[-1]) + tol
    A, B = o[(o >= lo) & (o <= hi)], b[(b >= lo) & (b <= hi)]
    if A.size < 2 or B.size < 2:
        return 0.0
    return float((near(A, B).sum() + near(B, A).sum()) / (A.size + B.size))


def estimate(Y, hr, t0=0.0, gate=GATE):
    """One period -> (beats in s on t0's clock or None = abstain, SNR of the kept run, info)."""
    Yp = prep(Y, hr)
    runs = []
    for c, phase, factor, load0 in STARTS:
        marks, snr = track(Yp, hr, start_markers(Yp, hr, c, phase, factor), load0)
        runs.append((np.unique(t0 + marks / FS), snr))
    agree = [np.mean([agreement(b, runs[j][0]) for j in range(len(runs)) if j != i]) for i, (b, _) in enumerate(runs)]
    ok = [i for i, (b, snr) in enumerate(runs) if snr >= gate and b.size >= 2]
    if not ok:
        return None, max(r[1] for r in runs), dict(pick=-1)
    i = max(ok, key=lambda i: agree[i])                           # first max on ties
    return runs[i][0], runs[i][1], dict(pick=i, agree=float(agree[i]))


def estimate_p(p):
    import bh
    return estimate(bh.Y(p), p["hr"], p["t0"])
