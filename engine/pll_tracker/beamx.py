"""beamx.py -- arm pll: the arm-C beams (FROZEN recipe, beams.py imported read-only) plus a per-harmonic RING output of the
same spatial weights, used as an ECG-free local noise-level reference for the tracker.

    beams_ring(p, centre, K=5, Y=None) -> (Z [K, T] complex = beams.beams(p, centre, K, Y) exactly,
                                           Nr [K, T] complex = the same beam in the ring k*fc*e^{+-[r0, r1]} (heart-free),
                                           q [n_cand, K] beams' cross-fitted contrast proxy)
Must be imported after the harness bh (ng511 or dev), like beams.py.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "beams"))
import numpy as np
import beams as BM

R0, R1 = 0.22, 0.40                       # ring in |ln f / (k fc)|: outside the +-20 % (+4 % taper) beam band


def _ring(F, fr, n, P, T, fk):
    x = np.full(fr.size, np.inf); pos = fr > 0
    x[pos] = np.abs(np.log(fr[pos] / fk))
    h = ((x >= R0) & (x <= R1) & (fr < 0.48 * BM.FS)).astype(float)
    full = np.zeros(n, complex); full[:fr.size] = 2 * F * h
    full[0] = 0
    return np.fft.ifft(full)[P:P + T]


def beams_ring(p, centre, K=5, Y=None):
    import variants
    cfg = dict(BM.BASE, **variants.FROZEN)
    W, S = BM.weights(p, centre, K, Y, cfg)
    inf = BM.last_info()
    F, fr, n, P = BM.spectrum(S)
    fc = centre / 60.0; T = S.shape[0]
    Z = np.stack([BM._beam_from(S, F, fr, n, P, k * fc, W[k - 1], cfg["bw_out"]) for k in range(1, K + 1)])
    Nr = np.stack([_ring(F @ W[k - 1][1][0].conj(), fr, n, P, T, k * fc) for k in range(1, K + 1)])
    return Z, Nr, np.asarray(inf.get("score", np.full((1, K), np.nan)), float)


def half_k(k, half1, nonover):
    """Output band half-width (ln) of harmonic k: k = 1 -> half1; k >= 2 -> ln 1.2, or with nonover the non-overlapping
    limit min(ln 1.2, 0.45 ln((k + 1) / k)) (k >= 3 bands of +-20 % overlap their neighbours)."""
    if k == 1:
        return half1
    h = float(np.log(1.2))
    return min(h, 0.45 * float(np.log((k + 1) / k))) if nonover else h


def beams_wide(p, centre, K=5, Y=None, half1=float(np.log(1.2)), nonover=False):
    """As beams.beams (FROZEN weights) but the k = 1 output band is k*fc*e^{+-half1} (+4 % taper); k >= 2 unchanged.
    Returns (Z [K, T], q [n_cand, K])."""
    import variants
    cfg = dict(BM.BASE, **variants.FROZEN)
    W, S = BM.weights(p, centre, K, Y, cfg)
    inf = BM.last_info()
    F, fr, n, P = BM.spectrum(S)
    fc = centre / 60.0
    Z = np.stack([BM._beam_from(S, F, fr, n, P, k * fc, W[k - 1], half_k(k, half1, nonover))
                  for k in range(1, K + 1)])
    return Z, np.asarray(inf.get("score", np.full((1, K), np.nan)), float)


def k1_edges(fc, half1, rr_bpm, margin):
    """(lo, hi) ln half-widths of the k = 1 band: hi = half1, lo = half1 limited so the band (plus its 4 % taper)
    stays `margin` (ln) above the breathing fundamental RR (a_priori: the breath line must not enter the heart band)."""
    if rr_bpm is None or margin is None:
        return half1, half1
    lo = min(half1, float(np.log(fc / (rr_bpm / 60.0))) - margin - float(np.log(1.04)))
    return max(lo, 0.05), half1


def _asym(F, fr, n, P, T, fk, lo, hi, tap=float(np.log(1.04))):
    x = np.full(fr.size, np.inf); pos = fr > 0
    x[pos] = np.log(fr[pos] / fk)
    h = np.where(x >= 0, BM.taper(np.abs(x), hi, hi + tap), BM.taper(np.abs(x), lo, lo + tap))
    h[~pos] = 0.0
    full = np.zeros(n, complex); full[:fr.size] = 2 * F * h
    full[0] = 0
    return np.fft.ifft(full)[P:P + T]


def beams_guard(p, centre, K=5, Y=None, half1=float(np.log(1.2)), margin=0.10):
    """beams_wide with the k = 1 band lower edge guarded against the breathing fundamental (k1_edges, p["rr"])."""
    import variants
    cfg = dict(BM.BASE, **variants.FROZEN)
    W, S = BM.weights(p, centre, K, Y, cfg)
    inf = BM.last_info()
    F, fr, n, P = BM.spectrum(S)
    fc = centre / 60.0; T = S.shape[0]
    lo, hi = k1_edges(fc, half1, p["rr"], margin)
    Z = [_asym(F @ W[0][1][0].conj(), fr, n, P, T, fc, lo, hi)]
    Z += [BM._beam_from(S, F, fr, n, P, k * fc, W[k - 1], cfg["bw_out"]) for k in range(2, K + 1)]
    return np.stack(Z), np.asarray(inf.get("score", np.full((1, K), np.nan)), float), (lo, hi)
