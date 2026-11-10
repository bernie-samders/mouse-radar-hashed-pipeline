r"""multivoxel.py -- anchored multi-voxel COMBINED breathing signal, then the SAME two-template DP on it.

Ported verbatim (same numbers) from the validated `apneacheck/multivoxel_combine.py --anchor 2 4`.

Why: the per-episode chosen voxel sits at the EDGE of the breathing blob, and a slight body shift slides the blob a few
bins -- the edge voxel loses breathing while voxels across the blob keep breathing. A combination over the blob keeps
the breathing waveform through such shifts and still goes flat when breathing truly stops (all voxels stop).

Per 8 s DP window (the blocks segmentation.py uses, aligned to each episode's t0):
  1. rank voxels within +-2 angle / +-4 range bins of the episode's chosen voxel by resp-band fundamental POWER
     (anchoring matters: in free-behaving mice the globally strongest resp-band cells are movement, gotcha #1);
  2. take the top-12, condition each exactly as the DP input (`cond_decimate`, 125 Hz);
  3. PC1 (unnormalised SVD -> strong voxels dominate, sign-mixed polarities handled) over window +-2 s;
  4. sign-align to the previous window over their overlap; keep the 8 s core;
then `fast_dp.segment_two_fast(core, 125, **SEG_KW)` per window -- identical to segmentation._episode_segments.
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy.signal import butter, filtfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import fast_dp as F
from conditioning import cond_decimate, SEG_KW, FLOOR_BPM, RESP_HI_BPM

WIN_S, CTX_S = 8.0, 2.0
K_VOX, ANCHOR = 12, (2, 4)                    # top-k voxels; +-angle / +-range-bin neighbourhood of the chosen voxel


def _fund_power(d50, fs50):
    """Per-column resp-band fundamental peak power (power, not SNR: the ratio favours faint clean sidelobes)."""
    b, a = butter(2, [1.2 / (fs50 / 2), 9.0 / (fs50 / 2)], "bandpass")
    x = filtfilt(b, a, d50 - d50.mean(0), axis=0)
    nfft = 1 << int(np.ceil(np.log2(max(256, x.shape[0]))))
    fr = np.fft.rfftfreq(nfft, 1 / fs50)
    fund = (fr >= FLOOR_BPM / 60) & (fr <= RESP_HI_BPM / 60)
    P = np.abs(np.fft.rfft(x * np.hanning(x.shape[0])[:, None], nfft, axis=0)) ** 2
    return P[fund].max(0)


def combine_episode(power, fs, t0e, t1e, k=K_VOX, allowed=None):
    """Returns (xc125 combined over [t0e, t1e) at 125 Hz, fs125, per-window chosen columns)."""
    q125 = int(round(fs / 125.0)); fs125 = fs / q125
    nwin = int(np.floor((t1e - t0e) / WIN_S + 1e-9))
    if (t1e - t0e) - nwin * WIN_S >= 3.0:
        nwin += 1                                             # trailing partial block (>=3 s), as _episode_segments
    out = np.zeros(int(round((t1e - t0e) * fs125)))
    prev_ext, prev_rng, cols_used = None, None, []
    for w in range(nwin):
        a = t0e + w * WIN_S; b = min(t1e, a + WIN_S)
        ea, eb = max(0.0, a - CTX_S), min(power.shape[0] / fs, b + CTX_S)
        i0, i1 = int(round(ea * fs)), int(round(eb * fs))
        mag = np.sqrt(np.maximum(np.asarray(power[i0:i1], np.float64), 0.0))
        n5 = mag.shape[0] // 5
        d50 = mag[:n5 * 5].reshape(n5, 5, -1).mean(1)
        fp = _fund_power(d50, fs / 5)
        if allowed is not None:
            fp = np.where(np.isin(np.arange(fp.size), allowed), fp, -np.inf)
        top = np.argsort(-fp)[:k]
        X = np.stack([cond_decimate(mag[:, c], fs)[0] for c in top], 1)          # (T125, k)
        X = X - X.mean(0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        y = X @ Vt[0]
        t_ext = ea + np.arange(y.size) / fs125
        if prev_ext is not None:                                               # sign-align on the overlap
            lo, hi = max(t_ext[0], prev_rng[0]), min(t_ext[-1], prev_rng[1])
            if hi - lo > 0.5:
                m = (t_ext >= lo) & (t_ext <= hi)
                pv = np.interp(t_ext[m], np.arange(prev_ext.size) / fs125 + prev_rng[0], prev_ext)
                if np.dot(pv - pv.mean(), y[m] - y[m].mean()) < 0:
                    y = -y
        prev_ext, prev_rng = y, (t_ext[0], t_ext[-1])
        s0 = int(round((a - t0e) * fs125)); s1 = min(out.size, int(round((b - t0e) * fs125)))
        c0 = int(round((a - ea) * fs125))
        out[s0:s1] = y[c0:c0 + (s1 - s0)]
        cols_used.append([int(c) for c in top])
    return out, fs125, cols_used


def segments_for(xc, fs125, t0e):
    """segmentation._episode_segments, verbatim, on an already-conditioned 125 Hz signal."""
    n = xc.size; w = int(WIN_S * fs125); lo = int(3 * fs125)
    out = []
    for a in range(0, n, w):
        seg = xc[a:a + w]
        if seg.size < lo:
            break
        res = F.segment_two_fast(seg, fs125, **SEG_KW)
        fitmap = {(p["a"], p["b"]): p["fit_tol"] for p in res["per"]}
        base = t0e + a / fs125
        for (s0, s1, kd) in res["segments"]:
            fit = fitmap.get((s0, s1), -1.0) if kd == 0 else -1.0
            out.append((base + s0 / fs125, base + s1 / fs125, float(kd), (s1 - s0) / fs125, float(fit)))
    return np.asarray(out, np.float32) if out else np.zeros((0, 5), np.float32)


def combine_session(power, meta, seg_json, k=K_VOX, anchor=ANCHOR, log=lambda *a: None):
    """Anchored combine + DP for every episode of a single-voxel substrate (its json dict). Returns (json, arrays) in
    the substrate format (trace/raw = combined signal @ fs; ibi/breath_times empty; cols = per-window voxels)."""
    import time
    fs = float(meta["fs"]); nb, na = int(meta["n_bin"]), int(meta["n_angle"])
    arrays = dict(segments=[], trace=[], raw=[], ibi=[], breath_times=[], cols=[])
    t_start = time.time()
    for ei, e in enumerate(seg_json["episodes"]):
        t0e, t1e = float(e["t0"]), float(e["t1"])
        ai0, bj0 = divmod(int(e["dom_col"]), nb)
        allowed = [ai * nb + bj for ai in range(max(0, ai0 - anchor[0]), min(na, ai0 + anchor[0] + 1))
                   for bj in range(max(0, bj0 - anchor[1]), min(nb, bj0 + anchor[1] + 1))]
        xc, fs125, cols = combine_episode(power, fs, t0e, t1e, k=k, allowed=allowed)
        segs = segments_for(xc, fs125, t0e)
        n250 = int(round((t1e - t0e) * fs))
        x250 = np.interp(np.arange(n250) / fs, np.arange(xc.size) / fs125, xc).astype(np.float32)
        arrays["segments"].append(segs); arrays["trace"].append(x250); arrays["raw"].append(x250)
        arrays["ibi"].append(np.zeros(0, np.float32)); arrays["breath_times"].append(np.zeros(0, np.float32))
        arrays["cols"].append(np.asarray(cols, np.int32))
        if ei % 10 == 0 or ei == len(seg_json["episodes"]) - 1:
            log(f"    [combine] ep {ei + 1}/{len(seg_json['episodes'])}  [{time.time() - t_start:.0f}s]")
    jj = dict(seg_json, method=f"multivoxel_pc1_k{k}_anchor{anchor[0]}x{anchor[1]}+wide_two_template")
    return jj, arrays
