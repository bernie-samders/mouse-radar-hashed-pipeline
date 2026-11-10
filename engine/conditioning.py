"""conditioning.py -- signal conditioning + breath/rate primitives for the radar magnitude trace.

Every function here is vendored VERBATIM from the working pipeline (originally spread across
`ml_apnea/mouse_voxel_infer.py`, `ml_apnea/mouse_voxel_candidates.py`, `ml_apnea/cohort/clean_substrate.py`,
`output/eqradio_resp/multivox.py`, and `output/wide_resp/wide_prescreen.py`) so the numbers are identical.
They are pure numpy/scipy with no cross-module dependencies -- the reusable DSP core.

The one convention to know: the `.f32` cube stores raw beamformed POWER; we work on MAGNITUDE
(`sqrt(max(power,0))`) and then `condition_trace` removes slow body-motion drift and abrupt step/spike
artifacts while preserving breathing down to ~40 bpm.  Everything is amplitude-invariant.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, find_peaks, decimate as sp_decimate
from scipy.ndimage import median_filter, uniform_filter1d

# --- constants (verbatim) --------------------------------------------------------------------------
SLOW_FLOOR_BPM = 40.0     # slowest breathing to PRESERVE through the de-drift / high-pass
AC_MIN = 0.45             # min de-jumped-bandpassed autocorr for a "clean" episode (clean_substrate.AC_MIN)
K_BIN_NEIGH = 1           # refine the chosen cell over +/- this many range bins
MIN_SEG_S = 8.0           # minimum breathing-episode length (cohort/core.MIN_SEG_S)
DEC_TT = 125.0            # decimation target (Hz) for the two-template DP (the "125 Hz" signal)
SEG_KW = dict(max_iter=8, bmaxP_s=2.5, amp_invariant=True)   # two-template DP knobs (multivox.SEG_KW)

# wide-prescreen grid/band constants (wide_prescreen.py) -- used by segmentation.py
DEC_HZ = 50.0             # decimation rate for the cheap all-voxel SNR shortlist
NFFT = 256                # rFFT length for the shortlist (next pow2 >= 4 s @ 50 Hz)
FLOOR_BPM = 96.0          # resp fundamental floor (~1.6 Hz)
RESP_HI_BPM = 480.0       # resp fundamental ceiling (~8 Hz)
HF_LO, HF_HI = 1.6, 8.0   # broadband floor band for the SNR denominator


# --- high-pass (vectorised over all voxels; mouse_voxel_infer._highpass) ---------------------------
def _highpass(x, fs, floor_bpm, axis=0):
    b, a = butter(4, (floor_bpm / 60.0) / (fs / 2.0), btype="highpass")
    return filtfilt(b, a, x - x.mean(axis, keepdims=True), axis=axis)


# --- de-drift / de-jump conditioning (mouse_voxel_candidates.condition_trace) ----------------------
def condition_trace(x, fs, bpm=None, hp_hz=0.3, k_jump=6.0):
    """Remove slow body-motion SLOPES and abrupt JUMPS from a voxel-magnitude trace, robustly, while
    preserving breathing down to SLOW_FLOOR_BPM.  Three scale-equivariant stages: (1) de-jump by
    clipping the derivative's outliers and re-integrating; (2) de-drift by subtracting a running median
    sized from the slowest breathing to keep; (3) a light high-pass below SLOW_FLOOR.  `bpm` is accepted
    for call-site compatibility but no longer sizes the drift window."""
    x = np.asarray(x, np.float64)
    if x.size < 8:
        return x - x.mean()
    # 1) de-jump via derivative clipping + re-integration
    d = np.diff(x, prepend=x[0])
    mad = np.median(np.abs(d - np.median(d))) + 1e-12
    xj = np.cumsum(np.clip(d, -k_jump * mad, k_jump * mad))
    # 2) de-drift: median window ~2.5 periods of the SLOWEST breathing to preserve
    w = max(3, int(round(2.5 * (60.0 / SLOW_FLOOR_BPM) * fs)) | 1)     # ~3.75 s at 40 bpm
    xj = xj - median_filter(xj, size=w, mode="nearest")
    # 3) light high-pass safely BELOW SLOW_FLOOR so slow breaths survive
    hp = min(hp_hz, 0.6 * SLOW_FLOOR_BPM / 60.0, 0.4 * fs)
    b, a = butter(2, hp / (fs / 2.0), btype="highpass")
    return filtfilt(b, a, xj - xj.mean())


# --- decimation to the two-template rate (multivox._cond_decimate) ----------------------------------
def cond_decimate(mag, fs, dec_to=DEC_TT):
    """Condition then anti-alias-decimate a magnitude trace to `dec_to` Hz.  Returns (xd, fs_out).
    This is the exact signal the two-template DP segmenter operates on."""
    q = int(round(fs / dec_to))
    c = condition_trace(mag, fs)
    return (sp_decimate(c, q, ftype="fir", zero_phase=True) if q > 1 else c), fs / max(1, q)


def _decimate(blk, q):
    """Mean-decimate a [T, V] slab by integer q (wide_prescreen._decimate)."""
    if q > 1 and blk.shape[0] >= 2 * q:
        n2 = (blk.shape[0] // q) * q
        return blk[:n2].reshape(-1, q, blk.shape[1]).mean(axis=1)
    return blk


# --- bandpass + autocorrelation cleanliness (clean_substrate) --------------------------------------
def _bandpass(x, fs, lo=1.6, hi=8.0):
    hi = min(hi, 0.45 * fs)
    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], "bandpass")
    return filtfilt(b, a, np.asarray(x, float) - np.mean(x))


def _ac_strength(x, fs, lo_bpm=100, hi_bpm=480):
    x = x - x.mean(); T = len(x)
    if T < 8:
        return 0.0, 0.0
    ac = np.correlate(x, x, "full")[T - 1:]; ac /= ac[0] + 1e-12
    lo, hi = int(fs / (hi_bpm / 60)), int(fs / (lo_bpm / 60))
    if hi + 1 > T:
        return 0.0, 0.0
    sub = ac[lo:hi + 1]
    return float(sub.max()), 60 * fs / (lo + int(np.argmax(sub)))


# --- instantaneous rate + rate-adaptive breath detection (mouse_voxel_candidates) ------------------
def _rfft_autocorr(seg):
    """Normalized autocorrelation via rFFT (O(n log n))."""
    n = len(seg)
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(seg - seg.mean(), nf)
    ac = np.fft.irfft(F * np.conj(F), nf)[:n]
    return ac / (ac[0] + 1e-30)


def instantaneous_rate(x, fs, floor_bpm=SLOW_FLOOR_BPM, hi_bpm=480.0, win_s=3.0, hop_s=0.25):
    """Track the instantaneous respiratory rate via the STRONGEST autocorrelation-peak lag (the dominant
    chest-motion period = breathing, robust to a smaller/faster cardiac ripple).  A sub-multiple check
    guards harmonic doubling.  Returns (t_rel, rate_bpm, strength) on the hop grid."""
    win = int(win_s * fs); hop = max(1, int(hop_s * fs))
    lo_lag = max(2, int(fs / (hi_bpm / 60.0)))
    hi_lag = int(fs / (floor_bpm / 60.0))
    t, rate, stg = [], [], []
    for s in range(0, max(1, len(x) - win), hop):
        ac = _rfft_autocorr(x[s:s + win])
        sub = ac[lo_lag:hi_lag + 1]
        tc = (s + win / 2) / fs
        if sub.size < 3:
            t.append(tc); rate.append(np.nan); stg.append(0.0); continue
        pk, props = find_peaks(sub, height=0.08, prominence=0.03)
        if pk.size:
            j = int(pk[np.argmax(props["peak_heights"])])
            best_lag = lo_lag + j; best_h = sub[j]
            for kk in (2, 3):
                sl = best_lag // kk
                if sl >= lo_lag:
                    wv = sub[max(0, sl - lo_lag - 2):sl - lo_lag + 3]
                    if wv.size and wv.max() >= 0.85 * best_h:
                        best_lag = sl; break
            t.append(tc); rate.append(60.0 * fs / best_lag); stg.append(float(best_h))
        else:
            t.append(tc); rate.append(np.nan); stg.append(0.0)
    return np.asarray(t), np.asarray(rate), np.asarray(stg)


def adaptive_breaths(x, fs, t_rate, rate_bpm, floor_bpm=SLOW_FLOOR_BPM, hi_bpm=480.0):
    """Rate-adaptive breath detection: candidate peaks on a wide respiratory bandpass, then greedily
    enforce the LOCAL minimum spacing (0.6 x local breath period) and a LOCAL amplitude gate (>= 30% of
    the sustained local envelope).  `t_rate` MUST share `x`'s time base.  Returns (peaks, ibi, peak_amp)."""
    lo = max(floor_bpm / 60.0, 0.4); hi = min(hi_bpm / 60.0, 0.45 * fs)
    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    xb = filtfilt(b, a, x - np.mean(x))
    env_s = uniform_filter1d(np.abs(hilbert(xb)), max(3, int(0.5 * fs)))
    fill = np.nanmedian(rate_bpm) if np.isfinite(rate_bpm).any() else floor_bpm
    ridx = np.clip(np.interp(np.arange(len(x)) / fs, t_rate,
                             np.nan_to_num(rate_bpm, nan=fill)), floor_bpm, hi_bpm)

    def cand(sig):
        amp = np.percentile(np.abs(sig), 75) + 1e-12
        p, _ = find_peaks(sig, distance=max(1, int(0.5 * 60.0 / hi_bpm * fs)), prominence=0.15 * amp)
        return p

    def reg(p):
        if p.size < 4:
            return 9.0
        d = np.diff(p) / fs; m = np.median(d)
        return float(np.median(np.abs(d - m)) / m) if m > 0 else 9.0

    pos, neg = cand(xb), cand(-xb)
    peaks = neg if reg(neg) < reg(pos) - 0.02 else pos
    kept, last = [], -1e9
    for p in peaks:
        if abs(xb[p]) < 0.30 * env_s[p]:                  # amplitude gate (reject in-pause noise)
            continue
        min_gap = 0.6 * (60.0 / ridx[p]) * fs             # local spacing from instantaneous rate
        if (p - last) >= min_gap:
            kept.append(p); last = p
        elif kept and abs(xb[p]) > abs(xb[kept[-1]]):     # dedupe a doubled detection -> keep larger
            kept[-1] = p; last = p
    pk = np.asarray(kept, int)
    ibi = np.diff(pk) / fs if pk.size >= 2 else np.array([])
    return pk, ibi, (np.abs(xb[pk]) if pk.size else np.array([]))
