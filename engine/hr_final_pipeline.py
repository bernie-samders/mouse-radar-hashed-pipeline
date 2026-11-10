"""final_pipeline.py -- FROZEN ECG-free mouse HR pipeline (2026-10-03; see FROZEN.md).

Per run: breath-order residual of the 330-voxel chest-patch acceleration -> B_spec band chooser (whitened cross-voxel
lam1 spectrogram, averaged over 6 s windows and over each tight band) -> polarity/delay-aware complex beam of that
band's 20 agreement picks -> 10 % trimmed mean of the beam's 1 s-median-filtered instantaneous frequency.
If any step fails on a run (exception / non-finite / too few samples) that run falls back to S1 (tight band with the
highest mean adjacent-beat correlation of its picks, read as the picks' pooled HR) and is counted in fallbacks().

ECG-free: uses only H.runs / H.all_runs, H.tight / H.tight_all, H.patch_accel, H.block, H.accel, H.patch_cols,
H.breath_rate, H.hr_fn, H.ibis_from_onsets. No H.wide, no tprior dump, no run times or per-mouse constants.
Self-contained: imports only harness (data access) + numpy/scipy. Every constant below carries its provenance.

  import final_pipeline as FP
  FP.estimate_runs(rec)               -> {round(t0, 2): HR bpm} for every run in H.runs(rec)
  FP.window_fns(rec)                  -> {round(t0, 2): f(a, b) -> HR bpm} (trimmed mean of the 10 Hz track in [a, b))
  FP.beat_onsets(rec)                 -> {round(t0, 2): beam beat onsets, absolute radar s}
  FP.fallbacks(rec)                   -> {round(t0, 2): reason} runs that fell back to S1
  FP.details(rec)                     -> {round(t0, 2): dict(method, band centre, ...)}
  which="all" (any of the above)      -> every clean-breathing run (H.all_runs + H.tight_all; unlabelled use)
"""
import os, sys, functools
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_k, "1")                       # before numpy: BLAS thrashes on the shared machine otherwise
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))               # .../hr_overnight (harness.py)
import numpy as np
import harness as H
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import median_filter
from scipy.stats import trim_mean

FS = H.FS                                                # 250 Hz voxel cube (pack constant)

# ---- frozen constants (provenance in brackets; "r1/r2/..." = overnight round; dev mice = 508 + 507d2) -------------
# Breath clock + order regression [a priori, round-1 ideation/machinery look_a + within_run_ridge; Hilbert-PC1 family
#   chosen over DP-onset clocks by the ECG-free leakage criterion in breath_clock_shootout (replicated)]
BREATH_BP = (0.55, 1.60)      # PC1 of patch amplitude band-passed to [0.55, 1.6] x RR [a priori]
BREATH_ORDER = 3              # Butterworth order of that band-pass [a priori]
ORDER_MAX_BPM = 2400.0        # regress breath orders k = 1 .. floor(2400 / RR) [a priori]
KNOT_S = 4.0                  # hat-spline gain knots (time-varying order amplitudes) [a priori; r3 knot 2/8 s pass]
RIDGE = 1e-10                 # relative Tikhonov term in the normal equations (numerical = lstsq) [a priori]
# Windowed cross-voxel lam1 spectrogram [a priori, within_run_ridge round 1]
WIN_S, HOP_S, SEG_S = 6.0, 1.0, 2.0   # 6 s windows, 1 s hop, 2 s Hann segments (5 per window) [a priori]
NFFT_MULT = 4                 # zero-padding factor (7.5 bpm bins) [a priori]
F_LO, F_HI = 350.0, 850.0     # spectrogram range, bpm [a priori]
# B_spec chooser [rule promoted post hoc in r2 (best of 11 rows scored on BOTH dev mice); robust in r3 (17/17
#   one-at-a-time perturbations beat S1 on both mice; replicated); largest weight in the r2 cross-mouse logit fitted on
#   each mouse separately]
CH = 1.08                     # band half-width for the band mean = the tight-band definition [a priori, method 2;
                              #   1.12 tested cross-mouse in r5, did not transfer]
# Beam [a priori, round-1 array ideation / within_run_ridge wrr.beam]
NPK = 20                      # agreement picks of the chosen band (all the tight scan keeps) [a priori]
BEAM_WID = 1.20               # Butterworth band-pass centre / 1.2 .. centre * 1.2 [a priori; 1.15/1.25 did not transfer]
BEAM_ORDER = 4                # [a priori]
FINST_MED_S = 1.0             # median filter of the instantaneous frequency [a priori]
OUT_FS = 10.0                 # output track rate [a priori]
# Readout [10 % two-sided trimmed mean = pre-declared r3 perturbation P17; ADOPTED because r5 cross-mouse axis-wise
#   selection picked it in BOTH directions: fit on 508 -> test 507d2 -0.73 bpm [-1.35, -0.06]; fit on 507d2 -> test 508
#   -1.09 [-2.17, -0.08] vs the median readout]
TRIM = 0.10
EDGE_S = 1.0                  # run readout ignores the first / last 1 s (filter edges) [a priori]
MIN_SAMPLES = 10              # fewer readout samples than this -> S1 fallback [a priori]
ACCEL_PAD_S = 1.0             # no-cache accel path: pad 1 s each side then trim (bit-matches the cached patch accel)


# ---- data ------------------------------------------------------------------------------------------------------------

def _run_accel(rec, r):
    """[T, 330] patch acceleration over [t0, t1): the pack cache if present, else recomputed from the live cube."""
    p = r.get("accel_patch")
    if p and os.path.exists(p):
        return np.asarray(H.patch_accel(rec, r), float)
    n = int(round((r["t1"] - r["t0"]) * FS))
    X = H.accel(H.block(rec, r["t0"] - ACCEL_PAD_S, r["t1"] + ACCEL_PAD_S, H.patch_cols(rec)))
    i0 = int(round(ACCEL_PAD_S * FS))
    return np.asarray(X[i0:i0 + n], float)


def _rr(rec, r):
    rr = r.get("rr")
    rr = float(rr) if rr is not None else np.nan
    if not np.isfinite(rr) or rr <= 0:
        rr = H.breath_rate(rec, r["t0"], r["t1"])
    if not np.isfinite(rr) or rr <= 0:
        raise ValueError("no breathing rate")
    return rr


# ---- breath-order residual -------------------------------------------------------------------------------------------

def breath_phase(rec, r, T, rr):
    """Unwrapped Hilbert phase of PC1 of the band-passed patch amplitude (continuous breath clock)."""
    amp = H.block(rec, r["t0"], r["t0"] + T / FS, H.patch_cols(rec))
    f0 = rr / 60.0
    sos = butter(BREATH_ORDER, [BREATH_BP[0] * f0, BREATH_BP[1] * f0], "bandpass", fs=FS, output="sos")
    x = sosfiltfilt(sos, amp - amp.mean(0), axis=0)
    x = x - x.mean(0)
    _, V = np.linalg.eigh(x.T @ x)                          # PC1 via the 330 x 330 Gram (= SVD PC1 up to sign)
    return np.unwrap(np.angle(hilbert(x @ V[:, -1])))


def harmonic_design(phi, kmax, knot_s=KNOT_S):
    T = phi.size; t = np.arange(T) / FS
    nk = max(2, int(np.ceil(t[-1] / knot_s)) + 1)
    cent = np.linspace(0, t[-1], nk); w = cent[1] - cent[0]
    G = np.maximum(0, 1 - np.abs(t[:, None] - cent[None]) / w)   # hat-spline (linear-interpolated) gains
    cols = [G]
    for k in range(1, kmax + 1):
        cols += [G * np.cos(k * phi)[:, None], G * np.sin(k * phi)[:, None]]
    return np.concatenate(cols, 1)


def residual(X, phi, rr):
    """X minus its time-varying breath-order (k * phi) fit, every voxel."""
    D = harmonic_design(phi, int(ORDER_MAX_BPM // rr), KNOT_S)
    G = D.T @ D
    return X - D @ np.linalg.solve(G + RIDGE * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), D.T @ X)


# ---- B_spec chooser --------------------------------------------------------------------------------------------------

def lam_spectrogram(Y):
    """Per 6 s window (1 s hop): lam1(f) = top eigenvalue of the [5 segment x 330 voxel] FFT Gram (z-scored voxels)."""
    nper = int(SEG_S * FS); hop = int(HOP_S * FS); nfft = NFFT_MULT * nper
    A = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    f = np.fft.rfftfreq(nfft, 1 / FS) * 60
    sel = np.where((f >= F_LO) & (f <= F_HI))[0]
    win = np.hanning(nper)[:, None]
    starts = np.arange(0, A.shape[0] - nper + 1, hop)
    S = np.stack([np.fft.rfft(A[s:s + nper] * win, n=nfft, axis=0)[sel] for s in starts])   # [nseg, nb, nv]
    nps = int(round((WIN_S - SEG_S) / HOP_S)) + 1
    nw = len(starts) - nps + 1
    if nw < 1:
        raise ValueError("run shorter than one spectrogram window")
    L = np.empty((nw, sel.size))
    for w in range(nw):
        Yw = S[w:w + nps]
        G = np.einsum("anv,bnv->nab", Yw, Yw.conj())
        L[w] = np.linalg.eigvalsh(G)[:, -1]
    return f[sel], L


def whiten(f, L):
    """Per window: log lam1 minus its straight-line fit in log-log."""
    lf = np.log(f); out = np.empty_like(L)
    for w in range(L.shape[0]):
        ly = np.log(L[w]); out[w] = ly - np.polyval(np.polyfit(lf, ly, 1), lf)
    return out


def choose_bspec(f, E, centres):
    """argmax over tight-band centres of the window-mean whitened lam1 averaged over centre/CH .. centre*CH."""
    m = E.mean(0); B = []
    for c in centres:
        s = (f >= c / CH) & (f <= c * CH)
        B.append(float(np.interp(c, f, m)) if s.sum() == 0 else float(m[s].mean()))
    return int(np.argmax(B)), np.asarray(B)


# ---- beam readout ----------------------------------------------------------------------------------------------------

def beam_track(Y, picks, c0):
    """Complex beam (top eigenvector of the picks' analytic Gram; sign/delay-free) -> 1 s-median f_inst (bpm)."""
    c = c0 / 60.0
    sos = butter(BEAM_ORDER, [c / BEAM_WID, c * BEAM_WID], "bandpass", fs=FS, output="sos")
    Z = hilbert(sosfiltfilt(sos, Y[:, picks], axis=0), axis=0)
    Z /= np.std(np.abs(Z), 0) + 1e-30
    _, V = np.linalg.eigh(Z.conj().T @ Z)
    b = Z @ V[:, -1]
    fi = np.diff(np.unwrap(np.angle(b))) * FS / (2 * np.pi) * 60
    return b, median_filter(np.r_[fi, fi[-1]], size=int(FINST_MED_S * FS), mode="nearest")


def crossings(t, theta):
    """Times where the unwrapped phase crosses a multiple of 2 pi upward (beam beat onsets)."""
    k = np.floor(theta / (2 * np.pi)); i = np.where(np.diff(k) > 0)[0]
    tgt = (k[i] + 1) * 2 * np.pi
    return t[i] + (tgt - theta[i]) / (theta[i + 1] - theta[i]) * (t[i + 1] - t[i])


# ---- S1 fallback -----------------------------------------------------------------------------------------------------

def s1_band(bands):
    s1 = [np.nanmean(np.asarray(b["r1"])[np.asarray(b["picks"])]) for b in bands]
    return int(np.nanargmax(s1))


def _s1_result(r, bands, reason):
    j = s1_band(bands); b = bands[j]
    hr = float(b["cons"])
    fn = None
    if "on" in b:
        try:
            fn = H.hr_fn(*zip(*[H.ibis_from_onsets(b["on"][i]) for i in b["picks"]]))
        except Exception:                                    # noqa: BLE001 -- window fn is optional
            fn = None
    if fn is None:
        fn = (lambda a, bb, _h=hr: _h)
    on = np.sort(np.concatenate([np.asarray(b["on"][i], float) for i in b["picks"]])) if "on" in b else np.array([])
    return dict(method="S1_fallback", reason=reason, band=j, centre=float(b["centre"]), hr=hr, wfn=fn, onsets=on,
                t=None, trk=None)


# ---- one run ---------------------------------------------------------------------------------------------------------

def process_run(rec, r, bands):
    """ECG-free estimate for one run; falls back to S1 on any failure (reason recorded)."""
    try:
        rr = _rr(rec, r)
        X = _run_accel(rec, r); T = X.shape[0]
        Y = residual(X, breath_phase(rec, r, T, rr), rr)
        f, L = lam_spectrogram(Y)
        C = np.array([b["centre"] for b in bands], float)
        j, B = choose_bspec(f, whiten(f, L), C)
        b, fi = beam_track(Y, np.asarray(bands[j]["picks"])[:NPK], float(C[j]))
        t_full = np.arange(T) / FS; t_out = np.arange(0, T / FS, 1 / OUT_FS)
        trk = np.interp(t_out, t_full, fi).astype(np.float32).astype(float)
        t = r["t0"] + t_out
        m = (t >= r["t0"] + EDGE_S) & (t < r["t1"] - EDGE_S)
        if m.sum() < MIN_SAMPLES or not np.all(np.isfinite(trk[m])):
            return _s1_result(r, bands, "too few / non-finite readout samples")
        hr = float(trim_mean(trk[m], TRIM))
        if not np.isfinite(hr):
            return _s1_result(r, bands, "non-finite estimate")

        def wfn(a, bb, _t=t, _k=trk):
            s = (_t >= a) & (_t < bb)
            return float(trim_mean(_k[s], TRIM)) if s.any() else np.nan
        return dict(method="bspec_beam_tmean", reason="", band=j, centre=float(C[j]), hr=hr, wfn=wfn,
                    onsets=r["t0"] + crossings(t_full, np.unwrap(np.angle(b))), t=t, trk=trk, rr=rr,
                    bscore=B.tolist())
    except Exception as e:                                   # noqa: BLE001 -- any failure -> counted S1 fallback
        return _s1_result(r, bands, f"{type(e).__name__}: {e}")


# ---- recording level -------------------------------------------------------------------------------------------------

def _band_map(rec, which):
    """{round(t0, 2): bands} from the tight-band scan (scored set first, then the all-runs scan)."""
    srcs = (H.tight, H.tight_all) if which == "scored" else (H.tight_all,)
    out = {}
    for src in srcs:
        try:
            for x in src(rec):
                out.setdefault(round(float(x["t0"]), 2), x["bands"])
        except (FileNotFoundError, KeyError, OSError):
            continue
    return out


def _run_list(rec, which):
    if which == "scored":
        return [dict(r) for r in H.runs(rec)]
    return [dict(t0=float(x["t0"]), t1=float(x["t1"]), rr=x.get("rr"), accel_patch=x.get("accel_patch"))
            for x in H.all_runs(rec)]


@functools.lru_cache(None)
def _results(rec, which="scored"):
    BM = _band_map(rec, which); out = {}
    for r in _run_list(rec, which):
        k = round(r["t0"], 2)
        bands = BM.get(k)
        if bands is None:                                     # tolerate t0 rounding differences
            near = [kk for kk in BM if abs(kk - r["t0"]) < 0.02]
            bands = BM[near[0]] if near else None
        if bands is None:
            out[k] = dict(method="none", reason="no tight-band scan for this run", band=-1, centre=np.nan, hr=np.nan,
                          wfn=(lambda a, b: np.nan), onsets=np.array([]), t=None, trk=None)
            continue
        out[k] = process_run(rec, r, bands)
    return out


def estimate_runs(rec, which="scored"):
    return {k: v["hr"] for k, v in _results(rec, which).items()}


def window_fns(rec, which="scored"):
    return {k: v["wfn"] for k, v in _results(rec, which).items()}


def beat_onsets(rec, which="scored"):
    return {k: v["onsets"] for k, v in _results(rec, which).items()}


def fallbacks(rec, which="scored"):
    return {k: v["reason"] for k, v in _results(rec, which).items() if v["method"] != "bspec_beam_tmean"}


def details(rec, which="scored"):
    return {k: {kk: vv for kk, vv in v.items() if kk not in ("wfn", "t", "trk", "onsets")}
            for k, v in _results(rec, which).items()}


if __name__ == "__main__":
    for rec in H.RECS:
        e = estimate_runs(rec)
        print(rec, f"{len(e)} runs, S1 fallbacks {len(fallbacks(rec))}; first 3:",
              {k: round(v, 1) for k, v in list(e.items())[:3]})
