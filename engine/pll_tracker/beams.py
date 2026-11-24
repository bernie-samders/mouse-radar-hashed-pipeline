"""beams.py -- arm C: ECG-free per-period multi-voxel beams per heart harmonic for ng511 (and any bh-style harness).

    beams(p, centre, K=5, Y=None, cfg=None) -> complex [K, T]   analytic beam around k * centre (bpm), k = 1..K
                                                                    (cfg None = FROZEN recipe; see RESULT.md)
    weights(p, centre, K, Y, cfg) -> (W, S)                        W[k] = (edges [nf+1] samples, w [nf, V] complex)
                                                                    S = z-scored Y the weights apply to (y = S @ w.conj())
Spatial weights are estimated per period from the radar only (no ECG). Spectral quantities via one zero-padded FFT of
the z-scored [T, V] block (Parseval: band covariance R = sum_f m(f)^2 F(f) F(f)^H, no filtering needed).
  ref = "none"  : top eigenvector of the in-band Gram of per-voxel |Z|-normalised analytic signals (= frozen beam_track
                  generalised to harmonic k; sel="picks" uses the frozen 20 agreement picks)
  ref = "ring"  : GEV  R_in w = l (R_ref + lam I) w, R_in = band k*c*exp(+-din) (cos taper dtap), R_ref = ring
                  din + dtap < |ln f/(k c)| < dout (noise spatial covariance next to the line); lam = Marchenko-Pastur-scaled
                  ridge  lamf * tr(R_ref)/V * clip(V / Neff_ref, 0.05, 10)  (Neff = ENBW * duration, complex dof)
  ref = "broad" : as ring with R_ref = 4..100 Hz
  sel = "topM"  : restrict to the M voxels with the largest in-band / ring power ratio summed over k = 1..K (log)
  hk = "k1sel"   : k >= 2 restricted to the Mk voxels with the largest k = 1 heart-to-noise |a1|^2 / B_vv
  hk = "pickeig" : k >= 2 = frozen 20 picks, top eigvec;  choose = [cfgs]: per harmonic the candidate with the largest
                  time-cross-fitted excess in-band power (FROZEN = choose between k1sel5_lamk3 and pickeig)
  dd = n passes : decision-directed MRC (time-fold cross-fitted, nfold) of the in-band analytic signal against u^k,
                  u = unit phasor of the k = 1 beam of the previous pass (hph.py idea); weights per fold.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
NG = os.path.dirname(os.path.dirname(HERE))
if "bh" not in sys.modules:
    sys.path.insert(0, os.path.join(NG, "harness"))
import numpy as np
from scipy.fft import next_fast_len
import bh

FS = bh.FS
PAD_S = 2.0
F_BROAD = (4.0, 100.0)

BASE = dict(sel="all", M=20, ref="ring", din=float(np.log(1.10)), dtap=float(np.log(1.04)), dout=float(np.log(1.45)),
            lamf=1.0, dd=0, nfold=4, reg=0.1, bw_out=float(np.log(1.20)), contrast="sum", ddsrc="self")


def zscore(Y):
    Y = np.asarray(Y, float)
    return (Y - Y.mean(0)) / (Y.std(0) + 1e-12)


def spectrum(S):
    T = S.shape[0]; P = int(PAD_S * FS); n = next_fast_len(T + 2 * P)
    A = np.zeros((n, S.shape[1])); A[P:P + T] = S
    F = np.fft.rfft(A, axis=0)
    return F, np.fft.rfftfreq(n, 1.0 / FS), n, P


def taper(x, a, b):
    """1 for x <= a, raised-cosine to 0 at x = b (x >= 0)."""
    out = np.zeros_like(x)
    out[x <= a] = 1.0
    m = (x > a) & (x < b)
    out[m] = 0.5 * (1 + np.cos(np.pi * (x[m] - a) / (b - a)))
    return out


def masks(fr, fk, cfg):
    x = np.full(fr.size, np.inf); pos = fr > 0
    x[pos] = np.abs(np.log(fr[pos] / fk))
    m_in = taper(x, cfg["din"], cfg["din"] + cfg["dtap"])
    if cfg["ref"] == "broad":
        m_ref = ((fr >= F_BROAD[0]) & (fr <= F_BROAD[1])).astype(float)
    else:
        m_ref = ((x > cfg["din"] + cfg["dtap"]) & (x < cfg["dout"]) & (fr < 0.48 * FS)).astype(float)
        if cfg.get("ringcap"):                                          # stay between the neighbouring harmonics
            k = fk / cfg["_fc"]
            m_ref *= ((fr > (k - 0.5) * cfg["_fc"]) & (fr < (k + 0.5) * cfg["_fc"]))
    return m_in, m_ref


def band_cov(F, m, cols=None):
    i = np.flatnonzero(m > 0)
    G = F[i] if cols is None else F[i][:, cols]
    Gm = G * m[i][:, None]
    return Gm.T @ Gm.conj(), float((m[i] ** 2).sum())            # R[a, b] = sum_f m^2 F_a F_b^*  (scale-free use)


def analytic(F, fr, n, P, T, fk, half):
    """Analytic band (cos-tapered +-half in log f, 4 % taper) of a 1-channel spectrum F -> [T] complex."""
    x = np.full(fr.size, np.inf); pos = fr > 0
    x[pos] = np.abs(np.log(fr[pos] / fk))
    h = taper(x, half, half + float(np.log(1.04)))
    full = np.zeros(n, complex); full[:fr.size] = 2 * F * h
    full[0] = 0
    return np.fft.ifft(full)[P:P + T]


def _gev_top(Rin, Rref, lam):
    from scipy.linalg import eigh
    V = Rin.shape[0]
    B = Rref + lam * np.eye(V)
    l, U = eigh(Rin, B)
    return U[:, -1], l[-1]


def track_phase(S, F, fr, n, P, fc, w1, cfg):
    """Slow HR track from the k = 1 beam: f_inst median over track_s, clipped to fc*[1/1.2, 1.2] -> Phi(t) (rad)."""
    from scipy.ndimage import median_filter
    T = S.shape[0]
    z = analytic(F @ w1.conj(), fr, n, P, T, fc, cfg["bw_out"])
    fi = np.diff(np.unwrap(np.angle(z))) * FS / (2 * np.pi)
    fi = median_filter(np.r_[fi, fi[-1]], size=int(cfg["track_s"] * FS), mode="nearest")
    fi = np.clip(fi, fc / 1.2, fc * 1.2)
    return 2 * np.pi * np.cumsum(fi) / FS, float(np.mean(fi))


def tracked_cov(S, Phi, k, cbar, cfg, cols):
    """In-band / ring covariances of S demodulated by k Phi(t) (order tracking on the slow HR)."""
    T = S.shape[0]; Pp = int(PAD_S * FS); n = next_fast_len(T + 2 * Pp)
    D = np.zeros((n, len(cols)), complex)
    D[Pp:Pp + T] = S[:, cols] * np.exp(-1j * k * Phi)[:, None]
    G = np.fft.fft(D, axis=0)
    nu = np.fft.fftfreq(n, 1.0 / FS)
    f = k * cbar + nu
    x = np.full(n, np.inf); pos = f > 0
    x[pos] = np.abs(np.log(f[pos] / (k * cbar)))
    mi = taper(x, cfg["din"], cfg["din"] + cfg["dtap"])
    mr = ((x > cfg["din"] + cfg["dtap"]) & (x < cfg["dout"]) & (f < 0.48 * FS)).astype(float)
    if cfg.get("ringcap"):
        mr *= ((f > (k - 0.5) * cbar) & (f < (k + 0.5) * cbar))
    Rin, ein = band_cov(G, mi)
    Rr, er = band_cov(G, mr)
    return Rin, ein, Rr, er, FS / n


_PRIOR = {}


def _prior(cfg):
    """[K, V] steering prior: cfg["_prior"] (diagnostic injection) or fit/steer_fit.npz (fit_split, FIT-split ECG)."""
    if cfg.get("_prior") is not None:
        return cfg["_prior"]
    fn = cfg.get("prior_file", os.path.join(HERE, "fit", "steer_fit.npz"))
    if fn not in _PRIOR:
        _PRIOR[fn] = np.load(fn)["A"]
    return _PRIOR[fn]


def weights(p, centre, K=5, Y=None, cfg=None):
    """Per-harmonic spatial weights (cfg None = the FROZEN recipe, variants.FROZEN) and the z-scored S they apply to."""
    if cfg is None:
        import variants
        cfg = variants.FROZEN
    cfg = dict(BASE, **cfg)
    S = zscore(bh.Y(p) if Y is None else Y)
    if cfg.get("choose"):
        return _choose(p, centre, K, S, cfg), S
    return _core(p, centre, K, S, cfg), S


def contrast(Sh, w, fc, k, cfg):
    """Excess in-band power of the beam S_h @ w.conj(): P_in / P_ring - 1 (ECG-free per-harmonic SNR proxy)."""
    F, fr, n, P = spectrum(Sh)
    c = dict(BASE, **cfg); c["_fc"] = fc; c["ref"] = "ring"
    mi, mr = masks(fr, k * fc, c)
    b = np.abs(F @ w.conj()) ** 2
    return float((b * mi ** 2).sum() / (mi ** 2).sum() / ((b * mr).sum() / max(mr.sum(), 1)) - 1.0)


def _choose(p, centre, K, S, cfg):
    """Per harmonic, the candidate beam with the largest TIME-CROSS-FITTED contrast (weights fitted on one half of the
    period, contrast measured on the other half, both ways, mean)."""
    T = S.shape[0]; h = T // 2; fc = centre / 60.0
    cands = cfg["choose"]
    full = [_core(p, centre, K, S, dict(BASE, **c)) for c in cands]
    score = np.zeros((len(cands), K))
    for a, b in ((slice(0, h), slice(h, T)), (slice(h, T), slice(0, h))):
        for i, c in enumerate(cands):
            Wh = _core(p, centre, K, S[a], dict(BASE, **c))
            for k in range(1, K + 1):
                score[i, k - 1] += 0.5 * contrast(S[b], Wh[k - 1][1][0], fc, k, c)
    pick = np.argmax(score, 0)
    _LAST.update(score=score, pick=pick)
    return [full[pick[k]][k] for k in range(K)]


_LAST = {}


def _core(p, centre, K, S, cfg):
    cfg = dict(cfg)
    T, V = S.shape
    F, fr, n, P = spectrum(S)
    fc = centre / 60.0
    cfg["_fc"] = fc
    dur = T / FS
    df = fr[1] - fr[0]
    MS = [masks(fr, k * fc, cfg) for k in range(1, K + 1)]
    # voxel subset
    if cfg["sel"] == "picks":
        cols = np.asarray(bh.picks(p))[:20]
    elif cfg["sel"] == "topM":
        sc = np.zeros(V)
        ks = range(K) if cfg["contrast"] == "sum" else [0]
        for k in ks:
            mi, mr = MS[k]
            Pin = (np.abs(F) ** 2 * (mi ** 2)[:, None]).sum(0) / (mi ** 2).sum()
            Pr = (np.abs(F) ** 2 * mr[:, None]).sum(0) / max(mr.sum(), 1)
            sc += np.log(Pin / (Pr + 1e-30))
        cols = np.argsort(-sc)[:cfg["M"]]
    else:
        cols = np.arange(V)
    cols = np.sort(cols)
    W = []
    allc = cols
    a1 = None; a1c = None
    for k in range(1, K + 1):
        mi, mr = MS[k - 1]
        hk = cfg.get("hk", "gev") if k > 1 else ("prior" if cfg.get("prior_k1") else "gev")
        if hk == "k1sel" and a1 is not None:                            # voxels where the k = 1 beam sees the heart
            sc1 = np.log(a1 + 1e-30)
            if cfg.get("selmix", 0) > 0:                                # + this harmonic's in-band / ring contrast
                Pin = (np.abs(F[:, allc]) ** 2 * (mi ** 2)[:, None]).sum(0) / (mi ** 2).sum()
                Pr = (np.abs(F[:, allc]) ** 2 * mr[:, None]).sum(0) / max(mr.sum(), 1)
                sc1 = sc1 + cfg["selmix"] * np.log(Pin / (Pr + 1e-30))
            cols = allc[np.argsort(-sc1)[:cfg.get("Mk", 20)]]
            cols = np.sort(cols)
        elif hk == "pickeig":                                           # frozen 20 agreement picks, top eigvec
            cols = np.sort(np.asarray(bh.picks(p))[:20])
        else:
            cols = allc
        trk = cfg.get("track_s", 0) > 0 and k > 1 and cfg["ref"] != "none"
        if trk:
            if k == 2:
                Phi, cbar = track_phase(S, F, fr, n, P, fc, W[0][1][0], cfg)
            Rin, ein, Rr_t, er_t, df_t = tracked_cov(S, Phi, k, cbar, cfg, cols)
        else:
            Rin, ein = band_cov(F, mi, cols)
        if cfg["ref"] == "none" or hk == "pickeig":
            s = np.sqrt(np.real(np.diag(Rin)) / ein) + 1e-30             # per-voxel normalisation (frozen beam)
            _, U = np.linalg.eigh(Rin / np.outer(s, s))
            w = U[:, -1] / s
        else:
            if mr.sum() < 8:                                            # empty ring (capped, high k) -> broad
                mr = ((fr >= F_BROAD[0]) & (fr <= F_BROAD[1])).astype(float)
            if trk:
                Rr, er = Rr_t, er_t
            else:
                Rr, er = band_cov(F, mr, cols)
            Rin = Rin / ein; Rr = Rr / er
            Neff = max(er * (df_t if trk else df) * dur, 1.0)
            Vc = len(cols)
            lam = (cfg["lamf"] if k == 1 else cfg.get("lamk", cfg["lamf"])) * np.real(np.trace(Rr)) / Vc *                 np.clip(Vc / Neff, 0.05, 10)
            Bm = Rr + lam * np.eye(Vc)
            if hk == "prior" or (k == 1 and cfg.get("prior_k1")):     # fit_split session steering, per-period whitening
                ak = _prior(cfg)[k - 1][cols]
                Bp = (Rin if cfg.get("prior_R", "ring") == "in" else Rr)
                Bp = Bp + lam * np.eye(Vc)
                w = np.linalg.solve(Bp, ak)
            elif hk == "couple" and k > 1 and a1c is not None:            # delay model: a_k,v = |a_1,v| e^{i k arg a_1,v}
                ak = np.abs(a1c) * np.exp(1j * k * np.angle(a1c))
                w = np.linalg.solve(Bm, ak)
            else:
                w, _ = _gev_top(Rin, Rr, lam)
            if k == 1 and hk == "gev":
                st = Bm @ w                                             # steering (rank-1 model: R_in = B + s a a^H)
                a1c = st
                a1 = np.abs(st) ** 2 / np.real(np.diag(Bm))             # per-voxel heart-to-noise (k = 1)
        wf = np.zeros(V, complex); wf[cols] = w                         # y = S @ wf.conj()
        W.append((np.array([0, T]), wf[None, :]))
    cols = allc
    if cfg.get("comb"):                                                 # one REAL (waveform) beam shared by harmonics
        cks = [k for k in range(1, K + 1) if k >= cfg.get("comb_k0", 1)]
        cc = allc if not (cfg.get("comb_sel") and a1 is not None) else np.sort(allc[np.argsort(-a1)[:cfg["comb_sel"]]])
        Ri = np.zeros((len(cc), len(cc))); Rq = np.zeros_like(Ri); Ne = 0.0
        for k in cks:
            mi, mr = MS[k - 1]
            if mr.sum() < 8:
                mr = ((fr >= F_BROAD[0]) & (fr <= F_BROAD[1])).astype(float)
            a_, e_ = band_cov(F, mi, cc); b_, f_ = band_cov(F, mr, cc)
            g = k ** 2 if cfg.get("comb_w") == "k2" else 1.0
            Ri += g * np.real(a_) / e_; Rq += g * np.real(b_) / f_; Ne += f_ * df * dur
        Vc = len(cc)
        lam = cfg["lamf"] * np.trace(Rq) / Vc * np.clip(Vc / max(Ne, 1.0), 0.05, 10)
        w, _ = _gev_top(Ri, Rq, lam)
        wf = np.zeros(V, complex); wf[cc] = w
        for k in cks:
            W[k - 1] = (np.array([0, T]), wf[None, :])
    if cfg["dd"]:
        W = _dd(S, F, fr, n, P, fc, K, W, cfg, cols)
    return W


def _beam_from(S, F, fr, n, P, fk, Wk, half):
    T = S.shape[0]; edges, w = Wk
    out = np.empty(T, complex)
    for i in range(w.shape[0]):
        Fb = F @ w[i].conj()
        z = analytic(Fb, fr, n, P, T, fk, half)
        out[edges[i]:edges[i + 1]] = z[edges[i]:edges[i + 1]]
    return out


def _dd(S, F, fr, n, P, fc, K, W, cfg, cols):
    """Decision-directed, time-fold cross-fitted MRC per harmonic against u^k (u from the k = 1 beam)."""
    T = S.shape[0]; nf = cfg["nfold"]
    e = np.linspace(0, T, nf + 1).astype(int)
    for _ in range(cfg["dd"]):
        z1 = _beam_from(S, F, fr, n, P, fc, W[0], cfg["bw_out"])
        u = z1 / (np.abs(z1) + 1e-30)
        Wn = []
        for k in range(1, K + 1):
            Fs = np.zeros((n, len(cols)), complex)
            x = np.full(fr.size, np.inf); pos = fr > 0; x[pos] = np.abs(np.log(fr[pos] / (k * fc)))
            h = taper(x, cfg["din"], cfg["din"] + cfg["dtap"])
            Fs[:fr.size] = 2 * F[:, cols] * h[:, None]
            Z = np.fft.ifft(Fs, axis=0)[P:P + T]
            uk = u ** k
            G = [Z[a:b].conj().T @ Z[a:b] for a, b in zip(e[:-1], e[1:])]
            cz = [Z[a:b].T @ uk[a:b].conj() for a, b in zip(e[:-1], e[1:])]
            Gt, Ct = sum(G), sum(cz)
            w = np.zeros((nf, S.shape[1]), complex)
            for i, (a, b) in enumerate(zip(e[:-1], e[1:])):
                m = T - (b - a); g = (Ct - cz[i]) / m; R = (Gt - G[i]) / m
                R = R - np.outer(g.conj(), g)
                R = R + cfg["reg"] * np.trace(R).real / R.shape[0] * np.eye(R.shape[0])
                ww = np.linalg.solve(R, g).conj()                       # beam = Z @ ww  -> S-space weight conj
                w[i, cols] = ww.conj()
            Wn.append((e, w))
        W = Wn
    return W


def beams(p, centre, K=5, Y=None, cfg=None):
    """complex [K, T]: analytic beam per harmonic k (band k*centre*exp(+-bw_out)), ECG-free per-period weights.
    cfg None = the FROZEN recipe (variants.FROZEN = choose_k1sel5L3_pickeig); Y = a front-end [T, 330] (default bh.Y)."""
    if cfg is None:
        import variants
        cfg = variants.FROZEN
    cfg = dict(BASE, **cfg)
    W, S = weights(p, centre, K, Y, cfg)
    F, fr, n, P = spectrum(S)
    fc = centre / 60.0
    return np.stack([_beam_from(S, F, fr, n, P, k * fc, W[k - 1], cfg["bw_out"]) for k in range(1, K + 1)])


def last_info():
    """After a beams()/weights() call with cfg["choose"]: dict(score [n_cand, K] cross-fitted excess in-band power of each
    candidate beam (ECG-free per-harmonic SNR proxy), pick [K] chosen candidate per harmonic)."""
    return dict(_LAST)
