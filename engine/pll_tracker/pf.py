"""pf.py -- arm pll: particle-filter version of the harmonic phase tracker (ECG-free), fixed-lag smoothed.

    track(Z, centre, q1, cfg) -> dict(u, Pu posterior mean / var of log f - log fc - mu at the decimated rate, mu, A, ...)
    curve(Z, centre, q1, t0, cfg) -> (est entry dict(t=abs s, ibi=s), info)

Same model as pll.py (phase phi, log-f OU state u; z_k = A_k e^{ik phi} + band-limited noise), but the posterior is
carried by N particles, so the phase may be multimodal and every harmonic is evaluated EXACTLY at each particle's phase
(no linearisation; harmonics add evidence only where a particle's phase explains them). Systematic resampling when
ESS < N/2; fixed-lag smoothing over `lag_s` via per-particle ring buffers (copied on resampling). Deterministic: the
numba RNG is seeded per period (seed = 1 + period length in samples), so blind reruns are bit-identical.
Likelihood per decimated sample: sum_k 2 Re(conj(A_k e^{ik phi}) z_k) / (c_k s_k^2), c_k = fs_dec / ENBW_k (the
in-band noise is coloured; this keeps the information per second right). A_1 real, from the beams' cross-fitted k1
contrast (ECG-free, independent of the tracker); A_k (k >= 2) by EM against the smoothed phase posterior of a run
without harmonic k (leave-one-harmonic-out), regression-debiased, capped.
"""
import numpy as np
from numba import njit
import pll as PL

FS = 250.0

BASE = dict(K=5, ks=(1, 2, 3, 4, 5), sig_s=0.11, L_beats=5.0, sig_w=0.06, dec=5, N=1024, lag_s=1.0,
            qscale=1.0, em_iter=1, loho=True, mu_em=True, alpha=1.0, ckf=1.0, zgate=1.0, acap=0.5, outdec=2,
            a1_floor=0.0, k1=True, seed=1, dyn="ou", P_beats=23.0, pjump=0.0, sjump=4.0, nenv_s=0.0, nenv_b=1.0, bidir=0, awin_s=0.0, a1ml=None, half1=0.0, bandgain=False, nonover=False, lo1=0.0, nrep=1)


@njit(cache=True)
def _pf(zr, zi, ks, Ar, Ai, Rk, fc, mu, G, Lq, L0, sq_phi, dt, N, lag, seed, pj, sj, bh_, bt_):
    """Forward PF with fixed-lag smoothing. Returns smoothed mean/var of u and E[e^{-i k phi}] per step."""
    np.random.seed(seed)
    K, T = zr.shape
    um = np.zeros(T); uv = np.zeros(T)
    er = np.zeros((K, T)); ei = np.zeros((K, T))
    phi = np.random.uniform(0.0, 2 * np.pi, N)
    u = np.zeros(N); v = np.zeros(N)
    for i in range(N):
        e1 = np.random.normal(); e2 = np.random.normal()
        u[i] = L0[0, 0] * e1; v[i] = L0[1, 0] * e1 + L0[1, 1] * e2
    lw = np.zeros(N)
    w = np.ones(N) / N
    Ub = np.zeros((N, lag)); Pb = np.zeros((N, lag))
    Ub2 = np.zeros((N, lag)); Pb2 = np.zeros((N, lag))
    idx = np.zeros(N, np.int64)
    logZ = 0.0
    for n in range(T):
        if n > 0:
            for i in range(N):
                phi[i] += 2 * np.pi * fc * dt * np.exp(mu + u[i]) + sq_phi * np.random.normal()
                e1 = np.random.normal(); e2 = np.random.normal()
                if pj > 0.0:
                    if np.random.uniform(0.0, 1.0) < pj:            # heavy-tailed transition: rare larger jump in u
                        e1 = e1 * sj
                un = G[0, 0] * u[i] + G[0, 1] * v[i] + Lq[0, 0] * e1
                vn = G[1, 0] * u[i] + G[1, 1] * v[i] + Lq[1, 0] * e1 + Lq[1, 1] * e2
                u[i] = un; v[i] = vn
        for i in range(N):
            ll = 0.0
            x = abs(mu + u[i])                                            # |ln f / fc|: beam band response (log taper)
            for k in range(K):
                if Rk[k, n] <= 0.0:
                    continue
                kk = ks[k]
                gk = 1.0
                if bh_[k] > 0.0:
                    if x >= bh_[k] + bt_:
                        gk = 0.0
                    elif x > bh_[k]:
                        gk = 0.5 * (1.0 + np.cos(np.pi * (x - bh_[k]) / bt_))
                    if gk == 0.0:
                        continue
                c = np.cos(kk * phi[i]); s = np.sin(kk * phi[i])
                hr = gk * (Ar[k, n] * c - Ai[k, n] * s); hi = gk * (Ar[k, n] * s + Ai[k, n] * c)
                ll += 2.0 * (hr * zr[k, n] + hi * zi[k, n]) / Rk[k, n]
                if bh_[k] > 0.0:                                          # -|h|^2 / R varies with the band gain
                    ll -= (hr * hr + hi * hi) / Rk[k, n]
            lw[i] = np.log(w[i] + 1e-300) + ll
        m = lw.max()
        tot = 0.0
        for i in range(N):
            w[i] = np.exp(lw[i] - m); tot += w[i]
        logZ += m + np.log(tot)
        for i in range(N):
            w[i] /= tot
        b = n % lag
        for i in range(N):
            Ub[i, b] = u[i]; Pb[i, b] = phi[i]
        # smoothed output for step n - lag + 1 (oldest slot in the ring)
        mo = n - lag + 1
        if mo >= 0:
            bo = mo % lag
            s1 = 0.0; s2 = 0.0
            for i in range(N):
                s1 += w[i] * Ub[i, bo]; s2 += w[i] * Ub[i, bo] ** 2
            um[mo] = s1; uv[mo] = max(s2 - s1 * s1, 0.0)
            for k in range(K):
                a = 0.0; bb = 0.0
                for i in range(N):
                    a += w[i] * np.cos(ks[k] * Pb[i, bo]); bb -= w[i] * np.sin(ks[k] * Pb[i, bo])
                er[k, mo] = a; ei[k, mo] = bb
        # resample
        ess = 0.0
        for i in range(N):
            ess += w[i] * w[i]
        ess = 1.0 / ess
        if ess < 0.5 * N:
            cs = 0.0; j = 0; u0 = np.random.uniform(0.0, 1.0 / N)
            for i in range(N):
                pos = u0 + i / N
                while j < N - 1 and cs + w[j] < pos:
                    cs += w[j]; j += 1
                idx[i] = j
            for i in range(N):
                for q in range(lag):
                    Ub2[i, q] = Ub[idx[i], q]; Pb2[i, q] = Pb[idx[i], q]
            for i in range(N):
                for q in range(lag):
                    Ub[i, q] = Ub2[i, q]; Pb[i, q] = Pb2[i, q]
            ph2 = phi[idx].copy(); u2 = u[idx].copy(); v2 = v[idx].copy()
            for i in range(N):
                phi[i] = ph2[i]; u[i] = u2[i]; v[i] = v2[i]; w[i] = 1.0 / N
    # flush the last lag - 1 steps with the final weights
    for mo in range(max(0, T - lag + 1), T):
        bo = mo % lag
        s1 = 0.0; s2 = 0.0
        for i in range(N):
            s1 += w[i] * Ub[i, bo]; s2 += w[i] * Ub[i, bo] ** 2
        um[mo] = s1; uv[mo] = max(s2 - s1 * s1, 0.0)
        for k in range(K):
            a = 0.0; bb = 0.0
            for i in range(N):
                a += w[i] * np.cos(ks[k] * Pb[i, bo]); bb -= w[i] * np.sin(ks[k] * Pb[i, bo])
            er[k, mo] = a; ei[k, mo] = bb
    return um, uv, er, ei, logZ


def _dyn(cfg, fc, dt):
    """(G, chol Q, chol P0) of the (u, v) process at step dt: OU (v unused) or damped oscillator (pll.dyn_mats)."""
    Tb = 1.0 / fc
    if cfg.get("dyn", "ou") == "ou":
        tau = cfg["L_beats"] * Tb; a = np.exp(-dt / tau)
        G = np.array([[a, 0.0], [0.0, 0.0]])
        Lq = np.array([[cfg["sig_s"] * np.sqrt(1 - a * a), 0.0], [0.0, 0.0]])
        L0 = np.array([[cfg["sig_s"], 0.0], [0.0, 0.0]])
        return G, Lq, L0
    from scipy.linalg import expm
    L = cfg["L_beats"] * Tb; P = cfg["P_beats"] * Tb
    zw = 1.0 / L; wd = 2 * np.pi / P; w0 = np.sqrt(wd ** 2 + zw ** 2)
    Ac = np.array([[0.0, 1.0], [-w0 ** 2, -2 * zw]])
    q = cfg["sig_s"] ** 2 * 4 * zw * w0 ** 2
    M = np.zeros((4, 4)); M[:2, :2] = -Ac; M[:2, 2:] = np.array([[0.0, 0.0], [0.0, q]]); M[2:, 2:] = Ac.T
    E = expm(M * dt)
    G = E[2:, 2:].T; Q = G @ E[:2, 2:]; Q = 0.5 * (Q + Q.T) + 1e-15 * np.eye(2)
    P0 = np.array([[cfg["sig_s"] ** 2, 0.0], [0.0, cfg["sig_s"] ** 2 * w0 ** 2]])
    return G, np.linalg.cholesky(Q), np.linalg.cholesky(P0)


def _half(k, cfg):
    """Beam output half-width (ln) of harmonic k, as built by beamx.beams_wide (half1, nonover)."""
    if k == 1:
        return cfg["half1"] if cfg["half1"] else PL.LN12
    return min(PL.LN12, 0.45 * float(np.log((k + 1) / k))) if cfg["nonover"] else PL.LN12


def _run(Zd, ks, A, s2, cf, fc, mu, cfg, use, seed, rel=None):
    fsd = FS / cfg["dec"]; dt = 1.0 / fsd
    G, Lq, L0 = _dyn(cfg, fc, dt)
    sq_phi = np.sqrt((2 * np.pi * cfg["sig_w"]) ** 2 * fc * dt)
    Rk = np.where(use, cf * s2, 0.0)[:, None] * (np.ones(Zd.shape) if rel is None else rel)
    lag = max(2, int(round(cfg["lag_s"] * fsd)))
    A = np.broadcast_to(A[:, None] if A.ndim == 1 else A, Zd.shape)
    bg = cfg["bandgain"]                                               # True: all k; 2: harmonics k >= 2 only
    bands = np.array([_half(int(k), cfg) if (bg is True or bg == 1 or (bg == 2 and k >= 2)) else 0.0 for k in ks],
                     float)
    return _pf(np.ascontiguousarray(Zd.real), np.ascontiguousarray(Zd.imag), np.asarray(ks, np.float64),
               np.ascontiguousarray(A.real), np.ascontiguousarray(A.imag), np.ascontiguousarray(Rk, dtype=np.float64), float(fc),
               float(mu), G, Lq, L0, float(sq_phi), float(dt), int(cfg["N"]), int(lag), int(seed),
               float(cfg["pjump"]), float(cfg["sjump"]), bands, float(PL.LN104))


def track(Z, centre, q1, cfg=None, Nr=None):
    cfg = dict(BASE, **(cfg or {}))
    ks = np.asarray(cfg["ks"], int)
    Zd = np.asarray(Z)[ks - 1][:, ::cfg["dec"]].astype(complex)
    K, T = Zd.shape
    fc = centre / 60.0
    fsd = FS / cfg["dec"]
    Pk = np.mean(np.abs(Zd) ** 2, 1)
    h1 = cfg["half1"] if cfg["half1"] else PL.LN12                    # k1 output band of the beam (ln half-width)
    hk = np.array([_half(k, cfg) for k in ks])
    lo1 = cfg["lo1"] if cfg["lo1"] else None                            # asymmetric k1 band (beamx.beams_guard)
    cf = np.array([cfg["ckf"] * max(1.0, fsd / PL.enbw(k * fc, h, lo=lo1 if k == 1 else None)) for k, h in zip(ks, hk)])
    kin = PL.enbw(fc, PL.LN11) / PL.enbw(fc, h1, lo=lo1)
    snr1 = max(float(q1) * cfg["qscale"], cfg["a1_floor"]) * kin if np.isfinite(q1) else 0.0
    A = np.zeros((K, T), complex); s2 = Pk.copy()
    A[0] = np.sqrt(Pk[0] * snr1 / (1 + snr1)); s2[0] = Pk[0] - abs(A[0, 0]) ** 2
    seed = int(cfg["seed"]) + int(T)
    rel = None
    if Nr is not None and cfg["nenv_s"] > 0:                            # local noise level from the heart-free ring
        from scipy.ndimage import uniform_filter1d
        e = uniform_filter1d(np.abs(np.asarray(Nr)[ks - 1][:, ::cfg["dec"]]) ** 2, max(3, int(cfg["nenv_s"] * fsd)),
                             axis=1, mode="nearest")
        rel = (e / e.mean(1, keepdims=True)) ** cfg["nenv_b"]
        rel = rel / rel.mean(1, keepdims=True)
    ev = []
    if cfg["a1ml"]:                                                     # type-II ML k1 SNR: PF evidence over a grid
        u1 = np.zeros(K, bool); u1[0] = True
        for snr in cfg["a1ml"]:
            a2 = Pk[0] * snr / (1 + snr); sn2 = Pk[0] - a2
            Ag = np.zeros((K, T), complex); Ag[0] = np.sqrt(a2); s2g = s2.copy(); s2g[0] = sn2
            if snr > 0:
                lz = _run(Zd, ks, Ag, s2g, cf, fc, 0.0, cfg, u1, seed, rel)[4]
            else:
                lz = 0.0
            lz += float(np.sum((-a2 - np.abs(Zd[0]) ** 2) / (cf[0] * sn2)) - T * np.log(np.pi * sn2) / cf[0])
            ev.append(lz)
        snr1 = float(cfg["a1ml"][int(np.argmax(ev))])
        A[0] = np.sqrt(Pk[0] * snr1 / (1 + snr1)); s2[0] = Pk[0] - abs(A[0, 0]) ** 2
    use = np.zeros(K, bool); use[0] = abs(A[0, 0]) > 0
    mu = 0.0
    um, uv, er, ei, lz = _run(Zd, ks, A, s2, cf, fc, mu, cfg, use, seed, rel)
    if cfg["mu_em"] and use.any():
        mu += float(np.mean(um))
        um, uv, er, ei, lz = _run(Zd, ks, A, s2, cf, fc, mu, cfg, use, seed, rel)
    info = []
    for it in range(cfg["em_iter"]):
        if not use[0]:
            break
        newA = A.copy(); news2 = s2.copy()
        for j in range(1, K):
            if cfg["loho"] and it > 0:
                u2 = use.copy(); u2[j] = False
                _, _, erj, eij, _ = _run(Zd, ks, A, s2, cf, fc, mu, cfg, u2, seed, rel)
            else:
                erj, eij = er, ei
            reg = (erj[j] - 1j * eij[j])                                    # E[e^{+ik phi}] (er + i ei = E e^{-ik phi})
            den = float(np.sum(np.abs(reg) ** 2))
            if den <= 1.0:
                newA[j] = 0; continue
            a = np.sum(Zd[j] * reg.conj()) / den
            var = s2[j] * cf[j] / den
            info.append((it, int(ks[j]), float(abs(a) ** 2 / var)))
            if cfg["awin_s"] > 0:                                           # slowly varying amplitude + phase psi_k(t)
                from scipy.ndimage import uniform_filter1d
                wn = max(3, int(cfg["awin_s"] * fsd))
                pr = Zd[j] * reg.conj()
                num = (uniform_filter1d(pr.real, wn, mode="nearest") + 1j * uniform_filter1d(pr.imag, wn, mode="nearest")) * wn
                dt_ = np.maximum(uniform_filter1d(np.abs(reg) ** 2, wn, mode="nearest") * wn, 1e-12)
                at = num / dt_; vt = s2[j] * cf[j] / dt_
                a2t = np.clip(np.abs(at) ** 2 - cfg["zgate"] * vt, 0.0, cfg["acap"] * Pk[j])
                newA[j] = at * np.sqrt(a2t / np.maximum(np.abs(at) ** 2, 1e-300))
                news2[j] = max(Pk[j] - float(np.mean(np.abs(newA[j]) ** 2)), 0.05 * Pk[j])
                continue
            a2d = min(max(abs(a) ** 2 - cfg["zgate"] * var, 0.0), cfg["acap"] * Pk[j])
            newA[j] = a * np.sqrt(a2d / max(abs(a) ** 2, 1e-300))
            news2[j] = max(Pk[j] - a2d, 0.05 * Pk[j])
        A, s2 = newA, news2
        use = np.abs(A).max(1) > 0
        um, uv, er, ei, lz = _run(Zd, ks, A, s2, cf, fc, mu, cfg, use, seed, rel)
        if cfg["mu_em"]:
            mu += float(np.mean(um))
            um, uv, er, ei, lz = _run(Zd, ks, A, s2, cf, fc, mu, cfg, use, seed, rel)
    if cfg["bidir"] and use.any():                                       # time-reversed pass (conj data, conj A)
        Zb = np.conj(Zd[:, ::-1]); relb = None if rel is None else rel[:, ::-1]
        ub, vb, _, _, _ = _run(Zb, ks, np.conj(A[:, ::-1]), s2, cf, fc, mu, cfg, use, seed + 7919, relb)
        ub = ub[::-1]; vb = vb[::-1]
        if cfg["bidir"] == 2:
            wf = 1.0 / np.maximum(uv, 1e-8); wb = 1.0 / np.maximum(vb, 1e-8)
            um = (wf * um + wb * ub) / (wf + wb); uv = 1.0 / (wf + wb)
        else:
            um = 0.5 * (um + ub); uv = 0.5 * (uv + vb)
    return dict(u=um, Pu=uv, mu=mu, A=A, s2=s2, cf=cf, snr1=snr1, info=info, fc=fc, fsd=fsd, ev=ev,
                R1=np.sqrt(er[0] ** 2 + ei[0] ** 2))


def curve(Z, centre, q1, t0, cfg=None, Nr=None):
    cfg = dict(BASE, **(cfg or {}))
    r = track(Z, centre, q1, cfg, Nr)
    if cfg["nrep"] > 1:                                                 # Monte-Carlo ensemble: mean of log f over seeds
        rs = [r] + [track(Z, centre, q1, dict(cfg, seed=cfg["seed"] + 1000 * j), Nr) for j in range(1, cfg["nrep"])]
        r = dict(r, u=np.mean([x["mu"] + x["u"] for x in rs], 0) - r["mu"], Pu=np.mean([x["Pu"] for x in rs], 0))
    u = r["u"]
    u = np.mean(u) + cfg["alpha"] * (u - np.mean(u))
    f = r["fc"] * np.exp(r["mu"] + u)
    t = t0 + np.arange(f.size) / r["fsd"]
    d = cfg["outdec"]
    return dict(t=t[::d], ibi=(1.0 / f)[::d]), r
