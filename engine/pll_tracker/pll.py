"""pll.py -- arm pll (D'): continuous-state harmonic phase tracker (EKF + RTS smoother, EM), ECG-free.

    track(Z, centre, q1, cfg) -> dict(t_rel s, u posterior mean of log f - log fc, Pu posterior var, mu, A [K], s2 [K])
    curve(Z, centre, q1, t0, cfg) -> est entry dict(t=abs s, ibi=s)        (posterior-mean HR curve, shrunk)

Model (per period, 250 Hz, Z[k-1] = analytic beam around k * fc from beams.py):
  phi_{n+1} = phi_n + 2 pi fc dt exp(mu + u_n) + w_phi        (w_phi: per-beat timing jitter, (2 pi sig_w)^2 per beat)
  (u, v) linear Gaussian: OU on u (tau = L_beats beats, sd sig_s) or a damped oscillator (period P_beats)
  z_k(n) = A_k exp(i k phi_n) + n_k(n),  n_k ~ CN(0, s_k^2) band-limited -> information divided by c_k = fs / ENBW_k
EKF with statistical linearisation over the phase prior (E e^{ik phi} = e^{ik m} e^{-k^2 P/2}; the linearisation residual
is added to the noise), so uncertain phase automatically switches the high harmonics off. RTS smoother.
A_1 = real, |A_1|^2 / s_1^2 from the beams' cross-fitted k1 contrast proxy q1 (ECG-free, independent of the tracker, so
the tracker cannot inflate its own confidence by locking onto noise). A_k (k >= 2, complex: amplitude + phase psi_k)
by EM against the phase posterior of a run WITHOUT harmonic k (leave-one-harmonic-out), noise-debiased. mu by EM.
"""
import numpy as np
from numba import njit

FS = 250.0
LN12 = float(np.log(1.20)); LN104 = float(np.log(1.04)); LN11 = float(np.log(1.10))

BASE = dict(K=5, ks=(1, 2, 3, 4, 5), sig_s=0.11, L_beats=5.0, sig_w=0.06, dyn="ou", P_beats=23.0,
            a1="proxy", qscale=1.0, em_iter=2, loho=True, mu_em=True, alpha=1.0, dec=10, ckf=1.0,
            p0phi=1.0, s2loc=0.0, umax=0.0, kmask=None, a1_floor=0.0, slin=False, aeff=True, zgate=1.0, acap=0.5)


def _taper(x, a, b):
    out = np.zeros_like(x)
    out[x <= a] = 1.0
    m = (x > a) & (x < b)
    out[m] = 0.5 * (1 + np.cos(np.pi * (x[m] - a) / (b - a)))
    return out


def enbw(fk, half, tap=LN104, lo=None):
    """Noise bandwidth (Hz) of the log-f cosine-tapered band of half-width `half` (lower half-width `lo`) around fk."""
    lo = half if lo is None else lo
    f = np.linspace(fk * np.exp(-lo - tap - 0.01), fk * np.exp(half + tap + 0.01), 4001)
    x = np.log(f / fk)
    h = np.where(x >= 0, _taper(np.abs(x), half, half + tap), _taper(np.abs(x), lo, lo + tap))
    return float(np.trapezoid(h ** 2, f))


def dyn_mats(cfg, fc_eff):
    """Discrete (u, v) transition G and process covariance Q for dt = 1/FS; tau etc. in beats of fc_eff."""
    dt = 1.0 / FS; Tb = 1.0 / fc_eff
    if cfg["dyn"] == "ou":
        tau = cfg["L_beats"] * Tb
        a = np.exp(-dt / tau)
        G = np.array([[a, 0.0], [0.0, 0.0]]); Q = np.array([[cfg["sig_s"] ** 2 * (1 - a * a), 0.0], [0.0, 1e-12]])
        P0 = np.array([[cfg["sig_s"] ** 2, 0.0], [0.0, 1e-12]])
        return G, Q, P0
    # damped oscillator: u'' + 2 z w u' + w^2 u = noise; C(t) ~ e^{-t/L} cos(2 pi t / P)
    from scipy.linalg import expm
    L = cfg["L_beats"] * Tb; P = cfg["P_beats"] * Tb
    zw = 1.0 / L; wd = 2 * np.pi / P; w0 = np.sqrt(wd ** 2 + zw ** 2)
    Ac = np.array([[0.0, 1.0], [-w0 ** 2, -2 * zw]])
    # stationary variance of u for unit noise intensity q: var_u = q / (4 zw w0^2)
    q = cfg["sig_s"] ** 2 * 4 * zw * w0 ** 2
    Lc = np.array([[0.0], [1.0]])
    M = np.zeros((4, 4)); M[:2, :2] = -Ac; M[:2, 2:] = Lc @ Lc.T * q; M[2:, 2:] = Ac.T
    E = expm(M * dt)
    G = E[2:, 2:].T; Q = G @ E[:2, 2:]
    Q = 0.5 * (Q + Q.T)
    P0 = np.array([[cfg["sig_s"] ** 2, 0.0], [0.0, cfg["sig_s"] ** 2 * w0 ** 2]])
    return G, Q, P0


@njit(cache=True)
def _smooth(zr, zi, ks, Ar, Ai, R0, cfac, fc, mu, G, Q, qphi, x0, P0, umax, slin):
    """EKF forward + RTS backward. State x = (phi, u, v). zr/zi [K, T]; A [K]; R0 [K] noise var; cfac [K]."""
    K, T = zr.shape
    dt = 1.0 / 250.0
    xf = np.zeros((T, 3)); Pf = np.zeros((T, 3, 3))
    xp = np.zeros((T, 3)); Pp = np.zeros((T, 3, 3)); Fj = np.zeros((T, 3, 3))
    x = x0.copy(); P = P0.copy()
    for n in range(T):
        if n > 0:
            # predict from n-1
            w = 2 * np.pi * fc * dt * np.exp(mu + x[1])
            F = np.zeros((3, 3))
            F[0, 0] = 1.0; F[0, 1] = w
            F[1, 1] = G[0, 0]; F[1, 2] = G[0, 1]; F[2, 1] = G[1, 0]; F[2, 2] = G[1, 1]
            xn = np.zeros(3)
            xn[0] = x[0] + w
            xn[1] = G[0, 0] * x[1] + G[0, 1] * x[2]
            xn[2] = G[1, 0] * x[1] + G[1, 1] * x[2]
            Pn = F @ P @ F.T
            Pn[0, 0] += qphi * dt
            Pn[1, 1] += Q[0, 0]; Pn[1, 2] += Q[0, 1]; Pn[2, 1] += Q[1, 0]; Pn[2, 2] += Q[1, 1]
            Fj[n - 1] = F
            x = xn; P = Pn
        xp[n] = x; Pp[n] = P
        # measurement (statistically linearised over the phase prior)
        J = 0.0; g = 0.0
        P00 = P[0, 0]
        for k in range(K):
            a2 = Ar[k] * Ar[k] + Ai[k] * Ai[k]
            if a2 <= 0.0:
                continue
            kk = ks[k]
            sl = slin == 1 or (slin == 2 and kk > 1.5)
            gam = np.exp(-0.5 * kk * kk * P00) if sl else 1.0
            res = a2 * (1.0 - gam * gam - kk * kk * gam * gam * P00) if sl else 0.0
            if res < 0.0:
                res = 0.0
            R = cfac[k] * (R0[k, n] + res)
            c = np.cos(kk * x[0]); s = np.sin(kk * x[0])
            hr = Ar[k] * c - Ai[k] * s; hi = Ar[k] * s + Ai[k] * c
            # Im(conj(h) z) = hr*zi - hi*zr
            J += 2.0 * kk * kk * a2 * gam * gam / R
            g += 2.0 * kk * gam * (hr * zi[k, n] - hi * zr[k, n]) / R
        if J > 0.0:
            den = 1.0 + J * P00
            col = P[:, 0].copy()
            for i in range(3):
                x[i] += col[i] * g / den
            for i in range(3):
                for j in range(3):
                    P[i, j] -= col[i] * col[j] * J / den
        if umax > 0.0:
            if x[1] > umax:
                x[1] = umax
            elif x[1] < -umax:
                x[1] = -umax
        xf[n] = x; Pf[n] = P
    # RTS
    xs = xf.copy(); Ps = Pf.copy()
    for n in range(T - 2, -1, -1):
        Pinv = np.linalg.inv(Pp[n + 1])
        C = Pf[n] @ Fj[n].T @ Pinv
        xs[n] = xf[n] + C @ (xs[n + 1] - xp[n + 1])
        Ps[n] = Pf[n] + C @ (Ps[n + 1] - Pp[n + 1]) @ C.T
    return xs, Ps


def _local_s2(Z, A, s2, cfg):
    """[K, T] noise variance: constant s2, or (s2loc > 0) the s2loc-second moving mean of |z|^2 minus |A|^2."""
    K, T = Z.shape
    if cfg["s2loc"] <= 0:
        return np.repeat(np.asarray(s2, float)[:, None], T, 1)
    from scipy.ndimage import uniform_filter1d
    w = max(3, int(cfg["s2loc"] * FS))
    e = uniform_filter1d(np.abs(Z) ** 2, w, axis=1, mode="nearest")
    return np.maximum(e - np.abs(A)[:, None] ** 2, 0.1 * np.asarray(s2, float)[:, None])


def _run(Z, ks, A, s2, cf, fc, mu, cfg, G, Q, P0uv, qphi):
    K, T = Z.shape
    x0 = np.zeros(3)
    m = min(T, int(0.5 * FS))
    tt = np.arange(m) / FS
    x0[0] = float(np.angle(np.mean(Z[0, :m] * np.exp(-2j * np.pi * fc * np.exp(mu) * tt))))
    P0 = np.zeros((3, 3)); P0[0, 0] = cfg["p0phi"]; P0[1:, 1:] = P0uv
    return _smooth(np.ascontiguousarray(Z.real), np.ascontiguousarray(Z.imag), np.asarray(ks, np.float64),
                   np.ascontiguousarray(A.real), np.ascontiguousarray(A.imag), np.ascontiguousarray(_local_s2(Z, A, s2, cfg)),
                   np.asarray(cf, float), float(fc), float(mu), G, Q, float(qphi), x0, P0, float(cfg["umax"]), int(cfg["slin"]))


def track(Z, centre, q1, cfg=None):
    cfg = dict(BASE, **(cfg or {}))
    ks = np.asarray(cfg["ks"], int)
    Z = np.asarray(Z)[ks - 1].astype(complex)
    K, T = Z.shape
    fc = centre / 60.0
    Pk = np.mean(np.abs(Z) ** 2, 1)
    cf = np.array([cfg["ckf"] * FS / enbw(k * fc, LN12) for k in ks])
    # k1 amplitude from the cross-fitted contrast proxy: in-band (+-10 %) excess density -> share of the +-20 % band
    kin = enbw(fc, LN11) / enbw(fc, LN12)
    snr1 = max(float(q1) * cfg["qscale"], cfg["a1_floor"]) * kin if np.isfinite(q1) else 0.0
    A = np.zeros(K, complex); s2 = Pk.copy()
    if ks[0] == 1:
        A[0] = np.sqrt(Pk[0] * snr1 / (1 + snr1)); s2[0] = Pk[0] - abs(A[0]) ** 2
    mu = 0.0
    G, Q, P0uv = dyn_mats(cfg, fc)
    qphi = (2 * np.pi * cfg["sig_w"]) ** 2 * fc                       # rad^2 / s (per-beat jitter at fc)
    use = np.zeros(K, bool); use[0] = abs(A[0]) > 0
    xs, Ps = _run(Z, ks, np.where(use, A, 0), s2, cf, fc, mu, cfg, G, Q, P0uv, qphi)
    if cfg["mu_em"]:
        mu = mu + float(np.mean(xs[:, 1]))
        xs, Ps = _run(Z, ks, np.where(use, A, 0), s2, cf, fc, mu, cfg, G, Q, P0uv, qphi)
    info = []
    for it in range(cfg["em_iter"]):
        newA = A.copy(); news2 = s2.copy()
        for j in range(1, K):
            if cfg["loho"] and it > 0:
                u2 = use.copy(); u2[j] = False
                xj, Pj = _run(Z, ks, np.where(u2, A, 0), s2, cf, fc, mu, cfg, G, Q, P0uv, qphi)
            else:
                xj, Pj = (xs, Ps) if not cfg["loho"] or it == 0 else (xs, Ps)
            k = ks[j]
            if cfg["aeff"]:                                            # coherent amplitude AS SEEN by the tracker phase
                gam = np.ones(T)
            else:
                gam = np.exp(-0.5 * k * k * Pj[:, 0, 0])
            reg = gam * np.exp(1j * k * xj[:, 0])
            den = float(np.sum(gam ** 2))
            if den <= 1.0:
                newA[j] = 0; continue
            a = np.sum(Z[j] * reg.conj()) / den
            var = s2[j] * cf[j] / den                                  # noise variance of the regression estimate
            a2d = min(max(abs(a) ** 2 - cfg["zgate"] * var, 0.0), cfg["acap"] * Pk[j])
            newA[j] = a * np.sqrt(a2d / max(abs(a) ** 2, 1e-300))
            news2[j] = max(Pk[j] - abs(newA[j]) ** 2, 0.05 * Pk[j])
            info.append((it, int(k), float(abs(a) ** 2 / var)))
        A, s2 = newA, news2
        use = np.abs(A) > 0
        xs, Ps = _run(Z, ks, np.where(use, A, 0), s2, cf, fc, mu, cfg, G, Q, P0uv, qphi)
        if cfg["mu_em"]:
            mu = mu + float(np.mean(xs[:, 1]))
            xs, Ps = _run(Z, ks, np.where(use, A, 0), s2, cf, fc, mu, cfg, G, Q, P0uv, qphi)
    return dict(u=xs[:, 1], Pu=Ps[:, 1, 1], phi=xs[:, 0], Pphi=Ps[:, 0, 0], mu=mu, A=A, s2=s2, cf=cf,
                snr1=snr1, info=info, fc=fc)


def curve(Z, centre, q1, t0, cfg=None):
    cfg = dict(BASE, **(cfg or {}))
    r = track(Z, centre, q1, cfg)
    u = r["u"]
    u = np.mean(u) + cfg["alpha"] * (u - np.mean(u))
    f = r["fc"] * np.exp(r["mu"] + u)
    d = cfg["dec"]
    t = t0 + np.arange(f.size) / FS
    return dict(t=t[::d], ibi=(1.0 / f)[::d]), r
