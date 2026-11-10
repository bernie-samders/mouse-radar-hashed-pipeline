"""hr_bands.py -- recording plumbing around the frozen heart-rate code (hr_final_pipeline.py).

Port of the research runner radar_hr/hr_core.py (validated 2026-10-03: identical HR, band and beat onsets to the frozen
pipeline on all 93 clean-breathing runs of 508 / 507 day 2). It provides:
  - breathing(): EQ-Radio clean-breathing runs >= 20 s and their breathing rate, from the `segments` substrate;
  - ecg_free_patch(): the 330-voxel chest patch (10 beams x abs range bins 50-82) centred on the breathing-power beam;
  - build_accel(): EQ-Radio acceleration of the patch per run (float32 npy; the frozen code reads it back);
  - LazyBands: the 17 tight heartbeat bands (centre / 1.08 .. centre * 1.08, centres 380-780 bpm) of one run. A band is
    segmented (EQ-Radio one-template DP on all 330 voxels + beat-phase agreement picks) only when the frozen code first
    reads more than its centre, so the normal path segments 1 band instead of 17 with an identical result. The other 16
    are touched only by the S1 fallback (0 of 93 dev runs). Each voxel's template (mu), EQ-Radio confidence and
    snr = (energy - cost) / cost are kept for the template export; they do not feed the frozen code.
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import beatseg as BS                                        # noqa: E402
import eqradio_hr as eq                                     # noqa: E402
from harness import read_block                              # noqa: E402
from scipy.signal import butter, sosfiltfilt                # noqa: E402

FS = 250.0                                                  # the frozen constants assume the 250 Hz live cube
CENTRES = np.arange(380, 781, 25)                           # tight bands (method 2), centre / 1.08 .. centre * 1.08
HALF = 1.08
PATCH_BINS = np.arange(50, 83)                              # abs range bins 0.27-0.45 m (fixed for both dev mice)
N_BEAMS = 10                                                # patch = 10 beams x 33 bins = 330 voxels
MIN_RUN_S = 20.0                                            # clean-breathing runs >= 20 s
PAD_S = 2.0                                                 # accel padding (band_scan.build)
PH_FS, TOPN, K_PICK = 50.0, 20, 20                          # agreement (score_period): phase grid, partners, picks
NB_ANG, NB_BIN = 1, 3                                       # agreement ignores partners this close (shared noise)


# ---- breathing ---------------------------------------------------------------------------------------------------------

def breathing(seg_base, min_s=MIN_RUN_S):
    """EQ-Radio fit-ok cycles (onset, ttot) and clean runs >= min_s [(t0, t1)] from <segments>/<rec>.{json,npz}."""
    import dp_apnea as DA
    s = DA.RadarSession(seg_base + ".json", seg_base + ".npz")
    cyc = [c for c in DA.dp_cycles(s) if c["fit_ok"]]
    on = np.array([c["t_onset"] for c in cyc]); tt = np.array([c["ttot"] for c in cyc])
    _, _, rr = DA.dp_clean_mask(s)
    runs = [(float(a), float(b)) for a, b, _ in rr if b - a >= min_s]
    return on, tt, runs


def run_rr(on, tt, a, b):
    m = (on >= a) & (on < b)
    return float(60.0 / np.median(tt[m])) if m.any() else None


# ---- chest patch ---------------------------------------------------------------------------------------------------------

def patch_fits(meta):
    """True if the cube's range bins cover the fixed patch range (abs bins 50-82)."""
    b0, nb = int(meta["bins"][0]), int(meta["n_bin"])
    return b0 <= PATCH_BINS[0] and PATCH_BINS[-1] < b0 + nb and int(meta["n_angle"]) >= N_BEAMS


def ecg_free_patch(cube, meta, runs, rrs):
    """Patch beams centred on the beam with the most breathing-band power (0.5-1.5 x RR) at the patch ranges, summed
    over every clean run (validated: same patch on 507d2, 508 one beam over with <= 0.06 bpm change per run).
    Returns (beams, beam_power, peak bin inside the patch)."""
    nb, na, b0 = meta["n_bin"], meta["n_angle"], meta["bins"][0]
    cols = np.array([ai * nb + (b - b0) for ai in range(na) for b in PATCH_BINS])
    P = np.zeros(cols.size)
    for (a, b), rr in zip(runs, rrs):
        rr = rr or 180.0
        X = read_block(cube, a, b, cols).astype(float); X -= X.mean(0)
        sos = butter(3, [0.5 * rr / 60, 1.5 * rr / 60], "bandpass", fs=FS, output="sos")
        P += (sosfiltfilt(sos, X, axis=0) ** 2).mean(0)
    Pm = P.reshape(na, PATCH_BINS.size)
    beam = Pm.sum(1); bpk = int(np.argmax(beam))
    start = min(max(0, bpk - 4), na - N_BEAMS)
    return list(range(start, start + N_BEAMS)), beam, int(PATCH_BINS[int(np.argmax(Pm[bpk]))])


def patch_cols(meta, beams):
    nb, b0 = meta["n_bin"], meta["bins"][0]
    return np.array([ai * nb + (b - b0) for ai in beams for b in PATCH_BINS])


def build_accel(args):
    """[T, 330] float32 EQ-Radio acceleration of the patch over [t0, t1) (band_scan.build) -> npy path."""
    cube_spec, cols, a, b, path = args
    if os.path.exists(path):
        return path
    cube = np.memmap(cube_spec[0], dtype=np.float32, mode="r", shape=tuple(cube_spec[1]))
    rec_s = cube.shape[0] / FS
    pa, pb = min(PAD_S, a), min(PAD_S, rec_s - b)
    X = read_block(cube, a - pa, b + pb, cols)
    a0 = int(round((a - pa) * FS)); c0, c1 = int(round(a * FS)), int(round(b * FS))
    A = np.stack([eq.accel_holoborodko(X[:, v].astype(np.float64), FS)[c0 - a0:c1 - a0] for v in range(X.shape[1])], 1)
    np.save(path, A.astype(np.float32))
    return path


# ---- lazy tight-band segmentation --------------------------------------------------------------------------------------

def seg_voxels(args):
    """One-template segmentation of patch voxels `vox` in band centre c (band_template._job settings)."""
    path, vox, c, a = args
    A = np.load(path, mmap_mode="r")
    bmin, bmax, m = BS.bounds_samples(FS, c / HALF, c * HALF)
    out = []
    for v in vox:
        best, _ = BS.multistart([np.asarray(A[:, v], float)], FS, bmin, bmax, m, inits=("acf", f"p{c}"))
        on, ibi, _ = BS.beats(best, [int(round(a * FS))], FS)
        out.append((v, on.astype(np.float32), ibi.astype(np.float32), np.asarray(best["blist"][0]),
                    np.asarray(best["mu"], float), float(best["conf"]),
                    (best["energy"] - best["cost"]) / max(best["cost"], 1e-30)))
    return out


def _rho1(x, blist, m):
    """Mean adjacent-beat correlation of the interior beats (band_repeat.rho, lag 1) -- S1 fallback only."""
    segs = [x[a:b] for a, b in zip(blist[1:-2], blist[2:-1])]
    if len(segs) < 8:
        return np.nan
    S = np.array([BS.warp(s, m) for s in segs]); S -= S.mean(1, keepdims=True)
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-30
    return float(np.mean(np.sum(S[:-1] * S[1:], 1)))


def _phase_vec(on, grid, max_gap):
    """exp(2 pi i * beat phase) on grid; 0 where the gap between onsets is not a beat (score_period.phase_vec)."""
    e = np.zeros(grid.size, np.complex64)
    if on.size < 3:
        return e
    k = np.searchsorted(on, grid, side="right") - 1
    ok = (k >= 0) & (k < on.size - 1)
    kk = k[ok]; d = on[kk + 1] - on[kk]
    good = d <= max_gap
    ph = (grid[ok] - on[kk]) / d
    e[np.flatnonzero(ok)[good]] = np.exp(2j * np.pi * ph[good]).astype(np.complex64)
    return e


def _plv_matrix(E):
    M = (E != 0).astype(np.float32)
    num = np.abs(E @ E.conj().T); den = M @ M.T
    return np.where(den > 100, num / np.maximum(den, 1), np.nan)


class _Band:
    def __init__(self, owner, j):
        self.o, self.j, self.d = owner, j, None

    def _get(self):
        if self.d is None:
            self.d = self.o.segment(self.j)
        return self.d

    def __getitem__(self, k):
        if k == "centre":
            return int(self.o.centres[self.j])
        if k == "r1":
            return self.o.r1(self.j)
        return self._get()[k]

    def __contains__(self, k):
        return k in ("centre", "agree", "picks", "r1", "cons", "on", "ibi")


class LazyBands:
    """Sequence of the 17 tight bands for one run, segmented on first access (pool = executor or None = in-process)."""

    def __init__(self, path, a, b, cols, n_bin, pool=None, chunk=15):
        self.path, self.a, self.b, self.pool, self.chunk = path, a, b, pool, chunk
        self.centres = CENTRES
        ai, bj = np.divmod(np.asarray(cols), n_bin)
        self.near = (np.abs(ai[:, None] - ai[None, :]) <= NB_ANG) & (np.abs(bj[:, None] - bj[None, :]) <= NB_BIN)
        self.nv = len(cols)
        self._b = [_Band(self, j) for j in range(len(CENTRES))]
        self._r1 = {}
        self.segmented = []

    def __len__(self):
        return len(self._b)

    def __getitem__(self, j):
        return self._b[j]

    def __iter__(self):
        return iter(self._b)

    def segment(self, j):
        c = self.centres[j]
        tasks = [(self.path, list(range(s, min(self.nv, s + self.chunk))), c, self.a) for s in range(0, self.nv, self.chunk)]
        res = list(self.pool.map(seg_voxels, tasks)) if self.pool is not None else [seg_voxels(t) for t in tasks]
        rows = sorted((r for chunk in res for r in chunk), key=lambda r: r[0])
        on = [r[1] for r in rows]; ibi = [r[2] for r in rows]
        grid = np.arange(self.a, self.b, 1 / PH_FS)
        E = np.stack([_phase_vec(o.astype(float), grid, 60.0 / (c / HALF) + 0.01) for o in on])
        P = _plv_matrix(E); P[self.near] = np.nan
        agree = np.nanmean(-np.sort(-np.nan_to_num(P, nan=-1.0), 1)[:, :TOPN], 1)
        pa = np.argsort(-agree)[:K_PICK]
        onv = np.empty(len(on), object); onv[:] = on
        ibv = np.empty(len(ibi), object); ibv[:] = ibi
        self.segmented.append(int(c))
        return dict(agree=agree.astype(np.float32), picks=pa, on=onv, ibi=ibv, blist=[r[3] for r in rows],
                    cons=float(60 / np.median(np.concatenate([ibi[k] for k in pa]).astype(float))),
                    mu=[r[4] for r in rows], conf=np.array([r[5] for r in rows]), snr=np.array([r[6] for r in rows]))

    def r1(self, j):
        if j not in self._r1:
            d = self._b[j]._get(); A = np.load(self.path, mmap_mode="r")
            _, _, m = BS.bounds_samples(FS, self.centres[j] / HALF, self.centres[j] * HALF)
            self._r1[j] = np.array([_rho1(np.asarray(A[:, v], float), d["blist"][v], m) for v in range(self.nv)], np.float32)
        return self._r1[j]
