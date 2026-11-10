r"""witness.py -- adjudicate apnea candidates with WITNESS voxels (the same two-template DP re-run on other voxels).

Ported verbatim (same numbers) from the validated research code `output/apnea_cohort/witness_check.py` (features) and
`witness_v2.py` (votes + the FINAL rule), with pandas replaced by plain numpy/python.

For every candidate pause, the 10 best voxels by flank resp-band SNR within +-15 range bins (all angles) of the
episode's chosen voxel, plus the chosen voxel itself, are each conditioned exactly as the DP input (`cond_decimate`)
and segmented with `fast_dp.segment_two_fast(SEG_KW)` over pause +-4 s, so each voxel learns its template and 0.4 x
breath floor from ITS OWN flank breaths (segmenting the pause alone locks a bad seed). Per voxel:
  witness          >= 3 confident breaths (fit >= 0.5) in EACH 2.5 s flank, at 0.5-2x the animal's median cycle
                   (a heartbeat-locked voxel runs at ~0.3-0.5x);
  confirms         a witness whose own breath-free stretch covers >= 60 % of the pause;
  breathes through rhythm known from >= 1 flank (a posture change keeps breathing but can lose a flank) AND by the
                   apnea definition it sees no apnea: longest in-pause gap < max(2 own cycles, 3 x median cycle), with
                   in-pause breaths >= 0.5 x its flank breaths, spaced at its rhythm.
Single voxels occasionally fabricate a breath (heartbeat), so both votes need TWO voxels.
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import fast_dp as F
from conditioning import cond_decimate, SEG_KW, NFFT, FLOOR_BPM, RESP_HI_BPM, HF_LO, HF_HI

WIN_S, CTX_S = 4.0, 2.0                       # flank segmented each side; conditioning edge room
FL_S = 2.5                                    # flank for witness breaths
NB_HALF, TOPK = 15, 10                        # +-range-bin neighbourhood (all angles) / voxels ranked by flank SNR
FIT = 0.5                                     # confident breath (= dp_apnea.DP_CLEAN_FIT)
POOL_MULT, POOL_ABS = 2.5, 0.9                # candidates shorter than both can't pass any rule -> not adjudicated

# thresholds fixed a priori (before WBP); CON_APNEA_MULT added in v2 from B17 (one voxel missed the old limit by 0.03 s)
WIT_N, WIT_RHYTHM = 3, (0.5, 2.0)
CON_GAP, CON_AMP, CON_SPACING = 2.0, 0.5, (0.6, 1.6)
CONF_FRAC = 0.6
CON_APNEA_MULT = 3.0


def _snr(seg, fs):
    """Resp-band SNR per column (segmentation's wide-prescreen: fundamental peak / HF median)."""
    nfft = 1 << int(np.ceil(np.log2(max(seg.shape[0], NFFT))))
    fr = np.fft.rfftfreq(nfft, 1 / fs)
    fund = (fr >= FLOOR_BPM / 60) & (fr <= RESP_HI_BPM / 60); anar = (fr >= HF_LO) & (fr <= min(HF_HI, 0.49 * fs))
    P = np.abs(np.fft.rfft((seg - seg.mean(0)) * np.hanning(seg.shape[0])[:, None], nfft, axis=0)) ** 2
    return P[fund].max(0) / (np.median(P[anar], 0) + 1e-30)


def _voxel_features(b, t0, t1, cyc_ref):
    """b: confident breaths [(a_s, b_s, rms)] of one voxel's DP over [t0-WIN, t1+WIN]."""
    on = np.array([x[0] for x in b]); en = np.array([x[1] for x in b]); rms = np.array([x[2] for x in b])
    mid = 0.5 * (on + en)
    pre = (mid >= t0 - FL_S) & (mid < t0 + 0.05); post = (mid > t1 - 0.05) & (mid <= t1 + FL_S)
    inn = (mid > t0 + 0.05) & (mid < t1 - 0.05)
    d = np.concatenate([np.diff(on[pre]), np.diff(on[post])])
    d = d[(d > 0.3 * cyc_ref) & (d < 2.5 * cyc_ref)] if np.isfinite(cyc_ref) else d
    cyc = float(np.median(d)) if d.size else np.nan
    fr = float(np.median(rms[pre | post])) if (pre | post).any() else np.nan
    # longest breath-free stretch inside [t0, t1]: gaps between consecutive confident breaths, clipped
    iv = sorted((max(t0, a), min(t1, e)) for a, e in zip(on, en) if e > t0 and a < t1)
    gaps, cur = [], t0
    for a, e in iv:
        gaps.append(a - cur); cur = max(cur, e)
    gaps.append(t1 - cur)
    ons_in = on[inn]
    sp = np.diff(np.r_[on[pre][-1:], ons_in, on[post][:1]]) if ons_in.size else np.zeros(0)
    # 10 significant digits, as the validated voxel tables stored them: a ratio landing exactly on a threshold
    # (spacing 0.448 / 0.280 = 1.6) otherwise flips on the last ulp
    g10 = lambda x: float(f"{x:.10g}")
    return dict(n_pre=int(pre.sum()), n_post=int(post.sum()), cyc_v=g10(cyc), n_in=int(inn.sum()),
                maxgap=g10(max(gaps)), in_amp=g10(np.median(rms[inn]) / fr) if inn.any() and fr > 0 else 0.0,
                in_spacing=g10(np.median(sp) / cyc) if sp.size and np.isfinite(cyc) else np.nan)


def event_witnesses(power, meta, dom, t0, t1, cyc_ref, traces=False):
    """Per-voxel DP features around one candidate [t0, t1] (s). `dom` = the episode's chosen voxel (always included)."""
    fs = float(meta["fs"]); nb, na = int(meta["n_bin"]), int(meta["n_angle"])
    T = power.shape[0] / fs
    wa, wb = max(0.0, t0 - WIN_S), min(T, t1 + WIN_S)
    i0, i1 = int(max(0.0, wa - CTX_S) * fs), int(min(T, wb + CTX_S) * fs)
    mag = np.sqrt(np.maximum(np.asarray(power[i0:i1], np.float64), 0.0))
    q = 5; n50 = mag.shape[0] // q; fs50 = fs / q
    d50 = mag[:n50 * q].reshape(n50, q, -1).mean(1)
    tt = i0 / fs + np.arange(n50) / fs50
    fl = ((tt >= t0 - 3.0) & (tt < t0)) | ((tt > t1) & (tt <= t1 + 3.0))
    snr = _snr(d50[fl], fs50) if fl.sum() > 16 else np.zeros(d50.shape[1])
    bj0 = dom % nb
    near = np.zeros(na * nb, bool)
    for ai in range(na):
        near[ai * nb + max(0, bj0 - NB_HALF): ai * nb + min(nb, bj0 + NB_HALF + 1)] = True
    snr[~near] = -1
    rows = []
    top = list(np.argsort(-snr)[:TOPK])
    if int(dom) not in top:
        top.append(int(dom))                  # the chosen voxel is often outside the noisy 6 s flank-SNR top-K
    for rank, c in enumerate(top):
        x, f125 = cond_decimate(mag[:, c], fs)
        s0 = int(round((wa - i0 / fs) * f125)); core = x[s0:s0 + int(round((wb - wa) * f125))]
        if core.size < 3 * f125:
            continue
        res = F.segment_two_fast(core, f125, **SEG_KW)
        b = [(wa + p["a"] / f125, wa + p["b"] / f125, p["rms"]) for p in res["per"] if p["fit_tol"] >= FIT]
        rows.append(dict(rank=rank, col=int(c), is_dom=int(c) == int(dom), snr=float(snr[c]),
                         **_voxel_features(b, t0, t1, cyc_ref)))
        if traces:
            rows[-1].update(t=wa + np.arange(core.size) / f125, x=core, breaths=b)
    return rows


def _between(v, lo_hi):
    return bool(np.isfinite(v) and lo_hi[0] <= v <= lo_hi[1])


def voxel_role(v, t0, t1, cycle_med, conf_frac=CONF_FRAC, wit_n=WIT_N):
    """Flags for one voxel row: wit / con (breathes through) / conf (confirms) / shows (chosen voxel shows the pause)."""
    rhythm = _between(v["cyc_v"] / cycle_med, WIT_RHYTHM)
    wit = v["n_pre"] >= wit_n and v["n_post"] >= wit_n and rhythm
    half = (v["n_pre"] >= wit_n or v["n_post"] >= wit_n) and rhythm
    sp = v["in_spacing"]
    sp_ok = (not np.isfinite(sp)) or CON_SPACING[0] <= sp <= CON_SPACING[1]
    gap_lim = max(CON_GAP * v["cyc_v"], CON_APNEA_MULT * cycle_med)
    con = half and v["n_in"] >= 1 and v["maxgap"] < gap_lim and v["in_amp"] >= CON_AMP and sp_ok
    covers = v["maxgap"] >= conf_frac * (t1 - t0)
    return dict(wit=wit, con=con, conf=wit and covers, shows=bool(v["is_dom"]) and half and covers)


def votes(rows, t0, t1, cycle_med):
    """Candidate-level vote counts. n_conf_ind excludes the chosen voxel: for a single-voxel candidate the chosen voxel
    IS the detector, so its vote would be circular."""
    n = dict(n_vox=len(rows), n_wit=0, n_con=0, n_conf=0, n_conf_ind=0, dom_shows=False)
    for v in rows:
        r = voxel_role(v, t0, t1, cycle_med)
        n["n_wit"] += r["wit"]; n["n_con"] += r["con"]; n["n_conf"] += r["conf"]
        n["n_conf_ind"] += r["conf"] and not v["is_dom"]
        n["dom_shows"] |= r["shows"]
    return n


def final_rule(c):
    """The validated rule (witness_v2 FINAL): >= 2 voxels other than the chosen one confirm the pause and < 2 breathe
    through; or, when no voxel breathes cleanly enough to testify (post-sniff, noisy voxels -- B16 1420 s), an anchored-
    combine pause that no voxel breathes through and that the chosen voxel itself shows with breathing on >= 1 side."""
    return ((c["n_conf_ind"] >= 2 and c["n_con"] < 2) or
            (c["n_wit"] == 0 and c["n_con"] == 0 and c["src"] == "anchored" and c["dom_shows"]))
