r"""dp_apnea.py -- EQ-Radio-native apnea candidates from a two-template segmentation substrate.

Ported verbatim (same numbers) from the validated research code: `apneacheck/radar_apnea.py` (the B16 radar-vs-WBP
detector pieces) and the `detect` / `boundaries` steps of `output/apnea_cohort/run_apnea_cohort.py`.

An APNEA candidate is a long PAUSE-template segment (kind 1) flanked by resolved breathing. Everything here reads only
the DP `segments` channel -- no peak detector, no ACF (breath / apnea work must use the two-template segmentation).
"""
from __future__ import annotations
import os, sys, json
from dataclasses import dataclass
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from substrate import _pair_cycles, FIT_MIN   # the EQ-Radio B->P cycle builder (seam hygiene + fit gate)


class RadarSession:
    """One segmentation substrate (<rec>.json + <rec>.npz): the single-voxel `segments` one or the anchored combine."""
    def __init__(self, seg_json, seg_npz):
        self.j = json.load(open(seg_json))
        self.fs = float(self.j["fs"])
        self.episodes = self.j["episodes"]
        z = np.load(seg_npz, allow_pickle=True)
        self.segments = z["segments"]          # object[n_ep] -> [n_seg,5]
        self.trace = z["trace"]                 # object[n_ep] -> conditioned trace @ fs

    @property
    def rec_s(self):
        return float(self.j.get("rec_min", 60) * 60.0)


def dp_cycles(sess: RadarSession, fit_min=FIT_MIN):
    """EQ-Radio per-cycle series (breath-motion + following pause) for every episode; each cycle gets its episode `ep`."""
    out = []
    for ep in range(len(sess.episodes)):
        S = np.asarray(sess.segments[ep], float)
        if S.size == 0:
            continue
        for c in _pair_cycles(S, float(sess.episodes[ep]["t0"]), fit_min=fit_min):
            c["ep"] = ep
            out.append(c)
    out.sort(key=lambda c: c["t_onset"])
    return out


# Clean-breathing gate. Stricter than FIT_MIN (0.35 = "is this a breath"): vs WBP truth (09-21 B16/B17) per-cycle fit
# separates tracked/untracked breathing (AUC 0.94/0.89); fit>=0.5 & run>=8 -> precision 0.97/0.91 (ACF mask 0.62/0.61).
DP_CLEAN_FIT, DP_CLEAN_RUN = 0.5, 8


def dp_clean_mask(sess: RadarSession, fit_min=DP_CLEAN_FIT, min_run=DP_CLEAN_RUN, seam_gap_s=1.0, grid_hop=0.5,
                  cycles=None):
    """CLEAN breathing = time covered by a run of >= min_run consecutive fit-gated B->P cycles. The only tolerated hole
    is the one `_pair_cycles` seam hygiene punches at each 8 s DP block edge (bridged if <= seam_gap_s) -- never bridge
    within a block, or a real cessation would be called clean.  Returns (grid_t, clean_bool, runs[(t0, t1, n)])."""
    cyc = cycles if cycles is not None else dp_cycles(sess, fit_min)
    g = np.arange(0.0, sess.rec_s + grid_hop, grid_hop)
    runs, cur = [], []

    def _close():
        if len(cur) >= min_run:
            runs.append((cur[0]["t_onset"], cur[-1]["t_onset"] + cur[-1]["ttot"], len(cur)))

    for c in cyc:
        if not c["fit_ok"]:
            _close(); cur = []; continue
        if cur:
            p = cur[-1]
            gap = c["t_onset"] - (p["t_onset"] + p["ttot"])
            same = c["ep"] == p["ep"] and c["win_id"] == p["win_id"]
            seam = c["ep"] == p["ep"] and c["win_id"] == p["win_id"] + 1
            if not ((same and gap <= 0.02) or (seam and gap <= seam_gap_s)):
                _close(); cur = []
        cur.append(c)
    _close()
    clean = np.zeros(g.size, bool)
    for a, b, _n in runs:
        clean |= (g >= a) & (g < b)
    return g, clean, runs


def flanked_clean(g, clean, t0, t1, flank_s=1.5):
    """True if clean breathing exists within flank_s BEFORE t0 and AFTER t1."""
    pre = (g >= t0 - flank_s) & (g < t0)
    post = (g > t1) & (g <= t1 + flank_s)
    return bool(pre.any() and post.any() and clean[pre].any() and clean[post].any())


PAUSE_CAP_S = 2.49            # SEG_KW bmaxP_s = 2.5 -> a pause piece at the cap is right-censored


@dataclass
class DPPause:
    t0_s: float
    t1_s: float
    dur_s: float
    ep: int
    n_pieces: int             # pause segments merged (seam splits / absorbed inadmissible breaths)
    censored: bool            # some piece hit the 2.5 s pause cap


def dp_breath_onsets(sess: RadarSession, fit_min=DP_CLEAN_FIT):
    """Onset times (s, sorted) of all breath-template segments with fit_tol >= fit_min."""
    on = [np.asarray(S, float)[(np.asarray(S, float)[:, 2] == 0) & (np.asarray(S, float)[:, 4] >= fit_min), 0]
          for S in sess.segments if np.asarray(S).size]
    return np.sort(np.concatenate(on)) if on else np.zeros(0)


def dp_pauses(sess: RadarSession, absorb_fit=FIT_MIN, min_pause_share=0.5, min_dur_s=0.3):
    """Merged cessations from the pause template. Two DP artefacts are undone:
      * seam split -- the 8 s non-overlapping DP blocks cut a pause crossing a block edge in two -> join;
      * forced breath -- the DP must alternate B<->P and caps a pause at 2.5 s, so a long flat stretch gets an
        INADMISSIBLE breath (fit < absorb_fit) wedged between pauses -> absorb it, provided pauses still make up
        >= min_pause_share of the merged span."""
    out = []
    for ep, S in enumerate(sess.segments):
        S = np.asarray(S, float)
        if S.size == 0:
            continue
        n, i = S.shape[0], 0
        while i < n:
            if S[i, 2] != 1:
                i += 1; continue
            j, k, npc, ptime = i, i + 1, 1, S[i, 3]
            while k < n:
                if S[k, 2] == 1 and abs(S[k, 0] - S[j, 1]) < 1e-3:
                    j, k, npc, ptime = k, k + 1, npc + 1, ptime + S[k, 3]
                elif (S[k, 2] == 0 and S[k, 4] < absorb_fit and k + 1 < n and S[k + 1, 2] == 1
                      and (ptime + S[k + 1, 3]) / (S[k + 1, 1] - S[i, 0]) >= min_pause_share):
                    j, k, npc, ptime = k + 1, k + 2, npc + 1, ptime + S[k + 1, 3]
                else:
                    break
            dur = float(S[j, 1] - S[i, 0])
            if dur >= min_dur_s:
                cens = bool(np.any((S[i:j + 1, 2] == 1) & (S[i:j + 1, 3] >= PAUSE_CAP_S)))
                out.append(DPPause(float(S[i, 0]), float(S[j, 1]), dur, ep, npc, cens))
            i = j + 1
    out.sort(key=lambda p: p.t0_s)
    return out


def dp_flank(onsets, t0, t1, flank_s=2.0, n_min=3):
    """Resp-reappears gate: >= n_min fit-ok breath onsets within flank_s before t0 AND after t1.
    Returns (ok, local_cycle_s = median onset spacing across both flanks)."""
    pre = onsets[(onsets >= t0 - flank_s) & (onsets < t0)]
    post = onsets[(onsets > t1 - 0.05) & (onsets <= t1 + flank_s)]
    d = np.concatenate([np.diff(pre), np.diff(post)])
    d = d[d < 1.0]                                   # spacing across a short cessation isn't a cycle
    cyc = float(np.median(d)) if d.size else float("nan")
    return bool(pre.size >= n_min and post.size >= n_min), cyc


# Residual-amplitude veto: a true cessation is FLAT on the radar; a posture change instead SHRINKS the radar breathing
# below the DP's relative admissibility floor, so the DP calls it a pause. B16: true apneas 0.02-0.27, such false
# pauses 0.42-0.54.
DP_RESIDUAL_MAX = 0.30


def pause_residual(sess: RadarSession, p, flank_s=1.5, band=(1.2, 9.0), trim=0.1):
    """In-pause band-passed RMS (inner 1-2*trim of the pause) / RMS of the flank_s either side."""
    from scipy.signal import butter, filtfilt
    # cache lives ON the session (a global id(sess)-keyed cache leaked across sessions in multi-recording loops)
    _cache = sess.__dict__.setdefault("_resid_cache", {})
    key = (band, p.ep)
    if key not in _cache:
        x = np.asarray(sess.trace[p.ep], float)
        b, a = butter(2, [band[0] / (sess.fs / 2), band[1] / (sess.fs / 2)], "bandpass")
        _cache[key] = filtfilt(b, a, x - x.mean())
    xb = _cache[key]; fs = sess.fs; e0 = float(sess.episodes[p.ep]["t0"])
    i0, i1 = int((p.t0_s - e0) * fs), int((p.t1_s - e0) * fs)
    m = int(trim * (i1 - i0))
    inn = xb[max(0, i0 + m):max(0, i1 - m)]
    fl = np.concatenate([xb[max(0, i0 - int(flank_s * fs)):max(0, i0)], xb[i1:i1 + int(flank_s * fs)]])
    if inn.size < 4 or fl.size < 4:
        return float("nan")
    return float(np.sqrt(np.mean(inn ** 2)) / (np.sqrt(np.mean(fl ** 2)) + 1e-12))


# ---- capture-file step artifact (07-xx captures only) ------------------------------------------------------------------
# The old live capture rescales every voxel at each .bin boundary (1066 frames = 4.264 s @ 250 Hz); a pause spanning one
# depressed file chunk reads as an apnea. 08-06+ captures have no step, so their grid comes back empty.
FILE_P = 1066 / 250.0
FILE_TOL = 0.2                                # an apnea with either end within this of a boundary is vetoed


def file_boundaries(power, fs, n=None, slab_s=120.0, k=20.0):
    """Frames where the all-voxel mean |frame diff| jumps > k x its median (s). Reads the whole cube once."""
    n = power.shape[0] if n is None else int(n); step = int(slab_s * fs)
    d = np.zeros(n)
    for lo in range(0, n, step):
        X = np.asarray(power[max(0, lo - 1):min(n, lo + step)], np.float64)
        d[max(1, lo):min(n, lo + step)] = np.abs(np.diff(X, axis=0)).mean(1)[-(min(n, lo + step) - max(1, lo)):]
    idx = np.where(d > k * np.median(d[1:]))[0]
    keep = idx[np.r_[True, np.diff(idx) > int(0.5 * fs)]] if idx.size else idx     # one per jump
    return keep / fs


def file_grid(b, rec_s):
    """Boundary times of the capture-file grid if the recording carries the per-file step (jumps at >= 50 % of the
    4.264 s slots, exact period); else empty. Off-grid jumps are movement."""
    if b.size < 10:
        return np.zeros(0)
    hist, e = np.histogram(b % FILE_P, bins=np.linspace(0, FILE_P, 86))
    ph = (e[hist.argmax()] + e[hist.argmax() + 1]) / 2
    grid = ph + FILE_P * np.arange(int(rec_s / FILE_P) + 1)
    on = np.abs(((b - ph + FILE_P / 2) % FILE_P) - FILE_P / 2) < 0.06
    return grid if on.sum() >= 0.5 * grid.size else np.zeros(0)


# ---- candidate table (run_apnea_cohort.detect_one) ---------------------------------------------------------------------
FLANKS = {"strict": (2.0, 3), "relaxed": (2.5, 2)}   # (flank_s, n breaths); strict = the validated default
MIN_CAND_S = 0.3
ACCLIM_S = 20 * 60                                   # ApneaCheck ignores the first 20 min (acclimation)


def candidates(sess: RadarSession, grid):
    """Every merged DP pause >= 0.3 s flanked by breathing (strict or relaxed), with its flank test, flatness residual
    and distance to capture-file boundaries. Times rounded as in the validated tables (the rules compare these)."""
    g, clean, _ = dp_clean_mask(sess)
    on = dp_breath_onsets(sess)
    to_grid = lambda t: float(np.min(np.abs(grid - t))) if grid.size else float("nan")
    out = []
    for p in dp_pauses(sess, min_dur_s=MIN_CAND_S):
        row = dict(t0_s=round(p.t0_s, 3), t1_s=round(p.t1_s, 3), dur_s=round(p.dur_s, 3), ep=p.ep,
                   n_pieces=p.n_pieces, censored=p.censored)
        keep = False
        for name, (fl, nb) in FLANKS.items():
            f_ok, cyc = dp_flank(on, p.t0_s, p.t1_s, fl, nb)
            row[f"flank_{name}"] = f_ok; row[f"cycle_{name}"] = round(cyc, 4) if np.isfinite(cyc) else float("nan")
            keep |= f_ok
        if not keep:
            continue
        row["residual"] = round(pause_residual(sess, p), 4)
        row["in_clean_flanked"] = flanked_clean(g, clean, p.t0_s, p.t1_s)
        row["d0_file"], row["d1_file"] = round(to_grid(p.t0_s), 3), round(to_grid(p.t1_s), 3)
        out.append(row)
    return out


def base_stats(sess: RadarSession):
    """Denominators from the single-voxel substrate: clean-breathing minutes and the median clean breath cycle."""
    g, clean, runs = dp_clean_mask(sess)
    ok = [c for c in dp_cycles(sess) if c["fit_ok"]]
    hop = g[1] - g[0]
    in_clean = lambda t: bool(clean[min(clean.size - 1, max(0, int(round(t / hop))))])
    clean_ttot = np.array([c["ttot"] for c in ok if in_clean(c["t_onset"])])
    rate = float(60 / np.median(clean_ttot)) if clean_ttot.size else float("nan")
    return dict(rec_min=float(sess.j.get("rec_min", np.nan)),
                presence_min=float(sum(e["t1"] - e["t0"] for e in sess.episodes) / 60),
                breathing_min=float(sum(c["ttot"] for c in ok) / 60),
                clean_min=float(clean.sum() * hop / 60),
                clean_min_post20=float(clean[g >= ACCLIM_S].sum() * hop / 60),
                n_clean_runs=len(runs), clean_rate_bpm=rate,
                cycle_med_s=60.0 / rate if np.isfinite(rate) else float("nan"),
                clean_grid=g, clean_mask=clean)
