r"""substrate.py -- load the two-template Breath/Pause segmentation and build the per-cycle series.

This is the foundation of the Stettner-from-two-template port (see SPEC.md).  It reads the cached
wide substrate DIRECTLY (output/wide_resp/substrate/<rec>.{json,npz}) -- NOT via apnea_lib -- because
we need the `segments` channel (the DP breath/pause segmentation) and `raw` (to reproduce the exact
125 Hz signal the segmenter saw, for template morphology).  No G:\ access is required.

Per-cycle model (SPEC 1.2):  a respiratory CYCLE = one breath-MOTION segment (kind 0, template mu_B,
amplitude-invariant shape-fit) + the immediately-following PAUSE segment (kind 1, flat mu_P==0).
Strict B<->P alternation, so consecutive (B, P) rows pair into cycles:
    ttot   = tbreath + tpause              (cycle length; CLIP-INVARIANT -- SPEC 1.2/3.1)
    tbreath = B.dur_s                       (bounded [0.20,0.35] s; ceiling-clipped -> low-info alone)
    tpause  = P.dur_s                       (end-expiratory quiescence; bounded [0.02,2.5] s)

SEAM HYGIENE (gotcha G2): run_wide segmented each episode in independent NON-overlapping 8 s DP blocks
(seg = xd[a:a+8s]).  No segment crosses a block seam, but the first/last segment of each block is a
boundary artifact (the DP restarts D[0,0]=D[0,1]=0, usually opening on a partial floor-pause).  So we
tag each cycle with its 8 s window id and DROP the first and last cycle of every block.

Grouping is on COHORT (A=wild-type, B=mutant), never the legacy `genotype` string (SPEC 2.5).
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# The segmentation step (engine/segmentation.py) writes one <rec>.{json,npz} substrate per experiment
# here; every segment-based pipeline (stettner, pause_distribution) reads it back. Override SUB_DIR to
# point at a different segments directory if you keep your outputs elsewhere.
SUB_DIR = os.path.join(os.path.dirname(_HERE), "outputs", "segments")
OUT_DIR = _HERE
WIN_S = 8.0                                # the DP sub-block length run_wide used (fixed; NEVER whole-episode)
FIT_MIN = 0.35                             # breath decision fit gate (two_template.breathing_segments default)

# calm behavioural band + AC arousal floor for REGULARITY metrics (SPEC 4.5, gotcha #4)
CALM_BAND = (0.25, 1.0)                     # keep cycles with ttot in [0.25,1.0] s  <=>  1-4 Hz
AC_FLOOR = 0.45                            # per-episode autocorrelation floor (== clean_substrate.AC_MIN)


# ==================================================================================================
# recording enumeration + metadata (group on COHORT)
# ==================================================================================================
def list_recordings():
    """All recordings with a cached wide substrate json, sorted."""
    return sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(SUB_DIR, "*.json")))


def _geno_from_cohort(cohort):
    return {"A": "wild-type", "B": "mutant"}.get(str(cohort), "unknown")


def load_meta(rec):
    """The substrate json for one recording."""
    with open(os.path.join(SUB_DIR, rec + ".json")) as fh:
        return json.load(fh)


def rec_info(rec, meta=None):
    """Normalised metadata dict (cohort-derived genotype, matched-batch flag, rate)."""
    m = meta if meta is not None else load_meta(rec)
    eps = m.get("episodes", []) or []
    bpm = [e.get("med_bpm") for e in eps if e.get("med_bpm")]
    sev = bool(m.get("severity_ref") in (True, "TRUE", "True", 1))
    return dict(
        recording=rec, cohort=m.get("cohort"),
        genotype=_geno_from_cohort(m.get("cohort")),           # A->wild-type, B->mutant (authoritative)
        genotype_legacy=m.get("genotype"),
        age_weeks=m.get("age_weeks"), severity_ref=sev,
        is_matched=sev,                                        # severity_ref TRUE == matched 8wk Match4/5 batch
        fs=float(m.get("fs", 250.0)),
        n_episodes=int(m.get("n_episodes", len(eps))),
        clean_min=m.get("clean_min"),
        med_bpm=round(float(np.median(bpm)), 1) if bpm else None,
    )


# ==================================================================================================
# per-cycle series from the cached `segments` channel (Path A -- SPEC 3.1)
# ==================================================================================================
def _pair_cycles(S, t0e, fit_min=FIT_MIN):
    """Pair (B, following P) segments of one episode's [n_seg,5] array into cycles, with 8 s-seam
    hygiene.  Returns a list of dicts (empty if none survive).  Columns of S:
        [t_start_abs_s, t_end_abs_s, kind(0=B/1=P), dur_s, fit_tol(-1 for pause)]."""
    S = np.asarray(S, float)
    if S.ndim != 2 or S.shape[0] < 2:
        return []
    win_id = np.floor((S[:, 0] - t0e) / WIN_S + 1e-6).astype(int)     # 8 s DP block index
    cyc = []
    n = S.shape[0]
    for i in range(n - 1):
        if S[i, 2] != 0.0 or S[i + 1, 2] != 1.0:                     # need B immediately followed by P
            continue
        tbreath = float(S[i, 3]); tpause = float(S[i + 1, 3])
        cyc.append(dict(
            t_onset=float(S[i, 0]),                                  # absolute breath-onset time (s)
            tbreath=tbreath, tpause=tpause, ttot=tbreath + tpause,
            fit=float(S[i, 4]), win_id=int(win_id[i]),
        ))
    if not cyc:
        return []
    # SEAM HYGIENE: drop the first and last cycle of each 8 s block (boundary artifacts)
    from collections import defaultdict
    by_win = defaultdict(list)
    for c in cyc:
        by_win[c["win_id"]].append(c)
    kept = []
    for w, cs in by_win.items():
        cs = sorted(cs, key=lambda c: c["t_onset"])
        inner = cs[1:-1] if len(cs) >= 3 else []                     # need >=3 to keep an interior cycle
        for c in inner:
            c = dict(c); c["fit_ok"] = bool(c["fit"] >= fit_min)
            kept.append(c)
    kept.sort(key=lambda c: c["t_onset"])
    return kept


def build_cycle_series(rec, meta=None, npz=None):
    """Full per-cycle series for one recording, pooled across episodes (each cycle tagged with its
    episode index, episode AC, and a global window id).  Returns dict:
        cycles: list of per-cycle dicts (t_onset, tbreath, tpause, ttot, fit, fit_ok, ep, ac, med_bpm,
                gwin -- a globally-unique 8 s-window id), sorted by t_onset;
        info:   rec_info(rec).
    The wide substrate is deliberately NOT AC-gated; callers apply CALM_BAND / AC_FLOOR themselves
    (regularity) or use the full series (burden) -- SPEC 4.5."""
    m = meta if meta is not None else load_meta(rec)
    if npz is None:
        npz = np.load(os.path.join(SUB_DIR, rec + ".npz"), allow_pickle=True)
    segs_all = npz["segments"]
    eps = m.get("episodes", []) or []
    cycles = []
    for ep in range(len(segs_all)):
        S = np.asarray(segs_all[ep])
        if S.size == 0:
            continue
        e = eps[ep] if ep < len(eps) else {}
        t0e = float(e.get("t0", S[0, 0] if S.size else 0.0))
        ac = e.get("ac"); med_bpm = e.get("med_bpm")
        # win_id is PER-EPISODE (8 s DP block index); seam hygiene groups within (ep, win_id), so no
        # global window id is needed -- downstream windowing always regroups by episode.
        for c in _pair_cycles(S, t0e):
            c["ep"] = ep
            c["ac"] = float(ac) if ac is not None else np.nan
            c["med_bpm"] = float(med_bpm) if med_bpm else np.nan
            cycles.append(c)
    cycles.sort(key=lambda c: c["t_onset"])
    return dict(cycles=cycles, info=rec_info(rec, m))


def episode_signal(rec, ep, meta=None, npz=None):
    """Reproduce the exact 125 Hz conditioned signal the segmenter saw for episode `ep`, from cached
    `raw` (bit-identical to run_wide -- verified).  Returns (xd125, fs125, t0e)."""
    m = meta if meta is not None else load_meta(rec)
    if npz is None:
        npz = np.load(os.path.join(SUB_DIR, rec + ".npz"), allow_pickle=True)
    from conditioning import cond_decimate
    raw = np.asarray(npz["raw"][ep], float)
    fs = float(m.get("fs", 250.0))
    xd, fs125 = cond_decimate(raw, fs)
    e = (m.get("episodes") or [{}])[ep] if ep < len(m.get("episodes", [])) else {}
    return xd, float(fs125), float(e.get("t0", 0.0))


# ==================================================================================================
# convenience filters (SPEC 4.5 substrate split)
# ==================================================================================================
def regularity_cycles(cycles, calm_band=CALM_BAND, ac_floor=AC_FLOOR):
    """Arousal-matched substrate for REGULARITY/COUPLING: calm-rate cycles from AC>=floor episodes."""
    out = []
    for c in cycles:
        if not np.isfinite(c.get("ac", np.nan)) or c["ac"] < ac_floor:
            continue
        if not (calm_band[0] <= c["ttot"] <= calm_band[1]):
            continue
        out.append(c)
    return out


def burden_cycles(cycles):
    """FULL substrate for BURDEN (never AC-gated -- the gate removes the arrhythmic long-pause episodes)."""
    return list(cycles)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="inspect the two-template per-cycle substrate")
    ap.add_argument("--rec", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    recs = list_recordings() if args.all else [args.rec or "Match5Mouse1"]
    print(f"{'recording':22s} {'coh':3s} {'geno':10s} {'age':4s} {'mch':3s} {'n_cyc':>6s} "
          f"{'n_reg':>6s} {'medTtot':>8s} {'medPause':>8s}")
    for rec in recs:
        cs = build_cycle_series(rec)
        cyc = cs["cycles"]; info = cs["info"]
        reg = regularity_cycles(cyc)
        ttot = np.array([c["ttot"] for c in cyc]) if cyc else np.array([np.nan])
        pause = np.array([c["tpause"] for c in cyc]) if cyc else np.array([np.nan])
        print(f"{rec:22s} {str(info['cohort']):3s} {info['genotype']:10s} {str(info['age_weeks']):4s} "
              f"{'Y' if info['is_matched'] else '.':3s} {len(cyc):6d} {len(reg):6d} "
              f"{np.median(ttot)*1000:8.1f} {np.median(pause)*1000:8.1f}")
