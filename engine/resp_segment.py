"""resp_segment.py -- timing-based (EQ-Radio) respiration segmenter for B9.

Reuses the EQ-Radio DP + template + confidence primitives from ml_apnea/cohort/eqradio_hr.py
(imported, never edited) and adapts them to RESPIRATION:
  - no acceleration preprocessing (that suppresses breathing; it is cardiac-specific)
  - respiration length bounds from Stage 0 (breath cycle ~0.3-1.45 s at 50 Hz)

Stage 1 (this file, single-template): wrap eqradio_hr.segment_signal on the conditioned breathing
trace -- the "get lucky" path.  Stage 2 will add the second (variable pause) template here.

Run:  python output/eqradio_resp/resp_segment.py      # single-template demo on B9 clean spans + null
"""
from __future__ import annotations
import os, sys
import numpy as np

# --- import the EQ-Radio primitives (self-contained module; safe to import) -------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import eqradio_hr as eq  # noqa: E402
import common as C  # noqa: E402

# --- Stage-0 respiration bounds (seconds) ----------------------------------------------------------
BMIN_CYCLE_S, BMAX_CYCLE_S = 0.28, 1.45   # single-template whole breath cycle
# two-template breath MOTION bounds bracket the motion TIGHTLY on BOTH ends: a short bmaxB stops a
# breath segment from swallowing the following exhale-pause (which must become a P segment), and a
# non-trivial bminB stops one true breath being split into multiple short pieces (so breath/pause
# LENGTHS are recoverable -- the arrhythmia read-out).  See two_template stage2 self-test.
BMIN_B_S, BMAX_B_S = 0.20, 0.35           # two-template: breath MOTION (Stage 2)
# strict B<->P alternation means EVERY cycle has a pause, so bminP must allow a near-zero exhale-hold
# for fast breathing (else a 2-sample pause would be forced to steal from the motion); bmaxP up to a
# multi-second apnea (one P between the flanking breaths).
BMIN_P_S, BMAX_P_S = 0.02, 4.00           # two-template: exhale PAUSE  (Stage 2)


# ==================================================================================================
# Stage 1 -- single-template respiration segmentation
# ==================================================================================================
def segment_single(x, fs, bmin_s=BMIN_CYCLE_S, bmax_s=BMAX_CYCLE_S, m=None, max_iter=16):
    """One-template EQ-Radio segmentation of a conditioned breathing trace.  Returns the eqradio_hr
    result dict: boundaries (incl. 0,n), ibi_s, mu, var_final, confidence, n_iter, bmin, bmax, m."""
    x = np.asarray(x, float)
    return eq.segment_signal(x, fs, bmin_s=bmin_s, bmax_s=bmax_s, m=m, init="auto", max_iter=max_iter)


# ==================================================================================================
# Utilities: surrogate null, boundary/breath alignment
# ==================================================================================================
def phase_surrogate(x, rng):
    """Phase-randomised surrogate: same power spectrum, destroyed temporal structure.  A genuine
    template lock must COLLAPSE on this (confidence ~0)."""
    x = np.asarray(x, float) - np.mean(x)
    X = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, X.shape[0])
    ph[0] = 0.0
    if x.size % 2 == 0:
        ph[-1] = 0.0
    Xs = np.abs(X) * np.exp(1j * ph)
    return np.fft.irfft(Xs, n=x.size)


def boundary_stats(boundaries, breaths_idx, fs):
    """How well the recovered CYCLE boundaries line up with the lab breath peaks.
    Returns dict(n_bounds, n_breaths, count_ratio, offset_ms, jitter_ms, matched_frac).
      offset_ms = median signed boundary->nearest-breath distance (a constant phase is fine)
      jitter_ms = robust spread of that distance after removing the offset (the LOCK quality)
      matched_frac = fraction of breaths within +-0.5*median-IBI of some boundary (recall of breaths)"""
    b = np.asarray(list(boundaries)[1:-1], float)  # interior cut points
    g = np.asarray(sorted(breaths_idx), float)
    out = dict(n_bounds=int(b.size), n_breaths=int(g.size),
               count_ratio=float(b.size / max(1, g.size)),
               offset_ms=float("nan"), jitter_ms=float("nan"), matched_frac=float("nan"))
    if b.size < 3 or g.size < 3:
        return out
    d = []
    for p in b:
        j = int(np.clip(np.searchsorted(g, p), 1, g.size - 1))
        near = g[j - 1] if abs(p - g[j - 1]) <= abs(p - g[j]) else g[j]
        d.append(p - near)
    d = np.asarray(d)
    off = float(np.median(d))
    jit = float(1.4826 * np.median(np.abs(d - off)))
    med_ibi = float(np.median(np.diff(g)))
    tol = 0.5 * med_ibi
    # breath recall: each breath near some boundary
    matched = 0
    for gi in g:
        k = int(np.clip(np.searchsorted(b, gi), 1, b.size - 1))
        nb = min(abs(gi - b[k - 1]), abs(gi - b[k]))
        if nb <= tol:
            matched += 1
    out.update(offset_ms=off / fs * 1000.0, jitter_ms=jit / fs * 1000.0,
               matched_frac=matched / g.size)
    return out


# ==================================================================================================
# Stage-1 demo on B9 clean spans
# ==================================================================================================
def stage1_demo(max_spans=None, make_fig=True):
    prep = C.load_prep()
    labels = C.load_labels()
    rng = np.random.default_rng(0)

    rows = []       # per-span diagnostics
    null_conf = []  # surrogate confidences
    example = None
    spans = list(C.iter_clean_traces(prep, labels))
    if max_spans:
        spans = spans[:max_spans]

    for l, seg, tr, env, fs, t_abs in spans:
        if tr.size < int(2.0 * fs):
            continue
        res = segment_single(tr, fs)
        breaths_abs = np.asarray(seg["breaths"], float)
        bi = np.round((breaths_abs[(breaths_abs >= t_abs[0]) & (breaths_abs <= t_abs[-1])]
                       - t_abs[0]) * fs).astype(int)
        bstat = boundary_stats(res["boundaries"], bi, fs)
        rec_bpm = 60.0 / np.median(res["ibi_s"]) if res["ibi_s"].size else float("nan")
        rows.append(dict(seg_id=l["seg_id"], dur=t_abs[-1] - t_abs[0], conf=res["confidence"],
                         n_seg=len(res["boundaries"]) - 1, rec_bpm=rec_bpm,
                         lab_bpm=seg.get("med_bpm"), lab_ac=seg.get("ac"), **bstat))
        # matched surrogate null
        sur = phase_surrogate(tr, rng)
        null_conf.append(segment_single(sur, fs)["confidence"])
        if example is None and 8.0 < (t_abs[-1] - t_abs[0]) < 20.0 and bi.size >= 10:
            example = (l, tr, fs, t_abs, bi, res)

    _report_stage1(rows, null_conf)
    if make_fig and example is not None:
        _fig_stage1(example)
    return rows


def _report_stage1(rows, null_conf):
    import statistics as st
    print("=" * 78)
    print("STAGE 1 -- single-template respiration segmenter on B9 clean spans")
    print("=" * 78)
    conf = [r["conf"] for r in rows]
    jit = [r["jitter_ms"] for r in rows if np.isfinite(r["jitter_ms"])]
    mf = [r["matched_frac"] for r in rows if np.isfinite(r["matched_frac"])]
    cr = [r["count_ratio"] for r in rows]
    print(f"spans analysed: {len(rows)}")
    print(f"confidence      : median={np.median(conf):.3f}  IQR[{np.percentile(conf,25):.3f},"
          f"{np.percentile(conf,75):.3f}]")
    if null_conf:
        print(f"  surrogate null: median={np.median(null_conf):.3f}  p95={np.percentile(null_conf,95):.3f}"
              f"   (must be << real -> lock is genuine)")
    print(f"breath recall   : median={np.median(mf):.3f}  (frac of lab breaths near a recovered boundary)")
    print(f"boundary jitter : median={np.median(jit):.1f} ms  (lock phase spread)")
    print(f"seg/breath count: median ratio={np.median(cr):.2f}  (1.0 = one cycle per breath)")
    # rate agreement
    dbpm = [abs(r["rec_bpm"] - r["lab_bpm"]) for r in rows
            if r.get("lab_bpm") and np.isfinite(r["rec_bpm"])]
    if dbpm:
        print(f"rate |rec-lab|  : median={np.median(dbpm):.1f} bpm")
    # where it struggles: low-AC (arrhythmic) spans
    lowac = [r for r in rows if (r.get("lab_ac") or 1) < 0.3]
    if lowac:
        print(f"\nlow-AC (arrhythmic, ac<0.3) spans: n={len(lowac)}  "
              f"median conf={np.median([r['conf'] for r in lowac]):.3f}  "
              f"median recall={np.median([r['matched_frac'] for r in lowac if np.isfinite(r['matched_frac'])]):.3f}")


def _fig_stage1(example):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable: {e})"); return
    l, tr, fs, t_abs, bi, res = example
    bounds = np.asarray(res["boundaries"][1:-1], int)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(t_abs, tr, lw=0.7, color="0.4", label="trace")
    for k, bd in enumerate(bounds):
        if 0 <= bd < t_abs.size:
            ax.axvline(t_abs[bd], color="C0", lw=0.8, alpha=0.7,
                       label="recovered cycle bound" if k == 0 else None)
    ax.plot(t_abs[bi[bi < t_abs.size]], tr[bi[bi < t_abs.size]], "v", color="C1", ms=6, label="lab breaths")
    ax.set_title(f"Stage1 single-template: span[{l['t0']:.0f},{l['t1']:.0f}]s  "
                 f"conf={res['confidence']:.2f}  rec_bpm={60/np.median(res['ibi_s']):.0f}  lab_ac={l.get('seg_id')}")
    ax.set_xlabel("time (s)"); ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    os.makedirs(C.FIGS, exist_ok=True)
    fig.savefig(os.path.join(C.FIGS, "stage1_single_template.png"), dpi=120)
    plt.close(fig)
    print(f"\nfigure -> {os.path.join(C.FIGS, 'stage1_single_template.png')}")


if __name__ == "__main__":
    stage1_demo()
