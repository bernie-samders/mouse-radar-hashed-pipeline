r"""segmentation.py -- raw experiment -> two-template Breath/Pause segmentation substrate.

This is the front-end every metric pipeline builds on.  For one raw experiment directory it:

  1. WIDE PRESENCE TIMELINE (streamed): for each 4 s window, rank ALL voxels by resp-band SNR (cheap
     vectorised FFT), run the FAST two-template DP on the top-K shortlist, and pool the best fit as a
     breathing-presence score + the winning voxel.  (The breathing cell migrates over time; the wide
     shortlist recovers it without a narrow tracker.)
  2. EPISODES: threshold + bridge the presence timeline into breathing spans (>= MIN_SEG_S).
  3. PER-EPISODE EXTRACTION: refine the cell (+-1 range bin by autocorrelation), condition the trace,
     detect breaths, AND save the full DP breath/pause `segments` array -- the novel two-template output
     the metric pipelines consume.

Output: <out_dir>/<rec>.json + <rec>.npz  (one substrate per experiment).  Ported verbatim (same numbers)
from the validated `output/wide_resp/run_wide.py`; only the data loader + metadata were de-coupled from
the lab package and the experiment registry so it runs standalone on any raw experiment folder.

  NPZ arrays (object arrays, one entry per episode, aligned to json "episodes"):
    segments : [n_seg, 5] float32  -> [t_start_abs_s, t_end_abs_s, kind(0=breath,1=pause), dur_s, fit_tol]
    trace    : conditioned breathing trace @ fs (250 Hz)      raw : chosen-cell magnitude (already sqrt'd)
    breath_times, ibi, rate_t, rate_bpm, rate_stg, breath_idx : per-episode breath detection channels
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fast_dp as F
import conditioning as CD
from conditioning import (_highpass, _bandpass, _ac_strength, condition_trace, instantaneous_rate,
                          adaptive_breaths, cond_decimate, _decimate, SEG_KW,
                          AC_MIN, K_BIN_NEIGH, MIN_SEG_S, DEC_HZ, NFFT, FLOOR_BPM, RESP_HI_BPM, HF_LO, HF_HI)
import voxel_store as VS
import resp_segment as RS

# --- tunables (verbatim from run_wide.py) ----------------------------------------------------------
BATCH_S = 120.0
CTX_S, CORE_S, HOP_S = 8.0, 4.0, 4.0        # segmentation context / core / timeline hop
K = 16                                       # voxels segmented per window (K=16 validated)
PRES_THR = 0.55                              # best_fit presence threshold
BRIDGE_S = 8.0                               # bridge presence gaps up to this (interrupted breathing)
MAX_EP_S = 90.0                              # split longer presence runs into <=90 s episodes (bounds O(n^2) work)


def exp_meta(rec, cohort=None, age_weeks=None, severity_ref=False, genotype=None):
    """Per-experiment metadata baked into the substrate json.  Cohort A=wild-type, B=mutant (the metric
    pipelines map cohort -> genotype).  All fields are optional -- unknown experiments segment fine and
    just carry genotype='unknown' downstream."""
    return dict(recording=rec, cohort=cohort, genotype=genotype, age_weeks=age_weeks,
                severity_ref=bool(severity_ref), species="mouse")


def seg_kw(breath_bounds=None):
    """Two-template DP knobs; breath_bounds = (bminB_s, bmaxB_s) overrides the breath-MOTION bounds (default
    resp_segment 0.20-0.35 s). A breather faster than ~250/min needs breath_check.FAST_BOUNDS: its whole cycle is
    shorter than the 0.20 s minimum motion, so the default DP fuses breath pairs and halves the rate."""
    if breath_bounds is None:
        return SEG_KW
    return dict(SEG_KW, bminB_s=float(breath_bounds[0]), bmaxB_s=float(breath_bounds[1]))


def _episode_segments(xd, fs125, t0e, win_s=8.0, kw=SEG_KW):
    """FULL DP breath(0)/pause(1) segmentation over an episode's conditioned trace, in NON-overlapping
    8 s windows (the DP is superlinear in length).  Returns [n_seg, 5] float32; times are ABSOLUTE
    recording seconds.  Strict B<->P alternation, so consecutive rows pair into cycles."""
    n = xd.size; w = int(win_s * fs125); lo = int(3 * fs125)
    if n < lo:
        return np.zeros((0, 5), np.float32)
    out = []
    for a in range(0, n, w):
        seg = xd[a:a + w]
        if seg.size < lo:
            break
        res = F.segment_two_fast(seg, fs125, **kw)
        fitmap = {(p["a"], p["b"]): p["fit_tol"] for p in res["per"]}
        base = t0e + a / fs125
        for (s0, s1, kd) in res["segments"]:
            fit = fitmap.get((s0, s1), -1.0) if kd == 0 else -1.0
            out.append((base + s0 / fs125, base + s1 / fs125, float(kd), (s1 - s0) / fs125, float(fit)))
    return np.asarray(out, np.float32) if out else np.zeros((0, 5), np.float32)


# ==================================================================================================
# phase 1 -- wide presence timeline (streamed, overlapping slabs so edge windows keep full context)
# ==================================================================================================
def wide_presence_timeline(power, meta, fs, k=K, max_n=None, log=lambda *a: None, kw=SEG_KW):
    N = power.shape[0] if max_n is None else min(power.shape[0], int(max_n))
    q50 = max(1, int(round(fs / DEC_HZ))); fs50 = fs / q50
    fr = np.fft.rfftfreq(NFFT, 1.0 / fs50)
    fund = np.where((fr >= FLOOR_BPM / 60) & (fr <= RESP_HI_BPM / 60))[0]
    anar = np.where((fr >= HF_LO) & (fr <= min(HF_HI, 0.49 * fs50)))[0]
    step = int((BATCH_S - CTX_S) * fs)
    T, PRES, BCOL = [], [], []
    seen = set()
    for lo in range(0, N, step):
        hi = min(N, lo + int(BATCH_S * fs))
        if hi - lo < int(CTX_S * fs * 1.2):
            break
        mag = np.sqrt(np.maximum(np.asarray(power[lo:hi], np.float64), 0.0))     # (T, V)
        d50 = _decimate(mag.astype(np.float32), q50).astype(np.float64)
        hp50 = _highpass(d50, fs50, FLOOR_BPM, axis=0)
        base_t = lo / fs
        cond_cache = {}                                                          # col -> (xd125, fs125)
        c1 = (hi - lo) / fs - CTX_S / 2
        tc = CTX_S / 2
        while tc <= c1 + 1e-6:
            tabs = base_t + tc
            key = round(tabs, 1)
            if key in seen:
                tc += HOP_S; continue
            a50 = max(0, int((tc - CORE_S / 2) * fs50)); b50 = min(hp50.shape[0], int((tc + CORE_S / 2) * fs50))
            seg = hp50[a50:b50]
            if seg.shape[0] >= 8:
                hann = np.hanning(seg.shape[0])
                P = np.abs(np.fft.rfft(seg * hann[:, None], NFFT, axis=0)) ** 2
                ct = P[fund].max(0) / (np.median(P[anar], 0) + 1e-300)
                kk = min(k, ct.size)
                top = np.argpartition(ct, -kk)[-kk:]
                best_fit, best_c = 0.0, -1
                for v in top:
                    vv = int(v)
                    if vv not in cond_cache:
                        cond_cache[vv] = cond_decimate(mag[:, vv], fs)           # condition full slab once
                    xd, fs125 = cond_cache[vv]
                    sa = max(0, int((tc - CTX_S / 2) * fs125)); sb = min(xd.size, int((tc + CTX_S / 2) * fs125))
                    if sb - sa < int(3 * fs125):
                        continue
                    res = F.segment_two_fast(xd[sa:sb], fs125, **kw)
                    ca = int((CTX_S / 2 - CORE_S / 2) * fs125); cb = int((CTX_S / 2 + CORE_S / 2) * fs125)
                    fits = [p["fit_tol"] for p in res["per"] if ca <= (p["a"] + p["b"]) // 2 < cb]
                    mf = float(np.mean(fits)) if fits else 0.0
                    if mf > best_fit:
                        best_fit, best_c = mf, vv
                T.append(tabs); PRES.append(best_fit); BCOL.append(best_c); seen.add(key)
            tc += HOP_S
        log(f"    [wide-presence] {hi/fs:.0f}/{N/fs:.0f}s")
        del mag, d50, hp50, cond_cache
    order = np.argsort(T)
    return dict(t=np.asarray(T)[order], pres=np.asarray(PRES)[order], best_col=np.asarray(BCOL, int)[order])


# ==================================================================================================
# phase 2 -- episodes from the presence timeline (threshold + bridge + min length)
# ==================================================================================================
def episodes_from_presence(tl, pres_thr=PRES_THR, bridge_s=BRIDGE_S, min_seg_s=MIN_SEG_S):
    t, pres, col = tl["t"], tl["pres"], tl["best_col"]
    if t.size == 0:
        return []
    hop = np.median(np.diff(t)) if t.size > 1 else HOP_S
    bridge_w = max(1, int(round(bridge_s / hop)))
    keep = pres >= pres_thr
    idx = np.where(keep)[0]
    if idx.size == 0:
        return []
    runs = []; start = prev = idx[0]
    for i in idx[1:]:
        if i - prev <= bridge_w:
            prev = i
        else:
            runs.append((start, prev)); start = prev = i
    runs.append((start, prev))
    out = []
    for a, b in runs:
        t0, t1 = float(t[a]), float(t[b]) + hop        # include the last window's core
        if t1 - t0 < min_seg_s:
            continue
        cols = col[a:b + 1][keep[a:b + 1]]; cols = cols[cols >= 0]
        if cols.size == 0:
            continue
        dom = int(np.bincount(cols).argmax())
        dur = t1 - t0
        npc = max(1, int(np.ceil(dur / MAX_EP_S)))
        for p in range(npc):
            a2 = t0 + p * dur / npc; b2 = t0 + (p + 1) * dur / npc
            if b2 - a2 >= min_seg_s:
                out.append((a2, b2, dom))
    return out


# ==================================================================================================
# phase 3 -- per-episode extraction (refine cell, condition, detect breaths) + DP segments channel
# ==================================================================================================
def build_session(exp_dir, k=K, max_min=None, meta_extra=None, log=print, breath_bounds=None):
    power, meta, cfg = VS.load_experiment(exp_dir)
    rec = VS.rec_name(exp_dir)
    fs = float(meta["fs"]); nb = int(meta["n_bin"])
    hi_bpm = float(cfg.get("resp_hi_bpm", 480.0))
    info = dict(exp_meta(rec), **(meta_extra or {}))
    N = power.shape[0] if max_min is None else min(power.shape[0], int(max_min * 60 * fs))
    kw = seg_kw(breath_bounds)
    t0 = time.time()
    tl = wide_presence_timeline(power, meta, fs, k=k, max_n=N, log=log, kw=kw)
    segs = episodes_from_presence(tl)
    log(f"[segment] {rec}: {len(segs)} episodes from wide presence in {time.time()-t0:.0f}s")

    episodes = []
    arrays = dict(breath_times=[], ibi=[], rate_t=[], rate_bpm=[], rate_stg=[],
                  trace=[], raw=[], breath_idx=[], segments=[])
    for (t0e, t1e, dom) in segs:
        s0 = max(0, int(t0e * fs)); s1 = min(power.shape[0], int(t1e * fs))
        if s1 - s0 < int(MIN_SEG_S * fs):
            continue
        ai0, bj0 = divmod(int(dom), nb)
        # read ONLY the candidate columns (dom +-K_BIN_NEIGH bins); refine the cell by autocorrelation
        cand_cols = [ai0 * nb + bb for bb in range(bj0 - K_BIN_NEIGH, bj0 + K_BIN_NEIGH + 1) if 0 <= bb < nb]
        Wc = np.sqrt(np.maximum(np.asarray(power[s0:s1, cand_cols], np.float64), 0.0))    # (T, ~3)
        best = (-1.0, int(dom), 0.0, None)
        for j, c in enumerate(cand_cols):
            x = condition_trace(Wc[:, j], fs, None)
            acs, rate = _ac_strength(_bandpass(x, fs), fs)
            if acs > best[0]:
                best = (acs, c, rate, x)
        ac, cell, rate, trace = best
        if trace is None:
            continue
        ai, bj = divmod(cell, nb)
        med_bpm = float(rate if rate > 0 else 96.0)
        rate_t, rate_bpm, rate_stg = instantaneous_rate(trace, fs, hi_bpm=hi_bpm)
        pk, ibi, amp = adaptive_breaths(trace, fs, rate_t, rate_bpm, hi_bpm=hi_bpm)
        if pk.size < 6:
            continue
        cidx = cand_cols.index(cell)
        raw = Wc[:, cidx].astype(np.float32)                    # raw magnitude at the chosen cell (already sqrt'd)
        xd, fs125 = cond_decimate(Wc[:, cidx], fs)
        segments = _episode_segments(xd, fs125, t0e, kw=kw)     # the novel two-template output

        bt = t0e + pk / fs
        episodes.append(dict(t0=round(t0e, 2), t1=round(t1e, 2), dom_col=int(cell), ai=int(ai), bj=int(bj),
                             abs_bin=int(meta["bins"][bj]), angle=round(float(meta["angles_deg"][ai]), 1),
                             ac=round(float(ac), 3), med_bpm=round(med_bpm, 1), n_breaths=int(pk.size)))
        arrays["breath_times"].append(bt.astype(np.float32))
        arrays["ibi"].append(ibi.astype(np.float32))
        arrays["rate_t"].append((t0e + rate_t).astype(np.float32))
        arrays["rate_bpm"].append(rate_bpm.astype(np.float32))
        arrays["rate_stg"].append(rate_stg.astype(np.float32))
        arrays["trace"].append(trace.astype(np.float32))
        arrays["raw"].append(raw)
        arrays["breath_idx"].append(pk.astype(np.int32))
        arrays["segments"].append(segments)
    clean_min = sum(e["t1"] - e["t0"] for e in episodes) / 60.0
    n_ibi = int(sum(len(x) for x in arrays["ibi"]))
    log(f"[segment] {rec:20s} {len(episodes)} episodes, {clean_min:.1f} min, {n_ibi} IBIs, "
        f"median AC={np.median([e['ac'] for e in episodes]) if episodes else 0:.2f}")
    res = dict(method="wide_two_template", fs=fs, session_dir=os.path.abspath(exp_dir),
               n_episodes=len(episodes), clean_min=round(clean_min, 1), n_ibi=n_ibi,
               total_min=round(N / fs / 60.0, 2),                 # length actually processed (honours --max-min)
               rec_min=round(power.shape[0] / fs / 60.0, 2),      # full recording length on disk
               pres_thr=PRES_THR, k=k, breath_bounds_s=[kw.get("bminB_s", RS.BMIN_B_S), kw.get("bmaxB_s", RS.BMAX_B_S)],
               episodes=episodes, **info)
    return res, arrays


def segment_experiment(exp_dir, out_dir, k=K, max_min=None, skip_existing=False, meta_extra=None, log=print,
                       breath_bounds=None):
    """Segment one raw experiment directory and write <out_dir>/<rec>.{json,npz}.  Returns the json dict."""
    rec = VS.rec_name(exp_dir)
    os.makedirs(out_dir, exist_ok=True)
    jf = os.path.join(out_dir, rec + ".json")
    if skip_existing and os.path.exists(jf):
        log(f"[segment][skip] {rec} exists")
        return json.load(open(jf))
    res, arrays = build_session(exp_dir, k=k, max_min=max_min, meta_extra=meta_extra, log=log, breath_bounds=breath_bounds)
    json.dump(res, open(jf, "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(out_dir, rec + ".npz"),
                        **{k2: np.array(v, dtype=object) for k2, v in arrays.items()})
    log(f"  -> {jf}")
    return res


def parse_bounds(txt):
    """'0.12,0.25' -> (0.12, 0.25); None -> None."""
    if not txt:
        return None
    lo, hi = (float(v) for v in str(txt).split(","))
    if not 0 < lo < hi:
        raise ValueError(f"breath bounds must be 0 < min < max, got {txt!r}")
    return lo, hi


def main(argv=None):
    ap = argparse.ArgumentParser(description="raw experiment -> two-template segmentation substrate")
    ap.add_argument("--exp", required=True, help="raw experiment directory (contains *_voxels.f32/json)")
    ap.add_argument("--out", default=None, help="output segments dir (default: <package>/outputs/segments)")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--max-min", type=float, default=None, help="only process the first N minutes (for a quick test)")
    ap.add_argument("--cohort", default=None, help="A (wild-type) or B (mutant), if known -- optional")
    ap.add_argument("--age-weeks", type=float, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--breath-bounds", default=None,
                    help="breath-motion bounds in s, e.g. 0.12,0.21 for a fast (~300/min) breather (default 0.20,0.35)")
    args = ap.parse_args(argv)
    out_dir = args.out or os.path.join(os.path.dirname(_HERE), "outputs", "segments")
    meta_extra = {}
    if args.cohort is not None:
        meta_extra["cohort"] = args.cohort
    if args.age_weeks is not None:
        meta_extra["age_weeks"] = args.age_weeks
    segment_experiment(args.exp, out_dir, k=args.k, max_min=args.max_min,
                       skip_existing=args.skip_existing, meta_extra=meta_extra,
                       log=lambda *a: print(*a, flush=True), breath_bounds=parse_bounds(args.breath_bounds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
