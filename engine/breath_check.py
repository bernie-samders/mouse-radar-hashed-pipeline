r"""breath_check.py -- does the EQ-Radio segmentation count every breath? Rate-free check for fast breathers.

The two-template DP's breath-MOTION bounds (resp_segment 0.20-0.35 s) fit breathing up to ~250/min. A faster breather
(B17 ~300/min, ng511 ~320/min) has a whole cycle shorter than the 0.20 s minimum motion, so the DP fuses breath pairs and
reports HALF the rate -- and everything downstream (clean runs, breath-order removal for HR) inherits the half-rate clock.

Check: per clean run, the spectral peak of the segmenter's OWN conditioned trace (npz `trace`, the signal the DP fitted)
in a rate-free band vs the DP rate (60 / median ttot). ~1.0 = every breath counted; ~2 = halved -> re-segment with
FAST_BOUNDS (segmentation.py --breath-bounds 0.12,0.21).

CLIPPED (new in 1.6): a breather just above the default bounds' reach (ng511 10-21, spectral 267/min) is not halved
(ratio 1.03) -- the DP squeezes its breaths against the 0.20 s minimum motion instead (30 % of clean-run breath motions
at the bound vs 1-3 % normally; segmenter pinned at 250/min). Clipped -> the same fast re-segmentation.

    python breath_check.py <segments_dir>/<rec>      # prints the ratio and the verdict
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.signal import welch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import dp_apnea as DA
import resp_segment as RS

# bminB 0.12: B17 09-28 vs WBP (multivoxel_combine --bmin-b 0.12) per-breath median |err| 136 -> 9 ms, B16 (~170/min)
# unchanged. bmaxB 0.21 (~1.05 cycles at 300/min): at 0.25 ng511 (0.19 s cycle) locks ~5 % of cycles on a 1.3-1.5-breath
# template pinned at bmaxB; 0.21 gives one breath per cycle 95 % (= 508) vs 91 % (fast_breath/README.md). B17 vs WBP:
# bmaxB 0.22 = 0.25.
FAST_BOUNDS = (0.12, 0.21)
HALVED_RATIO = 1.3          # 508 / 507d2 / B16 1.00-1.01; ng511 1.89; B17 single-voxel 1.59 (mixed 1:1 / 2:1 cycles)
CLIPPED_PINNED = 0.15      # share of clean-run breath motions at the minimum bound: 7 normal ECG sessions 0.01-0.03;
                            # ng511 10-21 0.30, ng511 10-15 (default bounds, also halved) 0.22
CLIPPED_MIN_BPM = 240.0     # ...with a spectral rate this fast: normal <= 203/min, ng511 10-21 267, ng511 10-15 304
BAND_BPM = (48.0, 480.0)    # rate-free: the wide-presence resp band ceiling (conditioning.RESP_HI_BPM)
MIN_RUN_S = 10.0


def rate_ratio(seg_base, min_run_s=MIN_RUN_S):
    """Duration-weighted median of (spectral peak rate / DP rate) over clean runs >= min_run_s.
    Returns dict(ratio, n_runs, run_min, spec_bpm, dp_bpm, pinned_frac, bmin_b, runs=[(t0, t1, spec, dp)]);
    pinned_frac = share of those runs' breath motions within 1 sample of the segmentation's minimum motion bmin_b."""
    s = DA.RadarSession(seg_base + ".json", seg_base + ".npz")
    cyc = [c for c in DA.dp_cycles(s) if c["fit_ok"]]
    on = np.array([c["t_onset"] for c in cyc]); tt = np.array([c["ttot"] for c in cyc])
    tb = np.array([c["tbreath"] for c in cyc]); ep_of = np.array([c["ep"] for c in cyc])
    bb = json.load(open(seg_base + ".json", encoding="utf-8")).get("breath_bounds_s")
    bmin_b = float(bb[0]) if bb else RS.BMIN_B_S
    fs = s.fs; rows = []; used = np.zeros(on.size, bool)
    for a, b, _n in DA.dp_clean_mask(s, cycles=None)[2]:
        if b - a < min_run_s:
            continue
        m = (on >= a) & (on < b)
        if m.sum() < 5:
            continue
        ep = int(np.bincount(ep_of[m]).argmax())
        x = np.asarray(s.trace[ep], float); e0 = float(s.episodes[ep]["t0"])
        i0, i1 = max(0, int((a - e0) * fs)), min(x.size, int((b - e0) * fs))
        if i1 - i0 < int(4 * fs):
            continue
        used |= m
        f, P = welch(x[i0:i1] - x[i0:i1].mean(), fs=fs, nperseg=int(4 * fs), nfft=8192)
        sel = (f * 60 >= BAND_BPM[0]) & (f * 60 <= BAND_BPM[1])
        rows.append((float(a), float(b), float(f[sel][np.argmax(P[sel])] * 60), float(60.0 / np.median(tt[m]))))
    if not rows:
        return dict(ratio=float("nan"), n_runs=0, run_min=0.0, spec_bpm=float("nan"), dp_bpm=float("nan"),
                    pinned_frac=float("nan"), bmin_b=bmin_b, runs=[])
    R = np.array(rows); w = R[:, 1] - R[:, 0]; r = R[:, 2] / R[:, 3]
    o = np.argsort(r); cw = np.cumsum(w[o]) / w.sum()
    return dict(ratio=float(r[o][np.searchsorted(cw, 0.5)]), n_runs=len(rows), run_min=float(w.sum() / 60),
                spec_bpm=float(np.median(R[:, 2])), dp_bpm=float(np.median(R[:, 3])),
                pinned_frac=float(np.mean(tb[used] <= bmin_b + 1.5 / fs)), bmin_b=bmin_b, runs=rows)


def is_halved(chk):
    return bool(np.isfinite(chk["ratio"]) and chk["ratio"] >= HALVED_RATIO)


def is_clipped(chk):
    """Breaths squeezed against the minimum motion bound by a breather too fast for the bounds (not halved)."""
    p = chk.get("pinned_frac", float("nan"))
    return bool(np.isfinite(p) and p >= CLIPPED_PINNED and np.isfinite(chk["spec_bpm"])
                and chk["spec_bpm"] >= CLIPPED_MIN_BPM)


def needs_fast(chk):
    """Halved or clipped: the segmentation should be re-made with FAST_BOUNDS."""
    return is_halved(chk) or is_clipped(chk)


def problem(chk):
    """One-phrase description of what is wrong ('' if nothing)."""
    if is_halved(chk):
        return "HALVED (every other breath counted)"
    if is_clipped(chk):
        return (f"CLIPPED ({100 * chk['pinned_frac']:.0f}% of breath motions at the {chk['bmin_b']:g} s minimum, "
                f"spectral {chk['spec_bpm']:.0f}/min: too fast for these bounds)")
    return ""


def halved_note(chk, what="heart rates"):
    """Plain-English warning for a halved segmentation (the pipelines' notes / logs)."""
    return (f"BREATHING HALVED: the segmentation counted every other breath (spectral {chk['spec_bpm']:.0f}/min vs "
            f"segmenter {chk['dp_bpm']:.0f}/min, ratio {chk['ratio']:.2f}), so breath removal and the heart band run on a "
            f"half-rate clock. These {what} are NOT trustworthy (on ng511 they read ~110 bpm high: a breathing "
            f"harmonic, which also looks like a strong heart signal). Re-segment with --breath-bounds "
            f"{FAST_BOUNDS[0]:g},{FAST_BOUNDS[1]:g} (run.py segment / all do this automatically) and re-run")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    for base in argv:
        base = base[:-5] if base.endswith((".json")) else base
        c = rate_ratio(base)
        verdict = (problem(c) + " -> re-segment with --breath-bounds %g,%g" % FAST_BOUNDS) if needs_fast(c) else "ok"
        bb = json.load(open(base + ".json")).get("breath_bounds_s")
        print(f"{os.path.basename(base)}: spectral/DP rate ratio {c['ratio']:.2f} over {c['n_runs']} clean runs "
              f"({c['run_min']:.1f} min); spectral {c['spec_bpm']:.0f}/min, DP {c['dp_bpm']:.0f}/min; "
              f"{100 * c['pinned_frac']:.0f}% of breath motions at the {c['bmin_b']:g} s minimum; "
              f"bounds {bb or 'default'} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
