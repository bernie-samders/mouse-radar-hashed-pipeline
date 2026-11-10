r"""run.py -- top-level driver for the mouse radar pipeline.

    python run.py all      --exp <raw_experiment_dir> [--out DIR] [--max-min N] [--cohort A|B] [--age-weeks W]
                           [--workers N] [--breath-bounds auto|default|LO,HI] [--dejump auto|on|off] [--keep-dejumped]
    python run.py segment  --exp <raw_experiment_dir> [--out DIR] [--max-min N] [--cohort A|B]
                           [--breath-bounds auto|default|LO,HI]
    python run.py stettner --rec <recording_name>  [--seg DIR] [--out DIR]
    python run.py pauses   --rec <recording_name>  [--seg DIR] [--out DIR]
    python run.py motion   --exp <raw_experiment_dir> [--out DIR] [--max-min N]
    python run.py apnea    --rec <recording_name>  [--exp <raw_experiment_dir>] [--seg DIR] [--out DIR]
    python run.py hr       --rec <recording_name>  [--exp <raw_experiment_dir>] [--seg DIR] [--out DIR] [--workers N]
                           [--patch-beams A-B] [--min-period-s S] [--dejump auto|on|off]
    python run.py b2b      --rec <recording_name>  [--exp <raw_experiment_dir>] [--seg DIR] [--out DIR] [--workers N]
                           [--dejump auto|on|off]                     # after hr: beat-to-beat HR (B2B.md, new in 1.7)
    python run.py dejump   --exp <raw_experiment_dir> [--out DIR] [--bins hr|all|LO-HI] [--detect-only]
    python run.py combine  [--out DIR]        # merge every experiment's metric CSVs into master tables
    python run.py list                         # list registered pipelines

`all` = segment the raw experiment once, then run every registered metric pipeline on it.  Segments land
in <out>/segments/<rec>.{json,npz}; per-experiment metrics land in <out>/metrics/.  Default <out> is
<package>/outputs.  Use --max-min to process only the first N minutes for a quick end-to-end test.

Fast breathers: the segmenter's default breath bounds (0.20-0.35 s) count every other breath above ~250/min (or, just
above their reach, squeeze breaths against the 0.20 s minimum: 'clipped', new in 1.6), and
every breathing metric and heart rate then runs on a half-rate clock. With --breath-bounds auto (the default),
`segment` / `all` check every segmentation (engine/breath_check.py) and, if it is halved or clipped, re-segment with
the fast bounds 0.12,0.21 s; the default-bounds one is kept in <out>/segments/default_bounds/. `default` = never
re-segment.

July-2026 captures (new in 1.5, DEJUMP.md): that live-capture build steps every voxel at each 4.264 s capture file and
can drop single frames, which buries the heart. With --dejump auto (the default), `all` / `hr` / `b2b` check the
cube once and, when it has the file-step grid or dropouts, run the heart-rate steps on a repaired copy written to
<out>/dejumped/ (`all` deletes it afterwards unless --keep-dejumped). Breathing / apnea / motion always use the raw cube.
`on` = repair whenever any jump is found; `off` = never. `dejump` alone writes (or, --detect-only, just reports) a copy.
"""
from __future__ import annotations
import os, sys, csv, glob, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "engine"), os.path.join(_HERE, "pipelines"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import segmentation as SEG
import voxel_store as VS
import dejump as DJ
from pipelines import REGISTRY, PIPELINES, QC

DEF_OUT = os.path.join(_HERE, "outputs")
DJ_HELP = ("July-2026 capture-build jumps (DEJUMP.md): auto (default) = run the heart-rate steps on a repaired copy "
           "when the cube has the file-step grid or dropouts; on = whenever any jump is found; off = raw cube")
BB_HELP = ("segmenter breath-motion bounds: auto (default: 0.20,0.35 s, re-segmented with 0.12,0.21 when the breath "
           "check finds halved or clipped breathing), default (0.20,0.35, never re-segment) or LO,HI in s")


def _dirs(out_root):
    seg = os.path.join(out_root, "segments"); met = os.path.join(out_root, "metrics")
    os.makedirs(seg, exist_ok=True); os.makedirs(met, exist_ok=True)
    return seg, met


def _log(*a):
    print(*a, flush=True)


def cmd_segment(exp, out_root, max_min=None, cohort=None, age_weeks=None, skip_existing=False, breath_bounds="auto"):
    seg_dir, _ = _dirs(out_root)
    meta_extra = {}
    if cohort is not None:
        meta_extra["cohort"] = cohort
    if age_weeks is not None:
        meta_extra["age_weeks"] = age_weeks
    mode = str(breath_bounds or "auto").lower()
    bounds = None if mode in ("auto", "default") else SEG.parse_bounds(breath_bounds)
    res = SEG.segment_experiment(exp, seg_dir, max_min=max_min, skip_existing=skip_existing,
                                 meta_extra=meta_extra, log=_log, breath_bounds=bounds)
    return _breath_check(exp, seg_dir, res, auto=mode == "auto", max_min=max_min, meta_extra=meta_extra)


def _breath_check(exp, seg_dir, res, auto, max_min=None, meta_extra=None):
    """Halved / clipped breathing check on a segmentation (radar_hr/run_radar_hr.py fast_breath_step). auto: a halved or
    clipped default-bounds segmentation is moved to <seg_dir>/default_bounds/ and re-made with breath_check.FAST_BOUNDS.
    The outcome is stored in the segments json as `breath_check`."""
    import json, shutil
    import numpy as np
    import breath_check as BC
    import resp_segment as RS
    rec = res["recording"]; base = os.path.join(seg_dir, rec)
    try:
        chk = BC.rate_ratio(base)
    except Exception as e:                                    # advisory: never lose a segmentation over it
        _log(f"[breath-check] {rec}: skipped ({type(e).__name__}: {e})")
        return res
    bb = res.get("breath_bounds_s") or [RS.BMIN_B_S, RS.BMAX_B_S]
    default_bb = np.allclose(bb, [RS.BMIN_B_S, RS.BMAX_B_S])
    head = (f"[breath-check] {rec}: spectral {chk['spec_bpm']:.0f}/min vs segmenter {chk['dp_bpm']:.0f}/min over "
            f"{chk['n_runs']} clean runs -> ratio {chk['ratio']:.2f}")
    rnd = lambda v, n=1: round(float(v), n) if np.isfinite(v) else None
    info = dict(ratio=rnd(chk["ratio"], 3), spec_bpm=rnd(chk["spec_bpm"]), dp_bpm=rnd(chk["dp_bpm"]), n_runs=chk["n_runs"],
                pinned_frac=rnd(chk["pinned_frac"], 3), halved=BC.is_halved(chk), clipped=BC.is_clipped(chk),
                resegmented=False)
    prev = res.get("breath_check") or {}
    if prev.get("resegmented") and not default_bb:     # cached fast re-segmentation (--skip-existing): keep its record
        info.update({k: v for k, v in prev.items() if k == "resegmented" or k.endswith("_default_bounds")})
    if not BC.needs_fast(chk):
        _log(head + (" (every breath counted)" if np.isfinite(chk["ratio"]) else " (too few clean runs to check)"))
    elif not (auto and default_bb):
        _log(head + f" -> WARNING: {BC.problem(chk)}" +
             ("" if auto else f"; re-segment with --breath-bounds {BC.FAST_BOUNDS[0]:g},{BC.FAST_BOUNDS[1]:g}"))
    else:
        _log(head + f" -> {BC.problem(chk)}: a breather too fast for the default bounds. Re-segmenting with breath "
             f"bounds {BC.FAST_BOUNDS[0]:g}-{BC.FAST_BOUNDS[1]:g} s (the default-bounds segmentation is kept in "
             f"default_bounds/)")
        keep = os.path.join(seg_dir, "default_bounds"); os.makedirs(keep, exist_ok=True)
        for ext in (".json", ".npz"):
            shutil.move(base + ext, os.path.join(keep, rec + ext))
        res = SEG.segment_experiment(exp, seg_dir, max_min=max_min, skip_existing=False, meta_extra=meta_extra,
                                     log=_log, breath_bounds=BC.FAST_BOUNDS)
        chk2 = BC.rate_ratio(base)
        info = dict(ratio=rnd(chk2["ratio"], 3), spec_bpm=rnd(chk2["spec_bpm"]), dp_bpm=rnd(chk2["dp_bpm"]),
                    n_runs=chk2["n_runs"], pinned_frac=rnd(chk2["pinned_frac"], 3), halved=BC.is_halved(chk2),
                    clipped=BC.is_clipped(chk2), resegmented=True, ratio_default_bounds=info["ratio"],
                    dp_bpm_default_bounds=info["dp_bpm"], pinned_frac_default_bounds=info["pinned_frac"],
                    reason_default_bounds="halved" if info["halved"] else "clipped")
        _log(f"[breath-check] {rec}: after re-segmentation the segmenter counts {chk2['dp_bpm']:.0f}/min, ratio "
             f"{chk2['ratio']:.2f}, {100 * chk2['pinned_frac']:.0f}% of breath motions at the "
             f"{chk2['bmin_b']:g} s minimum" + (f" -> STILL {BC.problem(chk2)}: breathing metrics and heart rate are "
                                                f"unreliable" if BC.needs_fast(chk2) else " (every breath counted)"))
    res["breath_check"] = info
    json.dump(res, open(base + ".json", "w"), indent=1, default=float)
    return res


def _raw_kw(mod, exp, workers=None, hr_exp=None):
    # segment pipelines that also re-read the voxels (apnea, hr) get the raw folder; else they use the substrate's
    # session_dir. The heart-rate pipelines (DEJUMP = True) get the de-jumped copy when there is one.
    # Parallel pipelines (hr) get the worker count.
    if getattr(mod, "DEJUMP", False) and hr_exp:
        exp = hr_exp
    kw = {"exp_dir": exp} if getattr(mod, "NEEDS_RAW", False) and exp else {}
    if getattr(mod, "PARALLEL", False) and workers:
        kw["workers"] = workers
    return kw


def _run_pipeline(mod, rec, exp, seg_dir, met_dir, max_min=None, workers=None, hr_exp=None):
    if mod.KIND == "segments":
        return mod.run(rec, sub_dir=seg_dir, out_dir=met_dir, log=_log, **_raw_kw(mod, exp, workers, hr_exp))
    if mod.KIND == "raw":
        return mod.run(exp, out_dir=met_dir, max_min=max_min, log=_log)
    raise ValueError(f"pipeline {mod.NAME} has unknown KIND {mod.KIND!r}")


def _hr_exp(exp, out_root, rec, mode):
    """The cube folder for the heart-rate steps (engine/dejump.py prepare); the raw folder if the check fails."""
    try:
        return DJ.prepare(exp, out_root, rec=rec, mode=mode, log=_log)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[dejump][ERROR] {rec}: {e} -- the heart-rate steps use the raw cube")
        return exp


def cmd_all(exp, out_root, max_min=None, cohort=None, age_weeks=None, skip_existing=False, workers=None,
            breath_bounds=None, dejump="auto", keep_dejumped=False):
    seg_dir, met_dir = _dirs(out_root)
    res = cmd_segment(exp, out_root, max_min=max_min, cohort=cohort, age_weeks=age_weeks,
                      skip_existing=skip_existing, breath_bounds=breath_bounds)
    rec = res["recording"]
    hr_exp = None
    for mod in PIPELINES:
        if getattr(mod, "DEJUMP", False) and hr_exp is None:   # once, right before the first heart-rate step
            hr_exp = _hr_exp(exp, out_root, rec, dejump)
        try:
            _run_pipeline(mod, rec, exp, seg_dir, met_dir, max_min=max_min, workers=workers, hr_exp=hr_exp)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[{mod.NAME}][ERROR] {rec}: {e}")
    if hr_exp and hr_exp != exp and not keep_dejumped:
        DJ.cleanup(out_root, exp, log=_log)
    print(f"\n-> segments in {seg_dir}\n-> metrics  in {met_dir}\n   (run `python run.py combine` to build master tables)")
    # QC runs LAST, after every metric pipeline, so its PASS/REVIEW/FAIL banner is the final thing on
    # screen (it also folds the just-written metric JSONs into its report).
    try:
        _run_pipeline(QC, rec, exp, seg_dir, met_dir, max_min=max_min)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"[qc][ERROR] {rec}: {e}")


def cmd_one(name, rec=None, exp=None, out_root=None, seg=None, max_min=None, workers=None, dejump=None, **extra):
    mod = REGISTRY[name]
    seg_dir, met_dir = _dirs(out_root)
    seg_dir = seg or seg_dir
    if getattr(mod, "DEJUMP", False) and rec is not None and str(dejump or "auto").lower() != "off":
        import json
        raw = exp
        if raw is None and os.path.exists(os.path.join(seg_dir, rec + ".json")):
            raw = json.load(open(os.path.join(seg_dir, rec + ".json"))).get("session_dir")
        if raw and glob.glob(os.path.join(raw, "*_voxels.json")):
            exp = _hr_exp(raw, out_root, rec, dejump or "auto")
            if exp != raw:
                print(f"[dejump]   the repaired copy stays in {os.path.dirname(exp)} for the next heart-rate step; "
                      f"delete that folder when you are done (~3.5 GB per recording hour)")
    if mod.KIND == "segments":
        if rec is None:
            raise SystemExit(f"pipeline '{name}' needs --rec <recording_name>")
        return mod.run(rec, sub_dir=seg_dir, out_dir=met_dir, log=_log, **_raw_kw(mod, exp, workers),
                       **{k: v for k, v in extra.items() if v is not None})
    else:
        if exp is None:
            raise SystemExit(f"pipeline '{name}' needs --exp <raw_experiment_dir>")
        return mod.run(exp, out_dir=met_dir, max_min=max_min, log=_log)


def cmd_dejump(exp, out_root, bins="hr", detect_only=False):
    """Report the capture-build jumps of one raw experiment and write the repaired copy to <out>/dejumped/."""
    import json
    _, met_dir = _dirs(out_root)
    rec = VS.rec_name(exp)
    rep = DJ.run(exp, os.path.join(out_root, "dejumped"), bins=bins, detect_only=detect_only, mode="on", log=_log)
    rep["recording"] = rec
    json.dump(rep, open(os.path.join(met_dir, rec + ".dejump.json"), "w"), indent=1, default=float)
    if rep["repaired"]:
        print(f'-> run the heart-rate steps on the copy:  python run.py hr --rec {rec} --exp "{rep["out_dir"]}" '
              f'--dejump off   (then the same for b2b)')
    return rep


def cmd_combine(out_root):
    """Concatenate every experiment's per-pipeline CSVs into master tables in <out>/metrics/."""
    _, met_dir = _dirs(out_root)
    suffixes = {"stettner": ".stettner.csv", "pauses": ".pauses.csv", "motion": ".motion.csv",
                "apnea": ".apnea.csv", "apnea_events": ".apnea_events.csv", "hr": ".hr.csv",
                "hr_periods": ".hr_periods.csv", "b2b": ".b2b.csv", "b2b_periods": ".b2b_periods.csv",
                "qc": ".qc.csv", "dejump": ".dejump.csv"}
    for name, suf in suffixes.items():
        files = sorted(glob.glob(os.path.join(met_dir, "*" + suf)))
        if not files:
            continue
        cols, rows = [], []
        for f in files:
            with open(f, newline="") as fh:
                rd = csv.DictReader(fh)
                cols += [c for c in (rd.fieldnames or []) if c not in cols]   # union: older runs may lack columns
                rows += list(rd)                                               # apnea_events: one row per event
        if cols:
            out = os.path.join(met_dir, f"combined_{name}.csv")
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, cols, restval=""); w.writeheader(); w.writerows(rows)
            print(f"-> {out}  ({len(rows)} rows from {len(files)} experiments)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("all"); p.add_argument("--exp", required=True); p.add_argument("--out", default=DEF_OUT)
    p.add_argument("--max-min", type=float, default=None); p.add_argument("--cohort", default=None)
    p.add_argument("--age-weeks", type=float, default=None); p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--workers", type=int, default=None, help="parallel workers for the heart-rate steps")
    p.add_argument("--breath-bounds", default=None, help=BB_HELP)
    p.add_argument("--dejump", default="auto", choices=("auto", "on", "off"), help=DJ_HELP)
    p.add_argument("--keep-dejumped", action="store_true", help="keep the repaired cube copy in <out>/dejumped/")

    p = sub.add_parser("segment"); p.add_argument("--exp", required=True); p.add_argument("--out", default=DEF_OUT)
    p.add_argument("--max-min", type=float, default=None); p.add_argument("--cohort", default=None)
    p.add_argument("--age-weeks", type=float, default=None); p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--breath-bounds", default=None, help=BB_HELP)

    for name in REGISTRY:
        p = sub.add_parser(name)
        p.add_argument("--rec", default=None); p.add_argument("--exp", default=None)
        p.add_argument("--out", default=DEF_OUT); p.add_argument("--seg", default=None)
        p.add_argument("--max-min", type=float, default=None)
        if getattr(REGISTRY[name], "PARALLEL", False):
            p.add_argument("--workers", type=int, default=None, help="parallel worker processes")
        if name == "hr":
            p.add_argument("--patch-beams", default=None, help="override the chest-patch beams, e.g. 6-15")
            p.add_argument("--min-period-s", type=float, default=None,
                           help="shortest still period scored (default 20 s = validated)")
        if getattr(REGISTRY[name], "DEJUMP", False):
            p.add_argument("--dejump", default="auto", choices=("auto", "on", "off"), help=DJ_HELP)

    p = sub.add_parser("dejump"); p.add_argument("--exp", required=True); p.add_argument("--out", default=DEF_OUT)
    p.add_argument("--bins", default="hr", help="range bins to write: hr (50-82, the heart-rate chest patch; default), "
                                                "all, or LO-HI")
    p.add_argument("--detect-only", action="store_true", help="only report the jumps; write no copy")

    p = sub.add_parser("combine"); p.add_argument("--out", default=DEF_OUT)
    sub.add_parser("list")

    args = ap.parse_args(argv)
    if args.cmd == "list":
        print("registered pipelines:")
        for m in PIPELINES + [QC]:
            print(f"  {m.NAME:10s} [{m.KIND}]  {m.__doc__.splitlines()[0] if m.__doc__ else ''}")
        return 0
    if args.cmd == "all":
        cmd_all(args.exp, args.out, max_min=args.max_min, cohort=args.cohort, age_weeks=args.age_weeks,
                skip_existing=args.skip_existing, workers=args.workers, breath_bounds=args.breath_bounds,
                dejump=args.dejump, keep_dejumped=args.keep_dejumped); return 0
    if args.cmd == "segment":
        cmd_segment(args.exp, args.out, max_min=args.max_min, cohort=args.cohort, age_weeks=args.age_weeks,
                    skip_existing=args.skip_existing, breath_bounds=args.breath_bounds); return 0
    if args.cmd == "combine":
        cmd_combine(args.out); return 0
    if args.cmd == "dejump":
        cmd_dejump(args.exp, args.out, bins=args.bins, detect_only=args.detect_only); return 0
    if args.cmd in REGISTRY:
        cmd_one(args.cmd, rec=args.rec, exp=args.exp, out_root=args.out, seg=args.seg, max_min=args.max_min,
                workers=getattr(args, "workers", None), patch_beams=getattr(args, "patch_beams", None),
                min_period_s=getattr(args, "min_period_s", None), dejump=getattr(args, "dejump", None))
        return 0
    ap.error(f"unknown command {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
