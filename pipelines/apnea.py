r"""apnea.py -- radar APNEA count per recording, with the rule validated against whole-body plethysmography (WBP).

On the paired radar + WBP session of mutant B16 (09-21), this rule found every scorable WBP apnea (12/12) and every
apnea it called was a real WBP apnea (16/16); on B17 it called 0 false apneas. The rule was designed on those two
sessions -- read APNEA.md before citing numbers.

How one recording is processed (every step uses the EQ-Radio two-template Breath/Pause DP; no peak detector):
  1. CANDIDATES -- merged DP pauses from TWO signals: the episode's chosen breathing voxel (the `segments` substrate)
     and an anchored multi-voxel combine (PC1 of the 12 strongest breathing voxels near it, re-segmented; cached in
     <out>/segments_anchored/). A candidate needs >= 3 confident breaths within 2 s on BOTH sides (breathing stops,
     then comes back) and must be FLAT inside (in-pause RMS <= 0.30 x the flanking breaths).
  2. WITNESS VOXELS -- the same DP re-run on the 10 best breathing voxels around each candidate + the chosen voxel.
     A true apnea is flat in every voxel; a posture change just moves the breathing to other voxels.
     Apnea = >= 2 voxels other than the chosen one confirm the pause and < 2 keep breathing through it (or, when no
     voxel breathes cleanly enough to testify, a combine pause that no voxel breathes through and the chosen voxel shows).
  3. DURATION -- reported under two definitions: > 3 x the recording's median clean breath cycle (rate-relative;
     validated default) and > 1 s (the collaborators' ApneaCheck rule on WBP). Overlapping passes merge into one event.
  4. RATE -- apneas per minute of CLEAN breathing (>= 8 consecutive confident DP cycles). The radar only resolves
     breathing while the animal is calm, so apneas during activity are invisible: per-hour counts are not comparable.
Old (pre-08-06) live captures step every 4.264 s capture file; apneas starting/ending on a step are vetoed and that
recording's count is rescaled by 1/(1 - chance). Newer captures have no step and are untouched.

Outputs (<out> = metrics dir):  <rec>.apnea.{csv,json} (one row), <rec>.apnea_events.csv (one row per apnea),
<rec>.apnea_candidates.csv (every candidate + its votes, for audits / re-thresholding), <rec>.apnea.png (timeline +
example events). Needs the RAW voxels (run.py passes --exp; else the substrate's session_dir).
Research provenance: apneacheck/ (radar_apnea, multivoxel_combine) + output/apnea_cohort/witness_v2.py.
"""
from __future__ import annotations
import os, sys, json, csv, glob, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for _p in (os.path.join(_PKG, "engine"), _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import substrate as SUB
import voxel_store as VS
import dp_apnea as DA
import multivoxel as MV
import witness as W

NAME = "apnea"
KIND = "segments"
NEEDS_RAW = True              # run.py passes exp_dir= (the raw folder); falls back to the substrate's session_dir

DEFS = {"3x": ("rel", 3.0), "gt1s": ("fixed", 1.0)}
DEF_LABEL = {"3x": "pause > 3x median clean breath cycle", "gt1s": "pause > 1 s"}
FLAT_MAX = DA.DP_RESIDUAL_MAX
MIN_CLEAN_MIN = 3.0           # below this a per-clean-minute rate is a handful of events (cohort analysis excluded it)
# chance that a pause touches a capture-file step by accident: either end within +-0.2 s of a 4.264 s grid ~ 2 x 0.4 /
# 4.264 = 0.19 (pooled 07-xx short pauses: 0.188; per-recording estimates are just noisier, 0.15-0.21)
P_CHANCE_FILE = 0.188
FS_VALIDATED = 250.0

CAND_COLS = ["src", "t0_s", "t1_s", "dur_s", "ep", "n_pieces", "censored", "flank_strict", "cycle_strict",
             "flank_relaxed", "cycle_relaxed", "residual", "in_clean_flanked", "d0_file", "d1_file", "vetoed_file",
             "adjudicated", "n_vox", "n_wit", "n_con", "n_conf", "n_conf_ind", "dom_shows", "pass_3x", "pass_gt1s"]
EVENT_COLS = ["recording", "definition", "t0_s", "t1_s", "dur_s", "t0_min", "after_20min", "sources", "n_conf_ind",
              "n_con", "n_wit", "residual", "in_clean_flanked"]


CACHE_VERSION = 1             # bump when the combine / step scan changes so existing caches rebuild


def _anchored_substrate(rec, sub_dir, seg_json, power, vmeta, force, log):
    """<out>/segments_anchored/<rec>.{json,npz}: anchored combine + DP, and the capture-file boundary scan. Rebuilt if
    missing, unreadable, from another CACHE_VERSION, or if the segments substrate was re-made since (episodes differ).
    Written npz-then-json via os.replace, so an interrupted write never leaves a cache that looks valid."""
    adir = os.path.join(os.path.dirname(os.path.abspath(sub_dir)), "segments_anchored")
    os.makedirs(adir, exist_ok=True)
    jf, nf = os.path.join(adir, rec + ".json"), os.path.join(adir, rec + ".npz")
    jj = None
    if not force and os.path.exists(jf) and os.path.exists(nf):
        try:
            jj = json.load(open(jf))
            np.load(nf, allow_pickle=True).files
        except Exception:
            jj = None
        if jj is not None and not (jj.get("episodes") == seg_json.get("episodes")
                                   and jj.get("apnea_cache_version") == CACHE_VERSION):
            jj = None
    if jj is None:
        t0 = time.time()
        log(f"[apnea]    {rec}: anchored multi-voxel combine ({len(seg_json['episodes'])} episodes) ...")
        jj, arrays = MV.combine_session(power, vmeta, seg_json, log=log)
        fs = float(vmeta["fs"]); N = power.shape[0]
        tm, rm = seg_json.get("total_min"), seg_json.get("rec_min")
        n = N if (tm is None or rm is None or tm >= rm) else min(N, int(round(tm * 60 * fs)))   # honours --max-min
        log(f"[apnea]    {rec}: capture-file step scan ...")
        jj.update(file_boundaries_s=[round(float(x), 4) for x in DA.file_boundaries(power, fs, n=n)],
                  scan_s=n / fs, scan_full=bool(n >= N), apnea_cache_version=CACHE_VERSION)
        tmp = nf[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **{k: np.array(v, dtype=object) for k, v in arrays.items()})
        os.replace(tmp, nf)
        json.dump(jj, open(jf + ".tmp", "w"), indent=1, default=float)
        os.replace(jf + ".tmp", jf)
        log(f"[apnea]    {rec}: -> {jf}  [{time.time() - t0:.0f}s]")
    return jf, nf, jj


def _passes(c, key, cyc):
    kind, v = DEFS[key]
    thr = v if kind == "fixed" else v * cyc
    return bool(c["flank_strict"] and c["dur_s"] > thr and c["residual"] <= FLAT_MAX and not c["vetoed_file"]
                and c["adjudicated"] and W.final_rule(c))


def merge_events(C, key):
    """Overlapping passing candidates (either source) -> one event per cessation."""
    out, cur = [], None
    for r in sorted((c for c in C if c[f"pass_{key}"]), key=lambda c: (c["t0_s"], c["src"])):
        if cur is not None and r["t0_s"] < cur["t1_s"]:
            cur["t1_s"] = max(cur["t1_s"], r["t1_s"]); cur["sources"] += "+" + r["src"]
            cur["in_clean_flanked"] |= bool(r["in_clean_flanked"]); continue
        if cur is not None:
            out.append(cur)
        cur = dict(t0_s=r["t0_s"], t1_s=r["t1_s"], sources=r["src"], in_clean_flanked=bool(r["in_clean_flanked"]),
                   ep=r["ep"], n_conf_ind=r["n_conf_ind"], n_con=r["n_con"], n_wit=r["n_wit"], residual=r["residual"],
                   cand=(r["t0_s"], r["t1_s"]))
    if cur is not None:
        out.append(cur)
    for e in out:
        e["dur_s"] = round(e["t1_s"] - e["t0_s"], 3)
    return out


def _r(x, nd=4):
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


def analyze(rec, sub_dir, exp_dir=None, force=False, log=print):
    """Everything for one recording. Returns (metrics, events{key: [..]}, candidates, context for the figure)."""
    seg_jf = os.path.join(sub_dir, rec + ".json")
    seg_json = json.load(open(seg_jf))
    exp_dir = exp_dir or seg_json.get("session_dir")
    if not (exp_dir and glob.glob(os.path.join(exp_dir, "*_voxels.json"))):
        raise FileNotFoundError(f"raw experiment folder not found ({exp_dir!r}); the apnea metric re-reads the voxels "
                                f"-- pass --exp <folder with *_voxels.f32>")
    power, vmeta, _ = VS.load_experiment(exp_dir)
    fs = float(vmeta["fs"]); disk_min = power.shape[0] / fs / 60
    notes = []
    if seg_json.get("rec_min") is not None and abs(float(seg_json["rec_min"]) - disk_min) > 0.05:
        raise ValueError(f"{exp_dir!r} holds {disk_min:.2f} min of voxels but {rec} was segmented from "
                         f"{seg_json['rec_min']} min -- wrong --exp folder?")
    if VS.rec_name(exp_dir) != rec:
        notes.append(f"raw folder is named {VS.rec_name(exp_dir)!r}, not {rec!r} -- check it is the same recording")
    ajf, anf, ajson = _anchored_substrate(rec, sub_dir, seg_json, power, vmeta, force, log)
    single = DA.RadarSession(seg_jf, seg_jf[:-5] + ".npz")
    anch = DA.RadarSession(ajf, anf)
    for sess in (single, anch):
        sess.j.setdefault("rec_min", round(disk_min, 2))          # older substrates; rec_s would default to 60 min
    base = DA.base_stats(single)
    cyc = base["cycle_med_s"]
    # grid slots must span only what was scanned, or a --max-min run fails the >= 50 %-of-slots test (no veto)
    grid = DA.file_grid(np.asarray(ajson["file_boundaries_s"], float),
                        single.rec_s if ajson.get("scan_full", True) else float(ajson["scan_s"]))

    C = []
    for src, sess in (("single", single), ("anchored", anch)):
        for r in DA.candidates(sess, grid):
            C.append(dict(src=src, **r))
    eps = seg_json["episodes"]
    t0 = time.time(); n_adj = 0
    for c in C:
        c["vetoed_file"] = bool(c["d0_file"] <= DA.FILE_TOL or c["d1_file"] <= DA.FILE_TOL)   # NaN (no grid) -> kept
        c.update(n_vox=0, n_wit=0, n_con=0, n_conf=0, n_conf_ind=0, dom_shows=False, adjudicated=False)
        if c["flank_strict"] and np.isfinite(cyc) and (c["dur_s"] > W.POOL_MULT * cyc or c["dur_s"] > W.POOL_ABS):
            rows = W.event_witnesses(power, vmeta, int(eps[c["ep"]]["dom_col"]), c["t0_s"], c["t1_s"], cyc)
            c.update(W.votes(rows, c["t0_s"], c["t1_s"], cyc), adjudicated=True)
            n_adj += 1
        for key in DEFS:
            c[f"pass_{key}"] = _passes(c, key, cyc)
    log(f"[apnea]    {rec}: {len(C)} candidate pauses, {n_adj} adjudicated by witness voxels [{time.time() - t0:.0f}s]")
    E = {key: merge_events(C, key) for key in DEFS}

    step = bool(grid.size)
    p_chance = P_CHANCE_FILE if step else 0.0
    scale = 1.0 / (1.0 - p_chance)
    cm, cm20 = base["clean_min"], base["clean_min_post20"]
    info = SUB.rec_info(rec, seg_json)
    # clean_breathing_min = DP clean-run minutes (the rate denominator), NOT the QC banner's "recovered breathing"
    m = dict(recording=rec, cohort=info["cohort"], genotype=info["genotype"], age_weeks=info["age_weeks"],
             rec_min=_r(base["rec_min"], 2), analyzed_min=_r(seg_json.get("total_min") or base["rec_min"], 2),
             breathing_min=_r(base["breathing_min"], 2), clean_breathing_min=_r(cm, 2),
             clean_breathing_min_after20=_r(cm20, 2), clean_rate_bpm=_r(base["clean_rate_bpm"], 1),
             cycle_med_s=_r(cyc), rate_reliable=bool(cm >= MIN_CLEAN_MIN),
             rate_reliable_after20=bool(cm20 >= MIN_CLEAN_MIN))
    for key in DEFS:
        ev = E[key]; n = len(ev); n20 = sum(e["t0_s"] >= DA.ACCLIM_S for e in ev)
        m[f"apnea_{key}_n"] = n
        m[f"apnea_{key}_n_adj"] = _r(n * scale, 3)               # step-corrected count: sum THIS to pool per animal
        m[f"apnea_{key}_per_clean_min"] = _r(n * scale / cm) if cm > 0 else None
        m[f"apnea_{key}_n_after20"] = n20
        m[f"apnea_{key}_n_adj_after20"] = _r(n20 * scale, 3)
        m[f"apnea_{key}_per_clean_min_after20"] = _r(n20 * scale / cm20) if cm20 > 0 else None
        m[f"apnea_{key}_dur_med_s"] = _r(np.median([e["dur_s"] for e in ev]), 3) if ev else None
    m["apnea_3x_threshold_s"] = _r(3.0 * cyc, 3)
    m.update(capture_file_step=step, n_vetoed_file=sum(c["vetoed_file"] and c["flank_strict"] for c in C),
             count_scale=_r(scale), n_candidates=len(C), n_adjudicated=n_adj, fs=fs)

    if cm < MIN_CLEAN_MIN:
        notes.append(f"only {cm:.1f} clean-breathing min (< {MIN_CLEAN_MIN:.0f}): the rate rests on a handful of "
                     f"minutes -- report it, but don't pool it (the cohort analysis excluded such recordings)")
    if step:
        notes.append(f"old live-capture build: per-file step every {DA.FILE_P:.3f} s detected; apneas touching a step "
                     f"vetoed, counts scaled x{scale:.2f} (chance a real apnea touches one: {p_chance:.2f})")
    if abs(fs - FS_VALIDATED) > 1e-6:
        notes.append(f"fs = {fs:.0f} Hz, not the validated {FS_VALIDATED:.0f} Hz")
    if not np.isfinite(cyc):
        notes.append("no clean breathing recovered: no apnea can be scored")
    m["notes"] = " | ".join(notes)
    ctx = dict(power=power, vmeta=vmeta, eps=eps, base=base, anch=anch, cyc=cyc, fs=fs)
    return m, E, C, ctx


# ---- figure ------------------------------------------------------------------------------------------------------------
ROLE_C = {"confirms": "#2a78d6", "breathes through": "#d0342c", "witness": "#52514e", "": "#b9b8b3"}
INK, INK2, CLEAN_C, EV_C = "#0b0b0b", "#52514e", "#dcebdc", {"3x": "#8e4585", "gt1s": "#eb6834"}


def _role(v, t0, t1, cyc):
    r = W.voxel_role(v, t0, t1, cyc)
    return "breathes through" if r["con"] else "confirms" if r["conf"] else "witness" if r["wit"] else ""


def _figure(path, rec, m, E, ctx, n_ex=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ev = E["3x"] or E["gt1s"]
    ex_key = "3x" if E["3x"] else "gt1s"
    idx = sorted(set(np.linspace(0, len(ev) - 1, min(n_ex, len(ev))).round().astype(int))) if ev else []
    nrow = int(np.ceil(len(idx) / 3))
    fig = plt.figure(figsize=(14, 2.8 + 4.6 * nrow))
    gs = fig.add_gridspec(1 + nrow, 3, height_ratios=[1.0] + [2.6] * nrow, hspace=0.45, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    g, clean = ctx["base"]["clean_grid"], ctx["base"]["clean_mask"]
    ax.fill_between(g / 60, 0, 1, where=clean, transform=ax.get_xaxis_transform(), color=CLEAN_C, step="mid", lw=0)
    for key, mk, sz in (("gt1s", "o", 34), ("3x", "D", 22)):
        ax.scatter([e["t0_s"] / 60 for e in E[key]], [e["dur_s"] for e in E[key]], marker=mk, s=sz,
                   facecolor="none" if key == "gt1s" else EV_C[key], edgecolor=EV_C[key], lw=1.2, zorder=3)
    ax.axvline(DA.ACCLIM_S / 60, color=INK2, ls=":", lw=0.8)
    ax.set_xlim(0, m["analyzed_min"] or g[-1] / 60); ax.set_ylim(0, max([2.0] + [e["dur_s"] * 1.15 for e in E["gt1s"] + E["3x"]]))
    ax.set_xlabel("time in recording (min)"); ax.set_ylabel("apnea duration (s)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    rate = lambda k: f"{m[f'apnea_{k}_per_clean_min']:.3f}" if m[f"apnea_{k}_per_clean_min"] is not None else "n/a"
    ax.set_title(f"{rec}  ({m['genotype']})   clean breathing {m['clean_breathing_min']} of {m['analyzed_min']} min, "
                 f"{m['clean_rate_bpm']} bpm   |   apneas > 3x cycle ({m['apnea_3x_threshold_s']} s): "
                 f"{m['apnea_3x_n']} = {rate('3x')}/clean-min   |   > 1 s: {m['apnea_gt1s_n']} = {rate('gt1s')}/clean-min",
                 loc="left", fontsize=9, color=INK)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CLEAN_C, label="clean breathing (where apneas are scorable)"),
                       Line2D([], [], marker="D", ls="", color=EV_C["3x"], label=f"apnea, {DEF_LABEL['3x']}"),
                       Line2D([], [], marker="o", ls="", markerfacecolor="none", color=EV_C["gt1s"],
                              label=f"apnea, {DEF_LABEL['gt1s']}"),
                       Line2D([], [], ls=":", color=INK2, label="20 min (ApneaCheck acclimation)")],
              loc="upper left", bbox_to_anchor=(0, -0.28), ncol=4, frameon=False, fontsize=8)
    anch, fs, cyc = ctx["anch"], ctx["fs"], ctx["cyc"]
    for i, j in enumerate(idx):
        e = ev[j]; c0, c1 = e["cand"]
        axe = fig.add_subplot(gs[1 + i // 3, i % 3])
        rows = W.event_witnesses(ctx["power"], ctx["vmeta"], int(ctx["eps"][e["ep"]]["dom_col"]), c0, c1, cyc, traces=True)
        rows = sorted(rows, key=lambda v: (not v["is_dom"], v["rank"]))
        lo, hi = c0 - W.WIN_S, c1 + W.WIN_S
        y = 0.0
        e0 = float(anch.episodes[e["ep"]]["t0"]); xa = np.asarray(anch.trace[e["ep"]], float)
        ta = e0 + np.arange(xa.size) / fs; mm = (ta >= lo) & (ta <= hi)
        if mm.sum() > 10:
            xs = xa[mm] - np.median(xa[mm])
            axe.plot(ta[mm], np.clip(xs / (np.percentile(np.abs(xs), 95) + 1e-12), -1.6, 1.6) * 0.4 + y, color=INK, lw=0.8)
            axe.text(hi + 0.1, y, "combined", fontsize=6.5, color=INK, va="center")
            y -= 1.1
        for v in rows:
            t, x = v["t"], v["x"] - np.median(v["x"])
            fl = ((t >= c0 - W.FL_S) & (t < c0)) | ((t > c1) & (t <= c1 + W.FL_S))
            x = x / (np.percentile(np.abs(x[fl]), 95) + 1e-12) if fl.any() else x
            col = ROLE_C[_role(v, c0, c1, cyc)]
            axe.plot(t, np.clip(x, -1.6, 1.6) * 0.4 + y, color=col, lw=0.9 if v["is_dom"] else 0.6)
            for a, b, _ in v["breaths"]:
                axe.plot([a, b], [y - 0.55, y - 0.55], color=col, lw=2.0, solid_capstyle="butt")
            axe.text(hi + 0.1, y, ("chosen " if v["is_dom"] else "") + (_role(v, c0, c1, cyc) or "-"), fontsize=6.5,
                     color=col, va="center")
            y -= 1.0
        axe.axvspan(e["t0_s"], e["t1_s"], color="#f0efec", zorder=0)
        axe.set_xlim(lo, hi + 1.9); axe.set_ylim(y + 0.2, 0.8); axe.set_yticks([])
        for s in ("top", "right", "left"):
            axe.spines[s].set_visible(False)
        axe.tick_params(labelsize=7); axe.set_xlabel("time (s)", fontsize=7)
        axe.set_title(f"{e['t0_s'] / 60:.1f} min, pause {e['dur_s']:.2f} s ({e['sources']})\n"
                      f"{e['n_conf_ind']} other voxels confirm, {e['n_con']} breathe through", fontsize=8, loc="left")
    if idx:
        import textwrap
        fig.text(0.01, 0.005, textwrap.fill(
            f"Examples ({DEF_LABEL[ex_key]}), evenly spaced through the recording. Each row is one voxel's conditioned "
            "signal (top: the multi-voxel combine); bars = confident DP breaths; colour = the voxel's vote: blue confirms "
            "the pause, red breathes through it, dark grey breathes around it but doesn't cover the pause, light grey "
            "not a witness.", 190), fontsize=7.5, color=INK2)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


# ---- pipeline entry point ----------------------------------------------------------------------------------------------
def _write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            w.writerow([r.get(k) for k in cols])


def run(rec, sub_dir=None, out_dir=None, log=print, exp_dir=None, force=False, figure=True):
    """Apnea metric for one segmented recording. Returns the flat metrics dict."""
    sub_dir = sub_dir or SUB.SUB_DIR
    SUB.SUB_DIR = sub_dir
    t0 = time.time()
    if out_dir:
        for f in glob.glob(os.path.join(out_dir, glob.escape(rec) + ".apnea*")):
            os.remove(f)
    m, E, C, ctx = analyze(rec, sub_dir, exp_dir=exp_dir, force=force, log=log)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, rec + ".apnea.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(m.keys()); w.writerow(m.values())
        ev_rows = [dict(recording=rec, definition=DEF_LABEL[k], t0_min=round(e["t0_s"] / 60, 2),
                        after_20min=e["t0_s"] >= DA.ACCLIM_S, **{x: e[x] for x in e if x != "cand"})
                   for k in DEFS for e in E[k]]
        _write_csv(os.path.join(out_dir, rec + ".apnea_events.csv"), EVENT_COLS, ev_rows)
        _write_csv(os.path.join(out_dir, rec + ".apnea_candidates.csv"), CAND_COLS, C)
        json.dump(dict(metrics=m, definitions=DEF_LABEL, **{f"events_{k}": [[e["t0_s"], e["t1_s"]] for e in E[k]]
                                                            for k in DEFS}),
                  open(os.path.join(out_dir, rec + ".apnea.json"), "w"), indent=1, default=float)
        if figure:
            try:
                _figure(os.path.join(out_dir, rec + ".apnea.png"), rec, m, E, ctx)
            except Exception as ex:                       # a figure failure must never lose the numbers
                log(f"[apnea]    figure skipped ({ex})")
    rate = lambda k: "n/a" if m[f"apnea_{k}_per_clean_min"] is None else m[f"apnea_{k}_per_clean_min"]
    log(f"[apnea]    {rec:20s} clean {m['clean_breathing_min']} min | >3x cycle: {m['apnea_3x_n']} ({rate('3x')}/clean-min) | "
        f">1 s: {m['apnea_gt1s_n']} ({rate('gt1s')}/clean-min)  [{time.time() - t0:.0f}s]"
        + (f"\n           NOTE: {m['notes']}" if m["notes"] else ""))
    return m


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="radar apnea metric for one segmented recording")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--sub-dir", default=None)
    ap.add_argument("--exp", default=None, help="raw experiment folder (default: the substrate's session_dir)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="rebuild the anchored-combine cache")
    a = ap.parse_args()
    run(a.rec, sub_dir=a.sub_dir, out_dir=a.out, exp_dir=a.exp, force=a.force)
