r"""hr.py -- radar HEART RATE per still, regular-breathing period, with the frozen ECG-free method (2026-10-03).

Validated against implanted-ECG mice 508 and 507 day 2: per-period error 10.9 / 8.9 bpm (87 % / 94 % of periods within
5 % of the ECG). Those two mice are the ones the method was developed on (in-sample) -- read HEART_RATE.md before
citing numbers.

How one recording is processed (only still, regular-breathing periods are scored; the heart is invisible otherwise):
  1. PERIODS -- EQ-Radio clean breathing (>= 8 consecutive confident Breath/Pause DP cycles, the `segments` substrate)
     lasting >= 20 s. Breathing rate per period = 60 / median cycle length.
  2. CHEST PATCH -- 10 beams x range bins 50-82 (0.27-0.45 m) = 330 voxels, centred on the beam with the most
     breathing-band power over all periods. Acceleration of every voxel (EQ-Radio 2nd derivative).
  3. HEART RATE (engine/hr_final_pipeline.py, the frozen code, byte-identical to the validated file):
     a. remove every integer breathing harmonic from every voxel (breath clock = Hilbert phase of the patch's PC1);
     b. choose one of 17 heartbeat bands (380-780 bpm) from the whitened cross-voxel spectrogram (B_spec);
     c. EQ-Radio beat segmentation of all 330 voxels in that band; beam the 20 voxels whose beats agree most;
     d. read the beam's instantaneous frequency (10 Hz track); period HR = 10 % trimmed mean of the track.
     If any step fails, the period falls back to the best-repeating band ("S1_fallback", listed in the outputs).

Outputs (<out> = metrics dir):  <rec>.hr.{csv,json} (one row), <rec>.hr_periods.csv (one row per period),
<rec>.hr_track.csv (10 Hz HR inside each period), <rec>.hr_beats.csv (beam beat times), <rec>.hr.png,
<rec>.hr_templates.npz (per period: the 20 beamed voxels' EQ-Radio templates + their own beat times; TEMPLATES_README).
Needs the RAW voxels (run.py passes --exp; else the substrate's session_dir) and a 250 Hz cube.
Research provenance: output/ecg_0928/hr_overnight/final/ (FROZEN.md) + radar_hr/ (runner, lazy band segmentation).
"""
from __future__ import annotations
import os, sys, json, csv, glob, time, shutil, hashlib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
for _p in (os.path.join(_PKG, "engine"), _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import voxel_store as VS
import hr_bands as HB

NAME = "hr"
KIND = "segments"
NEEDS_RAW = True              # run.py passes exp_dir= (the raw folder); falls back to the substrate's session_dir
PARALLEL = True               # run.py passes workers=
DEJUMP = True                 # run.py hands it the de-jumped cube copy of a July-2026 capture (engine/dejump.py, DEJUMP.md)

FROZEN_FILE = os.path.join(_PKG, "engine", "hr_final_pipeline.py")
FROZEN_SHA256 = "ff87ce110acf85e43da484c0f6768d41342cd0009886d831d3905a3dea97fdb0"   # FROZEN.md, final_pipeline.py
FS_REQUIRED = 250.0
EDGE_S = 1.0                  # the frozen readout ignores each period's first / last second (filter edges)
_BLAS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")

N_PICK = 20                   # voxels the frozen beam uses (hr_final_pipeline.NPK)
M_REF = 31                    # templates also resampled to 31 samples, as in the research band scan (band_template.py)
TEMPLATES_README = """<rec>.hr_templates.npz -- the 20 voxels the heart-rate beam used in each still period (P periods).
Per period, in the chosen heartbeat band, every chest voxel was cut into beats by EQ-Radio one-template segmentation of
its ACCELERATION (2nd derivative of sqrt(power), before breathing removal). The 20 voxels whose beats agree best with
the other voxels were beamed; these arrays are theirs, ordered by agreement (rank 0 = best).
  period, t0_s, t1_s, band_centre_bpm, method   [P]      period number (= hr_periods.csv), start/end s, band, method
  voxel_col                                     [P, 20]  cube column (ang_idx * n_bin + bin_idx)
  voxel_beam, voxel_angle_deg                   [P, 20]  radar beam index / azimuth
  voxel_range_bin, voxel_range_m                [P, 20]  absolute range bin / metres from the radar
  agree                                         [P, 20]  beat-phase agreement with the other voxels (picks rank by it)
  template_conf                                 [P, 20]  EQ-Radio confidence of the template fit (0-1)
  template_snr                                  [P, 20]  (signal energy - fit residual) / fit residual
  template_len                                  [P]      native template length m (samples at fs); depends on the band
  templates                                     [P, 20, max m]  native template, NaN-padded past template_len
  templates_31                                  [P, 20, 31]     the same resampled to 31 samples (comparable across bands)
  beat_sample, beat_t_s                         [N]      every beat start of these voxels: sample index / seconds
  beat_period, beat_rank                        [N]      which period / which of its 20 voxels (rank) the beat belongs to
Templates are mean-removed waveforms in arbitrary units; their sign is arbitrary per voxel (the beam aligns sign/delay).
Beat starts are segmentation boundaries (first and last partial beats of each period excluded)."""

PERIOD_COLS = ["recording", "period", "t0_s", "t1_s", "start_min", "dur_s", "breaths_per_min", "hr_bpm", "hr_over_rr",
               "band_centre_bpm", "n_beats", "method", "fallback_reason"]


def default_workers():
    return max(1, min(12, (os.cpu_count() or 2) - 2))


def _period_worker(args):
    """One period in a worker process: patch acceleration -> frozen process_run with lazy bands (in-process)."""
    rec, cube_path, shape, cols, n_bin, onset, ttot, a, b, rr, path = args
    import harness as H
    import hr_final_pipeline as FP
    t = time.time()
    HB.build_accel(((cube_path, shape), cols, a, b, path))
    H.register(rec, cube_path, shape, cols, onset, ttot)
    bands = HB.LazyBands(path, a, b, cols, n_bin, pool=None)
    res = FP.process_run(rec, dict(t0=a, t1=b, rr=rr, accel_patch=path), bands)
    res.pop("wfn", None)                                    # a closure; not picklable
    res["segmented_bands"] = list(bands.segmented)
    j = int(res.get("band", -1))
    if 0 <= j < len(bands) and bands[j].d is not None:     # the band whose picks were beamed (or S1-read)
        d = bands[j].d; pk = np.asarray(d["picks"])[:N_PICK]; a0 = int(round(a * HB.FS))
        res["voxels"] = dict(picks=pk, agree=np.asarray(d["agree"])[pk], mu=[d["mu"][k] for k in pk],
                             conf=d["conf"][pk], snr=d["snr"][pk],
                             beats=[np.asarray(d["blist"][k][1:-1], np.int64) + a0 for k in pk])
    res["elapsed_s"] = time.time() - t
    return res


def _readout(res, t0, t1):
    if res.get("t") is None:
        return np.zeros(0), np.zeros(0)
    t, k = np.asarray(res["t"]), np.asarray(res["trk"])
    m = (t >= t0 + EDGE_S) & (t < t1 - EDGE_S)
    return t[m], k[m]


def _write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def _write_templates(path, rows, res, cols, vmeta):
    """<rec>.hr_templates.npz (layout in TEMPLATES_README)."""
    import beatseg as BS
    P, K = len(rows), N_PICK
    nb, b0, mpb = int(vmeta["n_bin"]), int(vmeta["bins"][0]), float(vmeta.get("mmpb", 5.48)) / 1000
    ang = np.asarray(vmeta["angles_deg"], float)
    vx = [res[r["period"]].get("voxels") for r in rows]
    L = max([len(v["mu"][0]) for v in vx if v] or [1])
    col = np.full((P, K), -1, np.int32); f = lambda: np.full((P, K), np.nan, np.float32)
    agree, conf, snr, tl = f(), f(), f(), np.zeros(P, np.int32)
    T = np.full((P, K, L), np.nan, np.float32); T31 = np.full((P, K, M_REF), np.nan, np.float32)
    bs, bp, br = [], [], []
    for i, v in enumerate(vx):
        if not v:
            continue
        n = len(v["picks"]); col[i, :n] = np.asarray(cols)[v["picks"]]
        agree[i, :n], conf[i, :n], snr[i, :n] = v["agree"], v["conf"], v["snr"]
        tl[i] = len(v["mu"][0])
        for k in range(n):
            T[i, k, :tl[i]] = v["mu"][k]; T31[i, k] = BS.warp(v["mu"][k], M_REF)
            bs.append(v["beats"][k]); bp.append(np.full(v["beats"][k].size, rows[i]["period"], np.int32))
            br.append(np.full(v["beats"][k].size, k, np.int8))
    cat = lambda z, dt: np.concatenate(z).astype(dt) if z else np.zeros(0, dt)
    beat_sample = cat(bs, np.int64); beam, rbin = np.divmod(col, nb)
    ok = col >= 0
    np.savez_compressed(
        path, README=np.array(TEMPLATES_README), fs=np.float64(vmeta["fs"]),
        period=np.array([r["period"] for r in rows], np.int32), t0_s=np.array([r["t0_s"] for r in rows]),
        t1_s=np.array([r["t1_s"] for r in rows]), band_centre_bpm=np.array([r["band_centre_bpm"] for r in rows], np.int32),
        method=np.array([r["method"] for r in rows]), voxel_col=col, voxel_beam=np.where(ok, beam, -1).astype(np.int32),
        voxel_angle_deg=np.where(ok, ang[np.clip(beam, 0, ang.size - 1)], np.nan).astype(np.float32),
        voxel_range_bin=np.where(ok, rbin + b0, -1).astype(np.int32),
        voxel_range_m=np.where(ok, (rbin + b0) * mpb, np.nan).astype(np.float32),
        agree=agree, template_conf=conf, template_snr=snr, template_len=tl, templates=T, templates_31=T31,
        beat_sample=beat_sample, beat_t_s=beat_sample / float(vmeta["fs"]), beat_period=cat(bp, np.int32),
        beat_rank=cat(br, np.int8))


def _figure(path, rec, rows, res, halved=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    for r in rows:
        t, k = _readout(res[r["period"]], r["t0_s"], r["t1_s"])
        if t.size:
            ax[0].plot(t / 60, k, color="#2a78d6", lw=0.6, alpha=0.6)
        ax[0].plot([r["t0_s"] / 60, r["t1_s"] / 60], [r["hr_bpm"]] * 2, color="#0b0b0b", lw=2.2, solid_capstyle="butt")
        if r["breaths_per_min"] is not None:
            ax[1].plot([r["t0_s"] / 60, r["t1_s"] / 60], [r["breaths_per_min"]] * 2, color="#0b0b0b", lw=2.2,
                       solid_capstyle="butt")
    hrs = np.array([r["hr_bpm"] for r in rows], float)
    ax[0].set_ylim(max(200, np.nanmin(hrs) - 120), np.nanmax(hrs) + 120)
    ax[0].set_ylabel("heart rate (bpm)"); ax[1].set_ylabel("breaths / min"); ax[1].set_xlabel("time (min)")
    ax[0].set_title(f"{rec}: radar heart rate per still, regular-breathing period "
                    f"(black = period value, blue = 10 Hz track)", loc="left", fontsize=11)
    if halved:
        ax[0].text(0.5, 0.95, "BREATHING HALVED: these are not trustworthy heart rates -- re-segment (HEART_RATE.md)",
                   transform=ax[0].transAxes, ha="center", va="top", fontsize=12, fontweight="bold", color="#c0392b",
                   bbox=dict(facecolor="white", edgecolor="#c0392b", boxstyle="round,pad=0.3"))
    for a in ax:
        a.grid(color="#e4e3df", lw=0.6); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _summary(rec, seg_json, rows, notes, extra):
    hr = np.array([r["hr_bpm"] for r in rows], float)
    rr = np.array([r["breaths_per_min"] for r in rows if r["breaths_per_min"] is not None], float)
    ratio = np.array([r["hr_over_rr"] for r in rows if r["hr_over_rr"] is not None], float)
    q = lambda x, p: round(float(np.percentile(x, p)), 1) if x.size else None
    m = dict(recording=rec, cohort=seg_json.get("cohort"), age_weeks=seg_json.get("age_weeks"),
             n_periods=len(rows), scored_min=round(sum(r["dur_s"] for r in rows) / 60, 2),
             hr_median_bpm=q(hr, 50), hr_p25_bpm=q(hr, 25), hr_p75_bpm=q(hr, 75),
             hr_min_bpm=round(float(hr.min()), 1) if hr.size else None,
             hr_max_bpm=round(float(hr.max()), 1) if hr.size else None,
             breaths_per_min_median=q(rr, 50), hr_over_rr_median=round(float(np.median(ratio)), 3) if ratio.size else None,
             n_fallback=sum(r["method"] != "bspec_beam_tmean" for r in rows))
    m.update(extra)
    m["notes"] = " | ".join(notes)
    return m


def _breath_check(seg_base, notes, extra, log):
    """Halved-breathing check of the segments (engine/breath_check.py): the method needs every breath counted."""
    try:
        import breath_check as BC
        chk = BC.rate_ratio(seg_base)
    except Exception as ex:                                   # advisory; never lose the heart rate over it
        notes.append(f"breath check skipped ({type(ex).__name__}: {ex})")
        return
    extra["breath_check_ratio"] = round(chk["ratio"], 3) if np.isfinite(chk["ratio"]) else None
    extra["breathing_halved"] = BC.is_halved(chk)
    if extra["breathing_halved"]:
        notes.append(BC.halved_note(chk))
        log("[hr]       " + "!" * 100); log(f"[hr]       WARNING: {notes[-1]}."); log("[hr]       " + "!" * 100)


def analyze(rec, sub_dir, out_dir, exp_dir=None, workers=None, patch_beams=None, keep_work=False, log=print,
            min_period_s=None):
    """Everything for one recording. Returns (summary metrics, period rows, per-period results, info).
    min_period_s: shortest still period scored (default 20 s, the validated floor)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    T0 = time.time()
    seg_base = os.path.join(sub_dir, rec)
    seg_json = json.load(open(seg_base + ".json"))
    exp_dir = exp_dir or seg_json.get("session_dir")
    if not (exp_dir and glob.glob(os.path.join(exp_dir, "*_voxels.json"))):
        raise FileNotFoundError(f"raw experiment folder not found ({exp_dir!r}); the heart-rate step re-reads the voxels "
                                f"-- pass --exp <folder with *_voxels.f32>")
    power, vmeta, _ = VS.load_experiment(exp_dir)
    cube_path, shape = str(power.filename), tuple(power.shape)
    fs = float(vmeta["fs"]); disk_min = shape[0] / fs / 60
    notes = []
    if seg_json.get("rec_min") is not None and abs(float(seg_json["rec_min"]) - disk_min) > 0.05:
        raise ValueError(f"{exp_dir!r} holds {disk_min:.2f} min of voxels but {rec} was segmented from "
                         f"{seg_json['rec_min']} min -- wrong --exp folder?")
    if VS.rec_name(exp_dir) != rec:
        notes.append(f"raw folder is named {VS.rec_name(exp_dir)!r}, not {rec!r} -- check it is the same recording")
    sha = hashlib.sha256(open(FROZEN_FILE, "rb").read()).hexdigest()
    min_s = float(min_period_s or HB.MIN_RUN_S)
    extra = dict(patch_beams=None, patch_deg=None, breath_peak_bin=None, patch_edge_warning=None,
                 frozen_code_ok=sha == FROZEN_SHA256, min_period_s=min_s, breath_check_ratio=None, breathing_halved=None,
                 cube_dejumped=bool(vmeta.get("dejump")))
    if vmeta.get("dejump"):
        dj = vmeta["dejump"]
        notes.append(f"cube de-jumped first (July-2026 capture build: {dj.get('n_step')} file steps + {dj.get('n_glitch')} "
                     f"dropouts repaired; DEJUMP.md)")
    _breath_check(seg_base, notes, extra, log)
    if sha != FROZEN_SHA256:
        notes.append("engine/hr_final_pipeline.py differs from the validated frozen file -- results are not the "
                     "validated method")
    if abs(fs - FS_REQUIRED) > 1e-6:
        notes.append(f"skipped: cube is {fs:g} Hz; the heart-rate method is defined for the 250 Hz live cube")
        return _summary(rec, seg_json, [], notes, extra), [], {}, {}
    if not HB.patch_fits(vmeta):
        notes.append(f"skipped: the cube's range bins ({vmeta['bins'][0]}-{vmeta['bins'][0] + vmeta['n_bin'] - 1}) or "
                     f"beams ({vmeta['n_angle']}) do not cover the chest patch (bins 50-82, 10 beams)")
        return _summary(rec, seg_json, [], notes, extra), [], {}, {}

    on, tt, runs = HB.breathing(seg_base, min_s)
    if min_s < HB.MIN_RUN_S:
        notes.append(f"periods down to {min_s:g} s (validated floor is {HB.MIN_RUN_S:g} s)")
    rrs = [HB.run_rr(on, tt, a, b) for a, b in runs]
    log(f"[hr]       {rec}: {len(runs)} still, regular-breathing periods >= {min_s:g} s "
        f"({sum(b - a for a, b in runs) / 60:.1f} min)")
    if not runs:
        notes.append(f"no clean-breathing period >= {min_s:g} s (the animal was never still long enough) -- no heart rate")
        return _summary(rec, seg_json, [], notes, extra), [], {}, {}

    t = time.time()
    ang = np.asarray(vmeta["angles_deg"], float)
    if patch_beams:
        lo, hi = (int(x) for x in str(patch_beams).split("-"))
        beams, bpw, pk_bin = list(range(lo, hi + 1)), None, None
        notes.append(f"patch beams set by hand ({lo}-{hi})")
    else:
        beams, bpw, pk_bin = HB.ecg_free_patch(power, vmeta, runs, rrs)
    cols = HB.patch_cols(vmeta, beams)
    edge = pk_bin is not None and pk_bin in (HB.PATCH_BINS[0], HB.PATCH_BINS[-1])
    extra.update(patch_beams=f"{beams[0]}-{beams[-1]}", patch_deg=f"{ang[beams[0]]:+.2f}..{ang[beams[-1]]:+.2f}",
                 breath_peak_bin=pk_bin, patch_edge_warning=bool(edge))
    log(f"[hr]       chest patch: beams {beams[0]}-{beams[-1]} ({ang[beams[0]]:+.1f}..{ang[beams[-1]]:+.1f} deg) x range bins "
        f"{HB.PATCH_BINS[0]}-{HB.PATCH_BINS[-1]}" + (f"; breathing power peaks at bin {pk_bin}" if pk_bin is not None else ""))
    if edge:
        notes.append("breathing power peaks at the edge of the fixed patch range (0.27-0.45 m): the mouse may sit "
                     "outside it -- treat HR as unreliable")
        log("[hr]       WARNING: " + notes[-1])
    t_patch = time.time() - t

    work = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "hr_work", rec)
    os.makedirs(work, exist_ok=True)
    workers = int(workers or default_workers())
    order = sorted(range(len(runs)), key=lambda i: -(runs[i][1] - runs[i][0]))       # longest first: load balance
    tasks = {i: (rec, cube_path, shape, cols, int(vmeta["n_bin"]), on, tt, runs[i][0], runs[i][1], rrs[i],
                 os.path.join(work, f"A_{runs[i][0]:.2f}.npy")) for i in order}
    saved = {k: os.environ.get(k) for k in _BLAS}
    for k in _BLAS:
        os.environ[k] = "1"                       # workers: 1 BLAS thread each (as validated; no oversubscription)
    res, t = {}, time.time()
    log(f"[hr]       heart rate on {len(runs)} periods with {workers} workers (a few minutes per hour of recording) ...")
    try:
        with ProcessPoolExecutor(workers) as pool:
            fut = {pool.submit(_period_worker, tasks[i]): i for i in order}
            for n, f in enumerate(as_completed(fut), 1):
                i = fut[f]; x = f.result(); res[i] = x
                log(f"[hr]         {n:3d}/{len(runs)}  period at {runs[i][0] / 60:6.1f} min ({runs[i][1] - runs[i][0]:5.1f} s): "
                    f"HR {x['hr']:6.1f} bpm, band {x['centre']:.0f}  [{x['elapsed_s']:.0f} s]"
                    + (f"  [{x['method']}: {x['reason']}]" if x["reason"] else ""))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
            try:
                os.rmdir(os.path.dirname(work))
            except OSError:
                pass
    t_hr = time.time() - t

    rows = []
    for i, ((a, b), rr) in enumerate(zip(runs, rrs)):
        x = res[i]; hr = float(x["hr"])
        rows.append(dict(recording=rec, period=i + 1, t0_s=round(a, 3), t1_s=round(b, 3), start_min=round(a / 60, 2),
                         dur_s=round(b - a, 1), breaths_per_min=round(rr, 1) if rr else None, hr_bpm=round(hr, 2),
                         hr_over_rr=round(hr / rr, 3) if rr else None, band_centre_bpm=int(x["centre"]),
                         n_beats=int(np.asarray(x["onsets"]).size), method=x["method"], fallback_reason=x["reason"]))
    res = {r["period"]: res[i] for i, r in enumerate(rows)}
    if any(r["method"] != "bspec_beam_tmean" for r in rows):
        notes.append(f"{sum(r['method'] != 'bspec_beam_tmean' for r in rows)} period(s) used the S1 fallback "
                     f"(see hr_periods.csv)")
    extra.update(workers=workers, runtime_s=round(time.time() - T0, 1))
    info = dict(exp_dir=exp_dir, cube=cube_path, segments=seg_base, beam_power=bpw.tolist() if bpw is not None else None,
                frozen_sha256=sha, timing_s=dict(patch=round(t_patch, 1), heart_rate=round(t_hr, 1),
                                                 total=round(time.time() - T0, 1)),
                hr_bpm_exact=[float(res[r["period"]]["hr"]) for r in rows],
                period_elapsed_s=[round(res[r["period"]]["elapsed_s"], 1) for r in rows],
                bands_segmented=[res[r["period"]]["segmented_bands"] for r in rows],
                _cols=cols, _vmeta=vmeta)                   # for the template export; popped before hr.json
    return _summary(rec, seg_json, rows, notes, extra), rows, res, info


def run(rec, sub_dir=None, out_dir=None, log=print, exp_dir=None, workers=None, patch_beams=None, keep_work=False,
        figure=True, min_period_s=None):
    """Heart rate for one segmented recording. Returns the flat summary dict."""
    import substrate as SUB
    sub_dir = sub_dir or SUB.SUB_DIR
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(sub_dir)), "metrics")
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, glob.escape(rec) + ".hr*")):
        os.remove(f)
    m, rows, res, info = analyze(rec, sub_dir, out_dir, exp_dir=exp_dir, workers=workers, patch_beams=patch_beams,
                                 keep_work=keep_work, log=log, min_period_s=min_period_s)
    ctx = {k: info.pop(k) for k in ("_cols", "_vmeta") if k in info}
    with open(os.path.join(out_dir, rec + ".hr.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(m.keys()); w.writerow(m.values())
    _write_csv(os.path.join(out_dir, rec + ".hr_periods.csv"), PERIOD_COLS, rows)
    with open(os.path.join(out_dir, rec + ".hr_track.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["period", "t_s", "hr_bpm"])
        for r in rows:
            for tt_, hh in zip(*_readout(res[r["period"]], r["t0_s"], r["t1_s"])):
                w.writerow([r["period"], f"{tt_:.2f}", f"{hh:.2f}"])
    with open(os.path.join(out_dir, rec + ".hr_beats.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["period", "t_s"])
        for r in rows:
            for b in np.asarray(res[r["period"]]["onsets"], float):
                w.writerow([r["period"], f"{b:.4f}"])
    if rows:
        _write_templates(os.path.join(out_dir, rec + ".hr_templates.npz"), rows, res, ctx["_cols"], ctx["_vmeta"])
    json.dump(dict(metrics=m, periods=rows, info=info), open(os.path.join(out_dir, rec + ".hr.json"), "w"), indent=1,
              default=float)
    if figure and rows:
        try:
            _figure(os.path.join(out_dir, rec + ".hr.png"), rec, rows, res, halved=bool(m.get("breathing_halved")))
        except Exception as ex:                       # a figure failure must never lose the numbers
            log(f"[hr]       figure skipped ({ex})")
    hr = "n/a" if m["hr_median_bpm"] is None else f"{m['hr_median_bpm']:.0f} bpm"
    log(f"[hr]       {rec:20s} {m['n_periods']} periods ({m['scored_min']} min) | median HR {hr} "
        f"(IQR {m['hr_p25_bpm']}-{m['hr_p75_bpm']}) | fallbacks {m['n_fallback']}  [{m.get('runtime_s', 0):.0f}s]"
        + (f"\n           NOTE: {m['notes']}" if m["notes"] else ""))
    return m


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="radar heart rate for one segmented recording")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--sub-dir", default=None)
    ap.add_argument("--exp", default=None, help="raw experiment folder (default: the substrate's session_dir)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--patch-beams", default=None, help="override the chest patch beams, e.g. 6-15")
    ap.add_argument("--min-period-s", type=float, default=None, help="shortest still period (default 20 s)")
    a = ap.parse_args()
    run(a.rec, sub_dir=a.sub_dir, out_dir=a.out, exp_dir=a.exp, workers=a.workers, patch_beams=a.patch_beams,
        min_period_s=a.min_period_s)
