r"""dejump.py -- remove the July-2026 capture-build jumps from a voxel cube before the heart-rate steps (new in 1.5).

WHY. The live-capture build used in July 2026 (07-02 / 07-16 / 07-20 recordings) writes the cube in 1066-frame files
(4.264 s) and puts a level step into EVERY voxel at each file boundary; some of its recordings also drop single frames
(every voxel dips for one frame). The heart-rate steps work on the 2nd derivative (acceleration) of the voxel amplitude,
which turns each jump into a broadband spike that buries the heart: on raw July cubes the heart signal is lost
and `hr`'s period heart rates are off by ~60 bpm. Later captures (08-06 on) do not have the step. See DEJUMP.md.

WHAT. Port of the validated research tool (output/july_dejump/dejump_cube.py; same constants and arithmetic):
  amplitude A = sqrt(power) (float32, exactly as harness.read_block hands it to the heart-rate code);
  d[i] = mean over ALL voxels of |A[i+1] - A[i]|; a frame difference is a JUMP when d > 20 x its running median (10 s);
  jump differences <= 3 frames apart are one event;
  per event and voxel, each jump difference is replaced by the mean of its two neighbouring differences and the trace
  is re-integrated (a step's offset is removed, a dropped frame becomes a straight line).
The repaired copy is POWER again ((A - O + c)^2), so `hr` / `b2b` read it like any cube. O = the re-integration
offset (constant between jumps); c = a per-voxel constant that keeps the amplitude >= 0. The heart-rate steps only use
the acceleration and the mean-removed, band-passed amplitude, so c changes nothing for them -- but the absolute level of
the copy is arbitrary: use it for heart rate only, never for motion / presence / level metrics. Samples the repair does
not touch are copied bit for bit. By default only range bins 50-82 (the heart-rate chest patch range, all beams) are
written: ~27 % of a 124-bin cube.

VALIDATED (output/july_dejump/README.md): a step-free cube (B16) comes out with identical heart rate (5e-6 bpm); the
same cube with July-sized steps injected comes back within 1.8 bpm with every band and SNR call unchanged; on July
cubes the copy reproduces the research per-period numbers.

    python run.py dejump --exp <raw experiment folder> [--out DIR] [--bins 50-82|all] [--detect-only]
"""
from __future__ import annotations
import os, glob, json, time, shutil
import numpy as np
from scipy.ndimage import median_filter

FS = 250.0
FILE_FRAMES = 1066                # capture-file length of the live-capture build (4.264 s at 250 Hz)
JUMP_K = 20.0                     # step-free still periods: <= 6x; July file steps: mostly 30-500x (weakest seen 21x)
WIN_S = 10.0                      # running-median window of d
MERGE = 3                         # jump differences this close together = one event
CHUNK = 15000                     # rows per read (60 s; ~156 MB of a 2604-voxel cube)
BLK = 50                          # block of the per-voxel amplitude minimum (positivity bound; CHUNK % BLK == 0)
EXACT_MAX = 400                   # <= this many blocks with a jump: exact bound (keeps step-free cubes bit-identical where
                                  # possible); July cubes have ~1200-3700 -> conservative bound (still guarantees >= 0)
HR_BINS = (50, 82)                # absolute range bins the heart-rate chest patch can use (hr_bands.PATCH_BINS)
AUTO_MIN_DROPOUTS_PER_MIN = 1.0   # auto mode: repair when the file-step grid is present or dropouts reach this rate
                                  # (July: 14 steps/min on 99 % of boundaries; later captures: no grid, <= 0.2 events/min)


def open_cube(exp):
    vj = glob.glob(os.path.join(exp, "*_voxels.json"))
    if not vj:
        raise FileNotFoundError(f"no *_voxels.json in {exp!r} -- is this a raw experiment directory?")
    meta = json.load(open(vj[0]))
    T, V = int(meta["n_time"]), int(meta["n_vox"])
    P = np.memmap(os.path.join(exp, meta["f32_file"]), np.dtype(meta.get("dtype", "float32")), "r", shape=(T, V))
    return P, meta, vj[0]


def read_rows(P, s, e):
    for _ in range(40):                                       # USB drives drop out on long reads (as harness.read_block)
        try:
            x = np.asarray(P[s:e], np.float32)
            if x.shape[0] == e - s:
                return x
        except (OSError, ValueError):
            pass
        time.sleep(15)
    raise RuntimeError(f"cube read failed at rows {s}-{e}")


def amp(x):
    return np.sqrt(np.maximum(x, 0))                          # float32, exactly as harness.read_block


def pass1(P, wcols):
    """d (all-voxel mean |diff| of A, length T-1) and the per-block minimum amplitude of the written columns."""
    T = P.shape[0]
    d = np.zeros(T - 1, np.float64)
    amin = np.zeros(((T + BLK - 1) // BLK, wcols.size), np.float32)
    prev = None
    for s in range(0, T, CHUNK):
        e = min(T, s + CHUNK)
        A = amp(read_rows(P, s, e))
        B = A if prev is None else np.vstack([prev, A])
        o = s if prev is None else s - 1
        d[o:o + B.shape[0] - 1] = np.abs(np.diff(B, axis=0)).mean(1)
        Aw = A[:, wcols]; nb = (e - s + BLK - 1) // BLK
        pad = nb * BLK - (e - s)
        if pad:
            Aw = np.vstack([Aw, np.repeat(Aw[-1:], pad, 0)])
        amin[s // BLK:s // BLK + nb] = Aw.reshape(nb, BLK, -1).min(1)
        prev = A[-1:]
    return d, amin


def find_events(d, k=JUMP_K, win_s=WIN_S):
    w = int(round(win_s * FS)) | 1
    med = median_filter(d, size=w, mode="nearest")
    r = d / np.maximum(med, 1e-30)
    groups = []
    for i in np.flatnonzero(r > k):
        if groups and i - groups[-1][-1] <= MERGE:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups, r


def event_deltas(P, groups, wcols):
    """Per event: kind ('step' = level change / 'glitch' = dip and recovery) and the per-difference corrections
    D[k] - D'[k] of the written columns (D' = mean of the neighbouring differences)."""
    T = P.shape[0]
    kinds, ks, deltas = [], [], []
    for g in groups:
        i0, i1 = g[0], g[-1]
        s, e = max(0, i0 - 1), min(T, i1 + 3)
        D = np.diff(amp(read_rows(P, s, e)).astype(np.float64), axis=0)       # D[j] = difference index s + j
        gi = np.arange(i0, i1 + 1) - s
        net = np.abs(D[gi].sum(0)).mean() / (np.abs(D[gi]).sum(0).mean() + 1e-30)
        kinds.append("step" if net > 0.5 else "glitch")
        nb = [j - s for j in (i0 - 1, i1 + 1) if 0 <= j <= T - 2]
        Dn = D[nb][:, wcols].mean(0) if nb else np.zeros(wcols.size)
        for j in gi:                                          # every difference in [i0, i1] is replaced
            ks.append(s + j); deltas.append(D[j, wcols] - Dn)
    return kinds, np.asarray(ks, np.int64), (np.asarray(deltas) if deltas else np.zeros((0, wcols.size)))


def offsets(P, wcols, ks, deltas, amin):
    """Change frames cf (= difference index + 1), cumulative offset after each change, and the per-voxel constant c."""
    T = P.shape[0]
    cf = ks + 1
    cum = np.cumsum(deltas, 0) if deltas.size else np.zeros((0, amin.shape[1]))
    nblk = amin.shape[0]
    starts = np.arange(nblk) * BLK
    cum0 = np.vstack([np.zeros((1, amin.shape[1])), cum])
    ostart = cum0[np.searchsorted(cf, starts, side="right")]  # O at each block start (constant in change-free blocks)
    omax = ostart.copy()
    for e in range(cf.size):                                  # every value O takes inside a block (conservative bound)
        np.maximum(omax[cf[e] // BLK], cum[e], out=omax[cf[e] // BLK])
    need = omax - amin                                        # c >= max_t (O - A) keeps A - O + c >= 0
    cblk = np.unique(cf // BLK)
    for b in (cblk if cblk.size <= EXACT_MAX else []):        # few change blocks: exact, from their own rows
        s, e = b * BLK, min(T, (b + 1) * BLK)
        A = amp(read_rows(P, s, e)[:, wcols]).astype(np.float64)
        sel = np.flatnonzero((cf >= s) & (cf < e))
        L = np.zeros_like(A)
        L[cf[sel] - s] = deltas[sel]                          # cf is strictly increasing (one entry per difference)
        need[b] = ostart[b] + (np.cumsum(L, 0) - A).max(0)
    need = need.max(0)
    edges = np.concatenate([[0], cf, [T]])
    longest = int(np.argmax(np.diff(edges)))                  # longest stretch with constant O -> left bit-identical
    pref = cum0[longest]
    c = np.where(pref >= need, pref, np.maximum(need, 0.0))
    return cf, cum0, c, (int(edges[longest]), int(edges[longest + 1]))


def write_cube(P, wcols, cf, cum0, c, path):
    T = P.shape[0]
    out = np.memmap(path + ".part", np.float32, "w+", shape=(T, wcols.size))           # renamed when complete
    n_changed = 0
    for s in range(0, T, CHUNK):
        e = min(T, s + CHUNK)
        X = read_rows(P, s, e)[:, wcols]
        rows = np.arange(s, e)
        sh = cum0[np.searchsorted(cf, rows, side="right")] - c                    # O - c per row and voxel
        keep = sh == 0
        A = amp(X).astype(np.float64)
        Y = np.where(keep, X, ((A - sh) ** 2).astype(np.float32))
        out[s:e] = Y; n_changed += int((~keep).sum())
    out.flush(); del out
    os.replace(path + ".part", path)
    return n_changed


def grid_phase(groups, kinds):
    """Frame residue (mod FILE_FRAMES) of the file steps, or None when the steps do not form a grid."""
    st = [g[-1] + 1 for g, k in zip(groups, kinds) if k == "step"]
    if len(st) < 10:
        return None
    v, n = np.unique(np.asarray(st) % FILE_FRAMES, return_counts=True)
    return int(v[np.argmax(n)]) if n.max() >= 0.5 * len(st) else None


def on_grid(groups, ph, tol=2):
    if ph is None:
        return np.zeros(len(groups), bool)
    f = np.array([g[-1] + 1 for g in groups])
    return np.abs(((f - ph + FILE_FRAMES // 2) % FILE_FRAMES) - FILE_FRAMES // 2) <= tol


def write_cols(meta, bins="hr"):
    """Cube columns to write: 'hr' = range bins 50-82 (all beams) when the cube covers them, else every bin; 'all';
    or 'LO-HI' absolute bins."""
    nb, na, b0 = int(meta["n_bin"]), int(meta["n_angle"]), int(meta["bins"][0])
    if bins in (None, "all"):
        lo, hi = b0, b0 + nb - 1
    elif bins == "hr":
        lo, hi = HR_BINS if (b0 <= HR_BINS[0] and HR_BINS[1] < b0 + nb) else (b0, b0 + nb - 1)
    else:
        lo, hi = (int(x) for x in str(bins).split("-"))
        if not (b0 <= lo and hi < b0 + nb):
            raise ValueError(f"bins {bins} outside the cube's bins {b0}-{b0 + nb - 1}")
    wbins = list(range(lo, hi + 1))
    return np.array([ai * nb + (b - b0) for ai in range(na) for b in wbins]), wbins


def needs_repair(rep, mode="auto"):
    """auto: the July file-step grid is present, or single-frame dropouts reach AUTO_MIN_DROPOUTS_PER_MIN.
    on: any jump at all. off: never."""
    if mode == "off":
        return False
    if mode == "on":
        return rep["n_events"] > 0
    return rep["grid_phase_frames"] is not None or rep["glitches_per_min"] >= AUTO_MIN_DROPOUTS_PER_MIN


def run(exp, cache_root=None, bins="hr", detect_only=False, mode="on", log=print):
    """Detect the jumps of one raw experiment and (unless detect_only, or mode says no repair is needed) write the
    repaired copy to <cache_root>/<experiment folder name>/. Returns the report dict (out_dir = the copy, or None)."""
    t0 = time.time()
    P, meta, vj = open_cube(exp)
    T, V = P.shape
    wcols, wbins = write_cols(meta, bins)
    log(f"[dejump]   {os.path.basename(os.path.normpath(exp))}: {T / FS / 60:.1f} min x {V} voxels -- looking for "
        f"capture-build jumps (one read of the cube)")
    d, amin = pass1(P, wcols)
    groups, r = find_events(d)
    kinds, ks, deltas = event_deltas(P, groups, wcols)
    ph = grid_phase(groups, kinds); og = on_grid(groups, ph)
    st = np.array([k == "step" for k in kinds], bool)
    rmin = T / FS / 60
    rep = dict(exp_dir=os.path.abspath(exp), n_time=T, rec_min=round(rmin, 2), jump_k=JUMP_K, win_s=WIN_S,
               n_events=len(groups), n_step=int(st.sum()), n_glitch=int((~st).sum()),
               grid_phase_frames=ph, n_grid_boundaries=int((T - 1 - ph) // FILE_FRAMES + 1) if ph is not None else 0,
               n_on_grid=int(og.sum()), n_off_grid=int((~og).sum()),
               steps_per_min=round(st.sum() / rmin, 2), glitches_per_min=round((~st).sum() / rmin, 2),
               n_diffs_replaced=int(ks.size), mode=mode, repaired=False, out_dir=None)
    rep["file_step_grid"] = ph is not None
    rep["grid_boundaries_hit"] = round(rep["n_on_grid"] / rep["n_grid_boundaries"], 3) if rep["n_grid_boundaries"] else None
    log(f"[dejump]   {rep['n_events']} jumps: {rep['n_step']} level steps ({rep['steps_per_min']}/min), "
        f"{rep['n_glitch']} single-frame dropouts ({rep['glitches_per_min']}/min); capture-file step grid: "
        + (f"YES ({100 * rep['grid_boundaries_hit']:.0f} % of the {rep['n_grid_boundaries']} file boundaries)"
           if ph is not None else "no"))
    if not detect_only and needs_repair(rep, mode):
        cf, cum0, c, span = offsets(P, wcols, ks, deltas, amin)
        dst = os.path.join(cache_root, os.path.basename(os.path.normpath(exp)))
        os.makedirs(dst, exist_ok=True)
        f32 = os.path.join(dst, meta["f32_file"])
        n_changed = write_cube(P, wcols, cf, cum0, c, f32)
        m2 = dict(meta)
        m2.update(n_vox=int(wcols.size), n_bin=len(wbins), bins=wbins,
                  content=meta.get("content", "") + "; de-jumped (mouse_radar_pipeline engine/dejump.py): per-voxel "
                                                    "level shifted by a constant, use for heart rate only",
                  dejump=dict(source=os.path.abspath(vj), n_events=rep["n_events"], n_step=rep["n_step"],
                              n_glitch=rep["n_glitch"], jump_k=JUMP_K, win_s=WIN_S, bins_kept=f"{wbins[0]}-{wbins[-1]}",
                              samples_changed_frac=round(n_changed / (T * wcols.size), 6),
                              bit_identical_span_s=[round(span[0] / FS, 3), round(span[1] / FS, 3)]))
        for f in glob.glob(os.path.join(exp, "*_summary.json")):
            shutil.copy2(f, dst)
        with open(os.path.join(dst, "dejump_events.csv"), "w") as fh:
            fh.write("frame,t_s,n_diffs,ratio,kind,on_grid\n")
            for g, k, o in zip(groups, kinds, og):
                fh.write(f"{g[-1] + 1},{(g[-1] + 1) / FS:.3f},{g[-1] - g[0] + 1},{float(r[g].max()):.1f},{k},{int(o)}\n")
        json.dump(m2, open(os.path.join(dst, os.path.basename(vj)), "w"), indent=1)   # last: marks the copy complete
        rep.update(repaired=True, out_dir=dst, bins_kept=f"{wbins[0]}-{wbins[-1]}",
                   samples_changed_frac=m2["dejump"]["samples_changed_frac"])
        log(f"[dejump]   wrote the repaired copy (range bins {wbins[0]}-{wbins[-1]}, "
            f"{os.path.getsize(f32) / 1e9:.1f} GB) -> {dst}")
    elif not detect_only:
        log("[dejump]   no capture-build jumps to repair -- the heart-rate steps use the raw cube")
    rep["runtime_s"] = round(time.time() - t0, 1)
    return rep


def _complete_copy(dst, exp):
    """True if dst holds a finished copy made from exp (its json, written last, names the source cube)."""
    vj = glob.glob(os.path.join(dst, "*_voxels.json"))
    if not vj:
        return False
    try:
        src = json.load(open(vj[0])).get("dejump", {}).get("source", "")
    except ValueError:
        return False
    return os.path.normcase(os.path.dirname(os.path.abspath(src))) == os.path.normcase(os.path.abspath(exp))


def prepare(exp, out_root, rec=None, mode="auto", log=print):
    """The cube folder the heart-rate steps should read. mode off: exp itself. auto / on: detect once (report cached
    in <out_root>/metrics/<rec>.dejump.json) and, if needed, write / reuse the repaired copy in <out_root>/dejumped/."""
    mode = str(mode or "auto").lower()
    if mode == "off" or not exp:
        return exp
    if mode not in ("auto", "on"):
        raise ValueError(f"--dejump must be auto, on or off, not {mode!r}")
    import voxel_store as VS
    rec = rec or VS.rec_name(exp)
    met = os.path.join(out_root, "metrics"); os.makedirs(met, exist_ok=True)
    cache = os.path.join(out_root, "dejumped")
    dst = os.path.join(cache, os.path.basename(os.path.normpath(exp)))
    rj = os.path.join(met, rec + ".dejump.json")
    if os.path.exists(rj):
        rep = json.load(open(rj))
        if os.path.normcase(rep.get("exp_dir", "")) == os.path.normcase(os.path.abspath(exp)):   # counts don't depend on mode
            if not needs_repair(rep, mode):
                log(f"[dejump]   {rec}: no repair needed (cached check: {rep['n_events']} jumps, no file-step grid)")
                return exp
            if _complete_copy(dst, exp):
                log(f"[dejump]   {rec}: using the repaired copy {dst}")
                return dst
    rep = run(exp, cache, mode=mode, log=log)
    rep["recording"] = rec
    json.dump(rep, open(rj, "w"), indent=1, default=float)
    cols = ["recording", "rec_min", "file_step_grid", "grid_boundaries_hit", "n_events", "n_step", "n_glitch",
            "steps_per_min", "glitches_per_min", "mode", "repaired", "out_dir"]
    with open(os.path.join(met, rec + ".dejump.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n" + ",".join("" if rep.get(k) is None else str(rep.get(k)) for k in cols) + "\n")
    return rep["out_dir"] if rep["repaired"] else exp


def cleanup(out_root, exp, log=print):
    """Delete the repaired copy of exp (it is a cache: ~3.5 GB per recording hour)."""
    dst = os.path.join(out_root, "dejumped", os.path.basename(os.path.normpath(exp)))
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
        log(f"[dejump]   removed the repaired copy {dst} (keep it with --keep-dejumped)")
        try:
            os.rmdir(os.path.dirname(dst))
        except OSError:
            pass
