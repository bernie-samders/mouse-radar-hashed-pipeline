r"""voxel_store.py -- read a raw mouse radar experiment (the `*_voxels.f32` power cube).

A raw experiment directory (e.g. one folder under `G:\Research\07-02-26_mouse_data_training\`) contains:
    <name>_voxels.json   -- header: fs, n_time, n_vox, n_angle, n_bin, angles_deg[], bins[], f32_file, dtype
    <name>_voxels.f32    -- the (n_time, n_vox) float32 beamformed-POWER cube, memmapped read-only
    <name>_summary.json  -- run summary; its "config" block carries floor_bpm / resp_hi_bpm (optional here)

`load_voxel_store` is vendored verbatim from the lab's `breath_processor.py` (the only piece of that
48 KB module the pipeline needs).  The voxel column index is angle-major, bin-minor: `col = ai*n_bin + bj`.
"""
from __future__ import annotations
import os, glob, json
from pathlib import Path
import numpy as np

# fallback config if a *_summary.json is missing (these are the standard mouse capture settings)
DEFAULT_CONFIG = {"floor_bpm": 100.0, "resp_hi_bpm": 480.0, "hf_lo_hz": 1.6, "hf_hi_hz": 25.0, "neigh": 0.12}


def load_voxel_store(path):
    """`path` = the `_voxels.json` or the `_voxels.f32`.  Returns (power, meta) where power is a
    read-only (n_time, n_vox) float32 memmap.  (Vendored from breath_processor.load_voxel_store.)"""
    path = Path(path)
    json_path = path if path.suffix == ".json" else path.with_suffix(".json")
    meta = json.loads(json_path.read_text())
    f32_path = json_path.parent / meta["f32_file"]
    n_time, n_vox = int(meta["n_time"]), int(meta["n_vox"])
    power = np.memmap(f32_path, dtype=np.dtype(meta.get("dtype", "float32")),
                      mode="r", shape=(n_time, n_vox))
    return power, meta


def _one(pattern, exp_dir):
    hits = glob.glob(os.path.join(exp_dir, pattern))
    return hits[0] if hits else None


def load_experiment(exp_dir):
    """Open a raw experiment directory.  Returns (power, meta, cfg):
        power : read-only (n_time, n_vox) float32 memmap of beamformed power
        meta  : the *_voxels.json header (fs, n_bin, n_angle, angles_deg, bins, ...)
        cfg   : the *_summary.json "config" block (floor_bpm, resp_hi_bpm, ...), or DEFAULT_CONFIG."""
    vj = _one("*_voxels.json", exp_dir)
    if vj is None:
        raise FileNotFoundError(f"no *_voxels.json in {exp_dir!r} -- is this a raw experiment directory?")
    power, meta = load_voxel_store(vj)
    sj = _one("*_summary.json", exp_dir)
    if sj is not None:
        try:
            cfg = json.load(open(sj)).get("config", {}) or {}
        except Exception:
            cfg = {}
    else:
        cfg = {}
    cfg = {**DEFAULT_CONFIG, **cfg}
    return power, meta, cfg


def rec_name(exp_dir):
    """Stable recording key from an experiment dir: the *_voxels.json basename minus `_voxels`,
    with any trailing `_YYYYMMDD_hhmmss` capture stamp stripped."""
    import re
    vj = _one("*_voxels.json", exp_dir)
    base = os.path.basename(vj)[:-len("_voxels.json")] if vj else os.path.basename(os.path.normpath(exp_dir))
    return re.sub(r"_\d{8}_\d{6}$", "", base)
