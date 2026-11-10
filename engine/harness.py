"""harness.py -- data access for the frozen heart-rate code (hr_final_pipeline.py does `import harness as H`).

The frozen file is shipped byte-identical to the validated one (sha256 checked by pipelines/hr.py), so it still
imports this module by its research name. Here it only serves ONE recording's raw cube, chest patch and breathing
cycles, registered with register(); the research version also held the ECG scoring, which is not shipped.
Functions keep the research signatures and arithmetic (block = sqrt(power) as float32, column = ang_idx * n_bin + bin).
"""
from __future__ import annotations
import time
import numpy as np

FS = 250.0                                   # the frozen constants are defined for the 250 Hz live cube
RECS = ()
_CTX = {}


def register(rec, cube_path, shape, patch_cols, onset, ttot):
    """Make recording `rec` readable: cube = (n_time, n_vox) float32 power memmap; breathing = fit-ok DP cycles."""
    _CTX[rec] = dict(cube=np.memmap(cube_path, dtype=np.float32, mode="r", shape=tuple(shape)),
                     patch_cols=np.asarray(patch_cols), onset=np.asarray(onset, float), ttot=np.asarray(ttot, float))


def read_block(cube, t0, t1, cols, amp=True, step_s=60.0):
    """[T, n_cols] float32 for radar seconds [t0, t1); sqrt(power) if amp. Chunked, retries on OSError (USB drives drop
    out on long reads)."""
    # Length from the duration, not round(t1 * FS) - round(t0 * FS): rounding both ends separately could return one
    # sample short when t0 * FS fell on a half sample (hr then fell back to S1, hr_curve crashed). Fixed in 1.7; the
    # same samples as before whenever the old length was right.
    a0 = int(round(t0 * FS))
    a1 = a0 + int(round((t1 - t0) * FS))
    out = np.empty((a1 - a0, len(cols)), np.float32); st = int(step_s * FS)
    for s in range(a0, a1, st):
        e = min(a1, s + st)
        for _ in range(40):
            try:
                x = np.asarray(cube[s:e], np.float32)[:, cols]
                if x.shape[0] != e - s:
                    raise OSError("short read")
                break
            except (OSError, ValueError):
                time.sleep(15)
        else:
            raise RuntimeError(f"cube read failed at samples {s}-{e}")
        out[s - a0:e - a0] = np.sqrt(np.maximum(x, 0)) if amp else x
    return out


def block(rec, t0, t1, cols=None, amp=True):
    """[T, n_cols] for radar seconds [t0, t1); amplitude = sqrt(power)."""
    c = _CTX[rec]["cube"]
    cols = np.arange(c.shape[1]) if cols is None else np.asarray(cols)
    return read_block(c, t0, t1, cols, amp)


def patch_cols(rec):
    return _CTX[rec]["patch_cols"]


def patch_accel(rec, run):
    """[T, n_patch] acceleration of the chest patch over exactly [t0, t1) (float32 npy written by hr_bands.build_accel)."""
    return np.load(run["accel_patch"], mmap_mode="r")


def accel(x):
    """EQ-Radio acceleration (Holoborodko 2nd derivative + 45 Hz low-pass), per column if 2-D."""
    import eqradio_hr as eq
    x = np.asarray(x, float)
    if x.ndim == 1:
        return eq.accel_holoborodko(x, FS)
    return np.stack([eq.accel_holoborodko(x[:, k], FS) for k in range(x.shape[1])], 1)


def breath_rate(rec, a, b):
    c = _CTX[rec]; m = (c["onset"] >= a) & (c["onset"] < b)
    return float(60.0 / np.median(c["ttot"][m])) if m.any() else np.nan


def ibis_from_onsets(on):
    on = np.asarray(on, float); return on[1:], np.diff(on)


def hr_fn(ibi_t_list, ibi_list, min_n=10):
    """callable(a, b) -> pooled HR in [a, b) (S1 fallback window function)."""
    it = np.concatenate([np.asarray(x, float) for x in ibi_t_list]); ib = np.concatenate([np.asarray(x, float) for x in ibi_list])
    o = np.argsort(it); it, ib = it[o], ib[o]

    def f(a, b):
        i, j = np.searchsorted(it, a), np.searchsorted(it, b)
        return float(60.0 / np.median(ib[i:j])) if j - i >= min_n else np.nan
    return f


def _research_only(*a, **k):
    raise NotImplementedError("research-pack access (ECG-scored runs / precomputed band scans) is not shipped; "
                              "pipelines/hr.py calls process_run() per run instead")


runs = all_runs = tight = tight_all = _research_only
