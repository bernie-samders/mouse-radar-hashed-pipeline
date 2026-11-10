"""core.py -- minimal shared constants + weighted-median / regular-run helpers.

This is a trimmed stand-in for the original `ml_apnea/cohort/core.py`: it keeps only what
`phase_lib` and `stettner_metrics` actually import (the breath-weighted median and the
regular-run splitter, plus the three pause/run constants).  Everything else in the original
(cohort roll-ups, wslib/voxel imports, genotype grouping) was research machinery and is dropped.
"""
from __future__ import annotations
import numpy as np

# pause / regular-run constants (verbatim from the original core.py)
PAUSE_MULT = 1.5       # IBI > this x LOCAL running median = a pause that BREAKS a regular-breathing run
RUN_MED_WIN = 11       # window (IBIs) of the local running median used to define a pause (rate-drift robust)
MIN_RUN = 10           # a regular-breathing run needs this many IBIs to yield a Poincare point


def _wmedian(vals, wts):
    """Weight-aware median (breath-weighted across episodes/windows, lab convention)."""
    vals = np.asarray(vals, float); wts = np.asarray(wts, float)
    ok = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    vals, wts = vals[ok], wts[ok]
    if vals.size == 0:
        return float("nan")
    order = np.argsort(vals); vals, wts = vals[order], wts[order]
    cdf = np.cumsum(wts) - 0.5 * wts
    cdf /= wts.sum()
    return float(np.interp(0.5, cdf, vals))


def _regular_runs(ibi_s, mult=PAUSE_MULT, med_win=RUN_MED_WIN, min_run=MIN_RUN):
    """Split an IBI series into contiguous runs of NORMAL breaths, breaking at every pause
    (IBI > mult x local running median).  Returns (list_of_runs, n_pauses)."""
    from scipy.ndimage import median_filter
    x = np.asarray(ibi_s, float)
    if x.size < 3:
        return [], 0
    loc = median_filter(x, size=max(3, med_win | 1), mode="nearest")
    is_pause = x > mult * loc
    runs, cur = [], []
    for v, p in zip(x, is_pause):
        if p:
            if len(cur) >= min_run:
                runs.append(np.asarray(cur))
            cur = []
        else:
            cur.append(v)
    if len(cur) >= min_run:
        runs.append(np.asarray(cur))
    return runs, int(is_pause.sum())
