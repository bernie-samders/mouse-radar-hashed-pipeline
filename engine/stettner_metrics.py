"""stettner_metrics.py -- the windowing / variance / coupling helpers used by the Stettner pipeline.

Trimmed to the six primitives that `pipelines/stettner.py` reuses so its numbers stay identical to the
original `ml_apnea/cohort/stettner_metrics.py`: window slicing, CV, variance-in-ms^2, the breath-weighted
median over per-window records, and the Fisher-z mean of per-window correlations.  The original module's
`analyze_session`/registry/genotype machinery is dropped (research layer).
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import core
import phase_lib as P

MIN_WIN = 20                # min cycles in a window to yield a variance/coupling point (paper used 60)
CALM_BAND = (0.25, 1.0)     # ttot in [0.25, 1.0] s  <=>  1-4 Hz (arousal/rate-matched view)


def _windows(a, b, w=P.WIN_CYCLES, min_win=MIN_WIN):
    """Non-overlapping windows of size w over [a, b); keep a trailing window if >= min_win."""
    out = []
    i = a
    while i < b:
        j = min(i + w, b)
        if j - i >= min_win or (not out and j - i >= max(8, min_win // 2)):
            out.append((i, j))
        i = j
    return out


def _var_ms2(x):
    x = np.asarray(x, float) * 1000.0
    return float(np.var(x, ddof=1)) if x.size >= 2 else float("nan")


def _cv(x):
    x = np.asarray(x, float)
    m = np.mean(x)
    return float(np.std(x, ddof=1) / m) if x.size >= 2 and m > 0 else float("nan")


def _wmed(recs, key):
    vals = [r[key] for r in recs]
    wts = [r["n"] for r in recs]
    return core._wmedian(vals, wts)


def _wz(recs, key):
    """Fisher-z weighted mean of per-window correlations -> back-transformed r."""
    z, w = [], []
    for r in recs:
        v = r[key]
        if v is not None and np.isfinite(v) and abs(v) < 0.9999:
            z.append(np.arctanh(v)); w.append(r["n"])
    if not z:
        return float("nan")
    return float(np.tanh(np.average(z, weights=w)))
