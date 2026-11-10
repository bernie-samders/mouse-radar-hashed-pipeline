"""pipelines -- the registry of analyses that run on a segmented (or raw) mouse experiment.

Each pipeline is a module exposing:
    NAME : str            short CLI name (e.g. "stettner")
    KIND : "segments" | "raw"
        "segments" -> run(rec, sub_dir=..., out_dir=...) : consumes the segmentation substrate
        "raw"      -> run(exp_dir, out_dir=..., max_min=...) : consumes the raw experiment directory
    run(...) -> dict      a flat {metric: value} dict (also written to <out_dir>/<rec>.<name>.csv)

TO ADD YOUR OWN PIPELINE:
    1. Drop a new module in this folder with NAME, KIND, and a run(...) that returns a flat dict.
    2. Add it to the PIPELINES list below.
That's it -- `python run.py all --exp <dir>` and `python run.py <name> ...` pick it up automatically.
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.dirname(_HERE)
for _p in (_os.path.join(_PKG, "engine"), _HERE, _PKG):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import stettner            # noqa: E402
import pause_distribution  # noqa: E402
import azimuth_motion      # noqa: E402
import apnea               # noqa: E402
import hr                  # noqa: E402
import b2b                 # noqa: E402
import qc                  # noqa: E402

# The ordered list of METRIC pipelines that `run.py all` executes.  Register new ones here.
PIPELINES = [stettner, pause_distribution, azimuth_motion, apnea, hr, b2b]   # b2b needs hr's outputs

# The quality-control step.  `run.py all` runs it LAST -- after every metric pipeline -- so it can fold
# their outputs into its verdict and print the headline PASS/REVIEW/FAIL banner as the final thing on
# screen.  It is also runnable on its own (`python run.py qc --rec <name>`) to re-check a recording.
QC = qc

REGISTRY = {m.NAME: m for m in PIPELINES + [QC]}
