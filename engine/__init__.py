"""engine -- the reusable radar signal-processing core (raw IO, conditioning, two-template DP,
segmentation substrate).  Importing `engine` puts the engine directory on sys.path so its modules
can be imported flatly (``import segmentation``, ``import substrate``) exactly as they import each other.

Public API (the two things most pipelines need):
    from engine import load_experiment, segment_experiment
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from voxel_store import load_experiment, load_voxel_store, rec_name        # noqa: E402
from segmentation import segment_experiment, build_session                 # noqa: E402

__all__ = ["load_experiment", "load_voxel_store", "rec_name",
           "segment_experiment", "build_session"]
