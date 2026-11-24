"""bh.py -- data shim for the frozen heart-rate-curve tracker (beams.py / beamx.py do `import bh` by their research name).

The other files in this folder are byte-identical to the validated research files (sha256 checked by
pipelines/hr_curve.py). In the research tree `bh` was the beat-to-beat harness (per-period cache + ECG scoring); here a
period dict carries its own data and there is no ECG:
  p["Y"]     [T, 330] float32 breath-order residual of the chest-patch acceleration, 250 Hz (hr_final_pipeline.residual)
  p["picks"] the period's 20 agreement picks in the chosen heartbeat band (indices into the 330 patch voxels)
Must be imported before beams.py, so `import bh` there resolves to this module.
"""
import numpy as np

FS = 250.0


def Y(p):
    return p["Y"]


def picks(p):
    return np.asarray(p["picks"])
