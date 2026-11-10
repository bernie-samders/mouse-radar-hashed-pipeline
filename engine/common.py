"""output/eqradio_resp/common.py -- shared I/O + substrate helpers for the timing-based (EQ-Radio)
respiration detector prototype.  G:-free: everything here reads the in-repo B9 prep bundle + labels.

The prep bundle (ml_apnea/apnea_label/prepare.py) stores, per display SEGMENT:
  trace : base64 <f4, WIDE-band [1,12] Hz breathing MORPHOLOGY, per-segment amplitude-normalised, 50 Hz
  env   : base64 <f4, [1,6] Hz in-band ENVELOPE (same norm, 50 Hz)
  breaths : absolute-recording-second breath-peak times (detected on the [1,6] Hz fundamental)
  t0,t1 : clean CORE span (real s);  z0,z1 : extended span the trace/env actually cover (real s)
  disp_dt : display sample interval (~0.02 s = 50 Hz)
The label file stores spans {type, t0, t1, seg_id} in real recording seconds; type 'clean' = a
hand-verified breathing region (may contain pauses/apneas -- still 'breathing present').
"""
from __future__ import annotations
import os, json, base64
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREP_B9 = os.path.join(REPO, "ml_apnea", "outputs", "apnea_label", "prep", "Animal_B_9_8wk.json")
LAB_B9 = os.path.join(REPO, "ml_apnea", "outputs", "apnea_label", "labels", "Animal_B_9_8wk.json")
FIGS = os.path.join(os.path.dirname(__file__), "figs")


def b64f32(s: str) -> np.ndarray:
    """Decode a prep-bundle base64 little-endian float32 array to float64."""
    return np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float64)


def load_prep(path: str = PREP_B9) -> dict:
    return json.load(open(path))


def load_labels(path: str = LAB_B9) -> list[dict]:
    return json.load(open(path))["labels"]


def segments_by_id(prep: dict) -> dict[int, dict]:
    return {int(s["seg_id"]): s for s in prep["segments"]}


def clean_spans(labels: list[dict]) -> list[dict]:
    return [l for l in labels if l["type"] == "clean"]


def decode_segment(seg: dict):
    """Return (trace, env, fs, z0, breaths_abs) for a prep segment, all at the 50 Hz display rate."""
    tr = b64f32(seg["trace"])
    env = b64f32(seg["env"]) if "env" in seg else np.abs(tr)
    dt = float(seg["disp_dt"])
    fs = 1.0 / dt
    z0 = float(seg.get("z0", seg["t0"]))
    breaths = np.asarray(seg.get("breaths", []), float)
    # env may be a different length than trace by +-1 sample after decimation -- clip to common length
    n = min(tr.size, env.size)
    return tr[:n], env[:n], fs, z0, breaths


def span_slice(seg: dict, t0: float, t1: float):
    """Sample index slice into a decoded segment array for the real-time window [t0, t1]."""
    dt = float(seg["disp_dt"])
    z0 = float(seg.get("z0", seg["t0"]))
    a = int(round((t0 - z0) / dt))
    b = int(round((t1 - z0) / dt))
    return max(0, a), b


def iter_clean_traces(prep: dict = None, labels: list[dict] = None):
    """Yield (label, seg, tr, env, fs, t_abs) for every labelled clean span, where tr/env are sliced to
    the label's [t0,t1] and t_abs is the matching absolute-time axis."""
    prep = prep or load_prep()
    labels = labels or load_labels()
    by_id = segments_by_id(prep)
    for l in clean_spans(labels):
        seg = by_id.get(int(l.get("seg_id", -1)))
        if seg is None:
            continue
        tr, env, fs, z0, breaths = decode_segment(seg)
        a, b = span_slice(seg, l["t0"], l["t1"])
        b = min(b, tr.size)
        if b - a < 4:
            continue
        t_abs = z0 + np.arange(a, b) / fs
        yield l, seg, tr[a:b], env[a:b], fs, t_abs
