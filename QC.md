# Quality control: why we changed how a recording passes or fails

**Short version:** the automatic pass/fail should be decided *after* we reprocess the recording from the
stored voxels — **not** from the numbers the live monitor prints during capture. The live numbers are
approximate and have drifted as the capture code changed, and two specific quirks make them fail perfectly
good recordings — including the *sickest, most valuable* mice. The pipeline now does the QC check itself,
on the reprocessed signal, and tells you plainly whether a recording is analyzable.

---

## What was going wrong

The live capture monitor writes a `*_summary.json` while it records, with quick metrics like
`rr_med_bpm` (breathing rate) and `coverage_pct`. Deciding pass/fail from those two numbers is unreliable
for two reasons:

### 1. The live breathing rate can be *doubled* ("octave error")
A mouse breathing at ~180 breaths/min produces a strong "second harmonic" at ~360. The live detector
sometimes locks onto that harmonic and reports **~2× the true rate**. So a healthy recording shows an
"impossible" rate and gets failed — even though the real breathing is completely normal. This is what
happened with **a14_6wks**: the *saved* live rate was wrong, but the stored voxels are fine, and when we
reprocess them the rate is normal.

### 2. Live "coverage" is not data quality — it's how *calm* the animal was
`coverage_pct` counts only the time the animal was breathing **regularly**. A stressed, active, or **sick**
mouse breathes the whole time but *irregularly*, so its "coverage" looks tiny. Failing on low coverage
therefore throws away exactly the recordings we care about most. Real example from our data — the sickest
mutant (Animal B9):

| | Live monitor said | Reprocessed from voxels |
|---|---|---|
| breathing "coverage" / present | **7.6 %** → looks like bad data | **96 % of the recording** — the animal is breathing almost the whole time |

The animal was fine and on-camera the entire session. It just breathes irregularly, which *is the disease*.
7.6 % was measuring calmness, not data quality.

**The common thread:** the live monitor's job is a rough real-time readout, not a verdict. Anything it flags
should be re-checked against a proper reprocessing before a recording is thrown out.

---

## What the pipeline does now

After it segments the recording **from the stored voxels** with the current algorithm, it runs one more
step — **`qc`** — that judges the recording on the *reprocessed* signal and prints a verdict:

```
========================================================================
  QC VERDICT:  [ PASS ]    Animal_B_9_8wk
========================================================================
  Analyzable. Recomputed from the stored voxels, this recording looks
  physiologically reasonable.

    Recovered breathing   : 84.5 min  (present 96% of 88 min)
    Breathing rate        : 129 bpm
    Cycles for metrics    : 7729 total, 1411 calm
    Sampling rate         : 250 Hz

  Live capture summary (what the OLD pass/fail used):
    rate 146 bpm   (recomputed: 129 bpm)
    "coverage" 7.6%   (recomputed breathing present: 96%)
    -> live "coverage" is the CALM fraction, not data quality; a low value means an
       active/sick animal, not a bad recording. Trust the recomputed numbers above.
========================================================================
```

It deliberately **only fails a recording when the reprocessed signal itself is missing or impossible** —
never because the animal was active or sick.

### The three verdicts

- **`[ PASS ]`** — analyzable. Use it. (Most recordings, including irregular/sick animals, land here.)
- **`[ REVIEW ]`** — usable, but take a quick look before trusting the numbers. Reasons are printed, e.g.
  *very little breathing was recovered* (a short capture), *nonstandard sampling rate*, *breathing rate
  outside the usual range*, or *breathing halved*. Review ≠ fail.
  - **Breathing halved** (new in 1.4): a mouse breathing faster than ~250/min is counted at half its rate by the
    default segmentation, so every breathing number and the heart rate are wrong. `run.py all` / `segment` now
    detect this and re-segment automatically. This flag appears only if the recording was segmented some other way
    (or with `--breath-bounds default`). Fix it with `python run.py segment --exp <dir> --breath-bounds 0.12,0.21`,
    then re-run the metrics.
  - **Breathing clipped** (new in 1.6): a mouse just above that speed (ng511 on 10-21: 267/min) is not halved, but
    the default segmentation squeezes its breaths against the shortest breath it allows (counted at 250/min; 30 % of
    breaths at the limit vs 1–3 % in normal mice). Breath timing, cycle variability and heart rate are distorted.
    `run.py all` / `segment` re-segment this automatically too; the flag and fix are the same as for halved breathing.
  - For such fast breathers, the "Breathing rate" line shows the segmenter's rate. The older per-episode estimate,
    printed next to it, also halves them.
- **`[ FAIL ]`** — the recording does **not** look analyzable: essentially no breathing was recovered from
  the voxels. That means a capture problem — the animal was out of the beam, the recording was cut short,
  or the wrong voxel file was loaded. This is a data problem, not a science finding.

### What will **not** fail a recording (on purpose)
These are physiology, and they are what the study is about — the QC calls them out but keeps the recording:

- irregular breathing / low "calm" fraction (an active or sick mouse),
- apneas and long pauses,
- low movement / a hypoactive animal,
- the arrhythmia metric (`ttot_cv`) coming out blank because there wasn't much *calm* breathing — that is
  expected for short or very active recordings; the other metrics are still valid.

---

## How to run it

You don't have to do anything extra — **`python run.py all --exp <recording folder>`** now runs the QC
step automatically and prints the verdict banner at the very end.

To re-check a recording you've already processed, without reprocessing:

```bash
python run.py qc --rec <recording_name>
```

Every run also writes:

- `outputs/metrics/<rec>.qc.json` — the full report (all checks + the signal values),
- `outputs/metrics/<rec>.qc.csv` — a one-line summary (verdict + reasons + key numbers),
- and `python run.py combine` rolls every recording's verdict into `outputs/metrics/combined_qc.csv`, so you
  can scan a whole batch at a glance.

---

## If you're unsure about a recording

Send me the printed QC banner (or the `<rec>.qc.json`). A **PASS** with a low live "coverage" is expected
and fine — that's the situation the old check got wrong. A **REVIEW** is worth a shared look. A **FAIL**
usually means something about the capture (animal position, run length, file selection) rather than the
analysis, and it's worth checking the rig before re-recording.
