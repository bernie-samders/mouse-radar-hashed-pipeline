# July-2026 captures: removing the capture-build jumps before heart rate

**Short version:** the live-capture build used in July 2026 (our 07-02, 07-16 and 07-20 recordings) put two artifacts
into the voxel cube.
- **A level step in every voxel** at each 4.264 s capture-file boundary (1066 frames).
- **In some recordings, single dropped frames,** where every voxel dips for one frame.

Breathing and apnea work is barely affected; the apnea step already rejects events on the file boundaries. The
heart-rate steps are a different story. They work on the 2nd derivative of the voxel signal, which turns every jump
into a spike that buries the heart. On raw July cubes the heart signal disappears, and `hr`'s per-period heart
rates are off by about 60 beats/min.

Version 1.5 removes the jumps from a copy of the cube before the heart-rate steps. **It is automatic:** `run.py all`,
`hr` and `b2b` check every cube and repair it only when it has the July artifacts. On our July cohort, 14 of the
20 recordings with still periods then passed the heart-SNR check (of the 1.5–1.6 `hr_curve` step, removed in 1.7);
before the repair, none of the 7 we tested did.
Captures from 08-06 on have no step and run on the raw cube, unchanged.

---

## How to run it

Nothing extra is needed: the check runs inside the normal run.

```bash
python run.py all --exp "<experiment_dir>" --cohort B --age-weeks 8       # checks, repairs if needed, runs everything
python run.py hr --rec <recording_name>                                   # same check when (re)running hr alone ...
python run.py b2b --rec <recording_name>                                  # ... and b2b (reuses hr's repaired copy)
```

**What happens.**
1. The cube is read once to look for jumps. That is about 1 minute per 10 GB, and the result is cached in
   `<rec>.dejump.json`.
2. The cube is repaired when it has the file-step grid (steps on at least half the file boundaries) or ≥ 1 dropped
   frame per minute.
3. The heart-rate steps then read a repaired copy in `outputs/dejumped/<experiment folder>/`.
   - Only range bins 50–82 are copied: the chest-patch range, about 3.5 GB per recording hour.
   - `run.py all` deletes the copy at the end.
   - When you run `hr` and `b2b` on their own, the copy stays so the second step can reuse it. Delete
     `outputs/dejumped/` when you are done.
4. Segmentation, apnea, motion and QC always use the raw cube.

**Options.**

```bash
python run.py all --exp "<dir>" --dejump off           # never repair (heart rate on the raw cube, as before 1.5)
python run.py all --exp "<dir>" --dejump on            # repair whenever any jump is found
python run.py all --exp "<dir>" --keep-dejumped        # keep the repaired copy after `all`
python run.py dejump --exp "<dir>"                     # just write the repaired copy (and report the jumps)
python run.py dejump --exp "<dir>" --detect-only       # just report: does this cube have the July artifacts?
```

`hr` and `b2b` must read the same cube. If one ran with `--dejump off` and the other did not, `b2b`'s notes say so:
re-run both the same way.

## What you get

| file / column | what |
|---|---|
| `outputs/metrics/<rec>.dejump.json` / `.csv` | the check: `file_step_grid` (True for July captures), `grid_boundaries_hit` (share of file boundaries with a step), `n_step`, `n_glitch` (dropped frames), per minute, `repaired` |
| `outputs/dejumped/<experiment folder>/` | the repaired copy: `*_voxels.f32` / `.json` (range bins 50–82; the json's `dejump` block says what was done) and `dejump_events.csv` (every jump: frame, time, size, step or dropout, on the file grid or not) |
| `cube_dejumped` in `<rec>.hr.csv` and `<rec>.b2b.csv` | True when that step read the repaired copy (the notes say how many steps / dropouts were repaired) |

## How it works

The repair runs on the voxel amplitude (the square root of the stored power), the same values the heart-rate code
reads.
1. **Find the jumps.** For every pair of consecutive frames, average the absolute change over all voxels. A frame
   change more than 20 times its running median (10 s window) is a jump. Changes ≤ 3 frames apart count as one event.
   - A real frame-to-frame change is small.
   - A July file step is 30–500 times larger, and a dropped frame 20–400 times.
   - On a step-free recording, still periods never exceed about 6 times.
2. **Repair them.** For each voxel, every jump's frame change is replaced by the average of the two neighbouring
   changes, and the signal is rebuilt from the corrected changes. This removes a step's offset and turns a dropped
   frame into a straight line.
3. **Write power again,** so `hr` and `b2b` read the copy like any cube.
   - Rebuilding shifts each voxel's overall level, so a constant is added per voxel to keep it positive.
   - The heart-rate steps only use the 2nd derivative and the band-passed signal, so that constant changes nothing for
     them.
   - Frames the repair does not touch are copied bit for bit.

The detection uses all voxels, not the chest patch, because it is far more robust to movement: on a 64-min step-free
recording, the chest-patch version fires about 70 times while the mouse moves, the all-voxel version once.

## How well it works

Measured in 1.5 with `hr` and the heart-SNR check of the `hr_curve` step (removed in 1.7).

| test | result |
|---|---|
| step-free recording (B16, 09-21) repaired anyway | identical heart rate (within 0.00001 beats/min), band, heart SNR and verdict in all 21 periods |
| B16 with July-sized steps added to every file, no repair | heart signal lost in every minute (verdict `LOW`), heart rate off by a median 75 beats/min |
| … the same, repaired | back to B16's own numbers: heart rate within 1.8 beats/min, every band and SNR grade unchanged |
| July recordings (Match5Mouse2, Match4Mouse2_Redo) | reproduces the research analysis period by period (same band in 26/26 and 61/61 periods) |
| this package vs the research tool | byte-identical repaired cube; identical `hr` / `hr_curve` results |

**On our July cohort** (22 recordings, all with the step):
- 14 of the 20 with still periods pass the heart-SNR check after the repair.
- The other 6 stayed `LOW`.
- Per-period heart rates moved by a median of 58 beats/min from the unrepaired values, in both directions.
- Four recordings also had dropped frames: 24–110 per minute.

**What this changed for wild-type vs mutant** (10 vs 11 free-behaving recordings, `ok` / `strong` periods; this
package does no group statistics):
- **Heart rate:** mutants are lower, 446 vs 473 beats/min, but not significantly (p = 0.25). The gap disappears at
  matched breathing rate.
- **Breathing rate:** mutants breathe more slowly, 171 vs 197 per min (p = 0.053).
- **Heartbeats per breath:** mutants have more, 2.88 vs 2.41 (p = 0.085). That is the breathing-rate phenotype.
- **Heart-rate spread across periods:** no difference. The earlier "mutants vary more" lead came from the artifact.
- **HR × breathing rate:** mutants 16 % lower (p = 0.058); in same-batch comparisons it was the most consistent
  separator. It was looked for after the fact, so treat it as a candidate for the next cohort.

## Read before using the numbers

1. **The repaired copy is for heart rate only.** Its absolute signal level per voxel is arbitrary. Don't run motion,
   presence or other level-based analyses on it. The package never does.
2. **Not every boundary change can be undone.** The July build also changes how each voxel's power follows motion at
   every file boundary. The repair cannot remove that; it cost about 1 dB of heart signal in our test. Six July
   recordings stayed `LOW` on that check; `b2b` abstains where the heart is not clear enough to time beats.
3. **No July mouse had an ECG.** "Heart signal present" means the SNR check passes, not that the heart rate is
   validated on those animals. The repair itself was validated by the tests above.
4. **A very weak step can slip through.** The threshold is 20×. The weakest July step we saw was 21×, and 1–13
   boundaries per recording (of ~400–1500) stayed below it and are not repaired.
5. **Zero-filled gaps are filled.** A capture hiccup that wrote a run of all-zero frames is bridged with a flat line.
   We saw one 0.3 s gap, in B16, outside any still period.
6. **Distrust ~800 beats/min, as before.** The 1.5–1.6 SNR check graded a constant line near the top band as `strong`
   on some recordings, e.g. all of A11 (08-26), which is not a July capture. `b2b` does not report periods seeded or
   read above 750 (B2B.md).

## Where it comes from

- `engine/dejump.py` is a port of the research tool `output/july_dejump/dejump_cube.py`: same constants, same
  arithmetic, checked byte-identical on a July and a step-free recording.
- The diagnosis is in `output/july_heart_snr/README.md`; the validation and the cohort rerun are in
  `output/july_dejump/README.md` (our lab copy).
- The frozen heart-rate files (`engine/hr_final_pipeline.py`, `engine/b2b_beats.py`) are untouched and still checked
  on every run.
