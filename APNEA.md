# Apnea metric: counting breathing stops from the radar

**Short version:** `apnea` counts how often a mouse stops breathing, from the stored radar voxels. Each recording
gets a count and a rate per minute of clean breathing. We checked the rule against whole-body plethysmography
(WBP) airflow on a sick mutant (B16) recorded with both at once. The radar found every apnea it could see
(**12 of 12**), and every apnea it reported was real (**16 of 16**). On a second mutant (B17) it reported no false
apneas. Both checks cover the time after the first 20 minutes; read "How well it works" for what happens before
that. It runs automatically as part of `python run.py all`.

---

## How to run it

It is part of the normal run, so nothing extra is needed:

```bash
python run.py all --exp "<experiment_dir>" --cohort B --age-weeks 8     # segment + every metric, incl. apnea
python run.py combine                                                   # after all experiments: master tables
```

To (re)run only the apnea step on a recording that is already segmented:

```bash
python run.py apnea --rec <recording_name>                     # finds the raw voxels via the segmentation file
python run.py apnea --rec <recording_name> --exp "<experiment_dir>"   # if the raw folder has moved
```

**It needs the raw voxels** (`*_voxels.f32`), not just the segmentation, because it re-reads other voxels around
each candidate apnea. It adds about 2 minutes per hour of recording on top of segmentation (more if the data sit on
a slow external drive).

## What you get

| file | what |
|---|---|
| `outputs/metrics/<rec>.apnea.csv` / `.json` | one row: apnea counts and rates (columns below) |
| `outputs/metrics/<rec>.apnea_events.csv` | one row per apnea: start/end time, duration, which checks it passed |
| `outputs/metrics/<rec>.apnea.png` | timeline of apneas over the recording, plus up to 6 example apneas to eyeball |
| `outputs/metrics/<rec>.apnea_candidates.csv` | every pause that was considered, with its votes, for audits |
| `outputs/segments_anchored/<rec>.{json,npz}` | cache of the multi-voxel signal (safe to delete to force a rebuild; rebuilt automatically when the segmentation changes) |
| `outputs/metrics/combined_apnea.csv`, `combined_apnea_events.csv` | all experiments, after `run.py combine` |

The columns that matter:

| column | meaning |
|---|---|
| `apnea_3x_per_clean_min` | **main result.** Apneas longer than 3 × the animal's median breath cycle, per minute of clean breathing |
| `apnea_gt1s_per_clean_min` | same, with the fixed "> 1 s" rule the WBP ApneaCheck tool uses |
| `apnea_3x_n`, `apnea_gt1s_n` | the raw counts |
| `apnea_*_n_adj` | the counts after the old-capture correction (= `apnea_*_n` for current captures). **Sum these to pool** |
| `apnea_*_per_clean_min_after20` | same rates, ignoring the first 20 min (ApneaCheck's acclimation window) |
| `clean_breathing_min` | minutes of clean, calm breathing the radar resolved: the denominator. Not the QC banner's "recovered breathing", which counts all breathing |
| `rate_reliable`, `rate_reliable_after20` | `False` if the matching denominator is < 3 min. The rate then rests on a few minutes; report it, don't pool it |
| `apnea_3x_threshold_s` | the duration the 3× rule used for this animal (3 × its median cycle) |
| `clean_rate_bpm` | breathing rate during clean breathing |
| `capture_file_step`, `count_scale` | old-capture artifact handling (see "Read before using the numbers", point 6) |
| `notes` | plain-English warnings for this recording, if any |

## How it decides what is an apnea

Every step uses the same two-template breath/pause segmentation as the rest of the package.

1. **Find long pauses.** A candidate is a pause where breathing stops, with at least 3 clear breaths within 2 s
   on *both* sides. So breathing stopped, then came back. It must also be **flat**: the signal during the pause
   is at most 0.30 × the breathing around it. Candidates come from two signals: the best breathing voxel, and a
   combination of the 12 strongest breathing voxels next to it. The combination rides through small body shifts.
2. **Ask the other voxels.** The radar sees the mouse in many voxels. When a mouse really stops breathing,
   *every* voxel that was breathing goes quiet. When it only shifts posture, the breathing moves to other voxels
   and keeps going there. So for each candidate the same segmentation is re-run on the 10 best breathing voxels
   around it:
   - **Keep** the pause if at least **2 other voxels also show it**, and fewer than 2 keep breathing through it.
   - Occasionally no voxel is clean enough to judge (e.g. right after sniffing). Then the pause from the
     combined signal is kept only if no voxel breathes through it and the best voxel shows it too.
3. **Duration.** Two definitions are reported side by side:
   - **> 3 × the median breath cycle** is the validated default. It is ~1 s for a 180 bpm mouse and ~1.7 s for a
     105 bpm one.
   - **> 1 s** is the rule the collaborators' ApneaCheck tool applies to WBP.
4. **Rate.** The count is divided by the minutes of **clean breathing**: stretches of at least 8 consecutive
   well-fit breaths.

## How well it works

**Against WBP airflow**, from the paired radar + WBP sessions of 09-21, after the 20 min acclimation:

| | radar apneas that WBP confirms | WBP apneas the radar found (of those it could see) | false apneas on B17 |
|---|---|---|---|
| > 3 × breath cycle | **16 / 16** | **12 / 12** | **0** |
| > 1 s | 15 / 15 | 13 / 17 | 0 |

The four "> 1 s" misses: two were 0.93 / 0.99 s on the radar (1.01 / 1.11 s on WBP), just under the cut-off.
The other two happened while the radar couldn't resolve the breathing well enough to measure the pause. B17 is a
fast breather (~250 bpm when calm); its 2 visible WBP apneas under the 3× rule were short (0.69 / 0.84 s), and the
radar found neither.

**The first 20 minutes are not validated.** These numbers are after the 20 min acclimation, as ApneaCheck scores.
B16 had no radar apneas in its first 20 min. B17 had 4, and only 2 of them line up with a WBP airflow stop (the
other two overlap one for 29 % and 36 % of their length).

**Across our free-behaving cohort** (the research version of this exact rule; 10 wild-type vs 11 mutant animals):
mutants have more apneas. Median 0.112 vs 0.034 per clean-breathing minute with the 3× rule (Cliff's δ +0.60,
p = 0.02), 0.187 vs 0.034 with > 1 s (δ +0.73, p = 0.005). 4 of 10 wild-types had none. It also points the right
way in the same-day, same-rig 8-week batch (3 vs 3 animals, δ +0.67), where earlier apnea counts pointed the wrong
way. **But half of those apneas (53 %) fall in the first 20 minutes**, which hold only 15 % of the clean-breathing
time. Using only the time after 20 minutes, the difference shrinks and is no longer significant: δ +0.29
(p = 0.24) with 3×, δ +0.42 (p = 0.09) with > 1 s, from 5 wild-type vs 24 mutant events.

## Read before using the numbers

1. **The radar only sees apneas while the mouse is calm.** A moving or sniffing mouse gives no clean breathing,
   so apneas during activity are invisible. That is why the rate is **per minute of clean breathing**, never per
   hour. The count is a lower bound on the true number of apneas.
2. **Pool per animal, don't average rates.** A recording has only a handful of apneas, so a single recording's
   rate is noisy. The same mutant (B9, 8 wk) recorded twice on one day gave 0.22 vs 0.44 (3×) and 0.40 vs 1.10
   (> 1 s) apneas per clean-breathing minute. When an animal has several recordings, add
   up its `apnea_*_n_adj` and its `clean_breathing_min`, then divide. Treat `rate_reliable = False` recordings
   (< 3 clean min) with care.
3. **The rule was designed on two mutants in a WBP chamber.** B16 and B17 were used to set the rule, so 12/12 and
   16/16 are best-case numbers. Free-behaving mice and wild-types have **not** been checked against WBP yet. A
   paired WBP session on a wild-type is the missing test.
4. **Pick one definition before looking at the data.** 3× is the validated default. Mutants breathe more slowly,
   so the 3× threshold is longer for them, which makes it conservative for mutants. > 1 s matches ApneaCheck and
   separates the genotypes a little more strongly.
5. **Compare like with like.** Wild-type and mutant animals are often recorded on different days, rigs or
   ages. A genotype difference should also hold within a same-day, same-rig batch.
6. **Old captures (before 08-06) have a recording artifact.** The old live-capture software jumps in level
   every 4.264 s (one capture file). A pause that starts or ends on a jump can look like an apnea. The metric
   finds these recordings automatically (`capture_file_step = True`) and discards apneas touching a jump. It then
   scales the count up to make up for real apneas lost that way (`count_scale` = ×1.23). Recordings from the
   current monitor are unaffected. Prefer them where you can.
7. **The first 20 minutes matter a lot. Report both.** The main columns use the whole recording. The
   `_after20` columns skip the first 20 min, as ApneaCheck does, and that is the only period checked against WBP.
   Apneas are ~6× denser in the first 20 min than later. That may be real (a newly placed, aroused mouse; pauses
   right after movement or sighs), or it may be extra false detections while the animal is active: 2 of B17's
   4 early apneas were not WBP airflow stops. Either way, most of the genotype difference comes from that window
   (see "How well it works"). Longer recordings give the after-20-min number more data.
8. **Durations.** In the B16 check, radar pause lengths were within about 0.03 s of WBP on average. A pause right at the cut-off (e.g. 0.98 s
   under "> 1 s") can land on either side. Pauses longer than 2.5 s are stitched together from pieces, so
   their exact length is less precise.
9. **This is not the Stettner `apnea_rate_per_min`.** That older number counts every pause ≥ 2 × the local cycle
   across all breathing, with no flatness or voxel check. Use `apnea` for apneas.
10. **250 Hz captures only.** That is the validated setup. Other rates get a note.
11. **Fast breathers.** The one fast breather in the validation (B17, ~250 bpm calm) had 0 false apneas after
    20 min, but its 2 visible WBP apneas were short (< 0.85 s) and the radar missed both. Glance at
    `<rec>.apnea.png` for new animals.

**Eyeball the figure.** In `<rec>.apnea.png` each example stacks the voxels around one apnea. In a real apnea
the blue voxels all go flat together and breathing resumes afterwards. A red voxel means it kept breathing
through the pause.

## For Claude Code / AI agents

- Code: `pipelines/apnea.py` (orchestration, outputs, figure), `engine/dp_apnea.py` (candidates, clean mask,
  capture-file grid), `engine/multivoxel.py` (anchored combine), `engine/witness.py` (voxel votes + final rule).
- Ported from the research repo: `apneacheck/radar_apnea.py`, `apneacheck/multivoxel_combine.py --anchor 2 4`,
  `output/apnea_cohort/run_apnea_cohort.py` (`detect`, `boundaries`), `output/apnea_cohort/witness_check.py` +
  `witness_v2.py` (FINAL rule). The port was checked on B16 (09-21) against the research tables: identical
  substrates, candidates (665), witness votes (72) and final events (16 / 15).
- Thresholds (flank 3 breaths / 2 s, flat 0.30, witness ≥ 3 breaths at 0.5–2× cycle, confirm ≥ 60 % of the
  pause, 2 votes) are fixed in `engine/witness.py` / `engine/dp_apnea.py`. Changing them changes the validated
  rule. `<rec>.apnea_candidates.csv` keeps every candidate's votes, so an alternative definition is a filter on
  that table, with no re-run needed.
