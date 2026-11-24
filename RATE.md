# Rate-level heart rate (`rate`)

**Short version:** `rate` (new in version 1.8) gives a heart rate every 2 seconds over as much of the still, regular
breathing as the radar can support. It takes the beat-to-beat value where `b2b` timed the beats, and fills the rest
from a **rate tracker** that follows heart rate without timing single beats. This is the paper's **rate level**; `b2b`
alone is its **beat level**. It runs automatically after `b2b` in `python run.py all`, or on its own:

```bash
python run.py rate --rec <recording_name> [--exp <raw_experiment_dir>] [--workers N]   # after b2b
```

Checked against an implanted ECG on eight sessions (four mice):
- **2 s accuracy:** 93.8 % of 2 s values within 5 % of the ECG, 1.4 % more than 10 % off, over 136 min.
- **More coverage than the beat level, larger errors.** `b2b` alone: 96.8 % within 5 % (B2B.md) on far fewer minutes.
  The rate tracker alone: 92.3 % within 5 %, 2.1 % more than 10 % off.
- **Worse at high heart rates.** Within 5 % in 97 % of values below 450 beats/min, 95 % at 450–525, 87 % at 525–600
  and 71 % above 600. Above 600 it is 54 % outside the two sessions the method was developed on.

Read "Read before using the numbers" below: the settings were chosen with the ECG of most of these sessions in view.

---

## How it works

Per recording (no ECG anywhere):

1. **`hr` again, on shorter periods.** The `hr` step (HEART_RATE.md) is run a second time on every still,
   regular-breathing period of **6 s or more**. The default `hr` run, which `b2b` uses, keeps its 20 s floor.
2. **The rate tracker.** `hr_curve` with `--band continuity` runs on those periods (HR_CURVE.md). It follows the heart
   rate inside each period with a particle filter over beat phase and heart rate, 25 values per second. The heartbeat
   band is chosen across all periods in time order, so that heart rate cannot jump far between neighbouring periods.
   Each period gets a heart-SNR grade: `none`, `low` (at or below −3 dB), `ok`, `strong`.
3. **Merge on a 2 s grid** (bin centres at 1, 3, 5 … s, the grid of `b2b_bins.csv`):
   - the `b2b` value, where `b2b` reported one;
   - else the mean of the tracker's curve within ±3 s, if the bin centre lies in a tracker period that is **trusted**:
     grade `ok` or `strong`, and a period mean of the curve of at most **750 beats/min**.

Steps 1 and 2 write the usual `hr` / `hr_curve` files to `outputs/metrics/rate_tracker/`, apart from the default `hr`
outputs. `engine/pll_tracker/` (the tracker) and `engine/hr_final_pipeline.py` are byte-identical to the validated
research files, and their SHA-256 is checked on every run (`frozen_code_ok`).

### The 750 beats/min guard

The tracker's SNR grade also passes a non-cardiac line at about 780–830 beats/min as a `strong` heart (rig vibration;
B2B.md has the same guard for the same reason). No ECG-confirmed heart rate in our data is above 612. A tracker period
whose curve averages above 750 is dropped (`reason = curve_above_max`).
- On 529 of 10 Nov, without the guard, the tracker read that line for the first 20 min while the ECG was at 667–763.
- In our home-cage cohort the guard removes a quarter of the wild-type tracker periods; three mice lose all of theirs.
- Values between 650 and 750 are kept but are above every heart rate we have confirmed. One recording read a flat
  733–757 there. Treat them as suspect.

## Outputs (`outputs/metrics/`)

| file | what |
|---|---|
| `<rec>.rate.csv` / `.json` | one row (columns below) |
| `<rec>.rate_bins.csv` | the 2 s values: `t_s` (bin centre, s from the recording start), `hr_bpm`, `source` (`beats` or `curve`), `period` (for `beats` the period number of `b2b_periods.csv`, for `curve` that of `rate_periods.csv`), `clean_breathing` (1 if at least half the bin lies in a clean-breathing run) |
| `<rec>.rate_periods.csv` | one row per still period of 6 s or more: heart-SNR grade, curve mean, `used` (yes / no), `reason` if not (`not_tracked`, `low_heart_snr`, `curve_above_max`, `breathing_halved`), and how many of its bins came from the curve and from beats |
| `rate_tracker/<rec>.hr*` | the tracker's own run: `hr` on the ≥ 6 s periods and `hr_curve` (HEART_RATE.md, HR_CURVE.md) |

The columns of `<rec>.rate.csv`:

| column | meaning |
|---|---|
| `hr_median_bpm`, `hr_p25_bpm`, `hr_p75_bpm` | median and quartiles of the 2 s values |
| `covered_min` | minutes with a value (number of 2 s values × 2 s) |
| `beats_min`, `curve_min` | of those, minutes from `b2b` and from the rate tracker |
| `clean_breathing_min` | minutes of clean breathing (2 s bins at least half inside a clean run) |
| `covered_clean_min`, `covered_clean_frac` | clean-breathing minutes with a value, and their share |
| `n_periods`, `n_used` | tracker periods (≥ 6 s), and how many are trusted |
| `n_not_tracked`, `n_low_snr`, `n_guard_hr` | periods dropped: no heart signal, at or below −3 dB, curve mean above 750 |
| `tracker_snr_verdict` | the tracker's recording-level SNR verdict (HR_CURVE.md): `OK`, `LOW`, `INVALID` |
| `b2b_merged` | False if the `b2b` step had not run: the values are then the tracker's only |
| `frozen_code_ok` | True if the frozen heart-rate files are the validated ones |
| `breathing_halved`, `cube_dejumped` | as in `hr` (README, DEJUMP.md) |
| `notes` | plain-English warnings |

To compare animals, report `hr_median_bpm` together with `covered_min` and `covered_clean_frac`, and the breathing
rate: coverage differs a lot between animals, and the rate level samples the calmest stretches.

## How well it works

**Against the implanted ECG** (2 s values; ECG truth per bin = 60 / median R-R within ±3 s; our lab copy:
`output/hr_ratecov/`, `output/paper_figs/ecg_validation/`):

| session | covered | within 5 % of the ECG |
|---|---|---|
| 508, 28 Sep | 18.0 min | 97.4 % |
| 507 day 2, 28 Sep | 13.4 min | 93.3 % |
| ng511, 15 Oct | 6.1 min | 78.6 % |
| ng511, 21 Oct | 8.6 min | 91.1 % |
| 507, 31 Oct (cage sub-volume) | 8.0 min | 92.9 % |
| 507, 2 Nov | 22.4 min | 93.6 % |
| 508, 2 Nov | 26.2 min | 94.3 % |
| 509, 16 Nov | 33.7 min | 95.6 % |
| **all eight** | **136.3 min** | **93.8 %** (1.4 % more than 10 % off) |

Two more sessions, 529 and 530 of 10 Nov, had a rig vibration line and an ECG that needed rescue, and `b2b` reported
almost nothing on them. The rate tracker alone covered 25.9 of 41.7 scorable minutes on 530 at 93 % within 5 %, and
10.8 of 53.9 on 529 at 94 %.

Where `b2b` and the rate tracker both gave a value for the same 2 s bin (11 home-cage recordings), the median ratio
was 1.00 (range 0.991–1.014).

**These numbers were produced with package 1.6's `hr` and `hr_curve` on version 1.7.0's segments and `b2b`.** Version
1.8 packages that same code as one step. On three home-cage recordings (one of them a July capture that runs through
the de-jump repair, one run from the raw voxels with `run.py all`), 1.8 gave the same tracker output as the package
1.6 run, bit for bit, and the same 2 s values as the analysis table behind the paper. On the recording run with
`run.py all`, every output that 1.7.0 also writes was identical to the 1.7.0 run.

## Read before using the numbers

1. **The settings were chosen with ECG in view.** The tracker and its −3 dB grade were tuned on the ECG of 508 and
   507 day 2 (28 Sep) and of ng511 (15 Oct). The configuration used here (6 s periods, continuity band, 750 guard,
   merge with the beats) was chosen on 11 Nov 2026, after looking at 529 and 530 of 10 Nov against their ECG. Of the
   eight sessions in the table only 509 of 16 Nov was recorded after that choice. No ECG session has been scored after
   this version was locked. The table is therefore closer to a best case than to a test.
2. **The SNR grade is necessary, not sufficient.** Heart-free noise with the spectral tilt of a real recording passes
   the grade in a share of periods, and the band choice alone then lands near the true rate in many of them. A trusted
   period is not proof of a heartbeat; agreement with ECG across the heart-rate range is. This is why the grade is
   combined with the 750 guard, and why high values deserve suspicion.
3. **Heart rate level only.** The tracker's curve is smoothed. Do not compute beat-to-beat HRV (RMSSD, pNN, SD1) or
   RSA from rate-level values. For beat intervals use `b2b`'s tracked periods (B2B.md).
4. **Above 600 beats/min it is not established.** See the short version: 71 % within 5 %, and 54 % outside the two
   development sessions.
5. **Resting, regular breathing only,** as for `hr` and `b2b`. A rate-level value exists only inside a clean-breathing
   period of 6 s or more. Animals differ in how much of their time that is, and it is the calmest part of the
   recording, so the median is lower than the animal's heart rate over all still time.
6. **Coverage is not a heart measurement.** How much of the clean breathing is covered depends on the heart signal's
   contrast, which differs between animals and recording builds. In our home-cage cohort it was 37 % of clean breathing
   in Mecp2-null mice and 44 % in wild-type (pooled), and a few mice decided that difference.
7. **Halved breathing** (`breathing_halved`, see the README): the tracker's grades become `invalid` and it contributes
   nothing; the `b2b` values that remain are not trustworthy either. Re-segment.
8. **July-2026 captures** go through the de-jump repair first, like `hr` and `b2b` (DEJUMP.md).
9. **It costs time.** The step runs `hr` a second time on more, shorter periods, and then the tracker. On a 44 min
   recording with 4 worker processes it took 5.5 min, next to 2.7 min for `hr` and 12 min for `b2b`.

## Where it comes from

- The rate tracker is `pll` (HR_CURVE.md): six research files frozen on 17 Oct 2026, shipped as the `hr_curve` step
  in versions 1.4–1.6, removed in 1.7 and restored unchanged in 1.8.
- The 6 s floor and the continuity band are the recipe first used on the fast breather ng511 (16 Oct). The 750 guard
  value is the one `b2b` has used since 1.7.
- The merge rule is the one behind the paper's rate-level figures (our lab copy: `output/mecp2_hr_rerun/analyze.py`,
  `output/paper_figs/ecg_validation/common.py`).
