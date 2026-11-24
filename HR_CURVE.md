# Heart-rate curve: how heart rate changes within each still period

> **Status in version 1.8.** This step was part of versions 1.4–1.6, was removed in 1.7 and is back in 1.8 with its code
> unchanged (byte for byte). It is now the **rate tracker** of the rate-level heart rate: `run.py all` no longer runs
> it by itself; the `rate` step runs it on still periods of 6 s or more with `--band continuity` and writes its files to
> `outputs/metrics/rate_tracker/`. **Read [`RATE.md`](RATE.md) first** for how its curve is used. This page describes the
> tracker itself and is otherwise as it was in 1.6; its numbers are from October 2026 (three ECG mice, periods of 20 s
> or more unless it says otherwise).

**Short version:** `hr_curve` (new in version 1.4) follows the mouse's heart rate *inside* every still,
regular-breathing period that the `hr` step scores. It gives 25 values per second; `hr` gives one value per period.
It is the best radar-only method we have found for this. We tested it on three mice that also carried an implanted ECG:
- **Mice 508 and 507 day 2 (strong heart signal):** within the periods it explains **79 % and 55 %** of
  the ECG's beat-to-beat heart-rate changes. Averaged over 9 beats it is within **10.9 and 19.3 beats/min**
  of the ECG, and correlates **0.94 / 0.78** with the ECG's ups and downs. The older 10-per-second track from
  `hr` explains 26 % / 12 %.
- **Mouse ng511 (weak heart signal):** it explains only **7 %** overall. That 7 % averages two kinds of period. In
  the 12 periods above the SNR minimum, the curve follows the ECG's 9-beat heart rate (correlation 0.61, level within
  ~9 beats/min). In the low-SNR periods it does not.

So it works where the radar sees the heart clearly. A built-in **heart-SNR check** tells you, per period and per
recording, whether the signal is strong enough to trust the curve (next section). Some settings were tuned on these
same three mice, so new animals may do worse (see "Read before using the numbers").

---

## The heart-SNR check (read this first)

Every period gets a heart signal-to-noise ratio, `heart_snr_db`. It measures how far the heartbeat stands above the
noise next to it in the combined chest beam. It is measured on the other half of the period, so the curve cannot
inflate it. From it:

| `snr_grade` | heart SNR | meaning |
|---|---|---|
| `none` | no heart signal | not tracked: no curve for this period |
| `low` | at or below −3 dB | **curve unreliable**: on our ECG mice it explained none of the beat-to-beat changes (19 periods, all three mice) |
| `ok` | −3 to +3 dB | usable (ECG mice: 50 % of the beat-to-beat changes explained, 34 periods) |
| `strong` | above +3 dB | best (ECG mice: 75 %, 61 periods) |
| `invalid` | any | **breathing was halved** (see below): not a heart rate, whatever the SNR says |

**Per recording,** `snr_verdict` is:
- `OK` when at least half of the still minutes are above −3 dB.
- `LOW` otherwise. A `LOW` recording prints a warning and writes it to `notes` (run through the `rate` step, the QC
  banner says so too). Trust heart rate only in its ok / strong periods: use those periods' curve and its mean
  (`curve_mean_bpm`), blue in the figure, and drop the rest. On ng511 (default `run.py all`, 14 periods with clean ECG):
  - the 5 ok/strong periods' `curve_mean_bpm` was within 9 beats/min of the ECG on average, all 5 within 5 %;
  - in the other periods both the curve and `hr`'s period value were about 64 beats/min off;
  - `hr`'s value for the 5 good periods was off by 20.

  Five periods are few, so read these numbers as direction, not accuracy.
- `INVALID` when the breath segmentation counted every other breath. The heart band then holds a leftover breathing
  harmonic, which looks like a *strong* heart signal. On ng511 with the default breath bounds, the curve read about
  110 beats/min too high (≈ 2 × the true breathing rate), with 90 % of still minutes graded ok or strong. Every grade
  becomes `invalid`, a warning is printed, and QC says *REVIEW*. `run.py all` / `segment` re-segment such recordings
  automatically (see "How to run it"), so you should only see this on a recording segmented another way.

**Which period heart rate.** With the default band on an `OK` recording, `hr_periods.csv` (the `hr` step) is the
validated per-period value. On a `LOW` recording, see above.
With `--band continuity`, use `curve_mean_bpm` in `hr_curve_periods.csv` instead. `hr_periods.csv` keeps `hr`'s own
band choice, which continuity overrides. On ng511 that choice was further off: an average error of 49 against 31
beats/min in the `ok` / `strong` periods.

How the thresholds were set, from the logged runs on the three ECG mice (114 periods with clean ECG;
`output/hr_curve_package/snr_gate.py` in our lab copy):

| heart SNR (`heart_snr_db`) | periods | share of beat-to-beat changes explained | 508 / 507 day 2 / ng511 |
|---|---|---|---|
| no signal (not tracked) | 8 | 0 | – / 0 / 0 |
| below −6 dB | 5 | −0.04 | – / −0.86 / 0.00 |
| −6 to −3 dB | 6 | 0.01 | 0.50 / −0.54 / 0.01 |
| −3 to 0 dB | 14 | 0.46 | 0.70 / 0.48 / 0.22 |
| 0 to +3 dB | 20 | 0.53 | 0.74 / 0.71 / 0.18 |
| +3 to +6 dB | 19 | 0.80 | 0.84 / 0.73 / 0.59 |
| above +6 dB | 42 | 0.71 | 0.78 / 0.63 / – |

What the check says on the three ECG mice (this package's own output):

| | 508 | 507 day 2 | ng511 |
|---|---|---|---|
| still minutes above −3 dB | 93 % (46 of 50 periods) | 92 % (38 of 43) | 44 % (23 of 58) |
| median heart SNR | +5.2 dB | +5.0 dB | −5.6 dB |
| verdict | OK | OK | **LOW** |

Two things to keep in mind:
- **The check is necessary, not sufficient.** Above −3 dB, ng511's curves do follow the ECG's trend: in 10 of 12
  periods the correlation with the ECG's 9-beat heart rate beats a time-shifted copy of the same curve. But they swing
  less than the ECG (about 70 % of its size) and miss the deepest dips, so the beat-to-beat share is lower (0.26
  against 0.5–0.8 on the other two mice). Below −3 dB the curve is flat or at the wrong level. Correlation 0.11, and
  1 of 8 periods beats the shifted copy. A recording that passes is not guaranteed to track like 508.
- **The thresholds come from these three mice.** −3 dB is where tracking stopped on all three. The 50 % recording
  rule separates the two good mice from ng511 with a wide margin (92–93 % vs 44 %), but it is set on three animals.

## How to run it

Since 1.8 the normal run (`run.py all`) reaches it through the `rate` step (RATE.md), which runs it with the settings
of the ng511 recipe below. To run it by itself on a recording whose `hr` step has run (the 1.4–1.6 behaviour: `hr`'s
periods of 20 s or more, `hr`'s band, files in `outputs/metrics/`):

```bash
python run.py hr_curve --rec <recording_name>
python run.py hr_curve --rec <recording_name> --exp "<experiment_dir>"   # if the raw folder has moved
python run.py hr_curve --rec <recording_name> --workers 4                # fewer parallel processes (less memory)
```

**Very fast breathers (more than ~250 breaths/min).** The default breath segmentation counts every other breath for
such mice, and every breathing number and heart rate built on it is wrong.
- **Automatic since 1.4.** `run.py all` and `run.py segment` check every segmentation. If the breathing is halved (or,
  since 1.6, clipped: squeezed against the shortest breath the default bounds allow, QC.md), they
  re-segment with the fast breath bounds (0.12–0.21 s), keep the halved one in `outputs/segments/default_bounds/`, and
  print a `[breath-check]` line. The result is stored in the segments json (`breath_check`).
- `--breath-bounds default` turns the re-segmentation off; `--breath-bounds LO,HI` sets the bounds by hand.
- If a halved segmentation still reaches `hr` / `hr_curve`, both warn and the heart SNR becomes `INVALID`.

On ng511 we also scored periods down to 6 s and chose the heartbeat band by continuity across periods. Since 1.8 the
`rate` step does exactly this for every recording (in `outputs/metrics/rate_tracker/`). By hand:

```bash
python run.py hr       --rec <recording_name> --min-period-s 6
python run.py hr_curve --rec <recording_name> --band continuity
```

With `--band continuity`, the band is chosen across all periods in time order, so that heart rate cannot jump far
between neighbouring periods. It helped ng511 (heart rate only 1.2–2.6 × the breathing rate, so a breathing harmonic sat
inside the heart band) but hurt 508, whose heart rate really jumps between periods. Run by itself, this step keeps
the band `hr` chose as its default. The `rate` step uses continuity for every recording (chosen on 11 Nov 2026, see
RATE.md). With continuity, take the period heart rate from `curve_mean_bpm` (see "Which period heart rate" above).

**What it needs:**
- **The `hr` step's outputs** (`<rec>.hr.json`, `<rec>.hr_templates.npz`). It reuses `hr`'s periods, chest patch,
  heartbeat band and 20 voxels.
- **The raw voxels** (`*_voxels.f32`): it re-reads the chest patch.
- **Time:** about 1–2 minutes per hour-long recording on our laptop with 12 worker processes (507 day 2: 62 s), on
  top of `hr`.

## What you get

| file | what |
|---|---|
| `outputs/metrics/<rec>.hr_curve.csv` / `.json` | one row: the heart-SNR verdict and the curve's summary (columns below) |
| `outputs/metrics/<rec>.hr_curve_periods.csv` | one row per period of `hr`: tracked or not, heart SNR and grade, band, curve mean / SD / min / max |
| `outputs/metrics/<rec>.hr_curve_track.csv` | the curve: `period`, `t_s` (seconds from the recording start), `hr_bpm`, `snr_grade`; 25 rows per second, tracked periods only, first and last second of each period dropped |
| `outputs/metrics/<rec>.hr_curve.npz` | the same curve at full precision (+ `ibi_s` = 60 / `hr_bpm`) and the per-period arrays; `print(numpy.load(f)["README"])` |
| `outputs/metrics/<rec>.hr_curve.png` | top: every period's curve over the session (orange = below the SNR minimum); bottom: the longest good period, with `hr`'s 10-per-second track for comparison |

Run through the `rate` step, the same files are in `outputs/metrics/rate_tracker/`.

The columns that matter in `<rec>.hr_curve.csv`:

| column | meaning |
|---|---|
| `snr_verdict` | **read first.** `OK`; `LOW` = use the curve only in its `ok` / `strong` periods; `INVALID` = breathing halved, not a heart rate (see above) |
| `breathing_halved` | `True` if the segmentation counted every other breath (then `snr_verdict` is `INVALID`) |
| `snr_ok_frac`, `n_snr_ok`, `snr_ok_min` | share of the still minutes / number of periods / minutes above −3 dB |
| `heart_snr_median_db` | median heart SNR over the still minutes |
| `n_periods`, `n_tracked`, `tracked_min` | periods from `hr`, and how many had any heart signal |
| `within_period_sd_median_bpm` | typical size of the heart-rate swings inside a period (median SD of the curve), from the periods above −3 dB |
| `curve_hr_median_bpm`, `curve_hr_p25_bpm`, `curve_hr_p75_bpm` | distribution of all curve values in the periods above −3 dB |
| `band_mode` | `bspec` (default: `hr`'s band) or `continuity` |
| `breath_check_ratio` | breathing rate seen in the spectrum / rate the segmentation counted; ≥ 1.3 means halved (see above) |
| `frozen_code_ok` | `True` if the tracker code is the exact validated version |
| `cube_dejumped` | `True` if it read the repaired copy of a July-2026 capture (new in 1.5, see DEJUMP.md); must match `hr`'s |
| `ok_n_periods`, `ok_min` | periods / minutes graded `ok` / `strong` with period heart rate ≤ 650 beats/min (new in 1.5) |
| `ok_hr_median_bpm`, `ok_breaths_per_min_median`, `ok_hr_over_rr_median`, `ok_hr_iqr_bpm` | the recording's heart rate, breathing rate, heartbeats per breath and heart-rate spread over those periods only: the numbers to compare animals with (empty below 3 periods / 2 min; spread needs 5) |
| `ok_hr_x_rr_median_k` | heart rate × breathing rate / 1000 over those periods; exploratory (DEJUMP.md) |
| `notes` | plain-English warnings |

## How it works

Within a still period the heart rate is not constant. On our ECG mice it swings by tens of beats/min (standard
deviation within periods 35–50 beats/min), much of it in a slow wave of about one cycle per 3 seconds. The period's average hides that; this step follows it.

1. **Start from `hr`'s work.** Same period, same 330 chest voxels with breathing removed, same heartbeat band, same 20
   best-agreeing voxels.
2. **One combined signal per heart harmonic.** The heartbeat is not a sine wave, so it shows up at 1×, 2×, … 5× the
   heart rate. For each harmonic, the chest voxels are weighted into one beam that maximises heart power over the
   neighbouring noise. The 1× band is wide (heart rate / 1.45 to × 1.45) so fast swings stay inside it, but its lower
   edge is kept at least 10 % above the breathing rate.
3. **Track the beat phase.** A particle filter (1,024 guesses at once) follows the beat phase and the heart rate
   through the period. Heart rate is allowed to drift smoothly, as measured on ECG. The higher harmonics only count where
   they are unambiguous. The filter runs forward and backward in time, twice with different random seeds, and the
   answers are averaged.
4. **Read out** the average heart rate of the particles, 25 times per second.

Where the 1× beam shows no heart signal at all, the filter has nothing to follow and returns the band centre. Those
periods are marked not tracked and get no curve.

## How well it works

**Against the implanted ECG**, on the still periods of at least 20 s with clean ECG (ng511: every period of ≥ 6 s with
clean ECG, using the ng511 recipe above). The ng511 numbers cover the 89 % of its ECG beats that fall in tracked periods:

| | mouse 508 | mouse 507, day 2 | mouse ng511 |
|---|---|---|---|
| periods (ECG beats) | 23 (5,522) | 18 (4,239) | 26 (5,267) |
| **beat-to-beat changes explained** (ve1) | **0.79** | **0.55** | **0.07** |
| the same, over 9 beats (ve9) | 0.92 | 0.61 | 0.11 |
| 9-beat average vs ECG: average error | 10.9 bpm | 19.3 bpm | 36.5 bpm |
| 9-beat average within 5 % of the ECG | 91 % | 84 % | 59 % |
| correlation with the ECG's within-period changes | 0.94 | 0.78 | 0.36 |
| for comparison: `hr`'s 10-per-second track (ve1 / correlation) | 0.26 / 0.49 | 0.12 / 0.35 | −0.01 / 0.21 |
| heart SNR verdict | OK | OK | LOW |

How to read **ve1**: for every ECG beat interval, how much of its deviation from the period's average the radar curve
reproduces, pooled over periods. 1 = perfect, 0 = no better than a flat line at the period average. Below 0 is worse
than a flat line. A flat line scores 0 by construction, so ve1 measures only within-period changes, not the period's
level (that is `hr`'s job, see HEART_RATE.md).

**This package reproduces the research tracker.**
- The tracker code (`engine/pll_tracker/`) is the research code, unchanged. Given the research's own inputs, it gives
  the same curve: bit for bit in 87 of the 151 periods of the three mice, and within 1e-10 beats/min (last-digit
  rounding) in the rest.
- Run end to end from the raw voxels, the package finds the same periods, bands and 20 voxels as the research on all
  three mice (50 / 43 / 58 periods). Its breath-removed chest signal matches the research's up to float32 rounding
  (1 part in 10⁸).
- The particle filter can amplify such last-digit differences the way a different random seed would. Per period, the
  package curve and the research curve agree to within 0.01 beats/min (RMS) in most periods; the worst 10 % differ
  by up to 2.4 beats/min RMS (507 day 2), against 20–45 beats/min of real swing. The ECG scores agree to the second
  decimal (ve1 0.793 / 0.550 / 0.074 package vs 0.793 / 0.551 / 0.074 research).
- Two package runs on the same data give identical output.

## Read before using the numbers

1. **Partly tuned on the same mice.** The tracker's settings were tuned on the ECG of 508 and 507 day 2 (how fast heart
   rate drifts, band width, harmonic rules). ng511's band recipe was designed after looking at its ECG. The SNR
   thresholds were set on all three. The tracker was named the best afterwards, by comparing it with four others on the
   same three mice. So these numbers are a best case. A new animal with an ECG is the real test.
2. **Trust it only where the heart SNR is good.** See the SNR section. On ng511 the curve tracked in its periods above
   −3 dB and not in the rest; its overall 7 % mixes the two.
3. **Not beat-to-beat HRV.** The curve is smoothed by the filter. Even on the best mouse it explains about three
   quarters of single-beat changes. Don't compute RMSSD, SD1/SD2, pNN or RSA from it. Use it for heart-rate changes
   over a few beats to seconds: the slow ~0.3 Hz wave, rises and falls within a period.
4. **Resting periods only, as for `hr`.** Everything in HEART_RATE.md applies: still, regular-breathing periods only;
   report breathing rate alongside; heart rate near the top of the range is suspect.
5. **Low-SNR periods can drift far off.** On 507 day 2, one `low` period's curve averaged 652 beats/min, while `hr`
   measured 525 for the same period. That is why `low` periods are flagged and left out of the summary columns.
6. **Some periods have no curve.** Periods without heart signal are not tracked; check `n_tracked` and `snr_ok_min`
   per recording before comparing animals.
7. **The curve's first and last second are dropped**, as for `hr`'s track (filter edges; the ECG scoring ignored them
   too).

## Where it comes from

- The tracker is `pll`, the harmonic particle-filter phase tracker. It was frozen on 17 Oct 2026 and chosen as the
  current best by the mean beat-to-beat score over the three ECG mice. `engine/pll_tracker/` holds its six files,
  byte-identical (`beams.py`, `variants.py`, `beamx.py`, `pf.py`, `pll.py`, `pll_variants.py`). The package checks
  their fingerprints on every run (`frozen_code_ok`). `bh.py` is a small adapter that hands them the period's data.
- Research files (our lab copy): `output/ng511_hr/b2b/pll/RESULT.md` (method, every setting and where it came from) and
  `output/hr_best/` (the comparison with the other trackers). `output/hr_curve_package/` holds the reproduction checks and
  the SNR calibration behind this page.
