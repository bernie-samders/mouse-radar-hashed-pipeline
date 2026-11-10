# Heart rate: beats per minute from the radar, without contact

**Short version:** `hr` measures the mouse's heart rate from the stored radar voxels in every period where the mouse is
still and breathing regularly. It reports one value per period, plus a 10-per-second track and the detected beat
times. We checked it against an implanted ECG on two mice recorded with both at once (508 and 507 day 2). The radar
heart rate per period was off by **10.9 and 8.9 beats/min on average**, with **87 % and 94 % of periods within 5 %** of
the ECG. Those two mice are also the ones the method was built on, so new animals may do somewhat worse (see "Read
before using the numbers"). It runs automatically as part of `python run.py all` (new in version 1.3).

**July-2026 captures (07-02 / 07-16 / 07-20):** that capture build puts jumps into the cube that hide the heart. Since
version 1.5 this step repairs them automatically on a copy of the cube before it runs (`--dejump auto`); see
[`DEJUMP.md`](DEJUMP.md). Earlier versions' heart rates for July captures are not trustworthy.

**For individual beats, heart rate within a period, and the heart rate the paper reports**, use the `b2b` step (new in version 1.7; see
[`B2B.md`](B2B.md)). It also runs after this one, seeded with this step's period heart rate, and abstains where the
heart is weak.

---

## How to run it

It is part of the normal run, so nothing extra is needed:

```bash
python run.py all --exp "<experiment_dir>" --cohort B --age-weeks 8     # segment + every metric, incl. heart rate
python run.py combine                                                   # after all experiments: master tables
```

To (re)run only the heart-rate step on a recording that is already segmented:

```bash
python run.py hr --rec <recording_name>                             # finds the raw voxels via the segmentation file
python run.py hr --rec <recording_name> --exp "<experiment_dir>"    # if the raw folder has moved
python run.py hr --rec <recording_name> --workers 4                 # fewer parallel processes (less memory)
```

**What it needs:**
- **The raw voxels** (`*_voxels.f32`), not just the segmentation: it re-reads 330 voxels around the mouse's chest.
- **A 250 Hz capture** (the standard live capture). Other rates are skipped with a note.
- **Time and memory.** On our laptop (16-core Intel Core Ultra 9, 12 worker processes) it adds about
  **4 minutes** per hour-long recording on top of segmentation; the whole `run.py all` took **19–22 minutes** per
  hour-long recording. Time grows with the minutes of still breathing. Each worker process uses about 0.5 GB of memory. The default is one
  worker per CPU core minus two, at most 12; use `--workers` to lower it on a smaller computer.

## What you get

| file | what |
|---|---|
| `outputs/metrics/<rec>.hr.csv` / `.json` | one row: the recording's heart rate (columns below) |
| `outputs/metrics/<rec>.hr_periods.csv` | one row per still, regular-breathing period: heart rate, breathing rate, band, method |
| `outputs/metrics/<rec>.hr.png` | heart rate (top) and breathing rate (bottom) per period over the session |
| `outputs/metrics/<rec>.hr_track.csv` | heart rate 10 times per second inside each period (first and last second dropped) |
| `outputs/metrics/<rec>.hr_beats.csv` | detected beat times (seconds from the start of the recording) |
| `outputs/metrics/<rec>.hr_templates.npz` | the 20 voxels used in each period: their heartbeat templates and their own beat times (see "Templates and per-voxel beats") |
| `outputs/metrics/combined_hr.csv`, `combined_hr_periods.csv` | all experiments, after `run.py combine` |

The columns that matter in `<rec>.hr.csv`:

| column | meaning |
|---|---|
| `hr_median_bpm` | **main result.** Median of the per-period heart rates, beats per minute |
| `hr_p25_bpm`, `hr_p75_bpm` | spread across periods (25th / 75th percentile) |
| `breaths_per_min_median` | breathing rate in the same periods. **Report it next to heart rate** (see point 6 below) |
| `hr_over_rr_median` | heart beats per breath |
| `n_periods`, `scored_min` | how many still periods were scored and how many minutes they cover |
| `n_fallback` | periods where the main method failed and a simpler one was used (see "How it works"). Normally 0 |
| `patch_beams`, `patch_deg`, `breath_peak_bin` | where the chest patch was placed (radar beams / angle, range bin of the strongest breathing) |
| `patch_edge_warning` | `True` if the breathing was strongest at the edge of the chest-patch range: the mouse may sit outside it, so treat the heart rate as unreliable |
| `frozen_code_ok` | `True` if the heart-rate code is the exact validated version |
| `notes` | plain-English warnings for this recording, if any |

`<rec>.hr_periods.csv` has one row per period: `start_min`, `dur_s`, `breaths_per_min`, `hr_bpm`, `hr_over_rr`,
`band_centre_bpm` (which heartbeat band was chosen), `n_beats`, `method` (`bspec_beam_tmean` normally,
`S1_fallback` if the fallback was used) and `fallback_reason`.

### Templates and per-voxel beats (`<rec>.hr_templates.npz`)

For downstream work, each period's 20 combined voxels are saved with their EQ-Radio heartbeat template and their own
beat times. Load it in Python with `z = numpy.load("<rec>.hr_templates.npz")`; `print(z["README"])` describes every
array. Per period (`P` periods, matching `hr_periods.csv`):

| array | shape | meaning |
|---|---|---|
| `period`, `t0_s`, `t1_s`, `band_centre_bpm`, `method` | P | the period and the heartbeat band that was used |
| `voxel_col`, `voxel_beam`, `voxel_angle_deg`, `voxel_range_bin`, `voxel_range_m` | P × 20 | where each voxel is (rank 0 = the voxel whose beats agreed best with the others) |
| `agree`, `template_conf`, `template_snr` | P × 20 | beat agreement with the other voxels, and how well the template fits that voxel |
| `templates`, `template_len` | P × 20 × m, P | each voxel's template at its native length m (samples at 250 Hz; m depends on the band, padded with NaN) |
| `templates_31` | P × 20 × 31 | the same templates resampled to 31 samples, so periods with different bands line up |
| `beat_sample`, `beat_t_s`, `beat_period`, `beat_rank` | one row per beat | every beat start of those voxels (sample index / seconds), with its period and voxel rank |

Things to know:
- The templates are cut from each voxel's **acceleration** (second derivative of the radar amplitude), before breathing
  is removed.
- Their units are arbitrary and their **sign is arbitrary per voxel**. The combined beam aligns sign and timing itself.
  Align them before averaging across voxels.
- The heart rate and `hr_beats.csv` come from the combined beam, not from these per-voxel beats. The per-voxel beats
  are the segmentation that chose the voxels. They are noisier than the beam's beats and can disagree with them.
- About 1 MB per hour-long recording (0.9 MB for 507 day 2: 43 periods, 263,500 voxel beats).

## How it works

The heartbeat moves the chest far less than breathing does, and the radar only resolves it when the mouse is still.
So the method first finds still periods, then removes breathing, then looks for a rhythm that many chest voxels share.

1. **Find still periods.** It uses the same two-template breath/pause segmentation as the rest of the package. A period
   is a stretch of clean, regular breathing (at least 8 consecutive well-fit breaths) lasting at least 20 s. Only these
   periods are scored; on our two ECG mice they covered about half of an hour-long recording (29–30 min).
2. **Place a chest patch.** 330 voxels: 10 neighbouring radar beams × the range bins 0.27–0.45 m from the radar,
   centred on the beam where breathing is strongest. Each voxel is turned into acceleration (second derivative), which
   favours the fast heartbeat over the slow breathing.
3. **Remove breathing.** A continuous breathing clock is taken from the whole patch. Then every whole-number multiple of
   the breathing rate (1×, 2×, 3×, ... up to 2400 per minute) is fitted and subtracted from every voxel. Without this
   step the method mostly finds breathing harmonics instead of the heart.
4. **Choose the heartbeat band.** There are 17 candidate bands between 380 and 780 beats/min. It picks the band where
   the chest voxels share the most rhythmic power (a whitened cross-voxel spectrum, computed in 6 s windows).
5. **Find beats and combine voxels.** In that band, each voxel is cut into heartbeats with the EQ-Radio heartbeat
   segmentation. The 20 voxels whose beats agree best with the other voxels are combined into one signal (a beam), with
   their timing and sign aligned automatically.
6. **Read the rate.** The beam's instantaneous beat frequency gives the 10-per-second track. The period's heart rate is
   the average of the track after dropping the top and bottom 10 % of values (a trimmed mean). Beat times are where the
   beam completes each cycle.

**Fallback.** If any step fails on a period, that period gets the band whose detected beats repeat most consistently
(the earlier "S1" method) and is marked `S1_fallback`. That never happened on our 93 test periods. S1 is clearly less
accurate (per-period error 26 / 77 beats/min on the two ECG mice), so look twice at fallback periods.

## How well it works

**Against the implanted ECG**, on every still period with clean ECG (28 Sep 2026 recordings):

| | mouse 508 (23 periods) | mouse 507, day 2 (18 periods) |
|---|---|---|
| **average error per period** | **10.9 beats/min** | **8.9 beats/min** |
| periods within 5 % of the ECG | 87 % | 94 % |
| correlation with the ECG across periods | 0.98 | 0.97 |
| for comparison: always guessing the ECG's own median | 49.4 | 42.5 |
| for comparison: the earlier method (S1) | 26.4 | 77.3 |
| 6 s windows inside the periods (average error) | 20.4 | 11.7 |

The true heart rate ranged from about 405 to 612 beats/min in these periods, so the method follows real changes rather
than sitting at a typical value. Figures: `docs/validation_508.png` and `docs/validation_507day2.png` (radar vs ECG over
the session, a beat-by-beat example, and radar vs ECG per period).

**This package reproduces those numbers exactly.** We ran it on the two raw recordings and compared it with the
research code the numbers came from:
- **Mouse 508:** heart rate, chosen band and beat times are identical to the last digit in all 23 ECG periods, so the
  average error is again 10.9 beats/min. This holds both for the chest patch the package places itself (beams 7–16)
  and for the hand-placed patch of the original analysis (beams 6–15, all 50 periods); the two give 10.90 and 10.91.
- **Mouse 507 day 2:** the same 43 still periods, the same chest patch, and the same band and beat times in every
  period. The heart rates agree to within 0.0000002 beats/min (rounding in the last digits), so the average error is
  again 8.9 beats/min.
- **From the raw data:** running the whole package on the two raw recordings (`python run.py all`, breath
  segmentation from scratch) gave the same still periods and exactly the same heart rates, bands and beat times as
  above (50 of 50 periods on 508, 43 of 43 on 507 day 2).

**Beat times.** In 18 of the 41 periods the radar beats line up with the ECG beats one for one (at least 90 % of beats
within 20 ms; timing jitter 7–9 ms). In those periods the heart rate over 8 beats is typically within 3 beats/min. In
the others the radar gains or drops a beat every few seconds, although the period's rate is often still right. (The
figures' "beats locked to ECG" is a looser test: the radar beats match the ECG better than chance and better than a
beat train locked to breathing. That holds in 22 of 23 and 16 of 18 periods.)

## Read before using the numbers

1. **Built and tested on the same two mice.** The method was developed on 508 and 507 day 2, so the numbers above are a
   best case. It has not yet been checked on a new animal with an ECG; that is the missing test.
2. **Resting heart rate only.** Only still, regular-breathing periods are scored. Heart rate during movement, sniffing
   or stress is invisible to the radar. Animals differ in how many still periods they have, so compare per-animal
   medians and look at `n_periods` / `scored_min`.
3. **Heart rate close to 3 × the breathing rate is the known weak spot.** On 507 day 2 between 27 and 33 min the heart
   sat near 3 × breathing. The rate was right in 4 of those 5 periods, but in 3 of them the radar beats were not locked
   to the true heartbeat, and one period read 579 vs 531 beats/min on the ECG. `hr_over_rr` is in the period table. A
   value near a whole number is not wrong by itself: on 508, 12 of 23 periods had a true heart rate within 5 % of 2, 3
   or 4 × breathing and were still measured well (error 8.6 beats/min). It is just where errors cluster.
4. **Fast changes within a period are smoothed.** When the heart rate swings within a period, the period value can come
   out one band too high (+24 to +41 beats/min on 4 periods of 508). The 10-per-second track follows changes over
   several seconds, not drops lasting only a second or two. For changes within a period use `b2b`
   ([`B2B.md`](B2B.md)), which times the individual beats.
5. **Not for beat-to-beat HRV or RSA.** The beat times give heart rate and slow trends, not beat-to-beat variability.
   The radar's beat timing error (about 7–18 ms) is larger than the true beat-to-beat change in these mice (RMSSD 7–9
   ms), and the method's filter removes the breathing-linked heart-rate swings (respiratory sinus arrhythmia). Don't
   compute RMSSD, SD1/SD2, pNN or RSA from `hr_beats.csv`.
6. **Report breathing rate alongside heart rate.** In our free-behaving cohort (radar only, no ECG; July captures
   repaired, `ok` / `strong` periods only; 10 wild-type vs 11 mutant), mutants had somewhat lower heart rates (median
   446 vs 473 beats/min, not significant), and their slower breathing explained the difference (DEJUMP.md). Compare heart rate with breathing rate taken into account (both are in the outputs), and check any
   group difference against movement and batch, as for every other metric.
7. **Distrust values near the top of the range.** Calm mouse heart rates here were 400–600 beats/min. In one cohort
   recording (A11) the method sat at about 800 beats/min (the top band) whatever the breathing did, which is not a
   heart rate. Treat periods above ~650 beats/min as suspect; `b2b` does not run on periods seeded above 750.
8. **Very fast breathers need the right breath segmentation** (new in 1.4). Above ~250 breaths/min the default
   segmentation counts every other breath, so breathing is removed at the wrong rate. The heart band then locks onto a
   breathing harmonic. On ng511 (~300 breaths/min, implanted ECG) that read about 110 beats/min too high.
   - `run.py all` / `segment` now detect this and re-segment automatically (QC.md).
   - If a halved segmentation reaches this step anyway, `breathing_halved` is `True`, the notes and the figure say so,
     and QC prints *"BREATHING HALVED -> NOT a heart rate"*.
   - Since 1.6 the same automatic re-segmentation also catches a breather just above the default bounds' reach, whose
     breaths are squeezed rather than halved (ng511 on 10-21: 267/min counted as 250/min; QC.md, "breathing clipped").
   - Even when segmented correctly, ng511's per-period heart rate was much less accurate than on 508 / 507 day 2:
     average error about 50 beats/min over its 14 periods with clean ECG. Its heart signal is weak; `b2b` abstains on
     almost all of its periods (B2B.md).
9. **Cage geometry is fixed.** The chest patch covers 0.27–0.45 m from the radar (our cage position). If
   `patch_edge_warning` is `True`, the mouse was probably at the edge or outside that range. A recording made at
   another sampling rate than 250 Hz is skipped.

## Where it comes from

- The method was frozen on 3 Oct 2026, after a systematic search of breath-removal, band-choice and read-out ideas on
  the two ECG mice. `engine/hr_final_pipeline.py` is that frozen file, unchanged. The package checks its fingerprint
  (sha256 `ff87ce11…97fdb0`) on every run and writes the result to `frozen_code_ok`.
- The beat segmentation follows EQ-Radio (Zhao et al., MobiCom 2016): a beat template is fitted, and the signal is cut
  into variable-length beats that best match it.
- Research files (our lab copy): `output/ecg_0928/hr_overnight/final/` (frozen code, `FROZEN.md`), `radar_hr/` (the
  runner this step is ported from) and `output/hr_package/` (the reproduction check behind the table above).
