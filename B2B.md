# Beat-to-beat heart rate (`b2b`)

**Short version:** `b2b` (new in version 1.7) finds the individual heartbeats inside every still, regular-breathing
period that the `hr` step scores, and reports each beat's time and the heart rate from the beat intervals. It is the
heart-rate method the paper reports. It runs automatically after `hr` in `python run.py all`, or on its own:

```bash
python run.py b2b --rec <recording_name> [--exp <raw_experiment_dir>] [--workers N]   # after hr
```

Checked against an implanted ECG on nine sessions (five mice):
- **2 s accuracy:** 96.8 % of 2 s bins within 5 % of the ECG, 0.5 % more than 10 % off, mean error 5.7 beats/min.
- **Beat level:** median interval error 1.6 ms over 28,086 intervals; 98 % of radar beats within 20 ms of an R-peak.
- **Coverage depends on the heart signal.** Where the radar sees the heart (507, 508) it reports 27–41 % of the
  ECG-checkable time. On weak-heart sessions it reports almost nothing (≤ 4 %), because it **abstains** rather than
  guess (129 of 136 weak-heart periods).

## How it works

Per period of the `hr` step (≥ 20 s of still, regular breathing; no ECG anywhere):
1. The breath-order residual of the chest-patch acceleration: the frozen `hr` front end.
2. `engine/b2b_beats.py`: a space-time whitened matched filter over all chest voxels, then a second-order beat
   Viterbi. It is run from 11 starts and the run that agrees best with the others is kept. The `hr` step's period
   heart rate is the seed.
3. **Abstain rule:** a period is reported only if the kept run's median matched-filter output at its beats is at least
   2.0 noise SDs (`snr` column). Otherwise `abstain_reason = snr_below_gate`.
4. The same estimate at seed × 0.998 / 0.999 / 1.001 / 1.002 gives `family_agree`, a stability flag only. The reported
   beats are the × 1 run.

`engine/b2b_beats.py` is byte-identical to the validated research file. Its SHA-256 is checked on every run
(`frozen_code_ok`).

## High heart-rate guard (new in 1.7)

A period is **not reported** when its seed or its b2b heart rate is above **750 beats/min**
(`abstain_reason = seed_above_max` / `hr_above_max`; the unguarded value is kept in `hr_unguarded_bpm`).
- Why: the `hr` step's top band (centre 780 beats/min) sometimes locks onto a non-cardiac line at about
  780–830 beats/min. One NAS recording (Animal_B_16_6wks) had b2b seeded there at 805 and it read 1058 beats/min.
- Safe for the validated numbers: no ECG-checked heart was above 612 beats/min. On the nine ECG sessions every seed
  above 650 was 823 or more, and b2b had already abstained on all of them, so the guard changes none of the ECG
  results above.
- Why 750 and not 800: on our 83 NAS recordings both report the same periods, but 750 keeps a margin below the
  ~800 line (at 800 one period seeded at 777 read 802 and was dropped by only 2 beats/min).
- Values between 612 and 750 are above anything the ECG has confirmed. On the NAS recordings four periods read 628–713,
  two of them from WBP-chamber animals (Test2B_7weeks, Test3A_6wks) and two from Animal_A10_6wks_Repeat.

## Outputs (`<out>/metrics/`)

| file | content |
|---|---|
| `<rec>.b2b.csv` / `.json` | one row: `hr_median_bpm` (median of the 2 s values below), `hr_covered_min` (their minutes), `hr_per_breath_median` (median over tracked periods of period HR / period breathing rate), tracked periods / minutes / beats, abstain counts by reason, `frozen_code_ok`, notes |
| `<rec>.b2b_bins.csv` | the 2 s heart-rate values: on a 2 s grid inside tracked periods, 60 / median of the beat intervals whose midpoints lie within ±3 s (≥ 5 intervals) |
| `<rec>.b2b_periods.csv` | one row per `hr` period: `tracked`, `abstain_reason`, `snr`, `n_beats`, `hr_bpm`, `seed_hr_bpm`, `family_agree` |
| `<rec>.b2b_beats.csv` | one row per beat: period, beat time (s, radar clock), interval and heart rate from the previous beat |
| `<rec>.b2b.npz` | `{period: beat times in s}` for the tracked periods |

## Read before using the numbers

- **Use only tracked periods.** An abstained period has no heart rate; do not fill it from `hr`.
- **Successive-difference HRV** (RMSSD, pNN) is valid only at `snr` ≥ 2.3; between 2.0 and 2.3 beat jitter inflates
  RMSSD about 1.6×.
- **Halved breathing** (`breathing_halved`, see the README): a breathing harmonic can pass for a heartbeat, so ignore
  the beats.
- Development: b2b was built on two sessions from 09-28 (508, 507 day 2); the other seven ECG sessions came after it
  was frozen.
