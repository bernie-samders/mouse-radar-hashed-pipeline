# Locked release: mouse_radar_pipeline v1.8.0

This release fixes the code of the breathing, apnea and heart-rate pipeline used for our paper, so that anyone can check
they have exactly that code, byte for byte. Locked on 2026-11-24.

## What changed since v1.7.0

v1.7.0 was locked on 2026-11-10, with lock digest
`d39c26ca7cb257298fcd4516628cf8d5194fa515cd194ed4a8310530c69528fb`.
It computed heart rate at the beat level only (`b2b`). v1.8.0 adds the **rate level** (`rate` step, `RATE.md`):
the beat-level values, and where there are none, a rate tracker on still periods of 6 s or more.
- The rate tracker is the `hr_curve` step of package versions 1.4–1.6, restored without change
  (`engine/pll_tracker/`, `pipelines/hr_curve.py`).
- Segmentation, breathing metrics, apnea, `hr` and `b2b` run the same code as in v1.7.0. Compared with
  v1.7.0, the package files that differ are:

  - `B2B.md` (changed)
  - `DEJUMP.md` (changed)
  - `HEART_RATE.md` (changed)
  - `HR_CURVE.md` (added)
  - `RATE.md` (added)
  - `README.md` (changed)
  - `engine/pll_tracker/beams.py` (added)
  - `engine/pll_tracker/beamx.py` (added)
  - `engine/pll_tracker/bh.py` (added)
  - `engine/pll_tracker/pf.py` (added)
  - `engine/pll_tracker/pll.py` (added)
  - `engine/pll_tracker/pll_variants.py` (added)
  - `engine/pll_tracker/variants.py` (added)
  - `pipelines/__init__.py` (changed)
  - `pipelines/hr_curve.py` (added)
  - `pipelines/qc.py` (changed)
  - `pipelines/rate.py` (added)
  - `run.py` (changed)

**What this lock date means.** The rate-level settings (6 s periods, continuity band, 750 beats/min guard, merge with
the beats) were chosen on 2026-11-11. Rate-level results produced before 2026-11-24 came from package 1.6's `hr`
and `hr_curve` run next to v1.7.0; this release packages that code as one step (`RATE.md` describes the
check that it gives the same output). The lock fixes the rate level from 2026-11-24 on. It is not a registration made
before the earlier results were seen.

## What is locked

Every file listed in `MANIFEST.sha256`, byte for byte:
- the pipeline code and its documentation;
- this file and the license;
- the environment the release was run in (`requirements-lock.txt`).

The **lock digest** is the SHA-256 of `MANIFEST.sha256`. It is published with the archived release (Zenodo record,
Software Heritage) and in the paper. It can't be written inside the release, since it is computed from it.

## How to verify a copy

```bash
python verify_lock.py                       # re-hashes every file, prints the lock digest
python verify_lock.py <published digest>    # also compares it with the published one
```

You can also check the files with standard tools: `sha256sum -c MANIFEST.sha256` on Linux, or `shasum -a 256 -c
MANIFEST.sha256` on macOS. Then `sha256sum MANIFEST.sha256` gives the lock digest.

`.gitattributes` (`* -text`) makes git check the files out exactly as stored on every OS. Without it, a Windows checkout
with `core.autocrlf=true` rewrites line endings and every hash changes.

## Checks built into the pipeline

The heart-rate method, the beat-to-beat tracker and the rate tracker are byte-identical copies of the validated research
files. The pipeline checks their SHA-256 on every run (`FROZEN_SHA256` in `pipelines/hr.py`, `pipelines/b2b.py` and
`pipelines/hr_curve.py`), and every heart-rate output records the result (`frozen_code_ok`).

| frozen component | SHA-256 |
|---|---|
| `engine/b2b_beats.py` | `a3ffffc38659b2e6c6672b4eac2a1ae4e0e75251cc968e3e072d4a53406d2960` |
| `engine/hr_final_pipeline.py` | `ff87ce110acf85e43da484c0f6768d41342cd0009886d831d3905a3dea97fdb0` |
| `engine/pll_tracker/beams.py` | `cd2d190d49f05d33fab350dcfd5e31ee33a9ee1b55514652782a425ac9b02e26` |
| `engine/pll_tracker/beamx.py` | `02d33b783037273e00c7fa7ada8a9fc541a96a5fa05fe59401d21535cd8e1ab8` |
| `engine/pll_tracker/pf.py` | `6539457d7cc8a4d153c20f4b4899426d6d1f5d354c3864aef5f27f4c9bb2f6a8` |
| `engine/pll_tracker/pll.py` | `ed0c1e99c8372961ca9dbc8d672936f462150cd71d7a29d742f830599f474ff6` |
| `engine/pll_tracker/pll_variants.py` | `f4557ee80312e7a835fd82d1164bbc63678d2733a55851856fb083271b37be59` |
| `engine/pll_tracker/variants.py` | `ec120946ca3de567e7417fd574423b064c95f21e5879c4fd75ba36512d5e40da` |

## Environment

`requirements-lock.txt` lists the environment the release was run in.
- Other numpy / scipy / numba versions have not been tested. Last-digit floating-point differences could move a
  beat-to-beat result near its abstain threshold, and the rate tracker's particle filter can amplify them.
- The paper's results use the defaults (`python run.py all --exp <recording folder>`).

## Provenance

The code was exported from commit `862068bf8abdb6888bf7ed122e5d6fe6ea5e13cb` of the authors' development repository. In that repository the
package tree is `e4508df4ed28fd88273cbd27141a04a88f12bcf1`. The exported bytes are the git blobs, not a working-tree copy.
