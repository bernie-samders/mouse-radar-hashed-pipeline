# Locked release: mouse_radar_pipeline v1.7.0

This release fixes the code of the breathing, apnea and heart-rate pipeline used for our paper, so that anyone can check
they have exactly that code, byte for byte. Locked on 2026-11-10, before the preprint was posted.

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

The heart-rate method and the beat-to-beat tracker are byte-identical copies of the validated research files. The
pipeline checks their SHA-256 on every run (`FROZEN_SHA256` in `pipelines/hr.py` and `pipelines/b2b.py`), and every
heart-rate output records the result (`frozen_code_ok`).

| frozen component | SHA-256 |
|---|---|
| `engine/b2b_beats.py` | `a3ffffc38659b2e6c6672b4eac2a1ae4e0e75251cc968e3e072d4a53406d2960` |
| `engine/hr_final_pipeline.py` | `ff87ce110acf85e43da484c0f6768d41342cd0009886d831d3905a3dea97fdb0` |

## Environment

`requirements-lock.txt` lists the environment the release was run in.
- Other numpy / scipy / numba versions have not been tested. Last-digit floating-point differences could move a
  beat-to-beat result near its abstain threshold.
- The paper's results use the defaults (`python run.py all --exp <recording folder>`).

## Provenance

The code was exported from commit `273684de849ec6b3c0a43228a6ea70b7943e337a` of the authors' development repository. In that repository the
package tree is `f41c2b1f5cec15bfd3aba6f6d97ae08bc6ac8542`. The exported bytes are the git blobs, not a working-tree copy.
