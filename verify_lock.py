"""verify_lock.py -- check that this folder is the locked mouse_radar_pipeline release, byte for byte.

    python verify_lock.py                  # re-hash every file listed in MANIFEST.sha256, print the lock digest
    python verify_lock.py <lock digest>    # ...and compare it with the digest published for the release

Standard library only (no numpy needed). The lock digest is the SHA-256 of MANIFEST.sha256; the same check works with
`sha256sum -c MANIFEST.sha256` (Linux) or `shasum -a 256 -c MANIFEST.sha256` (macOS). Files the pipeline creates when it
runs (outputs/, __pycache__/) are ignored. Exit code 0 = locked release intact, 1 = something differs.
"""
import ast
import hashlib
import os
import sys

VERSION = "1.7.0"
HERE = os.path.dirname(os.path.abspath(__file__))
IGNORE_DIRS = {".git", "__pycache__", "outputs", "hr_work", "b2b_work", "dejumped"}
IGNORE_EXT = {".pyc", ".pyo", ".nbi", ".nbc"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_constants():
    """The validated research-file hashes the pipeline itself checks at run time (pipelines/hr.py, b2b.py)."""
    out = {}
    for rel, name, base in (("pipelines/hr.py", "FROZEN_SHA256", "engine/hr_final_pipeline.py"),
                            ("pipelines/b2b.py", "FROZEN_SHA256", "engine/b2b_beats.py")):
        tree = ast.parse(open(os.path.join(HERE, rel), encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
                val = ast.literal_eval(node.value)
                if isinstance(val, str):
                    out[base] = val
                else:
                    out.update({f"{base}/{k}": v for k, v in val.items()})
    return out


def main(argv):
    man_path = os.path.join(HERE, "MANIFEST.sha256")
    raw = open(man_path, "rb").read()
    listed = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            listed[rel.lstrip("*").strip()] = digest
    bad, missing = [], []
    for rel, digest in sorted(listed.items()):
        p = os.path.join(HERE, *rel.split("/"))
        if not os.path.isfile(p):
            missing.append(rel)
        elif sha256(p) != digest:
            bad.append(rel)
    extra = []
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), HERE).replace(os.sep, "/")
            if rel != "MANIFEST.sha256" and rel not in listed and os.path.splitext(f)[1] not in IGNORE_EXT:
                extra.append(rel)
    frozen_bad = [rel for rel, h in frozen_constants().items()
                  if not os.path.isfile(os.path.join(HERE, *rel.split("/"))) or sha256(os.path.join(HERE, *rel.split("/"))) != h]
    lock = hashlib.sha256(raw).hexdigest()

    print(f"mouse_radar_pipeline v{VERSION}: {len(listed)} files in MANIFEST.sha256")
    for rel in missing:
        print(f"  MISSING  {rel}")
    for rel in bad:
        print(f"  CHANGED  {rel}")
    for rel in extra:
        print(f"  extra    {rel} (not part of the release)")
    print(f"  frozen heart-rate components: {'validated' if not frozen_bad else 'DIFFER: ' + ', '.join(frozen_bad)}")
    print(f"lock digest (SHA-256 of MANIFEST.sha256): {lock}")
    ok = not (bad or missing or frozen_bad)
    if len(argv) > 1:
        match = argv[1].strip().lower() == lock
        print("published digest:", "MATCH" if match else "DOES NOT MATCH")
        ok = ok and match
    print(f"LOCKED v{VERSION}: all files intact" if ok else "NOT the locked release (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
