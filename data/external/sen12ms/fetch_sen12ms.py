# SPDX-License-Identifier: MIT
"""Download the files of the SEN12MS study (data/external/sen12ms/sources.json) and verify their SHA-256 and git
blob ids, or verify a folder that already holds them (--check). Nothing is unpickled or run here: the step
only downloads and hashes the files."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=HERE / "files")
    ap.add_argument("--check", action="store_true", help="verify the files in --dest without downloading")
    a = ap.parse_args()
    src = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
    bad = []
    for e in src["files"]:
        p = a.dest / e["path"]
        if not a.check and not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(e["url"], timeout=120) as r:
                p.write_bytes(r.read())
        raw = p.read_bytes() if p.exists() else None
        got = hashlib.sha256(raw).hexdigest() if raw is not None else "missing"
        blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest() if raw is not None else "missing"
        if got != e["sha256"] or blob != e["git_blob_sha1"]:
            bad.append(f"{e['path']}: sha256 {got}, git blob {blob}")
    if bad:
        sys.exit("hash mismatch:\n" + "\n".join(bad))
    print(f"SEN12MS files verified: {len(src['files'])} in {a.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
