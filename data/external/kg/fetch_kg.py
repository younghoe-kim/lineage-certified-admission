# SPDX-License-Identifier: MIT
"""Download the files of the link-prediction study (data/external/kg/sources.json) and verify their SHA-256,
or verify a folder that already holds them (--check). Nothing is extracted or run: the step reads the archive
in memory and parses one list literal of the construction script."""
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
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"
        if got != e["sha256"]:
            bad.append(f"{e['path']}: {got}")
    if bad:
        sys.exit("SHA-256 mismatch:\n" + "\n".join(bad))
    print(f"link-prediction files verified: {len(src['files'])} in {a.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
