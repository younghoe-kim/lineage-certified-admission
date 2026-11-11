# SPDX-License-Identifier: MIT
"""Run the external MaDI-Bench study: source freeze, protocol, audit, independent verification and the
comparison with the benchmark's normalization gold."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca import external  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=REPO / "data" / "hub",
                    help="hub input view (this step reads only the pinned benchmark checkout)")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs",
                    help="output root; this step writes <outputs>/external/")
    ap.add_argument("--checkout", type=Path, default=REPO / "data" / "external" / "MaDI-Bench",
                    help="local checkout of the pinned benchmark commit (see data/external/FETCH.md)")
    ap.add_argument("--git-dir", type=Path, default=None,
                    help="Git repository holding the pinned commit (default: the checkout when it has .git)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="accept a checkout that lacks some pinned files (requires a Git directory); rows that "
                         "need a missing file are left out and the missing files are listed in source_freeze.json")
    a = ap.parse_args()
    git_dir = a.git_dir if a.git_dir is not None else (a.checkout if (a.checkout / ".git").exists() else None)
    t0 = time.perf_counter()
    res = external.run(a.checkout, git_dir, a.outputs / "external", allow_partial=a.allow_partial)
    print(f"external: complete={res['complete']} missing_files={res['missing']} skipped_rows={res['skipped_rows']} "
          f"checks={res['checks']} decision={res['decision']} qualifying={','.join(res['qualifying_domains'])} "
          f"gold_cells={res['gold_cells']} gold_contradictions={res['gold_contradictions']} "
          f"seconds={time.perf_counter() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
