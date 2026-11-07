# SPDX-License-Identifier: MIT
"""Run the link-prediction study: the held-out triples of the WN18 and WN18RR releases against the WordNet table of
reflexive pointers (paper Sec. VI-C, extended audit details Section C)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca import kg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs", help="output root; this step writes <outputs>/kg/")
    ap.add_argument("--kg", type=Path, default=REPO / "data" / "external" / "kg" / "files",
                    help="folder with the files of data/external/kg/sources.json (see FETCH.md there)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    r = kg.run(a.kg, a.outputs / "kg")
    heads = " ".join(f"{x['release']}/{x['split']}={x['l1']}/{x['held_out']}" for x in r["summary"])
    print(f"kg: {heads} filter_reproduces={r['filter']['reproduces_all_splits']} "
          f"certified_in_wordnet={r['raw']['certified_pairs_on_one_synset_pair']}/"
          f"{r['raw']['certified_triples']} (one synset pair each) seconds={time.perf_counter() - t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
