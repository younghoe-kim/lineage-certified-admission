# SPDX-License-Identifier: MIT
"""Controls step: partition identity, simpler checks and the family-3 permutation null."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

THREAD_LIMIT = 1
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, str(THREAD_LIMIT))

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import json  # noqa: E402

import pandas as pd  # noqa: E402

from lca import controls, io  # noqa: E402


def input_hashes(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, name = line.split(None, 1)
            out[name.strip()] = sha
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=REPO / "data" / "hub")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--permutations", type=int, default=controls.PERMUTATIONS,
                    help="number of permutations B (seeds 0..B-1)")
    a = ap.parse_args()
    if a.permutations < 1:
        raise SystemExit("--permutations must be at least 1")
    started = time.perf_counter()

    sums = input_hashes(REPO / "data" / "hub" / "SHA256SUMS")
    anchors = io.require_sha(a.inputs / "anchors.csv.gz", sums["anchors.csv.gz"])
    up = {
        "conformance/full_conform.csv.gz": a.outputs / "conformance" / "full_conform.csv.gz",
        "registry/populations.json": a.outputs / "registry" / "populations.json",
        "verdicts/verdicts.csv": a.outputs / "verdicts" / "verdicts.csv",
        "utility/probes.csv": a.outputs / "utility" / "probes.csv",
    }
    hashes = {"anchors.csv.gz": sums["anchors.csv.gz"]}
    hashes.update({k: io.sha256_file(p) for k, p in up.items()})

    pops = json.loads(up["registry/populations.json"].read_text(encoding="utf-8"))
    facet_order = list(pops["block_coordinate_members"]["facet::"])
    verdicts = pd.read_csv(up["verdicts/verdicts.csv"], dtype=str, keep_default_na=False)
    probes = pd.read_csv(up["utility/probes.csv"], dtype=str, keep_default_na=False)

    t0 = time.perf_counter()
    rows = controls.build_rows(anchors, up["conformance/full_conform.csv.gz"])
    print(f"[controls] anchor pass {time.perf_counter() - t0:.1f}s", flush=True)

    out = a.outputs / "utility"
    part = controls.partition_identity(rows)
    io.write_csv(out / "partition_identity.csv", part, controls.PARTITION_FIELDS)
    table, details, values = controls.simple_checks(rows, verdicts, probes, facet_order)
    io.write_csv(out / "simple_checks.csv", table, controls.TABLE_FIELDS)
    io.write_csv(out / "simple_checks_details.csv", details, controls.DETAIL_FIELDS)
    io.write_csv(out / "simple_checks_values.csv", values, controls.VALUE_FIELDS)
    print(f"[controls] partition identity and simple checks {time.perf_counter() - started:.1f}s", flush=True)

    t1 = time.perf_counter()
    engines = controls.permutation_engines(rows)
    observed = controls.observed_family3_scores(probes)
    null_rows, summary = controls.permutation_null(engines, observed, B=a.permutations)
    io.write_csv(out / "permutation_null.csv", null_rows, controls.NULL_FIELDS)
    io.write_csv(out / "permutation.csv", summary, controls.SUMMARY_FIELDS)
    print(f"[controls] permutation null B={a.permutations} {time.perf_counter() - t1:.1f}s", flush=True)

    io.write_json(out / "controls.json", {
        "step": "controls",
        "schema_version": "1",
        "inputs_sha256": hashes,
        "thread_limit": THREAD_LIMIT,
        "build_report": {"conforms": controls.BUILD_REPORT_CONFORMS,
                         "dropped_core_entries": controls.BUILD_REPORT_DROPPED_CORE_ENTRIES},
        "permutation_null": {"B": a.permutations, "seeds": f"0-{a.permutations - 1}", "rng": controls.NULL_RNG,
                             "permutation_order": controls.NULL_ORDER, "definition": controls.NULL_DEFINITION,
                             "stratification": controls.NULL_STRATIFICATION, "p_formula": "(k+1)/(B+1)",
                             "min_label_support": controls.MIN_LABEL_SUPPORT, "score_tie_atol": controls.SCORE_EPS},
        "outputs": ["partition_identity.csv", "simple_checks.csv", "simple_checks_details.csv",
                    "simple_checks_values.csv", "permutation_null.csv", "permutation.csv"],
    })
    print(f"[controls] done {time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
