# SPDX-License-Identifier: MIT
"""Scaling step: Path B grouping and integer-program time on nested 10%-100% hash samples."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

THREAD_LIMIT = 1
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "HIGHS_THREADS"):
    os.environ.setdefault(_k, str(THREAD_LIMIT))

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca import io, scaling  # noqa: E402


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
    a = ap.parse_args()
    started = time.perf_counter()

    sums = input_hashes(REPO / "data" / "hub" / "SHA256SUMS")
    anchors = io.require_sha(a.inputs / "anchors.csv.gz", sums["anchors.csv.gz"])
    sample_order = io.require_sha(a.inputs / "sample_order.csv.gz", sums["sample_order.csv.gz"])
    upstream = ["registry/populations.json", "registry/facet_map.csv", "registry/certificate_spec.csv",
                "registry/view_projection.csv", "registry/dataset_support.json",
                "conformance/full_conform.csv.gz", "conformance/query_masks.npz"]
    hashes = {"anchors.csv.gz": sums["anchors.csv.gz"], "sample_order.csv.gz": sums["sample_order.csv.gz"]}
    hashes.update({k: io.sha256_file(a.outputs / k) for k in upstream})

    out = a.outputs / "scaling"
    io.write_json(out / "protocol.json", scaling.protocol(hashes, THREAD_LIMIT))

    model = scaling.load_model(anchors, a.outputs)
    ranks = scaling.read_sample_rank(sample_order, model["row_id"])
    orders = scaling.nested_orders(ranks, model["dataset_code"])
    print(f"[scaling] inputs and sample order {time.perf_counter() - started:.1f}s", flush=True)

    raw, summary, agg = scaling.run_scaling(model, orders, log=lambda s: print(s, flush=True))
    io.write_csv(out / "scaling_runs.csv", raw, scaling.RAW_FIELDS)
    io.write_csv(out / "scaling_summary.csv", summary, scaling.SUMMARY_FIELDS)
    io.write_csv(out / "scaling.csv", agg, scaling.AGG_FIELDS)
    print(f"[scaling] done {time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
