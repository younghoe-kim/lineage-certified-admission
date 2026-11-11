# SPDX-License-Identifier: MIT
"""Verdicts step: fiber verdicts of the registered views, witnesses, decomposition, carrier equivalence, Fig. 1 anchor, exact checks of every Path B certificate and open fiber."""
import os

THREADS = 1  # numerical libraries and HiGHS run single-threaded
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "HIGHS_THREADS"):
    os.environ[_k] = str(THREADS)

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from lca import certificates as cert  # noqa: E402
from lca import io as lio  # noqa: E402
from lca import verdicts as vd  # noqa: E402

RESULT_FIELDS = ["view", "dataset", "basis", "path", "fibers", "m^U", "m^N", "m^V_obs", "m^V_defect", "near_pure", "purity"]
INDEX_FIELDS = ["array_index", "view_id", "task", "certificate_path", "verdict_codes", "row_order_source_sha256", "row_count"]
PROJECTION_FIELDS = ["view_id", "task", "exposed_blocks", "projection_json", "projection_source"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=REPO / "data" / "hub")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    a = ap.parse_args()
    t0 = time.perf_counter()
    sums = vd.read_sha256sums(REPO / "data" / "hub" / "SHA256SUMS")
    anchors = a.inputs / "anchors.csv.gz"
    out = a.outputs

    check = cert.solver_self_check()
    print(f"[verdicts] integer-program solver check: {check}", flush=True)
    reg = vd.load_registry(out)
    conf = vd.load_conformance(out, reg)
    ev = vd.load_evidence(anchors, sums["anchors.csv.gz"], reg, conf, select_fig1=True)
    print(f"[verdicts] evidence loaded ({time.perf_counter() - t0:.1f}s); Fig. 1 candidates={ev['fig1_candidates']}", flush=True)
    res = vd.run_verdicts(reg, conf, ev, log=lambda s: print(s, flush=True))
    decomposition = vd.decomposition(reg, conf, ev, res["verdict_cache"])
    carrier = vd.carrier_equivalence(reg, conf, ev, res["verdict_cache"])
    path_b_rows = vd.verify_path_b(reg, res["path_b_unsafe"])
    print(f"[verdicts] Path B certificates verified without a solver: {len(path_b_rows)}", flush=True)
    path_b_open_rows = vd.verify_path_b_open(reg, res["path_b_open"])
    print(f"[verdicts] open single-label Path B fibers verified without a solver: {len(path_b_open_rows)}", flush=True)

    inputs_sha = {"anchors.csv.gz": sums["anchors.csv.gz"], **reg["sha256"], **conf["sha256"]}
    meta = {"step": "verdicts", "schema_version": "1", "inputs_sha256": inputs_sha}
    lio.write_csv(out / "registry" / "view_projection.csv", reg["projection_rows"], PROJECTION_FIELDS)
    lio.write_csv(out / "verdicts" / "verdicts.csv", res["result_rows"], RESULT_FIELDS)
    row_order_sha = conf["sha256"]["conformance/full_conform.csv.gz"]
    index_rows = [{"array_index": i, "view_id": r["view_id"], "task": r["task"], "certificate_path": r["certificate_path"],
                   "verdict_codes": vd.VERDICT_CODES_TEXT, "row_order_source_sha256": row_order_sha, "row_count": conf["n"]}
                  for i, r in enumerate(reg["cert"])]
    lio.write_csv(out / "verdicts" / "anchor_verdicts_index.csv", index_rows, INDEX_FIELDS)
    lio.write_npz(out / "verdicts" / "anchor_verdicts.npz", verdict_code=res["verdict_cache"])
    lio.write_csv(out / "verdicts" / "decomposition.csv", decomposition, vd.DECOMPOSITION_FIELDS)
    lio.write_csv(out / "verdicts" / "carrier_equivalence.csv", carrier, vd.CARRIER_EQUIVALENCE_FIELDS)
    lio.write_csv(out / "verdicts" / "path_b_certificates.csv", path_b_rows, vd.PATH_B_FIELDS)
    # Open single-label Path B fibers: the full table, its subset without single-anchor DeepFashion fibers
    # (rechecked from outputs/ alone by verify/verify_tie_order.py) and a summary with counts and SHA-256 of both.
    lio.write_csv(out / "verdicts" / "path_b_open_witnesses.csv", path_b_open_rows, vd.PATH_B_OPEN_FIELDS)
    path_b_open_public = vd.path_b_open_public(path_b_open_rows)
    lio.write_csv(out / "verdicts" / "path_b_open_witnesses_public.csv", path_b_open_public, vd.PATH_B_OPEN_FIELDS)
    lio.write_json(out / "verdicts" / "path_b_open_summary.json", {
        **meta, **vd.path_b_open_summary(path_b_open_rows, path_b_open_public,
                                 lio.sha256_file(out / "verdicts" / "path_b_open_witnesses.csv"),
                                 lio.sha256_file(out / "verdicts" / "path_b_open_witnesses_public.csv"))})
    lio.write_json(out / "verdicts" / "witnesses.json", {
        **meta, "representation": vd.WITNESS_REPRESENTATION, "views": res["views"],
        "settings": {"threads": THREADS, "path_b_solver": cert.solver_settings(), "solver_check": check,
                     "witness_limits": {"paths_a_c_per_kind": vd.WITNESS_LIMIT_AC, "path_b": vd.WITNESS_LIMIT_B},
                     "verdict_array": {"name": "verdict_code", "dtype": str(res["verdict_cache"].dtype),
                                       "shape": list(res["verdict_cache"].shape),
                                       "sha256": lio.array_sha(res["verdict_cache"])}}})
    fig1 = vd.fig1_payload(ev)
    lio.write_json(out / "verdicts" / "fig1_instance.json", {**meta, **fig1})
    # Aggregate copy of the Fig. 1 selection: counts and rule only, without the anchor id or its evidence hash.
    lio.write_json(out / "verdicts" / "fig1_example.json", {
        **meta, **{k: fig1[k] for k in ("selection_uses_label", "selection_rule", "candidate_rows")},
        "selected": {k: fig1["selected"].get(k, {}) for k in ("facet_nonzero", "declared_W_family")}})
    counts = np.bincount(res["verdict_cache"].ravel(), minlength=3)
    print(f"[verdicts] done in {time.perf_counter() - t0:.1f}s; verdict counts "
          f"{dict(zip(('UNVERIFIED', 'UNSAFE', 'NONDETERMINED'), counts.tolist()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
