# SPDX-License-Identifier: MIT
"""Tie-order step: the UNVERIFIED mass that a declared event order could turn into UNSAFE mass, with a strictly winning
rival state, re-checked in integer arithmetic, for every other open Path B fiber, and the admission decisions that such
an order could change (Sec. III after Proposition 1; Sec. VI-A, Open fibers need a third verdict)."""
import os

THREADS = 1  # numerical libraries and HiGHS run single-threaded
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "HIGHS_THREADS"):
    os.environ[_k] = str(THREADS)

import argparse
import csv
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca import certificates as cert  # noqa: E402
from lca import io as lio  # noqa: E402
from lca import support as sp  # noqa: E402
from lca import tie_order as to  # noqa: E402
from lca import verdicts as vd  # noqa: E402

RESULT_FIELDS = ["view", "dataset", "basis", "path", "fibers", "m^U", "m^N", "m^V_obs", "m^V_defect", "near_pure", "purity"]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=REPO / "data" / "hub")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    a = ap.parse_args()
    t0 = time.perf_counter()
    sums = vd.read_sha256sums(REPO / "data" / "hub" / "SHA256SUMS")
    anchors = a.inputs / "anchors.csv.gz"
    out = a.outputs
    if not (out / "verdicts" / "verdicts.csv").is_file():
        raise SystemExit("missing upstream output verdicts/verdicts.csv")

    check = cert.solver_self_check()
    reg = vd.load_registry(out)
    conf = vd.load_conformance(out, reg)
    ev = vd.load_evidence(anchors, sums["anchors.csv.gz"], reg, conf, select_fig1=False)
    print(f"[tie_order] evidence loaded ({time.perf_counter() - t0:.1f}s)", flush=True)
    res = to.run(reg, conf, ev, log=lambda s: print(s, flush=True))

    # the recomputed verdicts must reproduce the verdict step exactly
    frozen = {(r["view"], r["dataset"]): r for r in _read(out / "verdicts" / "verdicts.csv")}
    same_rows = len(res["verdict_rows"]) == len(frozen) and all(
        {k: lio._cell(r[k]) for k in RESULT_FIELDS} == frozen[(r["view"], r["dataset"])] for r in res["verdict_rows"])
    if not same_rows:
        raise SystemExit("the recomputed verdicts do not reproduce the verdict step")

    dst = out / "tie_order"
    lio.write_csv(dst / "tie_order_views.csv", res["view_rows"], to.VIEW_FIELDS)
    lio.write_csv(dst / "tie_order_admission.csv", res["admission_rows"], to.ADMISSION_FIELDS)
    lio.write_csv(dst / "tie_order_fibers_public.csv", res["fiber_rows"], to.FIBER_FIELDS)
    inputs_sha = {"anchors.csv.gz": sums["anchors.csv.gz"], **reg["sha256"], **conf["sha256"],
                  "verdicts/verdicts.csv": lio.sha256_file(out / "verdicts" / "verdicts.csv")}
    lio.write_json(dst / "tie_order.json", {
        "step": "tie_order", "schema_version": "1", "inputs_sha256": inputs_sha,
        "question": "which UNVERIFIED mass could a declared event order turn into UNSAFE mass, and which admission "
                    "decisions could it change",
        "rule": {"declared_order": "an order that the released labels satisfy fixes z, so it only removes states; "
                                   "UNSAFE and NONDETERMINED verdicts and query defects do not depend on z",
                 "path_A": "every single-label tie fiber is tie-only (all states of a fiber share the carrier)",
                 "path_B": "an open single-label fiber is tie-only unless some rival strictly beats the observed label "
                           "at a fiber state; that state is re-checked in integer arithmetic",
                 "path_C": "no certificate path, so no fiber can become UNSAFE",
                 "public_fibers": "every open Path B fiber except the DF fibers with one query-conforming anchor, with its "
                                 "strictly winning rival state or, if tie-only, the exact optimum of every rival program",
                 "admission": "Problem 1: admitted at eps iff m^U = 0 and m^V <= eps in each dataset; a decision is "
                              "fixed under every declared order when the view has UNSAFE mass or no tie-only rows"},
        "counts": res["counts"], "epsilons": list(sp.EPSILONS),
        "recomputed_verdicts_reproduce_verdict_step": same_rows,
        "settings": {"threads": THREADS, "path_b_solver": cert.solver_settings(), "solver_check": check}})
    print(f"[tie_order] done in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
