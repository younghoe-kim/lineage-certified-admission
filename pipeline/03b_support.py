# SPDX-License-Identifier: MIT
"""Support step: every registered view audited on the released supports and on three counterfactual coordinate sets
(their union, their intersection, each released support with color added), with exact checks of every Path B result
(Remark 1; extended audit details Section E)."""
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
    for rel in ("verdicts/verdicts.csv", "verdicts/path_b_certificates.csv"):
        if not (out / rel).is_file():
            raise SystemExit(f"missing upstream output {rel}")

    check = cert.solver_self_check()
    reg = vd.load_registry(out)
    conf = vd.load_conformance(out, reg)
    ev = vd.load_evidence(anchors, sums["anchors.csv.gz"], reg, conf, select_fig1=False)
    print(f"[support] evidence loaded ({time.perf_counter() - t0:.1f}s)", flush=True)
    res = sp.run(reg, conf, ev, anchors, log=lambda s: print(s, flush=True))

    # the released coordinate sets must reproduce the verdict step exactly
    frozen = {(r["view"], r["dataset"]): r for r in _read(out / "verdicts" / "verdicts.csv")}
    released = [r for r in res["verdict_rows"] if r["regime"] == "released"]
    same_rows = len(released) == len(frozen) and all(
        {k: lio._cell(r[k]) for k in RESULT_FIELDS} == frozen[(r["view"], r["dataset"])] for r in released)
    frozen_certs = _read(out / "verdicts" / "path_b_certificates.csv")
    mine = [{k: lio._cell(r[k]) for k in vd.PATH_B_FIELDS} for r in res["certificate_rows"] if r["regime"] == "released"]
    key = lambda r: tuple(r[k] for k in vd.PATH_B_FIELDS)  # noqa: E731
    same_certs = sorted(mine, key=key) == sorted(frozen_certs, key=key)
    if not (same_rows and same_certs):
        raise SystemExit("the released coordinate sets do not reproduce the verdict step")

    dst = out / "support"
    lio.write_csv(dst / "support_regimes.csv", res["regime_rows"], sp.REGIME_FIELDS)
    lio.write_csv(dst / "support_verdicts.csv", res["verdict_rows"], sp.VERDICT_FIELDS)
    lio.write_csv(dst / "support_transitions.csv", res["transition_rows"], sp.TRANSITION_FIELDS)
    lio.write_csv(dst / "support_admission.csv", res["admission_rows"], sp.ADMISSION_FIELDS)
    lio.write_csv(dst / "support_certificates.csv", res["certificate_rows"], sp.CERTIFICATE_FIELDS)
    inputs_sha = {"anchors.csv.gz": sums["anchors.csv.gz"], **reg["sha256"], **conf["sha256"],
                  "verdicts/verdicts.csv": lio.sha256_file(out / "verdicts" / "verdicts.csv"),
                  "verdicts/path_b_certificates.csv": lio.sha256_file(out / "verdicts" / "path_b_certificates.csv")}
    lio.write_json(dst / "support.json", {
        "step": "support", "schema_version": "1", "inputs_sha256": inputs_sha, "regimes": list(sp.REGIMES),
        "fixed": ["registry maps", "C_i", "w(C_i)", "FULL-CONFORM", "evidence", "labels", "view projections",
                  "certificate paths", "Path B solver"],
        "varied": "facet-basis support S_i; QUERY-CONFORM recomputed on each coordinate set",
        "color": res["color"], "counts": res["counts"], "epsilons": list(sp.EPSILONS),
        "released_reproduces_verdict_step": {"verdict_rows": same_rows, "path_b_certificates": same_certs},
        "settings": {"threads": THREADS, "path_b_solver": cert.solver_settings(), "solver_check": check}})
    print(f"[support] done in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
