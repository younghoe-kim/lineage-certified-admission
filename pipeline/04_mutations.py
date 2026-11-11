# SPDX-License-Identifier: MIT
"""Mutations step: five controlled auditor mutations (M1-M5) and the M5 label-domain cross-check."""
import os

THREADS = 1  # numerical libraries and HiGHS run single-threaded
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "HIGHS_THREADS"):
    os.environ[_k] = str(THREADS)

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from lca import certificates as cert  # noqa: E402
from lca import io as lio  # noqa: E402
from lca import mutations as mu  # noqa: E402
from lca import verdicts as vd  # noqa: E402
from lca.names import DATASETS  # noqa: E402

RESULT_FIELDS = ["mutation_id", "pass", "primary_check", "detail"]
CHANGED_FIELDS = ["mutation_id", "task", "row_id"]


def load_verdicts(out: Path, reg: dict, conf: dict) -> tuple[np.ndarray, dict, dict]:
    """Verdicts step outputs: the row-level verdict array and the per-view defect counts."""
    files = {"anchor_verdicts.npz": out / "verdicts" / "anchor_verdicts.npz",
             "anchor_verdicts_index.csv": out / "verdicts" / "anchor_verdicts_index.csv",
             "witnesses.json": out / "verdicts" / "witnesses.json"}
    with files["anchor_verdicts_index.csv"].open(encoding="utf-8", newline="") as fh:
        index = sorted(csv.DictReader(fh), key=lambda r: int(r["array_index"]))
    if [r["view_id"] for r in index] != [r["view_id"] for r in reg["cert"]]:
        raise SystemExit("verdict index order differs from certificate order")
    with np.load(files["anchor_verdicts.npz"], allow_pickle=False) as z:
        verdict = z["verdict_code"]
    if verdict.shape != (len(reg["cert"]), conf["n"]):
        raise SystemExit("verdict array shape differs from views x rows")
    wit = json.loads(files["witnesses.json"].read_text(encoding="utf-8"))
    defects = {(vid, ds): int(v["datasets"][ds]["metrics_counts"]["unverified_defect_rows"])
               for vid, v in wit["views"].items() for ds in DATASETS}
    return verdict, defects, {f"verdicts/{k}": lio.sha256_file(p) for k, p in files.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=REPO / "data" / "hub")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    a = ap.parse_args()
    t0 = time.perf_counter()
    sums = vd.read_sha256sums(REPO / "data" / "hub" / "SHA256SUMS")
    out = a.outputs
    reg = vd.load_registry(out)
    conf = vd.load_conformance(out, reg)
    verdict, defects, verdict_sha = load_verdicts(out, reg, conf)
    ev = vd.load_evidence(a.inputs / "anchors.csv.gz", sums["anchors.csv.gz"], reg, conf)
    print(f"[mutations] evidence loaded ({time.perf_counter() - t0:.1f}s)", flush=True)

    changed: list[dict] = []
    l0 = [r["view_id"] for r in reg["cert"]].index("l0_facet__facet")
    m1, r1, b_mat, b_inv = mu.mutation_m1(reg, conf, ev, verdict[l0])
    print(f"[mutations] M1 pass={r1['pass']}", flush=True)
    m2, r2 = mu.mutation_m2(reg, conf, ev, changed)
    print(f"[mutations] M2 pass={r2['pass']}", flush=True)
    m3, r3 = mu.mutation_m3(reg, conf, ev)
    print(f"[mutations] M3 pass={r3['pass']}", flush=True)
    m4, r4 = mu.mutation_m4(reg, conf, ev)
    print(f"[mutations] M4 pass={r4['pass']}", flush=True)
    m5, r5, mutated, valid = mu.mutation_m5(reg, conf, ev, defects, changed)
    print(f"[mutations] M5 pass={r5['pass']}", flush=True)
    detail, settings, xsummary = mu.m5_label_domain_check(reg, conf, ev, mutated, valid)
    print(f"[mutations] M5 label-domain cross-check: {xsummary['bits_checked']} bits, "
          f"mismatches={xsummary['baseline_component_mismatches']}+{xsummary['label_rule_vs_audit_mismatches']}", flush=True)
    results = [r1, r2, r3, r4, r5]
    if sum(int(r["pass"]) for r in results) != 5:
        raise SystemExit(f"mutations passed {sum(int(r['pass']) for r in results)}/5")

    inputs_sha = {"anchors.csv.gz": sums["anchors.csv.gz"], **reg["sha256"], **conf["sha256"], **verdict_sha}
    registry_file = out / "registry" / "registry.yaml"
    inputs_sha["registry/registry.yaml"] = lio.sha256_file(registry_file)
    meta = {"step": "mutations", "schema_version": "1", "inputs_sha256": inputs_sha}
    mdir = out / "mutations"
    lio.write_csv(mdir / "mutation_results.csv", results, RESULT_FIELDS)
    logical_sha, gz_sha = lio.write_csv_gz(mdir / "changed_rows.csv.gz", changed, CHANGED_FIELDS)
    base_registry = yaml.safe_load(registry_file.read_text(encoding="utf-8"))
    base_registry.update(meta)
    base_registry["mutation_extension"] = {
        "mutation_id": "M1", "block": "y", "materialization": "y = facet:: * B",
        "registered_edge": {"source_block": "y", "target_block": "facet::", "matrix": "B_inverse"},
        "B": b_mat.tolist(), "B_inverse": b_inv.tolist(), "basis_support": "Supp_i(y)=S_i",
        "seed": m1["seed"]}
    lio.write_yaml(mdir / "m1_registry.yaml", base_registry)
    lio.write_csv(mdir / "m1_certificate_spec.csv", [m1["certificate_spec"]], mu.M1_CERT_FIELDS)
    lio.write_csv(mdir / "m5_label_domain_check.csv", settings, mu.M5_SETTING_FIELDS)
    lio.write_csv(mdir / "m5_view_checks.csv", detail, mu.M5_VIEW_FIELDS)
    lio.write_json(mdir / "mutations.json", {
        **meta, "seed_rule": mu.SEED_RULE, "design_note": mu.DESIGN_NOTE,
        "mutations": {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5},
        "changed_rows": {"row_selective_list": "mutations/changed_rows.csv.gz", "logical_csv_sha256": logical_sha,
                         "gzip_sha256": gz_sha, "global_mutations": mu.GLOBAL_MUTATIONS},
        "settings": {"threads": THREADS, "path_b_solver": cert.solver_settings(),
                     "m5_label_domain_check": xsummary}})
    print(f"[mutations] done in {time.perf_counter() - t0:.1f}s; 5/5 pass", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
