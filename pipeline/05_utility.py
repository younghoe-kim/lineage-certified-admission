# SPDX-License-Identifier: MIT
"""Utility step: evaluator definition, probes, utility bounds, controls, the parameter-free registry rule and the
verdict-class checks."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Single-threaded numerical libraries (set before numpy is imported).
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS")
for _k in _THREAD_ENV:
    os.environ[_k] = "1"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from lca import bounds, io, probes, verdict_checks  # noqa: E402
from lca import evaluation as ev  # noqa: E402

STEP = "utility"
UPSTREAM = (
    "registry/view_projection.csv",
    "registry/facet_map.csv",
    "registry/term_to_facet.csv",
    "registry/populations.json",
    "registry/certificate_spec.csv",
    "conformance/full_conform.csv.gz",
    "conformance/query_masks.npz",
    "verdicts/anchor_verdicts.npz",
    "verdicts/anchor_verdicts_index.csv",
)
# Header of source_candidates.csv: estimator sources outside this code base (none are used).
SOURCE_CANDIDATE_FIELDS = ["role", "basename", "origin_type", "origin", "member", "relative_hint",
                           "preferred_suffix_match", "size_bytes", "sha256"]
EXPECTED_PI = 2715
EXPECTED_FACETS = 27


def read_sums(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            out[name.strip()] = h
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_registry(out: Path) -> dict:
    Pi: dict[str, str] = {}
    for r in read_csv(out / "registry/term_to_facet.csv"):
        if r["match"] != "1":
            raise SystemExit("term-to-facet map has a nonmatching row")
        Pi[r["core_term_uid"]] = r["core_facet"]
    W = {r["facet"]: r["family"] for r in read_csv(out / "registry/facet_map.csv")}
    if len(Pi) != EXPECTED_PI or len(W) != EXPECTED_FACETS:
        raise SystemExit("registry maps differ from the declared sizes")
    pops = json.loads((out / "registry/populations.json").read_text(encoding="utf-8"))
    members = pops["block_coordinate_members"]
    coords = {"facet": list(members["facet::"]), "family": list(members["family::"])}
    certs = probes.rule_certificates(read_csv(out / "registry/certificate_spec.csv"))
    return {"Pi": Pi, "W": W, "coords": coords, "certificates": certs}


def load_masks(out: Path, cache: dict) -> tuple[dict, dict, list[dict[str, str]], np.ndarray]:
    n = len(cache["row_id"])
    index = read_csv(out / "verdicts/anchor_verdicts_index.csv")
    with np.load(out / "verdicts/anchor_verdicts.npz", allow_pickle=False) as z:
        code = np.asarray(z["verdict_code"])
    if code.shape != (ev.EXPECTED_VIEWS, n) or code.dtype != np.uint8 or not np.isin(code, (0, 1, 2)).all():
        raise SystemExit("verdict array has an unexpected shape, type or code")
    if sorted(int(r["array_index"]) for r in index) != list(range(ev.EXPECTED_VIEWS)):
        raise SystemExit("verdict index does not cover the 32 views")
    if any(int(r["row_count"]) != n for r in index):
        raise SystemExit("verdict index row count differs from the anchors")
    with np.load(out / "conformance/query_masks.npz", allow_pickle=False) as z:
        qviews = [str(v) for v in z["view_ids"]]
        qmask = np.asarray(z["query_conform"], dtype=bool)
        dcode = np.asarray(z["dataset_code"])
    if qmask.shape != (ev.EXPECTED_VIEWS, n):
        raise SystemExit("query-conformance masks have an unexpected shape")
    if not np.array_equal(dcode, np.where(cache["dataset"] == "DF", 0, 1)):
        raise SystemExit("query-conformance row order differs from the anchors")
    by_view = {r["view_id"]: int(r["array_index"]) for r in index}
    if by_view != {v: i for i, v in enumerate(qviews)}:
        raise SystemExit("verdict index and query-conformance masks list the views differently")
    verdicts = {v: code[i] for v, i in by_view.items()}
    query = {v: qmask[i] for v, i in by_view.items()}
    return verdicts, query, index, code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=REPO / "data/hub")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    a = ap.parse_args()
    if set(_THREAD_ENV) != set(ev.THREAD_ENV):
        raise SystemExit("thread environment list differs from the evaluator settings")
    t0 = time.perf_counter()
    inp, out = a.inputs, a.outputs
    dst = out / STEP

    sums = read_sums(REPO / "data/hub/SHA256SUMS")
    anchors = io.require_sha(inp / "anchors.csv.gz", sums["anchors.csv.gz"])
    for rel in UPSTREAM:
        if not (out / rel).is_file():
            raise SystemExit(f"missing upstream output {rel}")
    inputs_sha256 = {"anchors.csv.gz": sums["anchors.csv.gz"]}
    inputs_sha256.update({rel: io.sha256_file(out / rel) for rel in UPSTREAM})

    io.write_json(dst / "evaluator.json", {
        "step": STEP, "schema_version": "1", "inputs_sha256": inputs_sha256,
        "semantics": ev.SEMANTICS, "coverage_definitions": ev.COVERAGE_DEFINITIONS, "settings": ev.settings()})
    io.write_csv(dst / "source_candidates.csv", [], SOURCE_CANDIDATE_FIELDS)

    cache = ev.load_feature_cache(anchors, out / "conformance/full_conform.csv.gz")
    views = ev.load_views(out / "registry/view_projection.csv")
    facets = ev.facet_order(out / "registry/facet_map.csv")
    registry = load_registry(out)
    verdicts, query, verdict_index, verdict_code = load_masks(out, cache)
    print(f"[utility] inputs loaded: rows={len(cache['row_id'])} dims={len(cache['features'])} "
          f"nnz={cache['X'].nnz} ({time.perf_counter() - t0:.1f}s)", flush=True)

    def timed(name, fn, *args):
        t = time.perf_counter()
        res = fn(*args)
        print(f"[utility] {name}: {time.perf_counter() - t:.1f}s", flush=True)
        return res

    io.write_csv(dst / "tolerance_sensitivity.csv", timed("tolerance", probes.tolerance_table, cache, views),
                 probes.TOLERANCE_FIELDS)
    io.write_csv(dst / "calibration.csv", timed("calibration", probes.calibration_table, cache, facets),
                 probes.CALIBRATION_FIELDS)
    io.write_csv(dst / "support_check.csv", timed("support check", probes.support_check_table, cache, views),
                 probes.SUPPORT_FIELDS)
    probe_rows, cosine_preds = timed("probes", probes.probe_table, cache, views)
    io.write_csv(dst / "probes.csv", probe_rows, probes.PROBE_FIELDS)
    factorial = timed("factorial", probes.factorial_table, cache, views)
    io.write_csv(dst / "factorial_8cell.csv", factorial, probes.FACTORIAL_FIELDS)
    io.write_csv(dst / "volume_control.csv", timed("volume control", probes.volume_table, cache, views),
                 probes.VOLUME_FIELDS)
    io.write_csv(dst / "type_weight.csv", timed("type weight", probes.type_weight_table, cache, views),
                 probes.TYPE_WEIGHT_FIELDS)
    matrix, summary = timed("verdict classes", probes.verdict_class_tables, cache, views, verdict_code,
                            verdict_index, cosine_preds)
    io.write_csv(dst / "cosine_determination_matrix.csv", matrix, probes.DETERMINATION_FIELDS)
    io.write_csv(dst / "verdict_class_accuracy.csv", summary, probes.VERDICT_CLASS_FIELDS)
    bound_rows = timed("bounds", bounds.bounds_table, cache, views)
    io.write_csv(dst / "bounds.csv", bound_rows, bounds.BOUND_FIELDS)
    checks = bounds.probe_bound_check_table(probe_rows, factorial, bound_rows, ev.BOUND_SLACK)
    io.write_csv(dst / "probe_bound_check.csv", checks, bounds.CHECK_FIELDS)

    # The registry rule reads its comparators back from this step's probes.csv and bounds.csv.
    probe_file, bound_file = read_csv(dst / "probes.csv"), read_csv(dst / "bounds.csv")
    full_rows, shared_rows = timed("registry rule", probes.registry_rule_tables, cache, registry["certificates"],
                                   registry["Pi"], registry["W"], registry["coords"], query, verdicts,
                                   probe_file, bound_file)
    io.write_csv(dst / "registry_rule_fullview.csv", full_rows, probes.FULLVIEW_FIELDS)
    io.write_csv(dst / "registry_rule.csv", shared_rows, probes.RULE_FIELDS)
    io.write_csv(dst / "lookup_coverage.csv", probes.lookup_coverage_table(probe_file), probes.LOOKUP_COVERAGE_FIELDS)
    io.write_csv(dst / "verdict_checks.csv",
                 verdict_checks.check_rows(cache, verdicts, read_csv(dst / "cosine_determination_matrix.csv")),
                 verdict_checks.FIELDS)
    print(f"[utility] done: {time.perf_counter() - t0:.1f}s; probe-bound checks={len(checks)}, violations=0",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
