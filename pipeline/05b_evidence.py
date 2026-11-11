# SPDX-License-Identifier: MIT
"""Evidence step: refit decomposition into rows and shared coordinates, verdict classes of the correct family-3
predictions, coordinates dropped by the shared projection, the 999-draw volume null, the registry rule and the cosine
probe label by label, and the type and family blocks of the drifted anchors (Sections VI-A and VI-B; extended audit details
Sections C and D)."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

# Single-threaded numerical libraries (set before numpy is imported).
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca import evaluation as ev  # noqa: E402
from lca import evidence as evd  # noqa: E402
from lca import io  # noqa: E402
from lca import probes  # noqa: E402

UPSTREAM = ("registry/view_projection.csv", "registry/term_to_facet.csv", "registry/facet_map.csv",
            "registry/populations.json", "registry/certificate_spec.csv", "conformance/full_conform.csv.gz",
            "conformance/query_masks.npz", "conformance/conformance_summary.csv", "verdicts/anchor_verdicts.npz",
            "verdicts/anchor_verdicts_index.csv", "utility/volume_control.csv", "utility/probes.csv",
            "utility/registry_rule.csv")


def read_sums(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, n = line.split(None, 1)
            out[n.strip()] = h
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def registry_maps(out: Path) -> dict:
    Pi = {}
    for r in read_csv(out / "registry/term_to_facet.csv"):
        if r["match"] != "1":
            raise SystemExit("term-to-facet map has a nonmatching row")
        Pi[r["core_term_uid"]] = r["core_facet"]
    fmap = read_csv(out / "registry/facet_map.csv")
    T = {r["facet"]: r["term_type"] for r in fmap}
    W = {r["facet"]: r["family"] for r in fmap}
    members = json.loads((out / "registry/populations.json").read_text(encoding="utf-8"))["block_coordinate_members"]
    coords = {"facet": list(members["facet::"]), "family": list(members["family::"])}
    certs = probes.rule_certificates(read_csv(out / "registry/certificate_spec.csv"))
    return {"Pi": Pi, "T": T, "W": W, "coords": coords, "certificates": certs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=REPO / "data/hub")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    a = ap.parse_args()
    t0 = time.perf_counter()
    out, dst = a.outputs, a.outputs / "evidence"
    sums = read_sums(REPO / "data/hub/SHA256SUMS")
    anchors = io.require_sha(a.inputs / "anchors.csv.gz", sums["anchors.csv.gz"])
    for rel in UPSTREAM:
        if not (out / rel).is_file():
            raise SystemExit(f"missing upstream output {rel}")
    cache = ev.load_feature_cache(anchors, out / "conformance/full_conform.csv.gz")
    views = ev.load_views(out / "registry/view_projection.csv")
    verdicts = evd.load_verdicts(out, cache)
    print(f"[evidence] inputs loaded ({time.perf_counter() - t0:.1f}s)", flush=True)

    sep = evd.refit_separation(cache, views)
    comp = evd.verdict_class_composition(cache, views, verdicts)
    drop = evd.shared_basis_drop(cache)
    print(f"[evidence] separation, composition, drop ({time.perf_counter() - t0:.1f}s)", flush=True)
    reg = registry_maps(out)
    label_rows, label_summary = evd.metric_labels(cache, views, reg["Pi"], reg["W"], reg["coords"], reg["certificates"])
    drift = evd.drift_type_mechanism(cache, reg["Pi"], reg["T"], reg["W"])
    print(f"[evidence] metric labels, drift mechanism ({time.perf_counter() - t0:.1f}s)", flush=True)
    null = evd.volume_null(cache, views)
    print(f"[evidence] volume null: {len(null)} draws ({time.perf_counter() - t0:.1f}s)", flush=True)

    # label-by-label Macro-F1 must reproduce the scores of the utility step
    rule = read_csv(out / "utility/registry_rule.csv")
    cos = [r for r in read_csv(out / "utility/probes.csv") if r["status"] == "OK" and r["probe"] == "cosine"
           and r["training_protocol"] == "fixed_dataset_refit" and r["evaluation_scope"] == "full_conform_target"]
    for s in label_summary:
        src = rule if s["predictor"] == "registry_rule" else cos
        hit = [r for r in src if r["view_id"] == s["view_id"] and r["direction"] == s["direction"]]
        if len(hit) != 1 or io._cell(s["macro_f1"]) != hit[0]["macro_f1"]:
            raise SystemExit(f"label-by-label Macro-F1 differs from the utility step: {s['view_id']} {s['direction']}")
    # the drifted anchors are the FULL-CONFORM failures
    cs = read_csv(out / "conformance/conformance_summary.csv")
    for d in drift:
        fails = [r for r in cs if r["dataset"] == d["dataset"] and r["view_id"] == "core_plus_type__family"]
        if len(fails) != 1 or int(fails[0]["query_defect_rows"]) != d["drifted_rows"]:
            raise SystemExit(f"drifted rows differ from the FULL-CONFORM failures in {d['dataset']}")

    # the first draws are the volume control of the utility step
    with (out / "utility/volume_control.csv").open(encoding="utf-8", newline="") as fh:
        control = {int(r["seed"]): r for r in csv.DictReader(fh)}
    for r in null:
        if r["seed"] in control and io._cell(r["macro_f1"]) != control[r["seed"]]["macro_f1"]:
            raise SystemExit(f"volume null draw {r['seed']} differs from the volume control")
    if not set(control) <= {r["seed"] for r in null}:
        raise SystemExit("the volume null does not contain the volume control seeds")

    io.write_csv(dst / "refit_separation.csv", sep, evd.SEPARATION_FIELDS)
    io.write_csv(dst / "verdict_class_composition.csv", comp, evd.COMPOSITION_FIELDS)
    io.write_csv(dst / "shared_basis_drop.csv", drop, evd.DROP_FIELDS)
    io.write_csv(dst / "volume_null.csv", null, evd.NULL_FIELDS)
    io.write_csv(dst / "metric_label_f1.csv", label_rows, evd.LABEL_F1_FIELDS)
    io.write_csv(dst / "metric_label_sensitivity.csv", label_summary, evd.LABEL_SUMMARY_FIELDS)
    io.write_csv(dst / "drift_type_mechanism.csv", drift, evd.DRIFT_FIELDS)
    io.write_json(dst / "evidence.json", {
        "step": "evidence", "schema_version": "1",
        "inputs_sha256": {"anchors.csv.gz": sums["anchors.csv.gz"],
                          **{rel: io.sha256_file(out / rel) for rel in UPSTREAM}},
        "settings": {**ev.settings(), "volume_null_draws": evd.NULL_DRAWS,
                     "volume_null_seeds": f"0..{evd.NULL_DRAWS - 1}", "composition_views": list(evd.COMPOSITION_VIEWS),
                     "drop_bases": list(evd.BASES),
                     "metric_label_min_targets": evd.MIN_TARGETS, "metric_label_small_targets": evd.SMALL_TARGETS}})
    print(f"[evidence] done: {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
