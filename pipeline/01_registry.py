# SPDX-License-Identifier: MIT
"""Registry step: release registry, certificate specification, protected query and dataset support."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca import io as lio  # noqa: E402
from lca import registry as reg  # noqa: E402
from lca.names import DATASETS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=REPO / "data" / "hub")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    args = ap.parse_args()
    started = time.perf_counter()
    sums = reg.read_sha256sums(REPO / "data" / "hub" / "SHA256SUMS")
    out = args.outputs / "registry"

    def require(name: str) -> Path:
        return lio.require_sha(args.inputs / name, sums[name])

    # Registry maps from the released terms, edges and label rule.
    rule_path = require("label_rule.py")
    core_path = require("core_terms.csv")
    edge_path = require("registry_edges.csv")
    reg.verify_label_rule(rule_path.read_text(encoding="utf-8"))
    core_rows = reg.read_columns(core_path, reg.CORE_TERMS_COLUMNS)
    edge_rows = reg.read_columns(edge_path, reg.EDGE_COLUMNS)
    if len(core_rows) != reg.N_CORE_TERMS or len(edge_rows) != reg.N_REGISTRY_EDGES:
        raise SystemExit(f"input sizes {len(core_rows)}/{len(edge_rows)} != {reg.N_CORE_TERMS}/{reg.N_REGISTRY_EDGES}")
    maps = reg.build_registry_maps(core_rows, edge_rows)
    if len(maps["facets"]) != reg.N_FACETS:
        raise SystemExit(f"registry facets {len(maps['facets'])} != {reg.N_FACETS}")

    views = reg.declared_views()
    cert_rows, view_rows = reg.build_certificate_spec(views)
    registry_inputs = {n: sums[n] for n in ("core_terms.csv", "registry_edges.csv", "label_rule.py")}
    document = reg.build_registry_document(maps, len(core_rows), len(edge_rows), views, cert_rows, view_rows,
                                           registry_inputs)
    lio.write_yaml(out / "registry.yaml", document)
    lio.write_yaml(out / "protected_query.yaml", {"step": "registry", "schema_version": "1",
                                                  "inputs_sha256": {"label_rule.py": sums["label_rule.py"]},
                                                  **reg.PROTECTED_QUERY})
    lio.write_csv(out / "certificate_spec.csv", cert_rows, reg.CERT_FIELDS)
    lio.write_csv(out / "views.csv", view_rows, reg.VIEW_FIELDS)
    lio.write_csv(out / "facet_map.csv", [{"facet": f, "term_type": maps["T"][f], "family": maps["W"][f]}
                                          for f in maps["facets"]], reg.FACET_MAP_FIELDS)
    lio.write_csv(out / "term_to_facet.csv", maps["term_to_facet_rows"], reg.TERM_TO_FACET_FIELDS)

    # Dataset support from the released evidence (no label column) and the registry tables just written.
    tables = reg.load_registry_tables(out / "term_to_facet.csv", out / "facet_map.csv")
    anchors = require("anchors.csv.gz")
    scan = reg.scan_support(reg.iter_anchor_chunks(anchors, reg.SUPPORT_ANCHOR_COLUMNS), tables["W"])
    if dict(scan["rows"]) != reg.DATASET_ROWS:
        raise SystemExit(f"anchor rows per dataset {dict(scan['rows'])} != {reg.DATASET_ROWS}")
    support_inputs = {"anchors.csv.gz": sums["anchors.csv.gz"],
                      "registry/term_to_facet.csv": lio.sha256_file(out / "term_to_facet.csv"),
                      "registry/facet_map.csv": lio.sha256_file(out / "facet_map.csv")}
    res = reg.build_support_outputs(scan, tables, support_inputs)
    lio.write_json(out / "dataset_support.json", res["support"])
    lio.write_json(out / "populations.json", res["populations"])
    lio.write_csv(out / "declared_vs_materialized.csv", res["diagnostics"], reg.DIAG_FIELDS)

    sup = res["support"]["datasets"]
    print("registry: facets", len(maps["facets"]), "terms", len(core_rows), "views", len(views))
    for ds in DATASETS:
        print(f"registry: {ds} rows {sup[ds]['n_rows']} |C_i| {sup[ds]['n_dataset_active_terms']} "
              f"|S_i| {sup[ds]['n_dataset_support']} |w(C_i)| {sup[ds]['n_dataset_family_image']}")
    print(f"registry: done in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
