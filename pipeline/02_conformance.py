# SPDX-License-Identifier: MIT
"""Conformance step: label-free FULL-CONFORM per anchor, QUERY-CONFORM masks per declared view, drift anatomy."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca import conformance as conf  # noqa: E402
from lca import drift  # noqa: E402
from lca import io as lio  # noqa: E402
from lca import registry as reg  # noqa: E402
from lca.names import DATASETS  # noqa: E402


def load_views(cert_path: Path, view_path: Path) -> list[dict[str, str]]:
    """Certificate rows joined with the view schema, ordered by view_id."""
    cert = reg.read_columns(cert_path, reg.CERT_FIELDS)
    views = {r["view_id"]: r for r in reg.read_columns(view_path, reg.VIEW_FIELDS)}
    if len(views) != len(cert) or len({r["view_id"] for r in cert}) != len(cert):
        raise SystemExit("certificate_spec.csv and views.csv do not list the same unique views")
    merged = []
    for r in cert:
        v = views.get(r["view_id"])
        if v is None or v["task"] != r["task"]:
            raise SystemExit(f"view {r['view_id']} differs between certificate_spec.csv and views.csv")
        merged.append({**r, "view_group": v["view_group"], "exposed_blocks": v["exposed_blocks"]})
    return sorted(merged, key=lambda r: r["view_id"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=REPO / "data" / "hub")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    args = ap.parse_args()
    started = time.perf_counter()
    sums = reg.read_sha256sums(REPO / "data" / "hub" / "SHA256SUMS")
    reg_dir, conf_dir = args.outputs / "registry", args.outputs / "conformance"
    record = conf.AccessRecord()
    inputs_sha256 = {"anchors.csv.gz": sums["anchors.csv.gz"]}

    def read_upstream(step: str, name: str) -> Path:
        p = reg_dir / name
        digest = lio.sha256_file(p)
        inputs_sha256[f"registry/{name}"] = digest
        record.read(step, f"registry/{name}", digest)
        return p

    # ---------------- conformance: no label column is loaded ----------------
    if conf.label_columns(conf.CONFORMANCE_ANCHOR_COLUMNS):
        raise SystemExit("the conformance loader lists a label column")
    tables = reg.load_registry_tables(read_upstream("conformance", "term_to_facet.csv"),
                                      read_upstream("conformance", "facet_map.csv"))
    support = json.loads(read_upstream("conformance", "dataset_support.json").read_text(encoding="utf-8"))
    supports = {ds: {"C_i": set(support["datasets"][ds]["C_members"]),
                     "S_i": set(support["datasets"][ds]["S_members"]),
                     "w(C_i)": set(support["datasets"][ds]["W_img_members"])} for ds in DATASETS}
    anchors = lio.require_sha(args.inputs / "anchors.csv.gz", sums["anchors.csv.gz"])
    record.load("conformance", "anchors.csv.gz", conf.CONFORMANCE_ANCHOR_COLUMNS)
    full_res = conf.full_conform(reg.iter_anchor_chunks(anchors, conf.CONFORMANCE_ANCHOR_COLUMNS),
                                 tables["Pi"], tables["T"], tables["W"],
                                 {ds: supports[ds]["C_i"] for ds in DATASETS}, reg.DATASET_ROWS)
    full_path = conf_dir / "full_conform.csv.gz"
    full_csv_sha, full_gz_sha = lio.write_csv_gz(full_path, full_res["rows"], conf.FULL_CONFORM_FIELDS)
    record.write("conformance", "conformance/full_conform.csv.gz", full_gz_sha)
    lio.write_csv(conf_dir / "full_conform_summary.csv", full_res["summary"], conf.FULL_SUMMARY_FIELDS)
    record.write("conformance", "conformance/full_conform_summary.csv",
                 lio.sha256_file(conf_dir / "full_conform_summary.csv"))
    del full_res

    # ---------------- query: labels are loaded only after FULL-CONFORM is written ----------------
    views = load_views(read_upstream("query", "certificate_spec.csv"), read_upstream("query", "views.csv"))
    if lio.sha256_file(full_path) != full_gz_sha:
        raise SystemExit("full_conform.csv.gz changed after it was written")
    record.read("query", "conformance/full_conform.csv.gz", full_gz_sha)
    full_df = pd.read_csv(full_path, dtype=str, keep_default_na=False, na_filter=False)
    if full_df.columns.tolist() != conf.FULL_CONFORM_FIELDS:
        raise SystemExit(f"full_conform.csv.gz columns {full_df.columns.tolist()}")
    full = full_df["full_registry_conform"].astype(np.int8).to_numpy().astype(bool)
    record.load("query", "anchors.csv.gz", conf.QUERY_ANCHOR_COLUMNS)
    arrays = conf.query_arrays(reg.iter_anchor_chunks(anchors, conf.QUERY_ANCHOR_COLUMNS),
                               full_df["row_id"].tolist(), full_df["dataset"].tolist(),
                               tables["Pi"], tables["T"], tables["W"], supports)
    del full_df
    qres = conf.query_outputs(views, full, arrays, row_order_source=full_csv_sha)
    for name, rows, fields in (("conformance_summary.csv", qres["summary"], conf.QUERY_SUMMARY_FIELDS),
                               ("query_conform_index.csv", qres["index"], conf.QUERY_INDEX_FIELDS)):
        lio.write_csv(conf_dir / name, rows, fields)
        record.write("query", f"conformance/{name}", lio.sha256_file(conf_dir / name))
    lio.write_npz(conf_dir / "query_masks.npz", **qres["arrays"])
    record.write("query", "conformance/query_masks.npz", lio.sha256_file(conf_dir / "query_masks.npz"))

    # ---------------- data access checks ----------------
    written = record.first("conformance", "write_output", "conformance/full_conform.csv.gz")
    label_load = record.first("query", "load_columns", "anchors.csv.gz")
    loaded = {s: sorted({c for e in record.events if e["step"] == s and e["event"] == "load_columns"
                         for c in e["label_columns"]}) for s in ("conformance", "query")}
    checks = {
        "registry_label_columns_loaded": conf.label_columns(reg.SUPPORT_ANCHOR_COLUMNS + reg.CORE_TERMS_COLUMNS
                                                            + reg.EDGE_COLUMNS),
        "conformance_label_columns_loaded": loaded["conformance"],
        "query_label_columns_loaded": loaded["query"],
        "full_conform_written_at_seq": written,
        "query_label_load_at_seq": label_load,
        "query_labels_loaded_after_full_conform_written": label_load > written,
    }
    if checks["registry_label_columns_loaded"] or checks["conformance_label_columns_loaded"]:
        raise SystemExit("a label column is loaded before the query step")
    if not checks["query_labels_loaded_after_full_conform_written"]:
        raise SystemExit("labels are loaded before FULL-CONFORM is written")
    inputs_sha256["conformance/full_conform.csv.gz"] = full_gz_sha
    lio.write_json(args.outputs / "checks" / "data_access_checks.json", {
        "step": "conformance",
        "schema_version": "1",
        "inputs_sha256": inputs_sha256,
        "columns_loaded": {
            "registry": {"anchors.csv.gz": reg.SUPPORT_ANCHOR_COLUMNS, "core_terms.csv": reg.CORE_TERMS_COLUMNS,
                         "registry_edges.csv": reg.EDGE_COLUMNS},
            "conformance": {"anchors.csv.gz": conf.CONFORMANCE_ANCHOR_COLUMNS},
            "query": {"anchors.csv.gz": conf.QUERY_ANCHOR_COLUMNS},
        },
        "source_files_parsed": {"registry": {"label_rule.py": ["norm_str", "derive_family"]}},
        "events": record.events,
        "checks": checks,
    })

    # ---------------- drift anatomy: label-free, from the evidence blocks and the IN_FACET edges ----------------
    edges = lio.require_sha(args.inputs / "registry_edges.csv", sums["registry_edges.csv"])
    pi = drift.pi_from_edges(reg.read_columns(edges, reg.EDGE_COLUMNS))
    if pi != tables["Pi"]:
        raise SystemExit("IN_FACET edges and registry/term_to_facet.csv define different Pi maps")
    full_rows = pd.read_csv(full_path, dtype=str, keep_default_na=False, na_filter=False).to_dict("records")
    lio.write_csv(conf_dir / "drift_anatomy.csv",
                  drift.anatomy(reg.iter_anchor_chunks(anchors, conf.CONFORMANCE_ANCHOR_COLUMNS), pi, full_rows),
                  drift.FIELDS)
    del full_rows

    print(f"conformance: {len(views)} views x {len(full)} anchors; FULL-CONFORM rows {int(full.sum())}")
    print(f"conformance: done in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
