# SPDX-License-Identifier: MIT
"""Drift anatomy: where the materialized facet block of each anchor departs from core x Pi."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable

import pandas as pd

from .names import DATASETS

FIELDS = ["dataset", "quantity", "key", "value"]
FACET_UID_PREFIX = "hub:facet/"


def pi_from_edges(edge_rows: Iterable[dict[str, str]]) -> dict[str, str]:
    """Pi (core term -> facet) from the IN_FACET edges of the registry, facet namespace removed."""
    pi: dict[str, str] = {}
    for e in edge_rows:
        if e["edge_type"] != "IN_FACET":
            continue
        if e["src_uid"] in pi:
            raise SystemExit("IN_FACET is not a function")
        dst = e["dst_uid"]
        pi[e["src_uid"]] = dst[len(FACET_UID_PREFIX):] if dst.startswith(FACET_UID_PREFIX) else dst
    return pi


def anatomy(chunks: Iterable[pd.DataFrame], pi: dict[str, str], full_rows: Iterable[dict[str, str]]
            ) -> list[dict[str, Any]]:
    """Unsupported facet mass sum(max(f - g, 0)) with g = core x Pi and f the materialized facet block, rows with an
    excess or a deficit, row classes of the FULL-CONFORM failures, anchors with color counts by row class, and
    whether the anchors with an excess are exactly the FULL-CONFORM failures on Pi. No label column is read."""
    pi_fail: set[str] = set()
    row_class: dict[str, str] = {}
    class_counts: dict[str, Counter] = defaultdict(Counter)
    for r in full_rows:
        if "Pi" in json.loads(r["failed_maps"] or "[]"):
            pi_fail.add(r["row_id"])
        if r["row_class"]:
            row_class[r["row_id"]] = r["row_class"]
            class_counts[r["dataset"]][r["row_class"]] += 1

    excess: dict[str, Counter] = defaultdict(Counter)
    stats: dict[str, Counter] = defaultdict(Counter)
    color_rows: dict[str, Counter] = defaultdict(Counter)
    excess_rows: set[str] = set()
    for chunk in chunks:
        for ds, row_id, raw in zip(chunk["dataset"].tolist(), chunk["anchor_id"].tolist(),
                                   chunk["full_feats_json"].tolist()):
            feats = json.loads(raw)
            g: Counter = Counter()
            f: Counter = Counter()
            for k, v in feats.items():
                c = int(float(v))
                if c == 0:
                    continue
                if k.startswith("core::"):
                    g[pi[k[len("core::"):]]] += c
                elif k.startswith("facet::"):
                    f[k[len("facet::"):]] += c
            stats[ds]["rows"] += 1
            if f.get("color", 0) > 0:
                color_rows[ds][row_class.get(row_id, "registry_conforming")] += 1
            has_excess = has_deficit = False
            for phi in set(f) | set(g):
                d = f[phi] - g[phi]
                if d > 0:
                    excess[ds][phi] += d
                    has_excess = True
                elif d < 0:
                    has_deficit = True
            if has_excess:
                stats[ds]["rows_with_excess"] += 1
                excess_rows.add(row_id)
            if has_deficit:
                stats[ds]["rows_with_deficit"] += 1
    if set(stats) != set(DATASETS):
        raise SystemExit(f"drift anatomy found datasets {sorted(stats)}")

    out: list[dict[str, Any]] = []
    for ds in DATASETS:
        for q in ("rows", "rows_with_excess", "rows_with_deficit"):
            out.append({"dataset": ds, "quantity": q, "key": "", "value": stats[ds][q]})
        out.append({"dataset": ds, "quantity": "unsupported_facet_mass", "key": "total",
                    "value": sum(excess[ds].values())})
        for phi, v in excess[ds].items():
            out.append({"dataset": ds, "quantity": "unsupported_facet_mass", "key": phi, "value": v})
        for c, v in class_counts[ds].items():
            out.append({"dataset": ds, "quantity": "row_class", "key": c, "value": v})
        out.append({"dataset": ds, "quantity": "color_rows", "key": "total", "value": sum(color_rows[ds].values())})
        for c, v in color_rows[ds].items():
            out.append({"dataset": ds, "quantity": "color_rows_by_row_class", "key": c, "value": v})
    out.append({"dataset": "ALL", "quantity": "excess_rows_equal_pi_failures", "key": "",
                "value": int(excess_rows == pi_fail)})
    return out
