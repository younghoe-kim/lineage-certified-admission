# SPDX-License-Identifier: MIT
"""FULL-CONFORM (registry reproduction per anchor, label-free) and QUERY-CONFORM masks per declared view."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

import numpy as np
import pandas as pd

from lca.names import DATASETS
from lca.registry import LABEL_PREFIX, exact_count

# Input column lists (every anchor loader reads exactly these columns).
CONFORMANCE_ANCHOR_COLUMNS = ["dataset", "anchor_id", "full_feats_json"]
QUERY_ANCHOR_COLUMNS = ["dataset", "anchor_id", "full_feats_json", "gold_facet", "gold_family"]
QUERY_LABEL_COLUMNS = ["gold_facet", "gold_family"]

FULL_CONFORM_FIELDS = ["dataset", "row_id", "full_registry_conform", "failed_maps", "row_class"]
FULL_SUMMARY_FIELDS = ["dataset", "rows", "conform_rows", "defect_rows", "conform_rate", "Pi_fail_rows",
                       "tau_fail_rows", "w_fail_rows", "support_fail_rows", "no_rep_core", "ghost_color",
                       "unsupported_category_unit", "conform_without_rep_core", "core_zero_rows",
                       "facet_zero_rows", "family_zero_rows", "type_zero_rows"]
ROW_CLASSES = ("no_rep_core", "ghost_color", "unsupported_category_unit")
QUERY_SUMMARY_FIELDS = ["dataset", "task", "view_id", "view_group", "certificate_path", "basis", "basis_support",
                        "required_maps", "rows", "query_conform_rows", "query_defect_rows", "outside_Drel_rows",
                        "gold_is_carrier_max_false_rows", "carrier_zero_rows", "support_fail_rows",
                        "Pi_fail_rows", "T_fail_rows", "W_fail_rows"]
QUERY_INDEX_FIELDS = ["array_index", "view_id", "task", "view_group", "certificate_path", "basis",
                      "basis_support", "required_maps", "row_order_source", "row_count"]


def label_columns(columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c.startswith(LABEL_PREFIX)]


class AccessRecord:
    """Ordered record of the column lists each step passes to its loaders and of the outputs it writes."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def _add(self, **event: Any) -> int:
        event["seq"] = len(self.events) + 1
        self.events.append(event)
        return event["seq"]

    def load(self, step: str, source: str, columns: list[str]) -> int:
        return self._add(step=step, event="load_columns", source=source, columns=list(columns),
                         label_columns=label_columns(columns))

    def read(self, step: str, source: str, sha256: str) -> int:
        return self._add(step=step, event="read_output", source=source, sha256=sha256)

    def write(self, step: str, output: str, sha256: str) -> int:
        return self._add(step=step, event="write_output", source=output, sha256=sha256)

    def first(self, step: str, event: str, source: str) -> int:
        seqs = [e["seq"] for e in self.events if (e["step"], e["event"], e["source"]) == (step, event, source)]
        if not seqs:
            raise SystemExit(f"no {event} event for {source} in step {step}")
        return seqs[0]


# ----------------------------------------------------------------------------
# FULL-CONFORM
# ----------------------------------------------------------------------------

def extract_block(feats: dict[str, Any], prefix: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, raw in feats.items():
        if not isinstance(key, str):
            raise SystemExit("feature key is not a string")
        if not key.startswith(prefix):
            continue
        coord = key[len(prefix):]
        if not coord:
            raise SystemExit(f"empty coordinate after prefix {prefix!r}")
        value = exact_count(raw, key)
        if value != 0:
            out[coord] = value
    return out


def predict_roles(core: dict[str, int], Pi: dict[str, str], T: dict[str, str],
                  W: dict[str, str]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Facet, type and family blocks implied by the core block: core Pi, core Pi T, core Pi W."""
    facet: dict[str, int] = {}
    typ: dict[str, int] = {}
    family: dict[str, int] = {}
    for uid, count in core.items():
        if uid not in Pi:
            raise SystemExit(f"core term outside Pi: {uid}")
        f = Pi[uid]
        if f not in T or f not in W:
            raise SystemExit(f"facet outside T/W: {f}")
        facet[f] = facet.get(f, 0) + count
        typ[T[f]] = typ.get(T[f], 0) + count
        family[W[f]] = family.get(W[f], 0) + count
    return facet, typ, family


def classify_defect(rep_core: dict[str, int], observed_facet: dict[str, int],
                    predicted_facet: dict[str, int]) -> str:
    """Row class of a non-conforming anchor; precedence no_rep_core, ghost_color, unsupported_category_unit."""
    if not rep_core:
        return "no_rep_core"
    if observed_facet.get("color", 0) > 0:
        return "ghost_color"
    residual = {f: observed_facet.get(f, 0) - predicted_facet.get(f, 0)
                for f in set(observed_facet) | set(predicted_facet)}
    residual = {f: d for f, d in residual.items() if d != 0}
    if len(residual) == 1:
        f, d = next(iter(residual.items()))
        if f.startswith("category/") and d == 1:
            return "unsupported_category_unit"
    raise SystemExit(f"a FULL-CONFORM defect matches no row class; residual={residual}")


def full_conform(chunks: Iterable[pd.DataFrame], Pi: dict[str, str], T: dict[str, str], W: dict[str, str],
                 core_support: dict[str, set[str]], expected_rows: dict[str, int]) -> dict[str, Any]:
    """Per-anchor check facet = core Pi, type = core Pi T, family = core Pi W and core terms in C_i."""
    rows = Counter()
    conform = Counter()
    defect = Counter()
    map_fail = Counter()
    support_fail = Counter()
    row_class = Counter()
    conform_without_rep = Counter()
    zero_block = Counter()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        for ds, row_id, raw in zip(chunk["dataset"].tolist(), chunk["anchor_id"].tolist(),
                                   chunk["full_feats_json"].tolist()):
            if ds not in expected_rows:
                raise SystemExit(f"unexpected dataset value: {ds!r}")
            if not row_id or row_id in seen:
                raise SystemExit(f"blank or duplicate anchor_id: {row_id!r}")
            seen.add(row_id)
            rows[ds] += 1
            feats = json.loads(raw)
            if not isinstance(feats, dict):
                raise SystemExit(f"full_feats_json is not an object for {row_id}")
            core = extract_block(feats, "core::")
            rep_core = extract_block(feats, "rep_core::")
            obs_facet = extract_block(feats, "facet::")
            obs_family = extract_block(feats, "family::")
            obs_type = extract_block(feats, "type::")
            for name, block in (("core", core), ("facet", obs_facet), ("family", obs_family), ("type", obs_type)):
                if not block:
                    zero_block[(ds, name)] += 1
            p_facet, p_type, p_family = predict_roles(core, Pi, T, W)
            failed: list[str] = []
            for name, ok in (("Pi", obs_facet == p_facet), ("tau", obs_type == p_type),
                             ("w", obs_family == p_family)):
                if not ok:
                    failed.append(name)
                    map_fail[(ds, name)] += 1
            if not set(core).issubset(core_support[ds]):
                failed.append("support")
                support_fail[ds] += 1
            ok_row = bool(core) and not failed
            rc = ""
            if ok_row:
                conform[ds] += 1
                if not rep_core:
                    conform_without_rep[ds] += 1
            else:
                defect[ds] += 1
                rc = classify_defect(rep_core, obs_facet, p_facet)
                row_class[(ds, rc)] += 1
            out.append({"dataset": ds, "row_id": row_id, "full_registry_conform": 1 if ok_row else 0,
                        "failed_maps": json.dumps(failed, ensure_ascii=True, separators=(",", ":")),
                        "row_class": rc})
    if dict(rows) != expected_rows:
        raise SystemExit(f"anchor rows per dataset {dict(rows)} != {expected_rows}")

    summary = []
    for ds in DATASETS:
        n, c = rows[ds], conform[ds]
        summary.append({
            "dataset": ds, "rows": n, "conform_rows": c, "defect_rows": defect[ds],
            "conform_rate": format(c / n, ".17g"),
            "Pi_fail_rows": map_fail[(ds, "Pi")], "tau_fail_rows": map_fail[(ds, "tau")],
            "w_fail_rows": map_fail[(ds, "w")], "support_fail_rows": support_fail[ds],
            **{rc: row_class[(ds, rc)] for rc in ROW_CLASSES},
            "conform_without_rep_core": conform_without_rep[ds],
            **{f"{b}_zero_rows": zero_block[(ds, b)] for b in ("core", "facet", "family", "type")},
        })
    return {"rows": out, "summary": summary}


# ----------------------------------------------------------------------------
# QUERY-CONFORM
# ----------------------------------------------------------------------------

def extract_role_blocks(feats: dict[str, Any]) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    core: dict[str, int] = {}
    facet: dict[str, int] = {}
    family: dict[str, int] = {}
    typ: dict[str, int] = {}
    for key, raw in feats.items():
        if not isinstance(key, str):
            raise SystemExit("feature key is not a string")
        if not raw:
            continue
        if key.startswith("core::"):
            target, coord = core, key[6:]
        elif key.startswith("facet::"):
            target, coord = facet, key[7:]
        elif key.startswith("family::"):
            target, coord = family, key[8:]
        elif key.startswith("type::"):
            target, coord = typ, key[6:]
        else:
            continue
        if not coord:
            raise SystemExit(f"empty coordinate in {key!r}")
        value = exact_count(raw, key)
        if value:
            target[coord] = value
    return core, facet, family, typ


def project_core(core: dict[str, int], Pi: dict[str, str]) -> dict[str, int]:
    facet: dict[str, int] = {}
    for uid, cnt in core.items():
        if uid not in Pi:
            raise SystemExit(f"core term outside Pi: {uid}")
        facet[Pi[uid]] = facet.get(Pi[uid], 0) + cnt
    return facet


def project_facet(facet: dict[str, int], T: dict[str, str], W: dict[str, str]) -> tuple[dict[str, int], dict[str, int]]:
    typ: dict[str, int] = {}
    family: dict[str, int] = {}
    for f, cnt in facet.items():
        if f not in T or f not in W:
            raise SystemExit(f"materialized facet outside T/W: {f}")
        typ[T[f]] = typ.get(T[f], 0) + cnt
        family[W[f]] = family.get(W[f], 0) + cnt
    return typ, family


def gold_is_max(gold: str, carrier: dict[str, int]) -> bool:
    """The gold label attains the carrier maximum (ties included)."""
    if not carrier:
        return False
    return carrier.get(gold, 0) == max(carrier.values())


def support_ok(block: dict[str, int], allowed: set[str]) -> bool:
    return bool(block) and set(block).issubset(allowed)


def query_arrays(chunks: Iterable[pd.DataFrame], row_ids: list[str], datasets_text: list[str],
                 Pi: dict[str, str], T: dict[str, str], W: dict[str, str],
                 supports: dict[str, dict[str, set[str]]]) -> dict[str, np.ndarray]:
    """Per-anchor edge, support, gold-is-carrier-max and carrier mass arrays, in full_conform row order."""
    n = len(row_ids)
    a = {
        "dataset_code": np.zeros(n, dtype=np.uint8),
        "pi_ok": np.zeros(n, dtype=bool), "t_ok": np.zeros(n, dtype=bool), "w_ok": np.zeros(n, dtype=bool),
        "supp_c": np.zeros(n, dtype=bool), "supp_s": np.zeros(n, dtype=bool), "supp_f": np.zeros(n, dtype=bool),
        "gold_is_carrier_max_facet": np.zeros(n, dtype=bool),
        "gold_is_carrier_max_family": np.zeros(n, dtype=bool),
        "carrier_l1_facet": np.zeros(n, dtype=np.uint16), "carrier_l1_family": np.zeros(n, dtype=np.uint16),
        "row_id_sha256": np.empty(n, dtype="S32"),
    }
    idx = 0
    for chunk in chunks:
        for ds, rid, raw, gf, gfam in zip(chunk["dataset"].tolist(), chunk["anchor_id"].tolist(),
                                          chunk["full_feats_json"].tolist(), chunk["gold_facet"].tolist(),
                                          chunk["gold_family"].tolist()):
            if idx >= n or rid != row_ids[idx] or ds != datasets_text[idx]:
                raise SystemExit(f"anchor order differs from full_conform order at index {idx}")
            if ds not in DATASETS:
                raise SystemExit(f"unexpected dataset value: {ds!r}")
            a["dataset_code"][idx] = DATASETS.index(ds)
            a["row_id_sha256"][idx] = hashlib.sha256(rid.encode("utf-8")).digest()
            feats = json.loads(raw)
            if not isinstance(feats, dict):
                raise SystemExit("full_feats_json is not an object")
            core, facet, family, typ = extract_role_blocks(feats)
            p_facet = project_core(core, Pi)
            p_type, p_family = project_facet(facet, T, W)
            a["pi_ok"][idx] = facet == p_facet
            a["t_ok"][idx] = typ == p_type
            a["w_ok"][idx] = family == p_family
            a["supp_c"][idx] = support_ok(core, supports[ds]["C_i"])
            a["supp_s"][idx] = support_ok(facet, supports[ds]["S_i"])
            a["supp_f"][idx] = support_ok(family, supports[ds]["w(C_i)"])
            a["gold_is_carrier_max_facet"][idx] = gold_is_max(gf, facet)
            a["gold_is_carrier_max_family"][idx] = gold_is_max(gfam, family)
            a["carrier_l1_facet"][idx] = sum(facet.values())
            a["carrier_l1_family"][idx] = sum(family.values())
            idx += 1
    if idx != n:
        raise SystemExit(f"anchor rows {idx} != full_conform rows {n}")
    return a


def compute_query_mask(view: dict[str, str], full: np.ndarray, a: dict[str, np.ndarray]) -> np.ndarray:
    """QUERY-CONFORM of one view: basis support, then every required map, then the gold label at the carrier max.
    Path C views use FULL-CONFORM in place of a certificate."""
    gold_ok = a["gold_is_carrier_max_facet"] if view["task"] == "facet" else a["gold_is_carrier_max_family"]
    if view["certificate_path"] == "C":
        return full & gold_ok
    supp = {"C_i": "supp_c", "S_i": "supp_s", "w(C_i)": "supp_f"}.get(view["basis_support"])
    if supp is None:
        raise SystemExit(f"invalid basis_support {view['basis_support']!r} on path {view['certificate_path']}")
    out = a[supp].copy()
    for edge in [x for x in view["required_maps"].split("|") if x]:
        if edge not in ("Pi", "T", "W"):
            raise SystemExit(f"unknown certificate edge: {edge}")
        out &= a[{"Pi": "pi_ok", "T": "t_ok", "W": "w_ok"}[edge]]
    return out & gold_ok


def query_outputs(views: list[dict[str, str]], full: np.ndarray, a: dict[str, np.ndarray],
                  row_order_source: str) -> dict[str, Any]:
    n = len(full)
    masks = np.zeros((len(views), n), dtype=bool)
    summary: list[dict[str, Any]] = []
    for j, view in enumerate(views):
        q = compute_query_mask(view, full, a)
        masks[j, :] = q
        gold = a["gold_is_carrier_max_facet"] if view["task"] == "facet" else a["gold_is_carrier_max_family"]
        carrier = a["carrier_l1_facet"] if view["task"] == "facet" else a["carrier_l1_family"]
        required = [x for x in view["required_maps"].split("|") if x]
        for code, ds in enumerate(DATASETS):
            cm = a["dataset_code"] == code
            qn = int((q & cm).sum())
            if view["certificate_path"] == "C":
                sf = 0
            else:
                supp = {"C_i": "supp_c", "S_i": "supp_s", "w(C_i)": "supp_f"}[view["basis_support"]]
                sf = int((~a[supp] & cm).sum())
            summary.append({
                "dataset": ds, "task": view["task"], "view_id": view["view_id"], "view_group": view["view_group"],
                "certificate_path": view["certificate_path"], "basis": view["basis"],
                "basis_support": view["basis_support"], "required_maps": view["required_maps"],
                "rows": int(cm.sum()), "query_conform_rows": qn, "query_defect_rows": int(cm.sum()) - qn,
                "outside_Drel_rows": int((q & ~full & cm).sum()),
                "gold_is_carrier_max_false_rows": int((~gold & cm).sum()),
                "carrier_zero_rows": int((carrier == 0)[cm].sum()),
                "support_fail_rows": sf,
                "Pi_fail_rows": int((~a["pi_ok"] & cm).sum()) if "Pi" in required else 0,
                "T_fail_rows": int((~a["t_ok"] & cm).sum()) if "T" in required else 0,
                "W_fail_rows": int((~a["w_ok"] & cm).sum()) if "W" in required else 0,
            })
    index = [{"array_index": j, "view_id": v["view_id"], "task": v["task"], "view_group": v["view_group"],
              "certificate_path": v["certificate_path"], "basis": v["basis"], "basis_support": v["basis_support"],
              "required_maps": v["required_maps"], "row_order_source": row_order_source, "row_count": n}
             for j, v in enumerate(views)]
    arrays = {
        "view_ids": np.asarray([v["view_id"] for v in views], dtype="U64"),
        "tasks": np.asarray([v["task"] for v in views], dtype="U8"),
        "certificate_paths": np.asarray([v["certificate_path"] for v in views], dtype="U1"),
        "required_maps": np.asarray([v["required_maps"] for v in views], dtype="U8"),
        "row_id_sha256": a["row_id_sha256"],
        "dataset_code": a["dataset_code"],
        "full_registry_conform": full,
        "query_conform": masks,
        "gold_is_carrier_max_facet": a["gold_is_carrier_max_facet"],
        "gold_is_carrier_max_family": a["gold_is_carrier_max_family"],
        "carrier_l1_facet": a["carrier_l1_facet"],
        "carrier_l1_family": a["carrier_l1_family"],
    }
    return {"summary": summary, "index": index, "arrays": arrays}
