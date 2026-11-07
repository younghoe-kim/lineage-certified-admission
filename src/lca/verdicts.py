# SPDX-License-Identifier: MIT
"""Three-valued fiber verdicts of the registered views: evidence, exact fibers, Path A/B/C audits, decomposition, exact Path B checks, Fig. 1 anchor."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from array import array
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from . import certificates as cert
from .io import require_sha, sha256_file
from .names import DATASETS, FAMILY_ORDER

UNVERIFIED, UNSAFE, NONDETERMINED = np.uint8(0), np.uint8(1), np.uint8(2)
VERDICT_CODES_TEXT = "0=UNVERIFIED;1=UNSAFE;2=NONDETERMINED"
ROLE_BLOCKS = ("facet::", "family::", "type::")
SPARSE_BLOCKS = ("core::", "rep_core::", "inst_family::", "inst_sem::")
EVIDENCE_BLOCKS = frozenset(ROLE_BLOCKS + SPARSE_BLOCKS)
ANCHOR_COLUMNS = ["dataset", "anchor_id", "full_feats_json", "gold_facet", "gold_family"]
COUNT_MAX = int(np.iinfo(np.uint16).max)
WITNESS_LIMIT_AC = 20     # representative witnesses per kind and view/dataset (Paths A and C)
WITNESS_LIMIT_B = 40      # representative witnesses per view/dataset (Path B)
NEAR_PURE = 0.95
RANK_BASE = 0x100          # sparse-block keys encode each coordinate by its declared rank as one character

# Views that expose selected coordinates of a block; every other view exposes whole blocks.
COORDINATE_PROJECTIONS: dict[str, dict[str, list[str]]] = {
    "type_attr_part__family": {"type::": ["attribute", "part"]},
    "type3__family": {"type::": ["attribute", "category", "part"]},
    "family3__facet": {"family::": ["structural_expression", "surface_expression", "taxonomic_category"]},
    "family3_plus_type__facet": {"family::": ["structural_expression", "surface_expression", "taxonomic_category"],
                                 "type::": ["attribute", "category", "part"]},
}
PROJECTION_SOURCE_BLOCKS = "view schema: whole exposed blocks"
PROJECTION_SOURCE_COORDS = "declared coordinate projection"
# Path B marginals: a selected family coordinate is the W image, a selected type coordinate the T image.
BLOCK_MAP = {"family::": "W", "type::": "T"}

FIG1_RULE = ("candidates: DF FULL-CONFORM; 3-4 nonzero facet coordinates; >=2 nonzero declared-W family coordinates; "
             "unique declared-W maximum. Selected: the candidate with the smallest SHA-256 of its core/facet/family/type "
             "evidence in the released benchmark, bound in this input view by its zero-based row "
             "(src/lca/paper/source_release.json, fig1)")
SOURCE_RELEASE = Path(__file__).resolve().parent / "paper" / "source_release.json"


def fig1_binding() -> tuple[int, int]:
    """Zero-based row of the Fig. 1 anchor and the expected candidate count, from the source-release record."""
    f1 = json.loads(SOURCE_RELEASE.read_text(encoding="utf-8"))["fig1"]
    return int(f1["release_row_zero_based"]), int(f1["candidate_rows"])
WITNESS_REPRESENTATION = ("per-anchor verdicts are in anchor_verdicts.npz (view order: anchor_verdicts_index.csv); "
                          "this file stores per-view counts and representative witnesses")


# ====================================================================================================
# Inputs
# ====================================================================================================

def read_sha256sums(path: Path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            out[name.strip()] = h
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_registry(outputs: Path) -> dict[str, Any]:
    """Registry step outputs: Pi (core term -> facet), T and W (facet -> type / family), certificate and view
    specifications, dataset supports and the declared coordinate inventories."""
    files = {k: outputs / "registry" / v for k, v in {
        "term_to_facet": "term_to_facet.csv", "facet_map": "facet_map.csv", "certificate_spec": "certificate_spec.csv",
        "views": "views.csv", "dataset_support": "dataset_support.json", "populations": "populations.json"}.items()}
    pi: dict[str, str] = {}
    for r in _read_csv(files["term_to_facet"]):
        if r["match"] != "1":
            raise SystemExit("term_to_facet contains a non-matching Pi row")
        pi[r["core_term_uid"]] = r["core_facet"]
    t_map: dict[str, str] = {}
    w_map: dict[str, str] = {}
    for r in _read_csv(files["facet_map"]):
        t_map[r["facet"]] = r["term_type"]
        w_map[r["facet"]] = r["family"]
    cert_rows = _read_csv(files["certificate_spec"])
    views = _read_csv(files["views"])
    if [r["view_id"] for r in cert_rows] != [r["view_id"] for r in views] or \
            [r["task"] for r in cert_rows] != [r["task"] for r in views]:
        raise SystemExit("certificate_spec and views differ in row order")
    support = json.loads(files["dataset_support"].read_text(encoding="utf-8"))
    pops = json.loads(files["populations"].read_text(encoding="utf-8"))
    members = {b: list(v) for b, v in pops["block_coordinate_members"].items()}
    family_coords = [f for f in FAMILY_ORDER if f in set(members["family::"])]
    if family_coords != members["family::"]:
        raise SystemExit("family coordinates do not follow the declared family order")
    facet_coords = members["facet::"]
    if not set(facet_coords) <= set(t_map) or set(t_map) != set(w_map):
        raise SystemExit("released facet coordinates are not covered by T/W")
    exposed = {r["view_id"]: [b for b in r["exposed_blocks"].split("|") if b] for r in views}
    projection: dict[str, dict[str, Any]] = {}
    projection_rows = []
    for r in views:
        vid = r["view_id"]
        if vid in COORDINATE_PROJECTIONS:
            spec: dict[str, Any] = {b: list(c) for b, c in COORDINATE_PROJECTIONS[vid].items()}
            if list(spec) != exposed[vid]:
                raise SystemExit(f"coordinate projection does not match the exposed blocks of {vid}")
            source = PROJECTION_SOURCE_COORDS
        else:
            spec = {b: "ALL" for b in exposed[vid]}
            source = PROJECTION_SOURCE_BLOCKS
        projection[vid] = spec
        projection_rows.append({"view_id": vid, "task": r["task"], "exposed_blocks": r["exposed_blocks"],
                                "projection_json": json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                                "projection_source": source})
    return {"Pi": pi, "T": t_map, "W": w_map, "cert": cert_rows, "views": views, "support": support, "pops": pops,
            "members": members, "facet_coords": facet_coords, "family_coords": family_coords,
            "type_coords": members["type::"], "projection": projection, "projection_rows": projection_rows,
            "files": files, "sha256": {f"registry/{p.name}": sha256_file(p) for p in files.values()}}


def load_conformance(outputs: Path, reg: dict[str, Any]) -> dict[str, Any]:
    """Conformance step outputs: row order, FULL-CONFORM mask and the 32 QUERY-CONFORM masks."""
    files = {"full_conform": outputs / "conformance" / "full_conform.csv.gz",
             "index": outputs / "conformance" / "query_conform_index.csv",
             "masks": outputs / "conformance" / "query_masks.npz"}
    dataset_text, row_id, full_col = [], [], []
    with gzip.open(files["full_conform"], "rt", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            dataset_text.append(r["dataset"]); row_id.append(r["row_id"]); full_col.append(r["full_registry_conform"])
    n = len(row_id)
    if len(set(row_id)) != n:
        raise SystemExit("full_conform row ids are not unique")
    code = {d: i for i, d in enumerate(DATASETS)}
    if any(d not in code for d in set(dataset_text)):
        raise SystemExit("unexpected dataset in full_conform")
    dataset_code = np.fromiter((code[d] for d in dataset_text), dtype=np.uint8, count=n)
    full_csv = np.asarray([v == "1" for v in full_col], dtype=bool)
    index = sorted(_read_csv(files["index"]), key=lambda r: int(r["array_index"]))
    if [int(r["array_index"]) for r in index] != list(range(len(reg["cert"]))):
        raise SystemExit("query index is not 0..views-1")
    if [r["view_id"] for r in index] != [r["view_id"] for r in reg["cert"]]:
        raise SystemExit("query index order differs from certificate order")
    with np.load(files["masks"], allow_pickle=False) as z:
        qmask = z["query_conform"].astype(bool)
        full = z["full_registry_conform"].astype(bool)
    if qmask.shape != (len(reg["cert"]), n) or not np.array_equal(full, full_csv):
        raise SystemExit("query masks do not match full_conform")
    for ds in DATASETS:
        if int(np.count_nonzero(dataset_code == code[ds])) != int(reg["support"]["datasets"][ds]["n_rows"]):
            raise SystemExit(f"row count of {ds} differs from dataset_support")
    return {"n": n, "row_id": np.asarray(row_id, dtype=object), "dataset_code": dataset_code, "full": full,
            "qmask": qmask, "index": index, "files": files,
            "sha256": {f"conformance/{p.name}": sha256_file(p) for p in files.values()}}


def dataset_mask(conf: dict[str, Any], ds: str) -> np.ndarray:
    return conf["dataset_code"] == DATASETS.index(ds)


# ====================================================================================================
# Evidence
# ====================================================================================================

def exact_count(v: Any, key: str) -> int:
    if isinstance(v, bool):
        raise SystemExit(f"boolean count at {key}")
    x = float(v)
    if not math.isfinite(x) or x < 0 or not x.is_integer() or x > COUNT_MAX:
        raise SystemExit(f"invalid count at {key}: {v!r}")
    return int(x)


def _rank_key(pairs: list[tuple[int, int]]) -> str:
    """Canonical text of one sparse block: (coordinate rank, decimal count) pairs in declared coordinate order.
    Sorting these texts orders blocks lexicographically by (coordinate, count text) pairs."""
    pairs.sort()
    return "\x1f".join(f"{chr(RANK_BASE + r)}\x1e{c}" for r, c in pairs)


def _codes_sorted(values: list[str]) -> np.ndarray:
    uniq = sorted(set(values))
    pos = {v: i for i, v in enumerate(uniq)}
    return np.fromiter((pos[v] for v in values), dtype=np.int32, count=len(values))


def _dense_codes(arr: np.ndarray) -> np.ndarray:
    """Exact partition codes of dense count vectors, in sorted order of their little-endian uint16 bytes."""
    x = np.ascontiguousarray(arr.astype("<u2"))
    keys = x.view(f"S{2 * x.shape[1]}").reshape(-1)
    _, inv = np.unique(keys, return_inverse=True)
    return inv.reshape(-1).astype(np.int32)


def load_evidence(anchors: Path, expected_sha: str, reg: dict[str, Any], conf: dict[str, Any], *,
                  select_fig1: bool = False) -> dict[str, Any]:
    """One pass over the anchor table: role blocks as dense counts, sparse blocks as exact partition codes,
    the core block as (row, coordinate, count) triplets, gold labels as coordinate codes."""
    require_sha(anchors, expected_sha)
    n = conf["n"]
    facet_coords, family_coords, type_coords = reg["facet_coords"], reg["family_coords"], reg["type_coords"]
    fidx = {x: i for i, x in enumerate(facet_coords)}
    famidx = {x: i for i, x in enumerate(family_coords)}
    tidx = {x: i for i, x in enumerate(type_coords)}
    rank = {b: {x: i for i, x in enumerate(reg["members"][b])} for b in SPARSE_BLOCKS}
    dense_index = {"facet::": fidx, "family::": famidx, "type::": tidx}
    fam_of_facet = {f: reg["W"][f] for f in facet_coords}
    trip = {b: (array("q"), array("q"), array("q")) for b in ROLE_BLOCKS + ("core::",)}
    keys: dict[str, list[str]] = {b: [""] * n for b in SPARSE_BLOCKS}
    gold_facet = np.empty(n, dtype=np.int16)
    gold_family = np.empty(n, dtype=np.int16)
    ds_text = [DATASETS[c] for c in conf["dataset_code"]]
    full = conf["full"]
    fig_count, fig_best = 0, None
    fig_row, fig_expected = fig1_binding() if select_fig1 else (-1, -1)
    csv.field_size_limit(sys.maxsize)
    row = -1
    with gzip.open(anchors, "rt", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        if next(rd) != ANCHOR_COLUMNS:
            raise SystemExit("anchors.csv.gz has unexpected columns")
        for row, (ds, aid, raw, g_facet, g_family) in enumerate(rd):
            if row >= n or ds != ds_text[row] or aid != conf["row_id"][row]:
                raise SystemExit(f"anchor row order differs from full_conform at row {row}")
            feats = json.loads(raw)
            if not isinstance(feats, dict):
                raise SystemExit(f"full_feats_json is not an object at {aid}")
            sparse: dict[str, list[tuple[int, int]]] = {b: [] for b in SPARSE_BLOCKS}
            facet_row: list[tuple[str, int]] = []
            for key, rv in feats.items():
                block, sep, coord = key.partition("::")
                block += "::"
                if not sep or block not in EVIDENCE_BLOCKS:
                    continue
                c = exact_count(rv, key)
                if c == 0:
                    continue
                if block in dense_index:
                    j = dense_index[block].get(coord)
                    if j is None:
                        raise SystemExit(f"unknown {block} coordinate {coord}")
                    t = trip[block]
                    t[0].append(row); t[1].append(j); t[2].append(c)
                    if block == "facet::":
                        facet_row.append((coord, c))
                else:
                    r = rank[block].get(coord)
                    if r is None:
                        raise SystemExit(f"{block} coordinate outside the declared inventory: {coord}")
                    sparse[block].append((r, c))
                    if block == "core::":
                        t = trip["core::"]
                        t[0].append(row); t[1].append(r); t[2].append(c)
            for b in SPARSE_BLOCKS:
                keys[b][row] = _rank_key(sparse[b])
            if g_facet not in fidx or g_family not in famidx:
                raise SystemExit(f"gold label outside the coordinate universe at {aid}")
            gold_facet[row] = fidx[g_facet]
            gold_family[row] = famidx[g_family]
            if select_fig1 and ds == "DF" and full[row] and 3 <= len(facet_row) <= 4:
                fam_pred = np.zeros(len(family_coords), dtype=np.int64)
                for fac, c in facet_row:
                    fam_pred[famidx[fam_of_facet[fac]]] += c
                mx = int(fam_pred.max())
                if np.count_nonzero(fam_pred) >= 2 and mx > 0 and int(np.count_nonzero(fam_pred == mx)) == 1:
                    fig_count += 1
                    if row == fig_row:
                        if fig_best is not None:
                            raise SystemExit("the Fig. 1 row was met twice")
                        keep = {k: feats[k] for k in feats if k.startswith(("core::", "facet::", "family::", "type::"))}
                        h = hashlib.sha256(json.dumps(keep, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
                        fig_best = {"release_row_zero_based": row, "row_id": aid, "evidence_sha256_input_view": h,
                                    "facet_nonzero": {fac: int(c) for fac, c in sorted(facet_row, key=lambda z: fidx[z[0]])},
                                    "declared_W_family": {family_coords[j]: int(fam_pred[j]) for j in np.flatnonzero(fam_pred)}}
    if row + 1 != n:
        raise SystemExit(f"anchor rows {row + 1} != {n}")
    if select_fig1:
        # Fail closed: the candidate rule and the bound row must reproduce the recorded selection exactly.
        if fig_count != fig_expected:
            raise SystemExit(f"Fig. 1 candidates {fig_count} differ from the recorded {fig_expected}")
        if fig_best is None:
            raise SystemExit(f"the bound Fig. 1 row {fig_row} is not a candidate")

    def dense(block: str, width: int) -> np.ndarray:
        r, j, c = (np.frombuffer(v, dtype=np.int64) for v in trip[block])
        out = np.zeros((n, width), dtype=np.uint16)
        out[r, j] = c
        return out

    facet = dense("facet::", len(facet_coords))
    family = dense("family::", len(family_coords))
    typ = dense("type::", len(type_coords))
    core_row, core_uid, core_cnt = (np.frombuffer(v, dtype=np.int64).copy() for v in trip["core::"])
    core_members = reg["members"]["core::"]
    pi_facet = np.full(len(core_members), -1, dtype=np.int64)
    for i, uid in enumerate(core_members):
        fac = reg["Pi"].get(uid)
        if fac is not None and fac in fidx:
            pi_facet[i] = fidx[fac]
    if np.any(pi_facet[core_uid] < 0):
        raise SystemExit("an active core coordinate maps outside the released facets")
    w_mat = np.zeros((len(facet_coords), len(family_coords)), dtype=np.int8)
    t_mat = np.zeros((len(facet_coords), len(type_coords)), dtype=np.int8)
    all_support = set().union(*(set(reg["support"]["datasets"][d]["S_members"]) for d in DATASETS))
    for j, fac in enumerate(facet_coords):
        if reg["W"][fac] not in famidx:
            raise SystemExit(f"facet {fac} maps by W outside the released family coordinates")
        w_mat[j, famidx[reg["W"][fac]]] = 1
        if reg["T"][fac] in tidx:
            t_mat[j, tidx[reg["T"][fac]]] = 1
        elif fac in all_support:
            # A facet whose type is not a released type coordinate must lie outside every dataset support,
            # so it is fixed to zero in every Path B program.
            raise SystemExit(f"supported facet with a non-released type: {fac}")
    core_facet = np.zeros((n, len(facet_coords)), dtype=np.int64)
    np.add.at(core_facet, (core_row, pi_facet[core_uid]), core_cnt)
    core_family = core_facet @ w_mat.astype(np.int64)
    facet_family = facet.astype(np.int64) @ w_mat.astype(np.int64)
    for a in (core_facet, core_family, facet_family):
        if int(a.max(initial=0)) > COUNT_MAX:
            raise SystemExit("projected count exceeds uint16")
    block_code = {b: _codes_sorted(keys[b]) for b in SPARSE_BLOCKS}
    del keys
    for b, arr in (("facet::", facet), ("family::", family), ("type::", typ)):
        block_code[b] = _dense_codes(arr)
    return {"facet": facet, "family": family, "type": typ,
            "core_facet": core_facet.astype(np.uint16), "core_family": core_family.astype(np.uint16),
            "facet_family": facet_family.astype(np.uint16),
            "core_row": core_row, "core_uid": core_uid, "core_count": core_cnt, "pi_facet": pi_facet,
            "block_code": block_code, "Wmat": w_mat, "Tmat": t_mat,
            "gold_facet": gold_facet, "gold_family": gold_family,
            "fig1_candidates": fig_count, "fig1_selected": fig_best}


# ====================================================================================================
# Fibers
# ====================================================================================================

def view_keys(view_id: str, reg: dict[str, Any], ev: dict[str, Any]) -> np.ndarray:
    """Exact fiber key: partition codes of whole blocks and literal counts of selected coordinates."""
    comps = []
    for block, selector in reg["projection"][view_id].items():
        if selector == "ALL":
            comps.append(ev["block_code"][block][:, None].astype(np.int32))
        else:
            idx = {"family::": reg["family_coords"], "type::": reg["type_coords"]}[block]
            arr = ev["family"] if block == "family::" else ev["type"]
            comps.append(arr[:, [idx.index(x) for x in selector]].astype(np.int32))
    if not comps:
        raise SystemExit(f"view {view_id} exposes no coordinate")
    return np.ascontiguousarray(np.concatenate(comps, axis=1), dtype=np.int32)


def dense_projection(view_id: str, reg: dict[str, Any], ev: dict[str, Any]) -> np.ndarray:
    """Selected coordinate counts of a coordinate-selective view (the Path B fiber value)."""
    parts = []
    for block, selector in reg["projection"][view_id].items():
        if selector == "ALL":
            raise SystemExit(f"view {view_id} is not coordinate-selective")
        idx = {"family::": reg["family_coords"], "type::": reg["type_coords"]}[block]
        arr = ev["family"] if block == "family::" else ev["type"]
        parts.append(arr[:, [idx.index(x) for x in selector]])
    return np.concatenate(parts, axis=1).astype(np.int64)


def group_stats(keys: np.ndarray, mask: np.ndarray, labels: np.ndarray, nlabels: int) -> dict[str, Any]:
    """Fibers of the masked rows, in lexicographic order of their keys, with per-fiber label counts."""
    idx = np.flatnonzero(mask)
    sub = np.ascontiguousarray(keys[idx])
    if sub.ndim == 1:
        sub = sub[:, None]
    if len(sub) == 0:
        inv = np.empty(0, dtype=np.int64)
        uniq = np.empty((0, sub.shape[1]), dtype=sub.dtype)
        counts = np.empty(0, dtype=np.int64)
    else:
        order = np.lexsort(tuple(sub[:, j] for j in range(sub.shape[1] - 1, -1, -1)))
        ss = sub[order]
        boundary = np.empty(len(ss), dtype=bool)
        boundary[0] = True
        if len(ss) > 1:
            boundary[1:] = np.any(ss[1:] != ss[:-1], axis=1)
        inv = np.empty(len(ss), dtype=np.int64)
        inv[order] = np.cumsum(boundary, dtype=np.int64) - 1
        uniq = ss[boundary].copy()
        counts = np.bincount(inv, minlength=len(uniq)).astype(np.int64)
    mat = np.bincount(inv * nlabels + labels[idx].astype(np.int64), minlength=len(uniq) * nlabels).reshape(len(uniq), nlabels)
    diversity = np.count_nonzero(mat, axis=1)
    purity = mat.max(axis=1) / counts if len(counts) else np.empty(0, dtype=float)
    first_pos = np.full(len(uniq), len(mask), dtype=np.int64)
    if len(uniq):
        np.minimum.at(first_pos, inv, idx)
    return {"idx": idx, "unique": uniq, "inv": inv, "counts": counts, "label_counts": mat,
            "diversity": diversity, "purity": purity, "first_pos": first_pos}


def _pair_witness(stats: dict[str, Any], g: int, labels: np.ndarray, label_names: list[str], row_id: np.ndarray) -> dict[str, Any]:
    a, b, la, lb = cert.witness_pair(stats["idx"][stats["inv"] == g], labels)
    return {"kind": "NONDETERMINED", "group_index": int(g), "row_pair": [str(row_id[a]), str(row_id[b])],
            "label_pair": [label_names[la], label_names[lb]]}


def summarize(view_id: str, ds_mask: np.ndarray, q: np.ndarray, verdict: np.ndarray, stats: dict[str, Any],
              full: np.ndarray) -> dict[str, Any]:
    n = int(ds_mask.sum())
    unsafe = int(np.count_nonzero(ds_mask & (verdict == UNSAFE)))
    nondet = int(np.count_nonzero(ds_mask & (verdict == NONDETERMINED)))
    defect = int(np.count_nonzero(ds_mask & ~q))
    observed = n - unsafe - nondet - defect
    if observed < 0:
        raise SystemExit(f"negative observed-unverified count for {view_id}")
    if len(stats["counts"]):
        purity = float(stats["label_counts"].max(axis=1).sum() / stats["counts"].sum())
        near = int(stats["counts"][(stats["diversity"] >= 2) & (stats["purity"] >= NEAR_PURE - 1e-15)].sum())
    else:
        purity, near = float("nan"), 0
    return {"fibers": int(len(stats["unique"])), "unsafe_rows": unsafe, "nondetermined_rows": nondet,
            "unverified_observed_rows": observed, "unverified_defect_rows": defect,
            "mU": unsafe / n, "mN": nondet / n, "mV_obs": observed / n, "mV_defect": defect / n,
            "near_pure": near / n, "purity": purity, "query_rows": int(q.sum()),
            "outside_Drel_rows": int(np.count_nonzero(q & ~full & ds_mask))}


def carrier_for(basis: str, carrier_edges: str, ev: dict[str, Any]) -> np.ndarray:
    table = {("facet::", "W"): "facet_family", ("core::", "Pi|W"): "core_family", ("core::", "Pi"): "core_facet",
             ("family::", ""): "family", ("facet::", ""): "facet"}
    key = table.get((basis, carrier_edges))
    if key is None:
        raise SystemExit(f"unsupported Path A basis/carrier: {basis}, {carrier_edges}")
    return ev[key]


def audit_path_a(view_id: str, basis: str, carrier_edges: str, mask: np.ndarray, keys: np.ndarray,
                 labels: np.ndarray, label_names: list[str], ds_mask: np.ndarray, ev: dict[str, Any],
                 conf: dict[str, Any], strict_label: tuple[np.ndarray, np.ndarray] | None = None):
    """Path A: a fiber is UNSAFE iff the registered carrier has a strict unique maximum on every row (it is
    fiber-constant), NONDETERMINED iff otherwise it holds two labels, UNVERIFIED else."""
    q = mask & ds_mask
    strict, cert_label = strict_label if strict_label is not None else cert.strict_unique_max(carrier_for(basis, carrier_edges, ev))
    stats = group_stats(keys, q, labels, len(label_names))
    verdict = np.full(len(q), UNVERIFIED, dtype=np.uint8)
    unsafe_label = np.full(len(q), -1, dtype=np.int16)
    witnesses: list[dict[str, Any]] = []
    if len(stats["unique"]):
        strict_count = np.bincount(stats["inv"], weights=strict[stats["idx"]].astype(np.int64),
                                   minlength=len(stats["unique"])).astype(np.int64)
        if np.any((strict_count != 0) & (strict_count != stats["counts"])):
            raise SystemExit(f"Path A strict maximum is not fiber-constant for {view_id}")
        g_unsafe = strict_count == stats["counts"]
        g_nondet = ~g_unsafe & (stats["diversity"] >= 2)
        vg = np.full(len(stats["unique"]), UNVERIFIED, dtype=np.uint8)
        vg[g_unsafe] = UNSAFE
        vg[g_nondet] = NONDETERMINED
        verdict[stats["idx"]] = vg[stats["inv"]]
        rows_u = stats["idx"][vg[stats["inv"]] == UNSAFE]
        unsafe_label[rows_u] = cert_label[rows_u]
        for g in np.flatnonzero(g_nondet)[:WITNESS_LIMIT_AC]:
            witnesses.append(_pair_witness(stats, int(g), labels, label_names, conf["row_id"]))
        for g in np.flatnonzero(g_unsafe)[:WITNESS_LIMIT_AC]:
            pos = int(stats["first_pos"][g])
            witnesses.append({"kind": "UNSAFE", "group_index": int(g), "basis": basis, "carrier_edges": carrier_edges,
                              "certified_label": label_names[int(cert_label[pos])]})
    conflicts = q & (verdict == UNSAFE) & (unsafe_label != labels)
    metrics = summarize(view_id, ds_mask, q, verdict, stats, conf["full"])
    metrics["certificate_label_conflicts"] = int(conflicts.sum())
    metrics["infeasible_fibers"] = 0
    return verdict, unsafe_label, metrics, witnesses, []


def audit_path_c(view_id: str, mask: np.ndarray, keys: np.ndarray, labels: np.ndarray, label_names: list[str],
                 ds_mask: np.ndarray, conf: dict[str, Any]):
    """Path C: no registered carrier; a fiber is NONDETERMINED iff it holds two labels, UNVERIFIED else."""
    q = mask & ds_mask
    stats = group_stats(keys, q, labels, len(label_names))
    verdict = np.full(len(q), UNVERIFIED, dtype=np.uint8)
    unsafe_label = np.full(len(q), -1, dtype=np.int16)
    witnesses: list[dict[str, Any]] = []
    if len(stats["unique"]):
        g_nondet = stats["diversity"] >= 2
        vg = np.full(len(stats["unique"]), UNVERIFIED, dtype=np.uint8)
        vg[g_nondet] = NONDETERMINED
        verdict[stats["idx"]] = vg[stats["inv"]]
        for g in np.flatnonzero(g_nondet)[:WITNESS_LIMIT_AC]:
            witnesses.append(_pair_witness(stats, int(g), labels, label_names, conf["row_id"]))
    metrics = summarize(view_id, ds_mask, q, verdict, stats, conf["full"])
    metrics["certificate_label_conflicts"] = 0
    metrics["infeasible_fibers"] = 0
    return verdict, unsafe_label, metrics, witnesses, []


def path_b_system(view_id: str, value: np.ndarray, ds: str, reg: dict[str, Any], ev: dict[str, Any]):
    """Equality rows (selected W/T marginals) and upper bounds (zero outside the dataset support)."""
    rows, rhs = [], []
    k = 0
    for block, selector in reg["projection"][view_id].items():
        mat = ev["Wmat"] if block == "family::" else ev["Tmat"]
        coords = reg["family_coords"] if block == "family::" else reg["type_coords"]
        for x in selector:
            rows.append(mat[:, coords.index(x)].astype(np.float64))
            rhs.append(int(value[k]))
            k += 1
    support = set(reg["support"]["datasets"][ds]["S_members"])
    ub = np.asarray([np.inf if f in support else 0.0 for f in reg["facet_coords"]], dtype=np.float64)
    return np.vstack(rows), np.asarray(rhs, dtype=np.float64), ub


def audit_path_b(view_id: str, task: str, mask: np.ndarray, values: np.ndarray, labels: np.ndarray,
                 label_names: list[str], ds: str, ds_mask: np.ndarray, reg: dict[str, Any], ev: dict[str, Any],
                 conf: dict[str, Any], open_sink: list[dict[str, Any]] | None = None):
    """Path B: a single-label fiber is UNSAFE iff its label is the strict unique maximum over the whole integer
    fiber of the exposed marginals (decided by integer programs); two-label fibers are NONDETERMINED. With
    open_sink, every single-label fiber that stays UNVERIFIED adds the state at which its first non-dominated
    rival ties with or exceeds the observed label, with the number of query-conforming anchors of the fiber."""
    q = mask & ds_mask
    stats = group_stats(np.ascontiguousarray(values), q, labels, len(label_names))
    verdict = np.full(len(q), UNVERIFIED, dtype=np.uint8)
    unsafe_label = np.full(len(q), -1, dtype=np.int16)
    vg = np.full(len(stats["unique"]), UNVERIFIED, dtype=np.uint8)
    witnesses: list[dict[str, Any]] = []
    unsafe_certs: list[dict[str, Any]] = []
    infeasible = 0
    coeff = np.eye(len(reg["facet_coords"]), dtype=np.int8) if task == "facet" else ev["Wmat"].astype(np.int8).T
    for g in range(len(stats["unique"])):
        if stats["diversity"][g] >= 2:
            vg[g] = NONDETERMINED
            if len(witnesses) < WITNESS_LIMIT_B:
                witnesses.append(_pair_witness(stats, g, labels, label_names, conf["row_id"]))
            continue
        label = int(np.flatnonzero(stats["label_counts"][g])[0])
        val = values[int(stats["first_pos"][g])].astype(np.int64)
        a_eq, b_eq, ub = path_b_system(view_id, val, ds, reg, ev)
        solved = cert.path_b_fiber(a_eq, b_eq, ub, coeff, label, label_names, record_open=open_sink is not None)
        if solved["infeasible"]:
            infeasible += 1
            continue
        if solved["unsafe"]:
            vg[g] = UNSAFE
            rec = {"kind": "UNSAFE", "group_index": g, "fiber_value": [int(x) for x in val],
                   "certified_label": label_names[label], "rival_optima": solved["rivals"]}
            unsafe_certs.append(rec)
            if len(witnesses) < WITNESS_LIMIT_B:
                witnesses.append(rec)
        elif open_sink is not None:
            op = solved["open"]
            open_sink.append({"view_id": view_id, "task": task, "dataset": ds, "group_index": g,
                              "fiber_size": int(stats["counts"][g]),
                              "fiber_value": [int(x) for x in val], "observed_label": label_names[label],
                              "rival": op["rival"], "program_status": op["program_status"],
                              "rival_minus_observed_optimum": solved["rivals"][-1]["rival_minus_candidate_optimum"],
                              "state": {reg["facet_coords"][j]: int(x) for j, x in enumerate(op["state"]) if x}})
    if len(stats["idx"]):
        verdict[stats["idx"]] = vg[stats["inv"]]
        rows_u = stats["idx"][vg[stats["inv"]] == UNSAFE]
        unsafe_label[rows_u] = labels[rows_u]
    conflicts = q & (verdict == UNSAFE) & (unsafe_label != labels)
    metrics = summarize(view_id, ds_mask, q, verdict, stats, conf["full"])
    metrics["certificate_label_conflicts"] = int(conflicts.sum())
    metrics["infeasible_fibers"] = int(infeasible)
    return verdict, unsafe_label, metrics, witnesses, unsafe_certs


def labels_for(task: str, reg: dict[str, Any], ev: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    if task == "facet":
        return ev["gold_facet"], reg["facet_coords"]
    if task == "family":
        return ev["gold_family"], reg["family_coords"]
    raise SystemExit(f"unknown task {task}")


def audit_view(crow: dict[str, str], ds: str, mask: np.ndarray, labels: np.ndarray, label_names: list[str],
               reg: dict[str, Any], ev: dict[str, Any], conf: dict[str, Any], keys: np.ndarray | None,
               values: np.ndarray | None, open_sink: list[dict[str, Any]] | None = None):
    dm = dataset_mask(conf, ds)
    path = crow["certificate_path"]
    if path == "A":
        return audit_path_a(crow["view_id"], crow["basis"], crow["carrier_edges"], mask, keys, labels, label_names, dm, ev, conf)
    if path == "B":
        return audit_path_b(crow["view_id"], crow["task"], mask, values, labels, label_names, ds, dm, reg, ev, conf,
                            open_sink)
    if path == "C":
        return audit_path_c(crow["view_id"], mask, keys, labels, label_names, dm, conf)
    raise SystemExit(f"unknown certificate path {path}")


def view_inputs(crow: dict[str, str], reg: dict[str, Any], ev: dict[str, Any]):
    if crow["certificate_path"] == "B":
        return None, dense_projection(crow["view_id"], reg, ev)
    return view_keys(crow["view_id"], reg, ev), None


METRIC_COUNT_KEYS = ("fibers", "unsafe_rows", "nondetermined_rows", "unverified_observed_rows", "unverified_defect_rows",
                     "certificate_label_conflicts", "infeasible_fibers")


def run_verdicts(reg: dict[str, Any], conf: dict[str, Any], ev: dict[str, Any], log=print) -> dict[str, Any]:
    """Audit all registered views on both datasets with the stored query masks and gold labels."""
    nviews, n = len(reg["cert"]), conf["n"]
    verdict_cache = np.zeros((nviews, n), dtype=np.uint8)
    result_rows: list[dict[str, Any]] = []
    metrics_by: dict[tuple[str, str], dict[str, Any]] = {}
    views_out: dict[str, Any] = {}
    path_b_unsafe: list[dict[str, Any]] = []
    path_b_open: list[dict[str, Any]] = []
    for i, crow in enumerate(reg["cert"]):
        vid, task, path = crow["view_id"], crow["task"], crow["certificate_path"]
        log(f"[verdicts] view {i + 1:02d}/{nviews} {vid} path={path}")
        labels, names = labels_for(task, reg, ev)
        keys, values = view_inputs(crow, reg, ev)
        views_out[vid] = {"task": task, "path": path, "datasets": {}}
        for ds in DATASETS:
            dm = dataset_mask(conf, ds)
            v, _ul, metrics, witnesses, unsafe_certs = audit_view(crow, ds, conf["qmask"][i], labels, names,
                                                                  reg, ev, conf, keys, values, path_b_open)
            verdict_cache[i, dm] = v[dm]
            metrics_by[(vid, ds)] = metrics
            views_out[vid]["datasets"][ds] = {"metrics_counts": {k: metrics[k] for k in METRIC_COUNT_KEYS},
                                              "representative_witnesses": witnesses}
            for rec in unsafe_certs:
                path_b_unsafe.append({"view_id": vid, "task": task, "dataset": ds, **rec})
            result_rows.append({"view": vid, "dataset": ds, "basis": crow["basis"], "path": path,
                                "fibers": metrics["fibers"], "m^U": float(metrics["mU"]), "m^N": float(metrics["mN"]),
                                "m^V_obs": float(metrics["mV_obs"]), "m^V_defect": float(metrics["mV_defect"]),
                                "near_pure": float(metrics["near_pure"]), "purity": float(metrics["purity"])})
    conflicts = sum(m["certificate_label_conflicts"] for m in metrics_by.values())
    infeasible = sum(m["infeasible_fibers"] for m in metrics_by.values())
    if conflicts:
        raise SystemExit(f"UNSAFE certificates conflict with observed labels on {conflicts} rows")
    if infeasible:
        raise SystemExit(f"{infeasible} observed Path B fibers are infeasible")
    return {"verdict_cache": verdict_cache, "result_rows": result_rows, "metrics_by": metrics_by,
            "views": views_out, "path_b_unsafe": path_b_unsafe, "path_b_open": path_b_open}


# ====================================================================================================
# Decomposition, Path B verification, Fig. 1
# ====================================================================================================

def decomposition(reg: dict[str, Any], conf: dict[str, Any], ev: dict[str, Any], verdict_cache: np.ndarray,
                  view_id: str = "identity_admitted__family") -> list[dict[str, Any]]:
    """UNSAFE rows of the primary W (facet basis) certificate versus the alternative Pi|W (core basis)
    composition; the two coincide row-wise on FULL-CONFORM rows."""
    fam_labels, _ = labels_for("family", reg, ev)
    alt_unique, alt_label = cert.strict_unique_max(ev["core_family"])
    alt_unsafe = conf["full"] & alt_unique
    if np.any(alt_unsafe & (alt_label != fam_labels)):
        raise SystemExit("the Pi|W certificate conflicts with an observed family label")
    i = [r["view_id"] for r in reg["cert"]].index(view_id)
    primary = verdict_cache[i] == UNSAFE
    full = conf["full"]
    out = []
    for ds in DATASETS:
        dm = dataset_mask(conf, ds)
        if not np.array_equal(primary[dm & full], alt_unsafe[dm & full]):
            raise SystemExit(f"W and Pi|W UNSAFE sets differ on FULL-CONFORM rows of {ds}")
        out.append({"dataset": ds, "view_id": view_id, "primary_path": "W(facet basis)",
                    "alternative_path": "Pi|W(core basis)",
                    "primary_unsafe_rows": int(np.count_nonzero(dm & primary)),
                    "primary_unsafe_full_rows": int(np.count_nonzero(dm & primary & full)),
                    "primary_unsafe_outside_full_rows": int(np.count_nonzero(dm & primary & ~full)),
                    "alternative_unsafe_rows": int(np.count_nonzero(dm & alt_unsafe)),
                    "alternative_unsafe_outside_full_rows": int(np.count_nonzero(dm & alt_unsafe & ~full)),
                    "full_conform_rowwise_equal": 1})
    return out


CARRIER_PAIRS = {"family": ("l0_family__family", "identity_admitted__family"),
                 "facet": ("l0_facet__facet", "identity_admitted__facet")}
CARRIER_EQUIVALENCE_FIELDS = ["task", "dataset", "direct_view", "identity_view", "direct_unsafe_rows",
                              "identity_unsafe_rows", "direct_only_rows", "direct_only_identity_query_defect_rows",
                              "direct_only_outside_full_conform_rows", "direct_only_with_color_rows",
                              "identity_only_rows", "identity_query_conform_rows",
                              "mismatch_on_identity_query_conform_rows"]


def carrier_equivalence(reg: dict[str, Any], conf: dict[str, Any], ev: dict[str, Any],
                        verdict_cache: np.ndarray) -> list[dict[str, Any]]:
    """UNSAFE anchors of the identity-admitted view of each task versus the direct-carrier view of that task.

    On the anchors that pass the identity-admitted view's QUERY-CONFORM check the two UNSAFE sets coincide; every
    anchor that only the direct-carrier view certifies is a query defect of the identity-admitted view."""
    views = [r["view_id"] for r in reg["cert"]]
    fidx = {f: i for i, f in enumerate(reg["facet_coords"])}
    if "color" not in fidx:
        raise SystemExit("the facet block has no color coordinate")
    color = np.asarray(ev["facet"][:, fidx["color"]]) > 0
    rows = []
    for task, (direct_view, identity_view) in CARRIER_PAIRS.items():
        i, j = views.index(direct_view), views.index(identity_view)
        direct = verdict_cache[i] == UNSAFE
        identity = verdict_cache[j] == UNSAFE
        q = conf["qmask"][j]
        for ds in DATASETS:
            dm = dataset_mask(conf, ds)
            only = dm & direct & ~identity
            row = {"task": task, "dataset": ds, "direct_view": direct_view, "identity_view": identity_view,
                   "direct_unsafe_rows": int(np.count_nonzero(dm & direct)),
                   "identity_unsafe_rows": int(np.count_nonzero(dm & identity)),
                   "direct_only_rows": int(np.count_nonzero(only)),
                   "direct_only_identity_query_defect_rows": int(np.count_nonzero(only & ~q)),
                   "direct_only_outside_full_conform_rows": int(np.count_nonzero(only & ~conf["full"])),
                   "direct_only_with_color_rows": int(np.count_nonzero(only & color)),
                   "identity_only_rows": int(np.count_nonzero(dm & identity & ~direct)),
                   "identity_query_conform_rows": int(np.count_nonzero(dm & q)),
                   "mismatch_on_identity_query_conform_rows": int(np.count_nonzero(dm & q & (direct != identity)))}
            if (row["mismatch_on_identity_query_conform_rows"] or row["identity_only_rows"]
                    or row["direct_only_identity_query_defect_rows"] != row["direct_only_rows"]):
                raise SystemExit(f"{task} {ds}: the identity-admitted and direct-carrier UNSAFE sets differ on "
                                 "query-conforming anchors")
            rows.append(row)
    return rows


DECOMPOSITION_FIELDS = ["dataset", "view_id", "primary_path", "alternative_path", "primary_unsafe_rows",
                        "primary_unsafe_full_rows", "primary_unsafe_outside_full_rows", "alternative_unsafe_rows",
                        "alternative_unsafe_outside_full_rows", "full_conform_rowwise_equal"]
PATH_B_FIELDS = ["view_id", "dataset", "group_index", "fiber_value", "certified_label", "verification_method",
                 "integer_states_checked", "state_set_sha256", "free_coordinate_count", "minimum_strict_margin",
                 "maximum_strict_margin", "rival_optima_exact_match", "pass"]


def _support_facets(reg: dict[str, Any], ds: str) -> list[str]:
    support = set(reg["support"]["datasets"][ds]["S_members"])
    support_facets = [f for f in reg["facet_coords"] if f in support]
    if len(support_facets) != int(reg["support"]["datasets"][ds]["n_dataset_support"]):
        raise SystemExit(f"{ds}: support cardinality differs from the registry")
    return support_facets


def _requirements(reg: dict[str, Any], vid: str, fiber_value: list[int]) -> dict[str, dict[str, int]]:
    requirements: dict[str, dict[str, int]] = {}
    k = 0
    for block, selector in reg["projection"][vid].items():
        for x in selector:
            requirements.setdefault(BLOCK_MAP[block], {})[x] = int(fiber_value[k])
            k += 1
    return requirements


def verify_path_b(reg: dict[str, Any], path_b_unsafe: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Solver-independent check of every Path B UNSAFE certificate."""
    maps = {"W": reg["W"], "T": reg["T"]}
    rows = []
    for rec in path_b_unsafe:
        ds, vid = rec["dataset"], rec["view_id"]
        support_facets = _support_facets(reg, ds)
        requirements = _requirements(reg, vid, rec["fiber_value"])
        names = reg["facet_coords"] if rec["task"] == "facet" else reg["family_coords"]
        rows.append(cert.verify_path_b_certificate(
            view_id=vid, dataset=ds, group_index=rec["group_index"], task=rec["task"], fiber_value=rec["fiber_value"],
            requirements=requirements, certified_label=rec["certified_label"],
            reported_optima=cert.reported_optima(rec["rival_optima"]), support_facets=support_facets,
            label_names=names, maps=maps))
    if not all(int(r["pass"]) == 1 for r in rows):
        raise SystemExit("a Path B certificate failed verification")
    return rows


PATH_B_OPEN_FIELDS = ["view_id", "dataset", "group_index", "fiber_value", "observed_label", "rival", "program_status",
                      "rival_minus_observed_optimum", "witness_state", "rival_minus_observed_at_state",
                      "free_coordinates_to_rival", "query_conforming_fiber_size", "pass"]
# The public subset of the open-witness table leaves out every DeepFashion row whose query-conforming fiber holds a
# single anchor (DATA_LICENSE.md, section 3); the full table is restricted and identified by its SHA-256.
PATH_B_OPEN_PUBLIC_RULE = ("all rows of verdicts/path_b_open_witnesses.csv except the DF rows whose "
                           "query_conforming_fiber_size is 1")


def verify_path_b_open(reg: dict[str, Any], path_b_open: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Solver-independent check of every open single-label Path B fiber: its state lies in the fiber and the
    rival ties with or exceeds the observed label there."""
    maps = {"W": reg["W"], "T": reg["T"]}
    rows = []
    for rec in path_b_open:
        ds, vid = rec["dataset"], rec["view_id"]
        names = reg["facet_coords"] if rec["task"] == "facet" else reg["family_coords"]
        rows.append({"query_conforming_fiber_size": int(rec["fiber_size"])})
        rows[-1].update(cert.verify_open_witness(
            view_id=vid, dataset=ds, group_index=rec["group_index"], task=rec["task"], fiber_value=rec["fiber_value"],
            requirements=_requirements(reg, vid, rec["fiber_value"]), observed_label=rec["observed_label"],
            rival=rec["rival"], program_status=rec["program_status"],
            reported_optimum=rec["rival_minus_observed_optimum"], state=rec["state"],
            support_facets=_support_facets(reg, ds), label_names=names, maps=maps))
    if not all(int(r["pass"]) == 1 for r in rows):
        raise SystemExit("an open Path B fiber failed verification")
    return rows


def path_b_open_public(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Public subset of the open-witness rows (PATH_B_OPEN_PUBLIC_RULE)."""
    return [r for r in rows if not (r["dataset"] == "DF" and int(r["query_conforming_fiber_size"]) == 1)]


def path_b_open_summary(rows: list[dict[str, Any]], public: list[dict[str, Any]], full_sha: str,
                        public_sha: str) -> dict[str, Any]:
    """Counts of the open-witness table and of its public subset, without row values."""
    def tally(rs: list[dict[str, Any]]) -> dict[str, Any]:
        return {"rows": len(rs), "by_dataset": dict(sorted(Counter(r["dataset"] for r in rs).items())),
                "by_view_and_dataset": {f"{v}|{d}": n for (v, d), n in
                                        sorted(Counter((r["view_id"], r["dataset"]) for r in rs).items())},
                "by_program_status": dict(sorted(Counter(r["program_status"] for r in rs).items())),
                "rows_passing_exact_check": sum(int(r["pass"]) == 1 for r in rs)}
    withheld = [r for r in rows if r["dataset"] == "DF" and int(r["query_conforming_fiber_size"]) == 1]
    if len(public) + len(withheld) != len(rows):
        raise SystemExit("the public open-witness subset and the withheld rows do not partition the table")
    return {"full_table": {"path": "verdicts/path_b_open_witnesses.csv", "sha256": full_sha, **tally(rows),
                           "released": False,
                           "reason": f"contains {len(withheld)} rows that each describe a DeepFashion fiber with a "
                                     "single query-conforming anchor"},
            "public_table": {"path": "verdicts/path_b_open_witnesses_public.csv", "sha256": public_sha,
                             **tally(public), "rule": PATH_B_OPEN_PUBLIC_RULE},
            "withheld_rows": {**tally(withheld), "rule": "DF rows whose query_conforming_fiber_size is 1"}}


def fig1_payload(ev: dict[str, Any]) -> dict[str, Any]:
    return {"selection_uses_label": False, "selection_rule": FIG1_RULE, "candidate_rows": int(ev["fig1_candidates"]),
            "selected": ev["fig1_selected"] or {}}
