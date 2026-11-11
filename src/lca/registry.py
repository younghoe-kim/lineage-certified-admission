# SPDX-License-Identifier: MIT
"""Release registry (maps Pi, T, W), certificate specification, protected query and dataset support."""
from __future__ import annotations

import ast
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from lca import io as lio
from lca.names import BLOCKS, DATASETS, FAMILY_ORDER, INST_FAMILY_ORDER

# ----------------------------------------------------------------------------
# Input column lists (every loader reads exactly these columns).
# ----------------------------------------------------------------------------
CORE_TERMS_COLUMNS = ["core_term_uid", "term_type", "facet"]
EDGE_COLUMNS = ["src_uid", "edge_type", "dst_uid"]
SUPPORT_ANCHOR_COLUMNS = ["dataset", "full_feats_json"]
LABEL_PREFIX = "gold_"
ANCHOR_CHUNK_ROWS = 25000

# Sizes of the released inputs.
N_CORE_TERMS = 2715
N_REGISTRY_EDGES = 15006
N_FACETS = 27
N_ANCHORS = 631384
DATASET_ROWS = {"DF": 289202, "FP": 342182}

# Registry representation of facet identifiers on IN_FACET edges.
FACET_UID_PREFIX = "hub:facet/"

# Feature blocks of full_feats_json, in the declared block order.
BLOCK_PREFIXES = tuple(f"{b}::" for b in BLOCKS)
BLOCK_CORE = "core::"
BLOCK_FACET = "facet::"
BLOCK_TYPE = "type::"
BLOCK_FAMILY = "family::"
AUX_BLOCKS = {"rep_core::", "inst_family::", "inst_sem::"}

SUPPORT_NAME = {BLOCK_CORE: "C_i", BLOCK_FACET: "S_i", BLOCK_FAMILY: "w(C_i)"}
RULE_TEXT = ("shortest registered path from an exposed block to the carrier; "
             "facet before core on ties")
MAP_ORDER = ("Pi", "T", "W")

# Materialized facet -> family map: dense solve of the integer normal equations.
MATERIALIZED_SOLVER = {
    "method": "numpy.linalg.solve on integer normal equations X'X b = X'Y",
    "coefficient_tolerance": 1e-8,
    "proof": "exact integer residual sum of squares",
}

REGISTRY_ID = "L^rel"
REGISTRY_SCOPE = "relative to L^rel; does not reconstruct the generator's tie state"

CERTIFICATE_EDGES = {
    "Pi": {"source_block": "core::", "target_block": "facet::",
           "meaning": "cleaned authority IN_FACET membership"},
    "T": {"source_block": "facet::", "target_block": "type::",
          "meaning": "unique cleaned term_type per authority facet"},
    "W": {"source_block": "facet::", "target_block": "family::",
          "meaning": "derive_family(facet, unique facet term_type)"},
}

PROTECTED_QUERY = {
    "authority": "frozen benchmark generator semantics",
    "tasks": {
        "facet": {"carrier_block": "facet::", "query": "strict unique argmax; otherwise rho"},
        "family": {"carrier_block": "family::", "query": "strict unique argmax; otherwise rho"},
    },
    "rho": {
        "definition": "the maximizing label whose evidence occurs first in the event stream",
        "implementation_evidence": "Counter.most_common(1)",
        "generator_lines": [382, 383],
    },
    "latent_state": {
        "name": "z",
        "meaning": "event-stream evidence order used by the tie rule",
        "released": False,
    },
    "scope": ("The released authority does not contain z; tie-dependent fibers are therefore "
              "not certified UNSAFE by strict-max evidence alone."),
}

# Certificate rows that the design fixes for the named views.
DESIGN_CERTIFICATES = {
    "identity_admitted__family": ("facet::", "S_i", "A", "W", "W"),
    "core_plus_type__family": ("core::", "C_i", "A", "Pi|W", "Pi|W"),
    "type3__family": ("facet::", "S_i", "B", "T|W", "W"),
    "identity_admitted__facet": ("core::", "C_i", "A", "Pi", "Pi"),
    "family3__facet": ("facet::", "S_i", "B", "W", ""),
    "family3_plus_type__facet": ("facet::", "S_i", "B", "T|W", ""),
    "l0_family__family": ("family::", "w(C_i)", "A", "", ""),
    "l0_facet__facet": ("facet::", "S_i", "A", "", ""),
    "inst_family__family": ("", "", "C", "", ""),
    "rep_core__family": ("", "", "C", "", ""),
}
CERT_KEYS = ("basis", "basis_support", "certificate_path", "required_maps", "carrier_edges")

CERT_FIELDS = ["view_id", "task", "basis", "basis_support", "certificate_path",
               "required_maps", "carrier_edges", "rule"]
VIEW_FIELDS = ["view_id", "task", "view_group", "exposed_blocks"]
FACET_MAP_FIELDS = ["facet", "term_type", "family"]
TERM_TO_FACET_FIELDS = ["core_term_uid", "core_facet", "edge_facet_uid", "normalized_edge_facet", "match"]
DIAG_FIELDS = ["row_count", "record_type", "facet", "declared_family", "materialized_family",
               "mapping_equal", "rows_checked", "exact_rows", "df_rows_checked", "df_exact_rows",
               "fp_rows_checked", "fp_exact_rows", "declared_W_mismatch_df", "declared_W_mismatch_fp",
               "materialized_map_hash", "solver_rank", "proof_method"]


# ----------------------------------------------------------------------------
# Label rule (copy of label_rule.py; its AST must equal the released rule).
# ----------------------------------------------------------------------------

def norm_str(x) -> str:
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = " ".join(x.split())
    return x


def derive_family(facet: str, term_type: str) -> str:
    facet_n = norm_str(facet)
    tt = norm_str(term_type)

    if tt == "category" or facet_n.startswith("category/"):
        return "taxonomic_category"
    if tt == "part" or facet_n.startswith("part/"):
        return "structural_expression"
    if tt == "color" or facet_n.startswith("color/"):
        return "surface_expression"
    if "nickname" in facet_n or facet_n.startswith("tag/") or tt == "tag":
        return "tag_terms"

    if facet_n.startswith("attribute/"):
        if any(k in facet_n for k in [
            "pattern", "textile", "material", "print", "surface", "leather",
            "non textile material", "finishing", "manufacturing"
        ]):
            return "surface_expression"
        if any(k in facet_n for k in [
            "silhouette", "fit", "length", "neckline", "opening",
            "garment detail", "hem", "sleeve"
        ]):
            return "structural_expression"
        return "attribute_expression"

    if tt == "attribute":
        return "attribute_expression"

    return "other_expression"


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise SystemExit(f"expected one function {name!r} in the label rule, found {len(nodes)}")
    return nodes[0]


def verify_label_rule(rule_source: str) -> None:
    """Stop unless norm_str and derive_family in this module have the same AST as the released rule.
    The rule file is parsed, never imported or executed."""
    released = ast.parse(rule_source)
    local = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for name in ("norm_str", "derive_family"):
        a = ast.dump(_function_node(released, name), annotate_fields=True, include_attributes=False)
        b = ast.dump(_function_node(local, name), annotate_fields=True, include_attributes=False)
        if a != b:
            raise SystemExit(f"{name} differs from the released label rule")


# ----------------------------------------------------------------------------
# Orders and hashes
# ----------------------------------------------------------------------------

def family_sort_key(label: str) -> tuple[int, str]:
    """Position of a family label in the declared order (names.FAMILY_ORDER, then names.INST_FAMILY_ORDER)."""
    for k, order in enumerate((FAMILY_ORDER, INST_FAMILY_ORDER)):
        if label in order:
            return (k, f"{order.index(label):03d}")
    raise SystemExit(f"family label outside the declared orders: {label}")


# Blocks whose coordinates are family-like labels and follow the declared family orders.
FAMILY_LABEL_BLOCKS = {"family::": FAMILY_ORDER, "inst_family::": INST_FAMILY_ORDER}


def ordered_coords(block: str, values: Iterable[str]) -> list[str]:
    vals = set(values)
    if block in FAMILY_LABEL_BLOCKS:
        order = FAMILY_LABEL_BLOCKS[block]
        if not vals <= set(order):
            raise SystemExit(f"labels of {block} outside the declared order")
        return [x for x in order if x in vals]
    return sorted(vals)


def json_sha256(obj: Any) -> str:
    return lio.sha256_bytes(lio.canonical_json_bytes(obj, decorate=False))


# ----------------------------------------------------------------------------
# Input readers
# ----------------------------------------------------------------------------

def read_sha256sums(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.strip()] = digest
    return out


def read_columns(path: Path, columns: list[str]) -> list[dict[str, str]]:
    """Read exactly `columns` of a CSV file as strings."""
    df = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False, na_filter=False,
                     encoding="utf-8-sig")
    if set(df.columns) != set(columns):
        raise SystemExit(f"{Path(path).name}: loaded columns {list(df.columns)} != {columns}")
    return df[columns].to_dict("records")


def iter_anchor_chunks(path: Path, columns: list[str]) -> Iterator[pd.DataFrame]:
    """Stream exactly `columns` of anchors.csv.gz as strings, in release order."""
    reader = pd.read_csv(path, compression="gzip", usecols=columns, dtype=str, keep_default_na=False,
                         na_filter=False, chunksize=ANCHOR_CHUNK_ROWS, low_memory=False)
    for chunk in reader:
        if set(chunk.columns) != set(columns):
            raise SystemExit(f"anchor loader returned columns {list(chunk.columns)} != {columns}")
        yield chunk


def exact_count(v: Any, key: str) -> int:
    """A released feature value as a non-negative integer count."""
    if isinstance(v, bool):
        raise SystemExit(f"boolean feature count at {key!r}")
    try:
        x = float(v)
    except Exception as exc:
        raise SystemExit(f"non-numeric feature count at {key!r}: {v!r}") from exc
    if not math.isfinite(x) or x < 0 or not x.is_integer():
        raise SystemExit(f"feature count at {key!r} is not a non-negative integer: {v!r}")
    return int(x)


# ----------------------------------------------------------------------------
# Registry maps
# ----------------------------------------------------------------------------

def build_registry_maps(core_rows: list[dict[str, str]], edge_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Pi (core term -> facet), tau (term -> type), T (facet -> type), W (facet -> family), w = Pi W."""
    core: dict[str, dict[str, str]] = {}
    for row in core_rows:
        uid = row["core_term_uid"].strip()
        if not uid:
            raise SystemExit("blank core_term_uid")
        if uid in core:
            raise SystemExit(f"duplicate core_term_uid: {uid}")
        core[uid] = row

    relation_counts = dict(sorted(Counter(r["edge_type"] for r in edge_rows).items()))
    in_facet = [r for r in edge_rows if r["edge_type"] == "IN_FACET"]

    edge_uid: dict[str, str] = {}
    edge_facet: dict[str, str] = {}
    bad: list[tuple[str, str]] = []
    for row in in_facet:
        src, dst = row["src_uid"].strip(), row["dst_uid"].strip()
        if not src or not dst:
            raise SystemExit(f"blank IN_FACET endpoint: {src!r} -> {dst!r}")
        if src in edge_facet:
            raise SystemExit(f"several IN_FACET edges for {src}")
        sem = dst[len(FACET_UID_PREFIX):] if dst.startswith(FACET_UID_PREFIX) else ""
        if not sem or sem.startswith(FACET_UID_PREFIX):
            bad.append((src, dst))
            continue
        edge_uid[src] = dst
        edge_facet[src] = sem
    if bad:
        raise SystemExit(f"IN_FACET targets outside the {FACET_UID_PREFIX!r} namespace: {bad[:10]}")
    if set(edge_facet) != set(core):
        raise SystemExit("IN_FACET sources and core terms differ")

    pi: dict[str, str] = {}
    tau: dict[str, str] = {}
    equivalence: list[dict[str, Any]] = []
    for uid in sorted(core):
        facet = core[uid]["facet"].strip()
        match = edge_facet[uid] == facet
        if not match:
            raise SystemExit(f"IN_FACET facet differs from the core facet for {uid}")
        equivalence.append({"core_term_uid": uid, "core_facet": facet, "edge_facet_uid": edge_uid[uid],
                            "normalized_edge_facet": edge_facet[uid], "match": int(match)})
        pi[uid] = facet
        tau[uid] = core[uid]["term_type"].strip()

    facets = sorted(set(pi.values()))
    if sorted(set(edge_facet.values())) != facets or len(set(edge_uid.values())) != len(facets):
        raise SystemExit("IN_FACET facet vocabulary differs from the core facet vocabulary")
    types_by_facet: dict[str, set[str]] = {f: set() for f in facets}
    for uid in pi:
        types_by_facet[pi[uid]].add(tau[uid])
    multi = {f: sorted(t) for f, t in types_by_facet.items() if len(t) != 1}
    if multi:
        raise SystemExit(f"facets without a unique term_type: {multi}")

    T = {f: next(iter(types_by_facet[f])) for f in facets}
    W = {f: derive_family(f, T[f]) for f in facets}
    w = {uid: derive_family(pi[uid], tau[uid]) for uid in sorted(pi)}
    if any(tau[u] != T[pi[u]] for u in pi):
        raise SystemExit("tau != Pi T")
    if any(w[u] != W[pi[u]] for u in pi):
        raise SystemExit("w != Pi W")

    return {
        "Pi": pi, "tau": tau, "w": w, "T": T, "W": W, "facets": facets,
        "relation_counts": relation_counts,
        "n_in_facet_edges": len(in_facet),
        "n_facets_single_type": sum(1 for t in types_by_facet.values() if len(t) == 1),
        "n_in_facet_namespace_valid": len(edge_facet),
        "n_in_facet_matches": sum(r["match"] for r in equivalence),
        "n_unique_edge_facet_uids": len(set(edge_uid.values())),
        "n_unique_normalized_edge_facets": len(set(edge_facet.values())),
        "term_to_facet_rows": equivalence,
        "edge_uid": edge_uid,
    }


# ----------------------------------------------------------------------------
# Views and certificate specification
# ----------------------------------------------------------------------------

def certificate_for_view(task: str, exposed_blocks: tuple[str, ...]) -> dict[str, str]:
    """Certificate of a view: the shortest registered path from an exposed block to the carrier."""
    carrier = BLOCK_FACET if task == "facet" else BLOCK_FAMILY
    exposed = set(exposed_blocks)

    def cert(basis: str, path: str, maps: str, edges: str) -> dict[str, str]:
        return {"basis": basis, "basis_support": SUPPORT_NAME.get(basis, ""), "certificate_path": path,
                "required_maps": maps, "carrier_edges": edges}

    if carrier in exposed:
        return cert(carrier, "A", "", "")
    if task == "family":
        if BLOCK_FACET in exposed:
            return cert(BLOCK_FACET, "A", "W", "W")
        if BLOCK_CORE in exposed:
            return cert(BLOCK_CORE, "A", "Pi|W", "Pi|W")
    if task == "facet" and BLOCK_CORE in exposed:
        return cert(BLOCK_CORE, "A", "Pi", "Pi")

    # Path B: the facet block expresses every exposed block and the carrier through registered maps.
    facet_image = {BLOCK_FACET: [], BLOCK_TYPE: ["T"], BLOCK_FAMILY: ["W"]}
    if exposed and not (exposed & AUX_BLOCKS) and all(b in facet_image for b in exposed):
        req = {m for b in exposed for m in facet_image[b]}
        carrier_edges = ["W"] if carrier == BLOCK_FAMILY else []
        req.update(carrier_edges)
        return cert(BLOCK_FACET, "B", "|".join(m for m in MAP_ORDER if m in req), "|".join(carrier_edges))

    # Path C: no registered common basis.
    return cert("", "C", "", "")


def declared_views() -> list[dict[str, Any]]:
    """The 32 declared view-task pairs, ordered by view_id."""
    views: list[dict[str, Any]] = []
    for F in (0, 1):
        for M in (0, 1):
            for T in (0, 1):
                blocks = [BLOCK_CORE] + ([BLOCK_FACET] if F else []) + ([BLOCK_FAMILY] if M else []) \
                         + ([BLOCK_TYPE] if T else [])
                for task in ("facet", "family"):
                    views.append({"view_id": f"factorial_F{F}M{M}T{T}__{task}", "task": task,
                                  "exposed_blocks": tuple(blocks), "view_group": "factorial_8cell"})
    named = [
        ("identity_admitted__family", "family", (BLOCK_CORE, BLOCK_FACET, BLOCK_TYPE), "primary"),
        ("core_plus_type__family", "family", (BLOCK_CORE, BLOCK_TYPE), "primary"),
        ("facet_only__family", "family", (BLOCK_FACET,), "aggregate"),
        ("type3__family", "family", (BLOCK_TYPE,), "aggregate"),
        ("type_attr_part__family", "family", (BLOCK_TYPE,), "aggregate"),
        ("identity_admitted__facet", "facet", (BLOCK_CORE, BLOCK_FAMILY, BLOCK_TYPE), "primary"),
        ("family3__facet", "facet", (BLOCK_FAMILY,), "primary_negative"),
        ("family3_plus_type__facet", "facet", (BLOCK_FAMILY, BLOCK_TYPE), "aggregate"),
        ("l0_family__family", "family", (BLOCK_FAMILY,), "direct_carrier"),
        ("l0_facet__facet", "facet", (BLOCK_FACET,), "direct_carrier"),
    ]
    for view_id, task, blocks, group in named:
        views.append({"view_id": view_id, "task": task, "exposed_blocks": tuple(blocks), "view_group": group})
    for block in ("inst_family::", "rep_core::", "inst_sem::"):
        stem = block.replace("::", "")
        for task in ("facet", "family"):
            views.append({"view_id": f"{stem}__{task}", "task": task, "exposed_blocks": (block,),
                          "view_group": "path_c_control"})
    ids = [v["view_id"] for v in views]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate declared view_id")
    return sorted(views, key=lambda r: r["view_id"])


def build_certificate_spec(views: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    cert_rows, view_rows = [], []
    for v in views:
        c = certificate_for_view(v["task"], v["exposed_blocks"])
        cert_rows.append({"view_id": v["view_id"], "task": v["task"], **c, "rule": RULE_TEXT})
        view_rows.append({"view_id": v["view_id"], "task": v["task"], "view_group": v["view_group"],
                          "exposed_blocks": "|".join(v["exposed_blocks"])})
    by_id = {r["view_id"]: r for r in cert_rows}
    for view_id, expected in DESIGN_CERTIFICATES.items():
        got = tuple(by_id[view_id][k] for k in CERT_KEYS)
        if got != expected:
            raise SystemExit(f"certificate of {view_id} is {got}, design fixes {expected}")
    return cert_rows, view_rows


def build_registry_document(maps: dict[str, Any], n_core: int, n_edges: int, views: list[dict[str, Any]],
                            cert_rows: list[dict[str, str]], view_rows: list[dict[str, str]],
                            inputs_sha256: dict[str, str]) -> dict[str, Any]:
    pi, tau, w, T, W = maps["Pi"], maps["tau"], maps["w"], maps["T"], maps["W"]
    n_match = maps["n_in_facet_matches"]
    return {
        "step": "registry",
        "schema_version": "1",
        "inputs_sha256": inputs_sha256,
        "registry_id": REGISTRY_ID,
        "registry_scope": REGISTRY_SCOPE,
        "rule_id": "derive_family",
        "generator_imported": False,
        "generator_executed": False,
        "family_order": list(FAMILY_ORDER),
        "authority_counts": {
            "n_authority_terms": n_core,
            "n_authority_facets": len(maps["facets"]),
            "n_facets_with_single_declared_type": maps["n_facets_single_type"],
            "n_in_facet_edges": maps["n_in_facet_edges"],
            "n_in_facet_namespace_valid": maps["n_in_facet_namespace_valid"],
            "n_in_facet_matches_core_facet": n_match,
            "n_unique_edge_facet_uids": maps["n_unique_edge_facet_uids"],
            "n_unique_normalized_edge_facets": maps["n_unique_normalized_edge_facets"],
            "n_ontology_edges": n_edges,
            "relation_counts": maps["relation_counts"],
        },
        "certificate_edges": CERTIFICATE_EDGES,
        "facet_uid_representation": {
            "edge_target_prefix": FACET_UID_PREFIX,
            "normalization": f"remove exactly one leading {FACET_UID_PREFIX} namespace",
            "semantic_equivalence_required": f"{n_match}/{maps['n_in_facet_edges']}",
        },
        "Pi_edge_uid_hash": json_sha256([{"core_term_uid": u, "edge_facet_uid": maps["edge_uid"][u]}
                                         for u in sorted(maps["edge_uid"])]),
        "Pi_hash": json_sha256([{"core_term_uid": u, "facet": pi[u]} for u in sorted(pi)]),
        "tau_hash": json_sha256([{"core_term_uid": u, "term_type": tau[u]} for u in sorted(tau)]),
        "w_hash": json_sha256([{"core_term_uid": u, "family": w[u]} for u in sorted(w)]),
        "T_hash": json_sha256([{"facet": f, "term_type": T[f]} for f in sorted(T)]),
        "W_hash": json_sha256([{"facet": f, "family": W[f]} for f in sorted(W)]),
        "certificate_spec_hash": json_sha256(cert_rows),
        "view_schema_hash": json_sha256(view_rows),
        "maps": {
            "Pi_core_to_facet": pi,
            "tau_core_to_type": tau,
            "w_core_to_family": w,
            "T_facet_to_type": T,
            "W_facet_to_family": W,
        },
        "views": [{"view_id": v["view_id"], "task": v["task"], "view_group": v["view_group"],
                   "exposed_blocks": list(v["exposed_blocks"])} for v in views],
        "protected_query": PROTECTED_QUERY,
    }


# ----------------------------------------------------------------------------
# Registry tables written by the registry step and read by the subsequent steps
# ----------------------------------------------------------------------------

def load_registry_tables(term_to_facet: Path, facet_map: Path) -> dict[str, dict[str, str]]:
    """Pi from term_to_facet.csv, T and W from facet_map.csv."""
    pi: dict[str, str] = {}
    for r in read_columns(term_to_facet, TERM_TO_FACET_FIELDS):
        if r["match"] != "1":
            raise SystemExit("term_to_facet.csv contains a non-matching row")
        if r["core_term_uid"] in pi:
            raise SystemExit(f"duplicate core term in term_to_facet.csv: {r['core_term_uid']}")
        pi[r["core_term_uid"]] = r["core_facet"]
    T: dict[str, str] = {}
    W: dict[str, str] = {}
    for r in read_columns(facet_map, FACET_MAP_FIELDS):
        if r["facet"] in T:
            raise SystemExit(f"duplicate facet in facet_map.csv: {r['facet']}")
        T[r["facet"]] = r["term_type"]
        W[r["facet"]] = r["family"]
    return {"Pi": pi, "T": T, "W": W}


# ----------------------------------------------------------------------------
# Dataset support, populations and the materialized facet -> family map
# ----------------------------------------------------------------------------

def _feature_block(key: str) -> tuple[str, str]:
    for prefix in BLOCK_PREFIXES:
        if key.startswith(prefix):
            coord = key[len(prefix):]
            if not coord:
                raise SystemExit(f"empty coordinate after prefix: {key!r}")
            return prefix, coord
    raise SystemExit(f"unknown feature block in full_feats_json: {key!r}")


def _new_moments() -> dict[str, Any]:
    return {"xtx": defaultdict(int), "xty": defaultdict(int), "yty": defaultdict(int)}


def _add_moments(m: dict[str, Any], x: dict[str, int], y: dict[str, int]) -> None:
    for fa, xa in x.items():
        for fb, xb in x.items():
            m["xtx"][(fa, fb)] += xa * xb
    for f, xv in x.items():
        for g, yv in y.items():
            m["xty"][(f, g)] += xv * yv
    for g, yv in y.items():
        m["yty"][g] += yv * yv


def scan_support(chunks: Iterable[pd.DataFrame], W: dict[str, str]) -> dict[str, Any]:
    """One pass over (dataset, full_feats_json): coordinate populations, active sets per dataset,
    integer sufficient statistics for facet -> family, and rows where W reproduces the family block."""
    rows = Counter()
    coords: dict[str, set[str]] = {p: set() for p in BLOCK_PREFIXES}
    active: dict[str, dict[str, set[str]]] = {d: {p: set() for p in BLOCK_PREFIXES} for d in DATASETS}
    moments = {"all": _new_moments(), **{d: _new_moments() for d in DATASETS}}
    w_mismatch = Counter()
    for chunk in chunks:
        for ds, raw in zip(chunk["dataset"].tolist(), chunk["full_feats_json"].tolist()):
            if ds not in DATASETS:
                raise SystemExit(f"unexpected dataset value: {ds!r}")
            rows[ds] += 1
            feats = json.loads(raw)
            if not isinstance(feats, dict):
                raise SystemExit("full_feats_json is not an object")
            facet_counts: dict[str, int] = {}
            family_counts: dict[str, int] = {}
            for key, value in feats.items():
                prefix, coord = _feature_block(key)
                count = exact_count(value, key)
                coords[prefix].add(coord)
                if count <= 0:
                    continue
                active[ds][prefix].add(coord)
                if prefix == BLOCK_FACET:
                    facet_counts[coord] = count
                elif prefix == BLOCK_FAMILY:
                    family_counts[coord] = count
            _add_moments(moments["all"], facet_counts, family_counts)
            _add_moments(moments[ds], facet_counts, family_counts)
            predicted: dict[str, int] = defaultdict(int)
            for facet, count in facet_counts.items():
                if facet not in W:
                    raise SystemExit(f"materialized facet without a W entry: {facet}")
                predicted[W[facet]] += count
            if {k: v for k, v in predicted.items() if v} != {k: v for k, v in family_counts.items() if v}:
                w_mismatch[ds] += 1
    return {"rows": rows, "coords": coords, "active": active, "moments": moments, "w_mismatch": w_mismatch}


def solve_materialized_map(m: dict[str, Any], facets: list[str], families: list[str]) -> tuple[dict[str, str], int]:
    """Solve X'X b = X'Y on the materialized facets; every coefficient row must be one-hot integral."""
    xtx = np.array([[m["xtx"].get((a, b), 0) for b in facets] for a in facets], dtype=np.float64)
    xty = np.array([[m["xty"].get((f, g), 0) for g in families] for f in facets], dtype=np.float64)
    rank = int(np.linalg.matrix_rank(xtx))
    if rank != len(facets):
        raise SystemExit(f"materialized facet design is not full rank: {rank}/{len(facets)}")
    coef = np.linalg.solve(xtx, xty)
    tol = MATERIALIZED_SOLVER["coefficient_tolerance"]
    M: dict[str, str] = {}
    for i, facet in enumerate(facets):
        rounded = np.rint(coef[i, :])
        if float(np.max(np.abs(coef[i, :] - rounded))) > tol:
            raise SystemExit(f"materialized coefficient row is not integral for {facet}")
        ints = rounded.astype(int)
        ones = np.where(ints == 1)[0].tolist()
        if np.any((ints != 0) & (ints != 1)) or len(ones) != 1:
            raise SystemExit(f"materialized facet does not map one-hot to one family: {facet}")
        M[facet] = families[ones[0]]
    return M, rank


def residual_sumsq(m: dict[str, Any], facets: list[str], families: list[str], M: dict[str, str]) -> dict[str, int]:
    """Exact integer sum over rows of (y_g - sum_f x_f [M(f) = g])^2 for every family g."""
    out: dict[str, int] = {}
    for g in families:
        sel = [f for f in facets if M[f] == g]
        rss = int(m["yty"].get(g, 0))
        rss -= 2 * sum(int(m["xty"].get((f, g), 0)) for f in sel)
        rss += sum(int(m["xtx"].get((a, b), 0)) for a in sel for b in sel)
        out[g] = rss
    return out


def build_support_outputs(scan: dict[str, Any], maps: dict[str, dict[str, str]],
                          inputs_sha256: dict[str, str]) -> dict[str, Any]:
    """dataset_support.json, populations.json and the declared-vs-materialized rows."""
    pi, T, W = maps["Pi"], maps["T"], maps["W"]
    n_total = sum(scan["rows"].values())
    datasets: dict[str, Any] = {}
    for ds in DATASETS:
        act = scan["active"][ds]
        C = sorted(act[BLOCK_CORE])
        outside = [t for t in C if t not in pi]
        if outside:
            raise SystemExit(f"{ds}: active core terms outside Pi: {outside[:10]}")
        S = sorted({pi[t] for t in C})
        W_img = sorted({W[f] for f in S}, key=family_sort_key)
        obs_facets = sorted(act[BLOCK_FACET])
        obs_families = sorted(act[BLOCK_FAMILY], key=family_sort_key)
        obs_types = sorted(act[BLOCK_TYPE])
        datasets[ds] = {
            "n_rows": int(scan["rows"][ds]),
            "C_members": C, "C_hash": json_sha256(C), "n_dataset_active_terms": len(C),
            "S_members": S, "S_hash": json_sha256(S), "n_dataset_support": len(S),
            "W_img_members": W_img, "W_img_hash": json_sha256(W_img), "n_dataset_family_image": len(W_img),
            "observed_facet_members": obs_facets, "observed_facet_hash": json_sha256(obs_facets),
            "observed_facet_support": len(obs_facets),
            "observed_family_members": obs_families, "observed_family_hash": json_sha256(obs_families),
            "observed_type_members": obs_types, "observed_type_hash": json_sha256(obs_types),
            "observed_facet_extra_vs_S": sorted(set(obs_facets) - set(S)),
        }

    hub_core = sorted(set().union(*(scan["active"][d][BLOCK_CORE] for d in DATASETS)))
    if any(t not in pi for t in hub_core):
        raise SystemExit("hub core terms outside Pi")
    active_core_facets = sorted({pi[t] for t in hub_core})
    materialized = sorted(scan["coords"][BLOCK_FACET])
    authority = sorted(T)
    members = {p: ordered_coords(p, scan["coords"][p]) for p in BLOCK_PREFIXES}
    counts = {p: len(members[p]) for p in BLOCK_PREFIXES}
    families = members[BLOCK_FAMILY]

    M, rank = solve_materialized_map(scan["moments"]["all"], materialized, families)
    for key in ("all",) + DATASETS:
        rss = residual_sumsq(scan["moments"][key], materialized, families, M)
        if any(v != 0 for v in rss.values()):
            raise SystemExit(f"materialized facet -> family identity fails ({key}): {rss}")
    M_hash = json_sha256([{"facet": f, "family": M[f]} for f in materialized])

    t_records = [{"facet": f, "term_type": T[f]} for f in materialized]
    w_records = [{"facet": f, "family": W[f]} for f in materialized]
    support = {
        "step": "registry", "schema_version": "1", "inputs_sha256": inputs_sha256,
        "row_count": n_total, "label_columns_read": [], "datasets": datasets,
    }
    populations = {
        "step": "registry", "schema_version": "1", "inputs_sha256": inputs_sha256,
        "row_count": n_total, "n_anchor_total": n_total,
        "dataset_rows": {d: int(scan["rows"][d]) for d in DATASETS},
        "n_authority_facets": len(authority),
        "n_materialized_facet_coords": len(materialized),
        "n_active_core_facets": len(active_core_facets),
        "n_hub_core_terms": len(hub_core),
        "n_feature_columns": sum(counts.values()),
        "block_coordinate_counts": counts,
        "block_coordinate_members": members,
        "block_coordinate_hashes": {p: json_sha256(members[p]) for p in BLOCK_PREFIXES},
        "hub_core_members": hub_core, "hub_core_hash": json_sha256(hub_core),
        "active_core_facet_members": active_core_facets, "active_core_facet_hash": json_sha256(active_core_facets),
        "materialized_facet_members": materialized, "materialized_facet_hash": json_sha256(materialized),
        "facet_set_differences": {
            "authority_only": sorted(set(authority) - set(materialized)),
            "materialized_not_core_reachable": sorted(set(materialized) - set(active_core_facets)),
        },
        "T_materialized_records": t_records, "T_materialized_hash": json_sha256(t_records),
        "W_materialized_records": w_records, "W_materialized_hash": json_sha256(w_records),
        "label_columns_read": [],
        "materialized_map_solver": dict(MATERIALIZED_SOLVER, rank=rank),
    }

    blank = {k: "" for k in DIAG_FIELDS}
    diag: list[dict[str, Any]] = []
    for f in materialized:
        diag.append({**blank, "row_count": n_total, "record_type": "MAP", "facet": f, "declared_family": W[f],
                     "materialized_family": M[f], "mapping_equal": int(W[f] == M[f]),
                     "materialized_map_hash": M_hash, "solver_rank": rank})
    diag.append({**blank, "row_count": n_total, "record_type": "SUMMARY",
                 "rows_checked": n_total, "exact_rows": n_total,
                 "df_rows_checked": int(scan["rows"]["DF"]), "df_exact_rows": int(scan["rows"]["DF"]),
                 "fp_rows_checked": int(scan["rows"]["FP"]), "fp_exact_rows": int(scan["rows"]["FP"]),
                 "declared_W_mismatch_df": int(scan["w_mismatch"]["DF"]),
                 "declared_W_mismatch_fp": int(scan["w_mismatch"]["FP"]),
                 "materialized_map_hash": M_hash, "solver_rank": rank,
                 "proof_method": ("integer sufficient statistics; one-hot M; exact residual sum of squares = 0 "
                                  "for every family globally and within " + "/".join(DATASETS))})
    return {"support": support, "populations": populations, "diagnostics": diag, "M": M}
