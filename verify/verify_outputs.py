# SPDX-License-Identifier: MIT
"""Independent verifier (imports no pipeline code): reads Pi and T from the input term and relation tables and checks
the registry maps of the step outputs against them and against the W map that Table S2 of the extended audit details states; recomputes both
conformance layers of every anchor from the input view and compares them row by row with the step outputs; rebuilds the
fibers and witness pairs of all 32 views, tallies the verdict codes, checks the verdict rules on every anchor, recounts
the carrier-equivalence sets and rechecks every open Path B fiber; and compares the other quantities with the step
outputs or with values the paper states."""
from __future__ import annotations

import argparse
import os

THREAD_LIMIT = 1
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "HIGHS_THREADS"):
    os.environ.setdefault(_k, str(THREAD_LIMIT))

import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.optimize  # noqa: E402

# This file never imports the pipeline library (src/lca): it recomputes quantities from the input view and the step
# outputs, or compares them with the step outputs.
REPO = Path(__file__).resolve().parents[1]
SUMS_FILE = REPO / "data" / "hub" / "SHA256SUMS"
INPUT_FILES = ("anchors.csv.gz", "core_terms.csv", "registry_edges.csv", "label_rule.py", "sample_order.csv.gz")
SCHEMA_VERSION = "1"
WRITER_ID = "verify.writer"
CHUNK_ROWS = 25000

DATASETS = ("DF", "FP")
EXPECTED_ROWS = {"DF": 289202, "FP": 342182}
EXPECTED_TOTAL = 631384
EXPECTED_VIEWS = 32
EXPECTED_FULL = {"DF": 274622, "FP": 291686}
EXPECTED_FAIL = {"DF": 14580, "FP": 50496}
EXPECTED_SUPPORT = {"DF": 16, "FP": 24}
# Facet-to-family map W as the extended audit details state it (Table S2); the conformance layers below are recomputed with it.
W_STATED = {
    "attribute/fit": "structural_expression", "attribute/garment detail": "structural_expression",
    "attribute/leather": "surface_expression", "attribute/length": "structural_expression",
    "attribute/neckline type": "structural_expression", "attribute/nickname": "tag_terms",
    "attribute/non textile material type": "surface_expression", "attribute/opening type": "structural_expression",
    "attribute/silhouette": "structural_expression",
    "attribute/textile finishing manufacturing techniques": "surface_expression",
    "attribute/textile material type": "surface_expression", "attribute/textile pattern": "surface_expression",
    "attribute/waistline": "attribute_expression", "category/arms and hands": "taxonomic_category",
    "category/head": "taxonomic_category", "category/legs and feet": "taxonomic_category",
    "category/lowerbody": "taxonomic_category", "category/neck": "taxonomic_category",
    "category/others": "taxonomic_category", "category/upperbody": "taxonomic_category",
    "category/waist": "taxonomic_category", "category/wholebody": "taxonomic_category",
    "color": "surface_expression", "meta/attribute type": "other_expression", "part/closures": "structural_expression",
    "part/decorations": "structural_expression", "part/garment parts": "structural_expression",
}
FACET_UID_PREFIX = "hub:facet/"
EXPECTED_COORDINATE_SUBSET = {"core::": 1181, "facet::": 26, "family::": 5, "type::": 3}
EXPECTED_DECOMPOSITION = {
    "DF": (181119, 180715, 404, 180715, 0),
    "FP": (324407, 285859, 38548, 285859, 0),
}
# Carrier equivalence (task, dataset): direct-carrier UNSAFE rows, identity-admitted UNSAFE rows, rows that only the
# direct-carrier view certifies.
EXPECTED_CARRIER = {
    ("family", "DF"): (190335, 181119, 9216),
    ("family", "FP"): (324407, 324407, 0),
    ("facet", "DF"): (139290, 138908, 382),
    ("facet", "FP"): (184597, 149848, 34749),
}
CARRIER_VIEWS = {"family": ("l0_family__family", "identity_admitted__family"),
                 "facet": ("l0_facet__facet", "identity_admitted__facet")}
EXPECTED_M5 = {"checks": 64, "unsafe_on_invalid": 0, "delta_mismatches": 0, "nonvacuous": True}
EXPECTED_RAW_TO_SHARED_CHANGES = 368911
EXPECTED_HEADLINE_4DP = {
    "refit_f1": 0.3036,
    "volume_min": 0.6860,
    "volume_max": 0.6874,
    "type_rescale_min": 0.2576,
    "type_rescale_max": 0.4687,
}
EXPECTED_COS_ON_DET_3DP = {
    ("all_source", "DF->FP"): 0.873,
    ("all_source", "FP->DF"): 0.754,
    ("fixed_dataset_refit", "DF->FP"): 0.988,
    ("fixed_dataset_refit", "FP->DF"): 0.915,
}
EXPECTED_COS_ON_DET_ROWS = 68
EXPECTED_SCOPES = {
    "DF|ALL": (1, 8),
    "DF|DRIFT": (1, 8),
    "DF|FULL_CONFORM": (8, 8),
    "FP|ALL": (1, 8),
    "FP|DRIFT": (1, 8),
    "FP|FULL_CONFORM": (8, 8),
}
# Cell identifiers of utility/simple_checks_values.csv for the FD cells of each dataset.
FD_CELL_SUFFIX = {"DF": "df", "FP": "fp"}
EXPECTED_HEADLINE_CELLS = {
    "l0_rate": 1.0,
    "l0_hit": 1.0,
    "l0_hit_san": 0.0,
    "instfam_n": 268011,
    "instfam_den": 289202,
    "fd_g3_df": 0.0,
    "fd_g3_fp": 0.0,
}
EXPECTED_PERMUTATIONS = 999
EXPECTED_SCALING_SHAPE = (240, 80, 10)
EXPECTED_SCALING_ROWS = [63138, 631384]

# Regression values that the utility step asserts (fixed-dataset source, FULL-CONFORM target, rounded to 4).
COSINE_REFIT_4DP = {
    ("identity_admitted__family", "DF", "FP"): 0.5829,
    ("identity_admitted__family", "FP", "DF"): 0.3036,
    ("core_plus_type__family", "DF", "FP"): 0.2701,
    ("core_plus_type__family", "FP", "DF"): 0.2576,
    ("identity_admitted__facet", "DF", "FP"): 0.1305,
    ("identity_admitted__facet", "FP", "DF"): 0.1746,
}
FACTORIAL_CELLS = tuple(f"F{f}M{m}T{t}" for f in (0, 1) for m in (0, 1) for t in (0, 1))
FACTORIAL_4DP = {
    ("facet", "DF->FP"): [0.1389, 0.1068, 0.1319, 0.1305, 0.5578, 0.4171, 0.3986, 0.3691],
    ("facet", "FP->DF"): [0.1788, 0.1217, 0.1326, 0.1746, 0.5114, 0.4772, 0.4718, 0.4732],
    ("family", "DF->FP"): [0.1165, 0.2701, 0.6848, 0.6799, 0.3272, 0.5829, 0.6860, 0.6879],
    ("family", "FP->DF"): [0.2322, 0.2576, 0.6400, 0.5088, 0.4381, 0.3036, 0.6326, 0.5304],
}
TYPE_WEIGHT_4DP = [0.4264, 0.4687, 0.3036, 0.2576, 0.2576]
VOLUME_MINMAX_4DP = (0.6860, 0.6874)
VERDICT_CLASS_BASIS = "identity_admitted__family/UNSAFE"
BOUND_SLACK = 1e-12

# Views whose verdict accounting is recounted from the per-anchor verdict codes.
COUNT_VIEWS = (
    "core_plus_type__family",
    "facet_only__family",
    "family3__facet",
    "family3_plus_type__facet",
    "identity_admitted__facet",
    "identity_admitted__family",
    "inst_family__family",
    "l0_facet__facet",
    "l0_family__family",
    "type3__family",
    "type_attr_part__family",
)
DECOMPOSITION_VIEW = "identity_admitted__family"
COUNT_FIELDS = ("fibers", "unsafe_rows", "nondetermined_rows", "unverified_observed_rows", "unverified_defect_rows")
VERDICT_CODE_TEXT = "0=UNVERIFIED;1=UNSAFE;2=NONDETERMINED"
UNSAFE = 1
NONDETERMINED = 2

REFIT_PROBE = {"view_id": "identity_admitted__family", "source": "FP", "target": "DF", "probe": "cosine",
               "training_protocol": "fixed_dataset_refit", "evaluation_scope": "full_conform_target", "status": "OK"}
SANITIZED_PREFIXES = ("facet::", "family::", "inst_family::")
SIGNATURE_BLOCKS = ("core::", "facet::", "family::", "type::", "inst_family::", "inst_sem::", "rep_core::")

SCALING_KEYS = ("fraction", "view_id", "task", "dataset")
SCALING_TIMING = ("grouping_seconds", "milp_seconds", "total_seconds")
SCALING_GROUPING_COLUMNS = (
    "dataset_rows_total", "sampled_dataset_rows", "sampled_full_conform_rows", "sampled_query_conform_rows",
    "facet_dimension", "label_universe_size", "theoretical_competing_labels_per_candidate",
    "observed_unique_fibers", "multi_label_fibers", "single_label_fibers", "max_fiber_rows", "mean_fiber_rows",
)

CHECK_FIELDS = ["check", "primary_value", "independent_value", "exact_match", "note"]


# ----------------------------------------------------------------------------------------------------
# Hashing, reading and writing (self-contained; the pipeline writer is not used)
# ----------------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_sums(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, name = line.split(None, 1)
            out[name.strip()] = sha
    return out


class Sources:
    """Resolves input-view and output files and records the SHA-256 of every file read."""

    def __init__(self, inputs: Path, outputs: Path, sums: dict[str, str]) -> None:
        self.inputs, self.outputs, self.sums = inputs, outputs, sums
        self.hashes: dict[str, str] = {}

    def input(self, name: str) -> Path:
        path = self.inputs / name
        if name not in self.sums:
            raise SystemExit(f"{name} is not listed in SHA256SUMS")
        got = sha256_file(path)
        if got != self.sums[name]:
            raise SystemExit(f"SHA-256 mismatch for {name}: expected {self.sums[name]}, got {got}")
        self.hashes[name] = got
        return path

    def output(self, rel: str) -> Path:
        path = self.outputs / rel
        if not path.is_file():
            raise SystemExit(f"missing step output {rel}")
        self.hashes[rel] = sha256_file(path)
        return path


def read_table(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """CSV as strings; numbers are converted explicitly with int()/float() (exact round-trip parsing).
    Identities between tables are evaluated on these values."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)


def read_numeric(path: Path) -> pd.DataFrame:
    """CSV with pandas' default type inference and float converter.
    Reported quantities (headline values, margins, means, ratios) are computed on these values."""
    return pd.read_csv(path, keep_default_na=False)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_int(x: Any) -> int:
    v = float(x)
    if not v.is_integer():
        raise SystemExit(f"non-integer count {x!r}")
    return int(v)


def value_text(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=True, sort_keys=True)
    return v


def cell_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        if not math.isfinite(v):
            raise SystemExit("non-finite value in a check row")
        return repr(v)
    return str(v)


def write_checks_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=CHECK_FIELDS, lineterminator="\n", extrasaction="raise")
    w.writeheader()
    for r in rows:
        w.writerow({k: cell_text(r.get(k)) for k in CHECK_FIELDS})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue().encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def add_check(rows: list[dict[str, Any]], check: str, primary: Any, independent: Any, match: bool | None,
              note: str = "") -> None:
    rows.append({
        "check": check,
        "primary_value": value_text(primary),
        "independent_value": value_text(independent),
        "exact_match": "NA" if match is None else int(bool(match)),
        "note": note,
    })


# ----------------------------------------------------------------------------------------------------
# One pass over the anchor table
# ----------------------------------------------------------------------------------------------------

def _norm_lex(x: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(x)).strip().split()).casefold()


class Codebook:
    def __init__(self) -> None:
        self.codes: dict[Any, int] = {}

    def code(self, key: Any) -> int:
        c = self.codes.get(key)
        if c is None:
            c = len(self.codes)
            self.codes[key] = c
        return c


def anchor_pass(path: Path, row_ds: np.ndarray, row_id: np.ndarray, pi: dict[str, str], w: dict[str, str],
                family_coords: list[str], type_coords: list[str]) -> dict[str, Any]:
    """Block signatures, lexical and carrier hits, family/type counts and the core-basis family carrier of every anchor."""
    n = EXPECTED_TOTAL
    books = {b: Codebook() for b in SIGNATURE_BLOCKS}
    codes = {b: np.empty(n, dtype=np.int64) for b in SIGNATURE_BLOCKS}
    gold_facet = np.empty(n, dtype=object)
    gold_family = np.empty(n, dtype=object)
    fam_idx = {x: i for i, x in enumerate(family_coords)}
    type_idx = {x: i for i, x in enumerate(type_coords)}
    family_counts = np.zeros((n, len(family_coords)), dtype=np.int64)
    type_counts = np.zeros((n, len(type_coords)), dtype=np.int64)
    core_strict = np.zeros(n, dtype=bool)
    core_label = np.empty(n, dtype=object)
    facet_strict = np.zeros(n, dtype=bool)
    facet_color = np.zeros(n, dtype=bool)
    direct: dict[tuple[str, str], int] = defaultdict(int)
    lexical: dict[tuple[str, str, str], int] = defaultdict(int)
    instfam_n = instfam_den = 0
    key_info: dict[str, tuple[str, str, str]] = {}
    pos = 0
    reader = pd.read_csv(path, usecols=["dataset", "anchor_id", "full_feats_json", "gold_facet", "gold_family"],
                         dtype=str, keep_default_na=False, chunksize=CHUNK_ROWS)
    for chunk in reader:
        for ds, aid, raw, gf, gm in zip(chunk["dataset"], chunk["anchor_id"], chunk["full_feats_json"],
                                        chunk["gold_facet"], chunk["gold_family"]):
            if pos >= n:
                raise SystemExit("anchor table has more rows than the conformance output")
            if ds != row_ds[pos] or aid != row_id[pos]:
                raise SystemExit(f"anchor and conformance row order differ at row {pos}")
            feats = json.loads(raw)
            sig: dict[str, list[tuple[str, float]]] = {b: [] for b in SIGNATURE_BLOCKS}
            raw_names: set[str] = set()
            san_names: set[str] = set()
            inst_values: set[str] = set()
            core_family: dict[str, float] = defaultdict(float)
            for key, v in feats.items():
                x = float(v)
                if x == 0.0:
                    continue
                info = key_info.get(key)
                if info is None:
                    cut = key.find("::")
                    prefix, local = (key[:cut + 2], key[cut + 2:]) if cut >= 0 else ("", key)
                    info = (prefix, local, _norm_lex(local))
                    key_info[key] = info
                prefix, local, normed = info
                raw_names.add(normed)
                if prefix not in SANITIZED_PREFIXES:
                    san_names.add(normed)
                if prefix in sig:
                    sig[prefix].append((local, x))
                if prefix == "inst_family::" and x > 0:
                    inst_values.add(local)
                elif prefix == "family::":
                    if local not in fam_idx or not x.is_integer():
                        raise SystemExit(f"family coordinate outside the registry or non-integer: {key}")
                    family_counts[pos, fam_idx[local]] = int(x)
                elif prefix == "type::":
                    if local not in type_idx or not x.is_integer():
                        raise SystemExit(f"type coordinate outside the registry or non-integer: {key}")
                    type_counts[pos, type_idx[local]] = int(x)
                if prefix == "core::":
                    facet = pi.get(local)
                    if facet is None or facet not in w:
                        raise SystemExit(f"core coordinate without a registered facet: {key}")
                    core_family[w[facet]] += x
            for b in SIGNATURE_BLOCKS:
                codes[b][pos] = books[b].code(tuple(sorted(sig[b])))
            fvals = sorted((x for _, x in sig["facet::"]), reverse=True)
            facet_strict[pos] = bool(fvals) and fvals[0] > 0 and (len(fvals) == 1 or fvals[0] > fvals[1])
            facet_color[pos] = any(local == "color" and x > 0 for local, x in sig["facet::"])
            if core_family:
                top = max(core_family.values())
                winners = [f for f, s in core_family.items() if s == top]
                if top > 0 and len(winners) == 1:
                    core_strict[pos] = True
                    core_label[pos] = winners[0]
            direct[(ds, "facet")] += int(float(feats.get("facet::" + gf, 0.0)) > 0)
            direct[(ds, "family")] += int(float(feats.get("family::" + gm, 0.0)) > 0)
            for task, gold in (("facet", gf), ("family", gm)):
                target = _norm_lex(gold)
                lexical[(ds, task, "RAW")] += int(target in raw_names)
                lexical[(ds, task, "SANITIZED")] += int(target in san_names)
            if ds == "DF":
                instfam_den += 1
                instfam_n += int(gm in inst_values)
            gold_facet[pos] = gf
            gold_family[pos] = gm
            pos += 1
    if pos != n:
        raise SystemExit(f"anchor rows {pos} != {n}")
    return {
        "dataset": row_ds, "core": codes["core::"], "facet": codes["facet::"], "family": codes["family::"],
        "type": codes["type::"], "signatures": codes, "gold_facet": gold_facet.astype(str), "gold_family": gold_family.astype(str),
        "family_counts": family_counts, "type_counts": type_counts,
        "core_strict": core_strict, "core_label": core_label,
        "facet_strict": facet_strict, "facet_color": facet_color,
        "direct": dict(direct), "lexical": dict(lexical), "instfam_n": instfam_n, "instfam_den": instfam_den,
    }


def unique_count(arrays: list[np.ndarray], mask: np.ndarray) -> int:
    if len(arrays) == 1:
        return int(np.unique(arrays[0][mask]).size)
    return int(np.unique(np.column_stack([a[mask] for a in arrays]), axis=0).shape[0])


def partition_rows(cache: dict[str, Any], full: np.ndarray) -> list[dict[str, Any]]:
    """Fiber counts of the core block and of every facet/family/type refinement on ALL, FULL_CONFORM and DRIFT rows."""
    out = []
    for ds in DATASETS:
        base = cache["dataset"] == ds
        for scope, mask in (("ALL", base), ("FULL_CONFORM", base & full), ("DRIFT", base & ~full)):
            core_n = unique_count([cache["core"]], mask)
            for f in (0, 1):
                for m in (0, 1):
                    for t in (0, 1):
                        arrays = [cache["core"]]
                        if f:
                            arrays.append(cache["facet"])
                        if m:
                            arrays.append(cache["family"])
                        if t:
                            arrays.append(cache["type"])
                        cell_n = unique_count(arrays, mask)
                        out.append({"dataset": ds, "scope": scope, "cell_id": f"F{f}M{m}T{t}",
                                    "core_fibers": core_n, "cell_fibers": cell_n,
                                    "partition_equal": int(core_n == cell_n)})
    return out


def fd_stats(determinant: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[bool, float, float]:
    """Exactness, row-deletion g3 and singleton-row mass of the dependency determinant -> label on masked rows."""
    groups: dict[int, Counter] = defaultdict(Counter)
    for d, label in zip(determinant[mask].tolist(), y[mask].tolist()):
        groups[int(d)][str(label)] += 1
    exact = all(len(c) <= 1 for c in groups.values())
    n = int(np.count_nonzero(mask))
    keep = sum(max(c.values()) for c in groups.values())
    g3 = 0.0 if n == 0 else 1.0 - keep / n
    singleton = 0.0 if n == 0 else sum(sum(c.values()) for c in groups.values() if sum(c.values()) == 1) / n
    return bool(exact), float(g3), float(singleton)


# ----------------------------------------------------------------------------------------------------
# Checks by step
# ----------------------------------------------------------------------------------------------------

def check_registry(src: Sources, rows: list[dict[str, Any]]) -> dict[str, Any]:
    t2f = read_table(src.output("registry/term_to_facet.csv"))
    fmap = read_table(src.output("registry/facet_map.csv"))
    support = read_json(src.output("registry/dataset_support.json"))
    pops = read_json(src.output("registry/populations.json"))
    if not {"core_term_uid", "core_facet", "match"}.issubset(t2f.columns):
        raise SystemExit("registry/term_to_facet.csv lacks core_term_uid, core_facet or match")
    if not {"facet", "term_type", "family"}.issubset(fmap.columns):
        raise SystemExit("registry/facet_map.csv lacks facet, term_type or family")

    pi_count = len(t2f)
    pi_all_match = bool((t2f["match"] == "1").all())
    t_map = dict(zip(fmap["facet"], fmap["term_type"]))
    w_map = dict(zip(fmap["facet"], fmap["family"]))
    t_hash, w_hash = canonical_hash(t_map), canonical_hash(w_map)
    s_counts = {ds: len(support["datasets"][ds]["S_members"]) for ds in DATASETS}
    members = pops["block_coordinate_members"]
    coordinate_counts = {k: len(v) for k, v in members.items() if isinstance(v, list)}
    got_subset = {k: coordinate_counts.get(k) for k in EXPECTED_COORDINATE_SUBSET}

    add_check(rows, "registry.Pi_count", 2715, pi_count, pi_count == 2715)
    add_check(rows, "registry.Pi_all_match", 1, int(pi_all_match), pi_all_match)
    add_check(rows, "registry.T_domain_facets", 27, len(t_map), len(t_map) == 27)
    add_check(rows, "registry.W_domain_facets", 27, len(w_map), len(w_map) == 27)
    add_check(rows, "registry.T_map_hash", None, t_hash, None, "canonical JSON hash of the registry map")
    add_check(rows, "registry.W_map_hash", None, w_hash, None, "canonical JSON hash of the registry map")
    add_check(rows, "registry.support_facets", EXPECTED_SUPPORT, s_counts, s_counts == EXPECTED_SUPPORT)
    add_check(rows, "registry.coordinate_counts_subset", EXPECTED_COORDINATE_SUBSET, got_subset,
              got_subset == EXPECTED_COORDINATE_SUBSET)
    return {
        "details": {"T_hash": t_hash, "W_hash": w_hash, "support_counts": s_counts,
                    "coordinate_counts": coordinate_counts},
        "pi": dict(zip(t2f["core_term_uid"], t2f["core_facet"])),
        "w": w_map,
        "t": t_map,
        "family_coords": list(members["family::"]),
        "type_coords": list(members["type::"]),
        "facet_coords": list(members["facet::"]),
    }


def check_registry_inputs(src: Sources, reg: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pi and T read from the input term and relation tables; W compared with the map that the extended audit details state."""
    terms = read_table(src.input("core_terms.csv"), usecols=["core_term_uid", "term_type", "facet"])
    edges = read_table(src.input("registry_edges.csv"), usecols=["src_uid", "edge_type", "dst_uid"])
    in_facet = edges[edges["edge_type"] == "IN_FACET"]
    if in_facet["src_uid"].duplicated().any() or not in_facet["dst_uid"].str.startswith(FACET_UID_PREFIX).all():
        raise SystemExit("IN_FACET edges are not one facet per term in the facet namespace")
    pi_edges = dict(zip(in_facet["src_uid"], in_facet["dst_uid"].str[len(FACET_UID_PREFIX):]))
    pi_field = dict(zip(terms["core_term_uid"], terms["facet"]))
    pi_equal = pi_edges == pi_field == reg["pi"]
    types: dict[str, set[str]] = defaultdict(set)
    for f, tt in zip(terms["facet"], terms["term_type"]):
        types[f].add(tt)
    t_in = {f: next(iter(v)) for f, v in types.items() if len(v) == 1}
    t_ok = len(t_in) == len(types) and all(reg["t"].get(f) == tt for f, tt in t_in.items())
    w_ok = reg["w"] == W_STATED
    add_check(rows, "registry.Pi_from_input_relations", len(reg["pi"]), sum(pi_edges.get(k) == v for k, v in pi_field.items()),
              pi_equal, "IN_FACET targets = facet field of the term table = registry/term_to_facet.csv")
    add_check(rows, "registry.T_from_input_terms", len(types), len(t_in), t_ok,
              "one term type per facet in the term table = registry/facet_map.csv")
    add_check(rows, "registry.W_equals_stated_map", len(W_STATED), sum(reg["w"].get(f) == x for f, x in W_STATED.items()),
              w_ok, "registry/facet_map.csv = the facet-to-family map of Table S2 of the extended audit details")
    return {"pi": pi_field, "t": t_in, "w": dict(W_STATED)}


def conformance_pass(path: Path, row_ds: np.ndarray, row_id: np.ndarray, pi: dict[str, str], t_map: dict[str, str],
                     w_map: dict[str, str]) -> dict[str, Any]:
    """Both conformance layers recomputed per anchor from the input view: the map edges (facet = core Pi,
    type = facet T, family = facet W), FULL-CONFORM (all role blocks = registry of the core block), block supports
    and whether the released label maximizes the carrier."""
    n = EXPECTED_TOTAL
    facets = sorted(w_map)
    fbit = {f: 1 << i for i, f in enumerate(facets)}
    families = sorted(set(w_map.values()))
    mbit = {m: 1 << i for i, m in enumerate(families)}
    a = {k: np.zeros(n, dtype=bool) for k in ("full", "pi_ok", "t_ok", "w_ok", "core_nz", "gmax_facet", "gmax_family")}
    facet_bits = np.zeros(n, dtype=np.int64)
    family_bits = np.zeros(n, dtype=np.int64)
    core_terms: dict[str, set[str]] = {ds: set() for ds in DATASETS}
    pos = 0
    reader = pd.read_csv(path, usecols=["dataset", "anchor_id", "full_feats_json", "gold_facet", "gold_family"],
                         dtype=str, keep_default_na=False, chunksize=CHUNK_ROWS)
    for chunk in reader:
        for ds, aid, raw, gf, gm in zip(chunk["dataset"], chunk["anchor_id"], chunk["full_feats_json"],
                                        chunk["gold_facet"], chunk["gold_family"]):
            if ds != row_ds[pos] or aid != row_id[pos]:
                raise SystemExit(f"anchor and conformance row order differ at row {pos}")
            blocks: dict[str, dict[str, int]] = {"core::": {}, "facet::": {}, "family::": {}, "type::": {}}
            for key, v in json.loads(raw).items():
                x = float(v)
                if x == 0.0:
                    continue
                cut = key.find("::")
                block = blocks.get(key[:cut + 2]) if cut >= 0 else None
                if block is not None:
                    if not x.is_integer():
                        raise SystemExit(f"non-integer count in {key}")
                    block[key[cut + 2:]] = int(x)
            core, facet, family, typ = (blocks[b] for b in ("core::", "facet::", "family::", "type::"))
            p_facet: dict[str, int] = defaultdict(int)
            for uid, c in core.items():
                p_facet[pi[uid]] += c
            p_type: dict[str, int] = defaultdict(int)
            p_family: dict[str, int] = defaultdict(int)
            for f, c in p_facet.items():
                p_type[t_map[f]] += c
                p_family[w_map[f]] += c
            f_type: dict[str, int] = defaultdict(int)
            f_family: dict[str, int] = defaultdict(int)
            for f, c in facet.items():
                f_type[t_map[f]] += c
                f_family[w_map[f]] += c
            a["core_nz"][pos] = bool(core)
            a["pi_ok"][pos] = facet == dict(p_facet)
            a["full"][pos] = bool(core) and a["pi_ok"][pos] and typ == dict(p_type) and family == dict(p_family)
            a["t_ok"][pos] = typ == dict(f_type)
            a["w_ok"][pos] = family == dict(f_family)
            a["gmax_facet"][pos] = bool(facet) and facet.get(gf, 0) == max(facet.values())
            a["gmax_family"][pos] = bool(family) and family.get(gm, 0) == max(family.values())
            facet_bits[pos] = sum(fbit[f] for f in facet)
            family_bits[pos] = sum(mbit[m] for m in family)
            core_terms[ds].update(core)
            pos += 1
    if pos != n:
        raise SystemExit(f"anchor rows {pos} != {n}")
    s_sets = {ds: {pi[u] for u in core_terms[ds]} for ds in DATASETS}
    s_mask = {ds: sum(fbit[f] for f in s_sets[ds]) for ds in DATASETS}
    m_mask = {ds: sum(mbit[m] for m in {w_map[f] for f in s_sets[ds]}) for ds in DATASETS}
    a["supp_c"] = a["core_nz"]
    a["supp_s"] = np.zeros(n, dtype=bool)
    a["supp_f"] = np.zeros(n, dtype=bool)
    for ds in DATASETS:
        dm = row_ds == ds
        a["supp_s"][dm] = (facet_bits[dm] != 0) & ((facet_bits[dm] & ~s_mask[ds]) == 0)
        a["supp_f"][dm] = (family_bits[dm] != 0) & ((family_bits[dm] & ~m_mask[ds]) == 0)
    a["support_sizes"] = {ds: len(s_sets[ds]) for ds in DATASETS}
    return a


def check_conformance_recomputed(src: Sources, conf: dict[str, Any], a: dict[str, Any],
                                 rows: list[dict[str, Any]]) -> dict[str, Any]:
    """FULL-CONFORM and the QUERY-CONFORM masks of all views, recomputed from the input view with the registry maps
    above and the certificate specification of each view, compared row by row with the step outputs."""
    full_equal = bool(np.array_equal(a["full"], conf["full"]))
    fails = {ds: int(np.count_nonzero((conf["row_ds"] == ds) & ~a["full"])) for ds in DATASETS}
    add_check(rows, "conformance.FULL_CONFORM_recomputed_from_inputs", EXPECTED_FAIL, fails,
              full_equal and fails == EXPECTED_FAIL, "every anchor's FULL-CONFORM flag recomputed from the input view")
    spec = {r["view_id"]: r for r in read_table(src.output("registry/certificate_spec.csv")).to_dict("records")}
    edge = {"Pi": "pi_ok", "T": "t_ok", "W": "w_ok"}
    supp = {"C_i": "supp_c", "S_i": "supp_s", "w(C_i)": "supp_f"}
    differing = []
    for i, vid in enumerate(conf["qviews"]):
        s = spec[vid]
        gold_ok = a["gmax_facet"] if s["task"] == "facet" else a["gmax_family"]
        if s["certificate_path"] == "C":
            q = a["full"] & gold_ok
        else:
            q = a[supp[s["basis_support"]]].copy()
            for e in [x for x in s["required_maps"].split("|") if x]:
                q &= a[edge[e]]
            q &= gold_ok
        if not np.array_equal(q, conf["qmask"][i]):
            differing.append(vid)
    add_check(rows, "conformance.QUERY_CONFORM_recomputed_from_inputs", EXPECTED_VIEWS,
              EXPECTED_VIEWS - len(differing), not differing,
              "every view's QUERY-CONFORM mask recomputed from the input view and compared row by row")
    add_check(rows, "conformance.support_sizes_from_inputs", EXPECTED_SUPPORT, a["support_sizes"],
              a["support_sizes"] == EXPECTED_SUPPORT, "facet supports S_i recomputed from the core terms of each dataset")
    return {"details": {"full_conform_rows_equal": full_equal, "query_masks_differing": differing}}


def check_conformance(src: Sources, rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_path = src.output("conformance/full_conform.csv.gz")
    fc = read_table(full_path, usecols=["dataset", "row_id", "full_registry_conform"])
    if len(fc) != EXPECTED_TOTAL or fc["row_id"].nunique() != EXPECTED_TOTAL:
        raise SystemExit("conformance/full_conform.csv.gz row identity differs from the anchor table")
    if not set(fc["full_registry_conform"]).issubset({"0", "1"}):
        raise SystemExit("full_registry_conform is not 0/1")
    row_ds = fc["dataset"].to_numpy(dtype=str)
    row_id = fc["row_id"].to_numpy(dtype=str)
    full = fc["full_registry_conform"].to_numpy(dtype=str) == "1"

    counts, fails, fail_hash = {}, {}, {}
    for ds in DATASETS:
        dm = row_ds == ds
        n = int(dm.sum())
        ok = int(np.count_nonzero(dm & full))
        counts[ds], fails[ds] = ok, n - ok
        h = hashlib.sha256()
        for rid in row_id[dm & ~full].tolist():
            h.update(rid.encode("utf-8"))
            h.update(b"\n")
        fail_hash[ds] = h.hexdigest()

    with np.load(src.output("conformance/query_masks.npz"), allow_pickle=False) as qz:
        for name in ("query_conform", "full_registry_conform", "view_ids"):
            if name not in qz.files:
                raise SystemExit(f"conformance/query_masks.npz lacks {name}")
        qmask = np.asarray(qz["query_conform"]).astype(bool)
        full_npz = np.asarray(qz["full_registry_conform"]).astype(bool)
        qviews = [str(v) for v in qz["view_ids"]]
    csv_vs_npz = bool(full_npz.shape == full.shape and np.array_equal(full_npz, full))

    add_check(rows, "conformance.FULL_CONFORM_pass_rows", EXPECTED_FULL, counts, counts == EXPECTED_FULL)
    add_check(rows, "conformance.FULL_CONFORM_fail_rows", EXPECTED_FAIL, fails, fails == EXPECTED_FAIL)
    add_check(rows, "conformance.query_mask_shape", [EXPECTED_VIEWS, EXPECTED_TOTAL], list(qmask.shape),
              list(qmask.shape) == [EXPECTED_VIEWS, EXPECTED_TOTAL])
    add_check(rows, "conformance.full_mask_csv_vs_npz", 1, int(csv_vs_npz), csv_vs_npz)
    for ds in DATASETS:
        add_check(rows, f"conformance.fail_mask_hash_{ds}", None, fail_hash[ds], None,
                  "independent ordered failing-row hash")
    if len(qviews) != EXPECTED_VIEWS or len(set(qviews)) != EXPECTED_VIEWS:
        raise SystemExit("query masks do not name 32 distinct views")
    return {"details": {"fail_mask_hash": fail_hash}, "row_ds": row_ds, "row_id": row_id, "full": full,
            "qmask": qmask, "qviews": qviews, "full_sha256": src.hashes["conformance/full_conform.csv.gz"]}


def milp_counterexample(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Small integer program whose optimum (-5) differs from its directed-flow relaxation."""
    c = np.asarray([0.0, -2.0, -3.0])
    a = np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    b = np.asarray([1.0, 1.0])
    constraints = [
        scipy.optimize.LinearConstraint(a, b, b),
        scipy.optimize.LinearConstraint(np.ones((1, 3), dtype=float), np.asarray([1.0]), np.asarray([np.inf])),
    ]
    res = scipy.optimize.milp(
        c=c,
        integrality=np.ones(3, dtype=np.uint8),
        bounds=scipy.optimize.Bounds(np.zeros(3, dtype=float), np.full(3, np.inf, dtype=float)),
        constraints=constraints,
        options={"disp": False, "mip_rel_gap": 0.0},
    )
    if int(res.status) != 0 or res.x is None:
        raise SystemExit(f"integer-program counterexample not solved: {res.message}")
    x = np.rint(np.asarray(res.x)).astype(np.int64)
    obj = int(np.asarray([0, -2, -3], dtype=np.int64) @ x)
    add_check(rows, "verdicts.MILP_counterexample_objective", -5, obj, obj == -5)
    return {"status": int(res.status), "x": x.tolist(), "objective": obj}


def check_verdicts(src: Sources, conf: dict[str, Any], cache: dict[str, Any],
                   rows: list[dict[str, Any]]) -> dict[str, Any]:
    vindex = read_table(src.output("verdicts/anchor_verdicts_index.csv"))
    if set(vindex["verdict_codes"]) != {VERDICT_CODE_TEXT}:
        raise SystemExit("verdict codes differ from 0=UNVERIFIED;1=UNSAFE;2=NONDETERMINED")
    if set(vindex["row_order_source_sha256"]) != {conf["full_sha256"]}:
        raise SystemExit("verdict rows are not in the order of conformance/full_conform.csv.gz")
    with np.load(src.output("verdicts/anchor_verdicts.npz"), allow_pickle=False) as npz:
        codes = np.asarray(npz["verdict_code"])
    if codes.shape != (EXPECTED_VIEWS, EXPECTED_TOTAL):
        raise SystemExit(f"verdict array shape {codes.shape}")
    witnesses = read_json(src.output("verdicts/witnesses.json"))
    table = read_table(src.output("verdicts/verdicts.csv"))
    tab = {(r["view"], r["dataset"]): r for r in table.to_dict("records")}
    vpos = {str(v): int(i) for v, i in zip(vindex["view_id"], vindex["array_index"])}
    qpos = {v: i for i, v in enumerate(conf["qviews"])}
    row_ds = conf["row_ds"]

    exact_cells = 0
    keys = sorted(f"{v}|{ds}" for v in COUNT_VIEWS for ds in DATASETS)
    for key in keys:
        vid, ds = key.rsplit("|", 1)
        metrics = witnesses["views"][vid]["datasets"][ds]["metrics_counts"]
        primary = {k: int(metrics[k]) for k in COUNT_FIELDS}
        dm = row_ds == ds
        n = int(dm.sum())
        v = codes[vpos[vid]]
        unsafe = int(np.count_nonzero(dm & (v == UNSAFE)))
        nondet = int(np.count_nonzero(dm & (v == NONDETERMINED)))
        defect = int(np.count_nonzero(dm & ~conf["qmask"][qpos[vid]]))
        trow = tab[(vid, ds)]
        got = {
            "fibers": int(trow["fibers"]),
            "unsafe_rows": unsafe,
            "nondetermined_rows": nondet,
            "unverified_observed_rows": n - unsafe - nondet - defect,
            "unverified_defect_rows": defect,
        }
        # The verdict table reports the same counts as masses over the dataset rows.
        masses_ok = all(float(trow[col]) == got[k] / n for col, k in (
            ("m^U", "unsafe_rows"), ("m^N", "nondetermined_rows"), ("m^V_obs", "unverified_observed_rows"),
            ("m^V_defect", "unverified_defect_rows")))
        ok = got == primary and masses_ok
        exact_cells += int(ok)
        add_check(rows, f"verdicts.counts.{key}", primary, got, ok)

    conflicts = infeasible = 0
    modes: set[str] = set()
    for vobj in witnesses["views"].values():
        for dobj in vobj["datasets"].values():
            mc = dobj.get("metrics_counts", {})
            conflicts += int(mc.get("certificate_label_conflicts", 0))
            infeasible += int(mc.get("infeasible_fibers", 0))
            if vobj.get("path") == "B":
                for wit in dobj.get("representative_witnesses", []):
                    for rival in wit.get("rival_optima", []):
                        mode = str(rival.get("solver_mode", ""))
                        if mode:
                            modes.add(mode)
    add_check(rows, "verdicts.certificate_label_conflicts", 0, conflicts, conflicts == 0)
    add_check(rows, "verdicts.infeasible_fibers", 0, infeasible, infeasible == 0)
    add_check(rows, "verdicts.PathB_representative_solver_modes", ["direct-HiGHS-MILP"], sorted(modes),
              modes == {"direct-HiGHS-MILP"})

    # Decomposition: UNSAFE rows of the W (facet basis) view versus the Pi|W (core basis) carrier,
    # recounted from the verdict codes and from the core coordinates of the anchor table.
    dd = read_table(src.output("verdicts/decomposition.csv"))
    primary_unsafe = codes[vpos[DECOMPOSITION_VIEW]] == UNSAFE
    full = conf["full"]
    alt_unsafe = full & cache["core_strict"]
    alt_label_ok = bool(np.all(cache["core_label"][alt_unsafe] == cache["gold_family"][alt_unsafe]))
    decomposition = {}
    for r in dd.to_dict("records"):
        ds = str(r["dataset"])
        dm = row_ds == ds
        file_vals = tuple(int(r[k]) for k in ("primary_unsafe_rows", "primary_unsafe_full_rows",
                                               "primary_unsafe_outside_full_rows", "alternative_unsafe_rows",
                                               "alternative_unsafe_outside_full_rows"))
        arithmetic = (file_vals[0] == file_vals[1] + file_vals[2] and file_vals[3] == file_vals[1]
                      and file_vals[4] == 0 and int(r["full_conform_rowwise_equal"]) == 1)
        recount = (
            int(np.count_nonzero(dm & primary_unsafe)),
            int(np.count_nonzero(dm & primary_unsafe & full)),
            int(np.count_nonzero(dm & primary_unsafe & ~full)),
            int(np.count_nonzero(dm & alt_unsafe)),
            int(np.count_nonzero(dm & alt_unsafe & ~full)),
        )
        rowwise = bool(np.array_equal(primary_unsafe[dm & full], alt_unsafe[dm & full]))
        decomposition[ds] = list(recount)
        ok = (file_vals == EXPECTED_DECOMPOSITION[ds] and arithmetic and recount == file_vals
              and rowwise and alt_label_ok)
        add_check(rows, f"verdicts.decomposition.{ds}", list(EXPECTED_DECOMPOSITION[ds]), list(recount), ok)
    return {"details": {"counts_exact_cells": exact_cells, "conflicts": conflicts, "infeasible": infeasible,
                        "decomposition": decomposition}}


def check_carrier(src: Sources, conf: dict[str, Any], cache: dict[str, Any],
                  rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recounts, from the materialized carrier block of every anchor and the QUERY-CONFORM masks, where the
    direct-carrier and the identity-admitted view of each task are UNSAFE: an anchor is UNSAFE in either view exactly
    when its carrier block has a strict unique maximum and it passes that view's QUERY-CONFORM check."""
    vindex = read_table(src.output("verdicts/anchor_verdicts_index.csv"))
    with np.load(src.output("verdicts/anchor_verdicts.npz"), allow_pickle=False) as npz:
        codes = np.asarray(npz["verdict_code"])
    vpos = {str(v): int(i) for v, i in zip(vindex["view_id"], vindex["array_index"])}
    qpos = {v: i for i, v in enumerate(conf["qviews"])}
    fam = np.sort(cache["family_counts"], axis=1)[:, ::-1]
    strict = {"family": (fam[:, 0] > 0) & ((fam.shape[1] == 1) | (fam[:, 0] > fam[:, 1])),
              "facet": cache["facet_strict"]}
    table = {(str(r["task"]), str(r["dataset"])): r
             for r in read_table(src.output("verdicts/carrier_equivalence.csv")).to_dict("records")}
    details = {}
    for task, (direct_view, identity_view) in CARRIER_VIEWS.items():
        q_direct, q_identity = conf["qmask"][qpos[direct_view]], conf["qmask"][qpos[identity_view]]
        direct_codes = codes[vpos[direct_view]] == UNSAFE
        identity_codes = codes[vpos[identity_view]] == UNSAFE
        direct = strict[task] & q_direct
        identity = strict[task] & q_identity
        for ds in DATASETS:
            dm = conf["row_ds"] == ds
            only = dm & direct & ~identity
            recount = (int(np.count_nonzero(dm & direct)), int(np.count_nonzero(dm & identity)),
                       int(np.count_nonzero(only)))
            r = table[(task, ds)]
            file_vals = tuple(int(r[k]) for k in ("direct_unsafe_rows", "identity_unsafe_rows", "direct_only_rows"))
            extra_ok = (int(r["direct_only_identity_query_defect_rows"]) == recount[2]
                        == int(np.count_nonzero(only & ~q_identity))
                        and int(r["direct_only_outside_full_conform_rows"]) == int(np.count_nonzero(only & ~conf["full"]))
                        and int(r["direct_only_with_color_rows"]) == int(np.count_nonzero(only & cache["facet_color"]))
                        and int(r["identity_only_rows"]) == int(np.count_nonzero(dm & identity & ~direct)) == 0
                        and int(r["mismatch_on_identity_query_conform_rows"]) == 0)
            rowwise = (bool(np.array_equal(direct_codes[dm], direct[dm]))
                       and bool(np.array_equal(identity_codes[dm], identity[dm])))
            ok = recount == file_vals == EXPECTED_CARRIER[(task, ds)] and extra_ok and rowwise
            details[f"{task}|{ds}"] = list(recount)
            add_check(rows, f"verdicts.carrier_equivalence.{task}.{ds}", list(EXPECTED_CARRIER[(task, ds)]),
                      list(recount), ok, "strict unique maximum of the carrier block within QUERY-CONFORM, "
                      "row by row against the verdict codes of both views")
    return {"details": details}


# Rules that tie the verdicts to the certificate paths (Definition of the verdicts, Sec. III-IV).
SEMANTIC_RULES = (
    "defect_rows_unverified",          # an anchor outside QUERY-CONFORM is UNVERIFIED
    "nondetermined_iff_witness_pair",  # NONDETERMINED exactly on fibers with two observed labels, on every path
    "unsafe_only_on_paths_a_b",        # Path C has no determination solver, so it issues no UNSAFE
    "definite_verdicts_exclusive",     # no UNSAFE anchor lies in a fiber with an observed witness pair
    "verdict_constant_on_fiber",       # every query-conforming anchor of a fiber carries the fiber's verdict
)


def fiber_codes(projection: dict[str, Any], cache: dict[str, Any], fam_idx: dict[str, int],
                type_idx: dict[str, int]) -> np.ndarray:
    """One integer per anchor that identifies its view value (equal codes iff equal projected values)."""
    key = np.zeros(EXPECTED_TOTAL, dtype=np.int64)
    for block, coords in sorted(projection.items()):
        if coords == "ALL":
            part = cache["signatures"][block]
        elif block in ("family::", "type::"):
            counts = cache["family_counts" if block == "family::" else "type_counts"]
            idx = fam_idx if block == "family::" else type_idx
            part = np.unique(counts[:, [idx[c] for c in coords]], axis=0, return_inverse=True)[1].reshape(-1)
        else:
            raise SystemExit(f"unsupported projection {block}")
        key = np.unique(key * (int(part.max()) + 1) + part.astype(np.int64), return_inverse=True)[1].reshape(-1)
    return key


def check_semantics(src: Sources, conf: dict[str, Any], cache: dict[str, Any], reg: dict[str, Any],
                    rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recounts the verdict rules on every anchor of all views from the input view and the conformance masks."""
    vindex = read_table(src.output("verdicts/anchor_verdicts_index.csv"))
    with np.load(src.output("verdicts/anchor_verdicts.npz"), allow_pickle=False) as npz:
        codes = np.asarray(npz["verdict_code"])
    vpos = {str(v): int(i) for v, i in zip(vindex["view_id"], vindex["array_index"])}
    spec = {r["view_id"]: r for r in read_table(src.output("registry/certificate_spec.csv")).to_dict("records")}
    proj = {r["view_id"]: json.loads(r["projection_json"])
            for r in read_table(src.output("registry/view_projection.csv")).to_dict("records")}
    if not (set(spec) == set(proj) == set(vpos) == set(conf["qviews"])) or len(spec) != EXPECTED_VIEWS:
        raise SystemExit("views differ between the certificate spec, the projections, the verdicts and the masks")
    fam_idx = {x: i for i, x in enumerate(reg["family_coords"])}
    type_idx = {x: i for i, x in enumerate(reg["type_coords"])}
    labels = {t: np.unique(cache[f"gold_{t}"], return_inverse=True)[1].reshape(-1).astype(np.int64)
              for t in ("facet", "family")}
    qpos = {v: i for i, v in enumerate(conf["qviews"])}
    bad: dict[str, list[str]] = {r: [] for r in SEMANTIC_RULES}
    tally = {"view_datasets": 0, "nondetermined_rows": 0, "witness_rows": 0, "unsafe_rows": 0,
             "path_c_view_datasets": 0}
    for vid in sorted(spec):
        path, task = str(spec[vid]["certificate_path"]), str(spec[vid]["task"])
        key = fiber_codes(proj[vid], cache, fam_idx, type_idx)
        v = codes[vpos[vid]]
        for ds in DATASETS:
            tag = f"{vid}|{ds}"
            dm = conf["row_ds"] == ds
            q = conf["qmask"][qpos[vid]] & dm
            tally["view_datasets"] += 1
            tally["path_c_view_datasets"] += int(path == "C")
            if np.any(v[dm & ~q] != 0):
                bad["defect_rows_unverified"].append(tag)
            idx = np.flatnonzero(q)
            if not len(idx):
                continue
            inv = np.unique(key[idx], return_inverse=True)[1].reshape(-1)
            nf = int(inv.max()) + 1
            pairs = np.unique(np.column_stack([inv, labels[task][idx]]), axis=0)
            witness = np.bincount(pairs[:, 0], minlength=nf)[inv] >= 2
            vq = v[idx].astype(np.int64)
            if not np.array_equal(vq == NONDETERMINED, witness):
                bad["nondetermined_iff_witness_pair"].append(tag)
            if path == "C" and np.any(vq == UNSAFE):
                bad["unsafe_only_on_paths_a_b"].append(tag)
            if np.any((vq == UNSAFE) & witness):
                bad["definite_verdicts_exclusive"].append(tag)
            lo = np.full(nf, 9, dtype=np.int64)
            hi = np.full(nf, -1, dtype=np.int64)
            np.minimum.at(lo, inv, vq)
            np.maximum.at(hi, inv, vq)
            if np.any(lo != hi):
                bad["verdict_constant_on_fiber"].append(tag)
            tally["nondetermined_rows"] += int(np.count_nonzero(vq == NONDETERMINED))
            tally["witness_rows"] += int(np.count_nonzero(witness))
            tally["unsafe_rows"] += int(np.count_nonzero(vq == UNSAFE))
    for rule in SEMANTIC_RULES:
        add_check(rows, f"semantics.{rule}", 0, len(bad[rule]), not bad[rule],
                  f"view-dataset pairs violating the rule, of {tally['view_datasets']}; " + ";".join(bad[rule][:5]))
    return {"details": {**tally, "violations": {r: len(b) for r, b in bad.items()}}}


def check_mutations(src: Sources, rows: list[dict[str, Any]]) -> None:
    mr = read_table(src.output("mutations/mutation_results.csv"))
    passed = int(sum(str(x).strip().lower() in {"1", "true", "pass"} for x in mr["pass"]))
    add_check(rows, "mutations.mutations_passed", 5, passed, passed == 5)

    muts = read_json(src.output("mutations/mutations.json"))["mutations"]
    selector_ok = bool(muts["M1"].get("generic_registry_selector_ok"))
    add_check(rows, "mutations.M1_generic_selector", 1, int(selector_ok), selector_ok)

    m5 = muts["M5"]
    got = {
        "checks": int(m5["rerun_view_dataset_checks"]),
        "unsafe_on_invalid": int(m5["unsafe_certificates_on_invalid_rows"]),
        "delta_mismatches": int(m5["query_defect_delta_mismatches"]),
        "nonvacuous": bool(m5["nonvacuous"]),
    }
    # Every per-view M5 row must be present and passing.
    vc = read_table(src.output("mutations/m5_view_checks.csv"))
    rows_ok = len(vc) == got["checks"] and bool((vc["pass"] == "1").all())
    add_check(rows, "mutations.M5_full_reaudit", EXPECTED_M5, got, got == EXPECTED_M5 and rows_ok)


def _one(df: pd.DataFrame, what: str, **eq: str) -> dict[str, str]:
    sel = df
    for k, v in eq.items():
        sel = sel[sel[k] == v]
    if len(sel) != 1:
        raise SystemExit(f"{what}: {len(sel)} rows match {eq}")
    return sel.iloc[0].to_dict()


def check_utility(src: Sources, rows: list[dict[str, Any]]) -> dict[str, Any]:
    support = read_table(src.output("utility/support_check.csv"))
    probes = read_table(src.output("utility/probes.csv"))
    volume_path = src.output("utility/volume_control.csv")
    weights_path = src.output("utility/type_weight.csv")
    checks_path = src.output("utility/probe_bound_check.csv")
    matrix_path = src.output("utility/cosine_determination_matrix.csv")
    volume, volume_n = read_table(volume_path), read_numeric(volume_path)
    weights, weights_n = read_table(weights_path), read_numeric(weights_path)
    checks, checks_n = read_table(checks_path), read_numeric(checks_path)
    matrix, matrix_n = read_table(matrix_path), read_numeric(matrix_path)
    bounds = read_table(src.output("utility/bounds.csv"))
    factorial = read_table(src.output("utility/factorial_8cell.csv"))
    classes = read_table(src.output("utility/verdict_class_accuracy.csv"))
    evaluator = read_json(src.output("utility/evaluator.json"))

    # Shared-projection support: raw-to-shared prediction changes are reported, and every fit-row support
    # recomputation agrees with the shared support used by the probes.
    changes = int(sum(int(x) for x in support["raw_to_shared_prediction_changes"]))
    setting = ["training_protocol", "evaluation_scope", "view_id", "direction"]
    bad = support[(support["support_recompute_equal"] != "1")
                  | (support["shared_support_hash"] != support["independent_shared_support_hash"])]
    disagreements = int(bad.drop_duplicates(setting).shape[0])
    add_check(rows, "utility.raw_to_shared_prediction_changes", EXPECTED_RAW_TO_SHARED_CHANGES, changes,
              changes == EXPECTED_RAW_TO_SHARED_CHANGES)
    add_check(rows, "utility.support_disagreements", 0, disagreements, disagreements == 0)

    ok_probes = probes[probes["status"] == "OK"]
    pkey = ["view_id", "direction", "probe", "training_protocol", "evaluation_scope"]
    bkey = ["view_id", "direction", "training_protocol", "evaluation_scope"]
    probe_f1 = {tuple(r[k] for k in pkey): r["macro_f1"] for r in ok_probes.to_dict("records")}
    if len(probe_f1) != len(ok_probes):
        raise SystemExit("utility/probes.csv repeats a probe setting")
    for r in factorial.to_dict("records"):
        k = (r["view_id"], r["direction"], "cosine", r["training_protocol"], r["evaluation_scope"])
        if k in probe_f1:
            raise SystemExit(f"factorial cell also present in utility/probes.csv: {k}")
        probe_f1[k] = r["macro_f1"]
    bound_c2 = {tuple(r[k] for k in bkey): r["macro_f1_upper_bound_C2"] for r in bounds.to_dict("records")}
    if len(bound_c2) != len(bounds):
        raise SystemExit("utility/bounds.csv repeats a bound setting")

    # Scores of the probe/bound table equal the probe and 8-cell tables.
    score_hits = sum(probe_f1.get(tuple(r[k] for k in pkey)) is not None
                     and float(probe_f1[tuple(r[k] for k in pkey)]) == float(r["macro_f1"])
                     for r in checks.to_dict("records"))
    add_check(rows, "utility.probe_scores_match_probe_tables", len(checks), int(score_hits),
              int(score_hits) == len(checks), "probe/bound scores equal the probe tables")

    # Volume-matched refits draw as many source rows per label as the fixed-dataset refit and are scored
    # on the same FULL-CONFORM target rows.
    refit = _one(probes, "fixed-dataset refit probe", **REFIT_PROBE)
    seeds = [int(s) for s in evaluator["settings"]["volume_seeds"]]
    volume_hits = sum(
        int(r["seed"]) in seeds and r["training_protocol"] == f"volume_matched_seed_{int(r['seed'])}"
        and r["view_id"] == REFIT_PROBE["view_id"] and r["direction"] == "FP->DF"
        and r["source_rows_fit"] == refit["source_rows_fit"] and r["target_eval_rows"] == refit["target_eval_rows"]
        for r in volume.to_dict("records"))
    volume_ok = volume_hits == len(seeds) and len(volume) == len(seeds) and volume["seed"].nunique() == len(seeds)
    add_check(rows, "utility.volume_control_matched", len(seeds), int(volume_hits), volume_ok,
              "volume-matched rows equal the refit row counts")

    # Type rescaling uses the declared weight grid on the refit support; weight 1 reproduces the refit probe.
    grid = [float(x) for x in evaluator["settings"]["type_weights"]]
    weight_hits = 0
    for r in weights.to_dict("records"):
        wgt = float(r["type_weight"])
        same_support = r["shared_support_hash"] == refit["shared_support_hash"]
        unit_ok = wgt != 1.0 or (r["prediction_hash"] == refit["prediction_hash"]
                                 and float(r["macro_f1"]) == float(refit["macro_f1"]))
        weight_hits += int(wgt in grid and same_support and unit_ok)
    weight_ok = weight_hits == len(grid) == len(weights) and weights["type_weight"].nunique() == len(grid)
    add_check(rows, "utility.type_weight_unit_matches_refit", len(grid), int(weight_hits), weight_ok,
              "weight 1 reproduces the refit probe")

    # Bounds of the probe/bound table equal the bound table; margins and violation flags are recomputed.
    bound_hits = 0
    for r in checks.to_dict("records"):
        c2 = bound_c2.get(tuple(r[k] for k in bkey))
        f1, b = float(r["macro_f1"]), float(r["C2_upper_bound"])
        bound_hits += int(c2 is not None and float(c2) == b and float(r["margin_bound_minus_f1"]) == b - f1
                          and int(r["violation"]) == int(f1 > b + BOUND_SLACK))
    add_check(rows, "utility.bounds_match_bound_table", len(checks), int(bound_hits), bound_hits == len(checks),
              "probe/bound bounds equal the bound table")

    headline = {
        "refit_f1": float(refit["macro_f1"]),
        "volume_min": float(volume_n["macro_f1"].astype(float).min()),
        "volume_max": float(volume_n["macro_f1"].astype(float).max()),
        "type_rescale_min": float(weights_n["macro_f1"].astype(float).min()),
        "type_rescale_max": float(weights_n["macro_f1"].astype(float).max()),
    }
    rounded = {k: round(v, 4) for k, v in headline.items()}
    add_check(rows, "utility.headline_quantities_4dp", EXPECTED_HEADLINE_4DP, rounded,
              rounded == EXPECTED_HEADLINE_4DP)

    violations = 0
    min_margin = math.inf
    for r in checks_n.to_dict("records"):
        f1, b = float(r["macro_f1"]), float(r["C2_upper_bound"])
        violations += int(f1 > b + BOUND_SLACK)
        min_margin = min(min_margin, b - f1)
    add_check(rows, "utility.bound_violation_count", 0, violations, violations == 0)
    add_check(rows, "utility.minimum_bound_margin", None, min_margin, None, "independent recomputation")

    add_check(rows, "utility.cos_on_det_matrix_rows", EXPECTED_COS_ON_DET_ROWS, len(matrix_n),
              len(matrix_n) == EXPECTED_COS_ON_DET_ROWS)
    target = matrix_n[matrix_n["determination_view_id"].astype(str) == "core_plus_type__family"]
    got = {(str(r["training_protocol"]), str(r["direction"])): float(r["fraction_correct_on_unsafe"])
           for r in target.to_dict("records")}
    got3 = {k: round(v, 3) for k, v in got.items()}
    add_check(rows, "utility.cos_on_det_core_plus_type_3dp",
              {str(k): v for k, v in EXPECTED_COS_ON_DET_3DP.items()}, {str(k): v for k, v in got3.items()},
              got3 == EXPECTED_COS_ON_DET_3DP)

    # Regression values asserted by the utility step, re-evaluated on its outputs.
    cos_ok = all(round(float(_one(probes, "refit cosine", view_id=v, source=s, target=t, probe="cosine",
                                  training_protocol="fixed_dataset_refit",
                                  evaluation_scope="full_conform_target", status="OK")["macro_f1"]), 4) == e
                 for (v, s, t), e in COSINE_REFIT_4DP.items())
    fac_ok = True
    for (task, direction), expected in FACTORIAL_4DP.items():
        vals = [round(float(_one(factorial, "8-cell", task=task, direction=direction, cell_id=c,
                                 training_protocol="fixed_dataset_refit",
                                 evaluation_scope="full_conform_target")["macro_f1"]), 4) for c in FACTORIAL_CELLS]
        fac_ok &= vals == expected
    ordered = sorted(weights.to_dict("records"), key=lambda r: float(r["type_weight"]))
    tw_ok = [round(float(r["macro_f1"]), 4) for r in ordered] == TYPE_WEIGHT_4DP
    vol_vals = [float(x) for x in volume["macro_f1"]]
    vol_ok = (round(min(vol_vals), 4), round(max(vol_vals), 4)) == VOLUME_MINMAX_4DP
    matrix_ok = len(matrix) > 4 and all(0.0 <= float(x) <= 1.0 for x in matrix["fraction_correct_on_unsafe"])
    classes_ok = len(classes) == 4 and bool((classes["declared_determination_basis"] == VERDICT_CLASS_BASIS).all())
    fit_rows = {s: set(probes[(probes["training_protocol"] == "fixed_dataset_refit")
                              & (probes["source"] == s)]["fitting_mask_rows"]) for s in DATASETS}
    fit_ok = all(fit_rows[s] == {str(EXPECTED_FULL[s])} for s in DATASETS)
    regression = {"cosine_refit": cos_ok, "factorial_8cell": fac_ok, "type_weight": tw_ok, "volume_range": vol_ok,
                  "determination_matrix_bounded": matrix_ok, "verdict_class_cells": classes_ok,
                  "fixed_dataset_fit_rows": fit_ok}
    reg_ok = all(regression.values())
    add_check(rows, "utility.regression_assertions", 1, int(reg_ok), reg_ok,
              "re-evaluates the step's regression values on its outputs")

    # Bound premises: no violation, and every bound uses the shared support of the cosine probe it bounds.
    support_hash = {(r["view_id"], r["direction"], r["training_protocol"], r["evaluation_scope"]):
                    r["shared_support_hash"] for r in ok_probes[ok_probes["probe"] == "cosine"].to_dict("records")}
    for r in factorial.to_dict("records"):
        support_hash[(r["view_id"], r["direction"], r["training_protocol"], r["evaluation_scope"])] = \
            r["shared_support_hash"]
    shared_ok = all(support_hash.get(tuple(r[k] for k in bkey)) == r["shared_support_hash"]
                    for r in bounds.to_dict("records"))
    premises_ok = violations == 0 and shared_ok
    add_check(rows, "utility.bound_premises", 1, int(premises_ok), premises_ok,
              "re-evaluates the bound premises on the step's outputs")
    return {"details": {"headline_quantities": headline, "bound_violations": violations,
                        "min_bound_margin": min_margin, "regression_assertions": regression,
                        "cos_on_det_core_plus_type": {str(k): v for k, v in got.items()}}}


def check_controls(src: Sources, cache: dict[str, Any], full: np.ndarray,
                   rows: list[dict[str, Any]]) -> dict[str, Any]:
    part = read_table(src.output("utility/partition_identity.csv"))
    ind = pd.DataFrame(partition_rows(cache, full))
    merged = part.merge(ind, on=["dataset", "scope", "cell_id"], how="outer", suffixes=("_primary", "_ind"),
                        indicator=True)
    exact = bool(
        len(part) == 48 and len(ind) == 48 and (merged["_merge"] == "both").all()
        and all((merged[f"{c}_primary"].astype(int) == merged[f"{c}_ind"].astype(int)).all()
                for c in ("core_fibers", "cell_fibers", "partition_equal"))
    )
    add_check(rows, "controls.partition_48rows_independent", 1, int(exact), exact)

    scopes: dict[str, list[int]] = {}
    for (ds, scope), g in part.groupby(["dataset", "scope"], sort=True):
        scopes[f"{ds}|{scope}"] = [int(g["partition_equal"].astype(int).sum()), int(len(g))]
    expected_scopes = {k: list(v) for k, v in EXPECTED_SCOPES.items()}
    add_check(rows, "controls.scope_boundary", expected_scopes, scopes, scopes == expected_scopes)

    values = read_numeric(src.output("utility/simple_checks_values.csv"))
    primary_cells = {str(r["cell_id"]): float(r["value"]) for r in values.to_dict("records")}
    settings = [(ds, task) for ds in DATASETS for task in ("facet", "family")]
    cells: dict[str, float] = {
        "l0_rate": min(cache["direct"][(ds, task)] / EXPECTED_ROWS[ds] for ds, task in settings),
        "l0_hit": min(cache["lexical"][(ds, task, "RAW")] / EXPECTED_ROWS[ds] for ds, task in settings),
        "l0_hit_san": max(cache["lexical"][(ds, task, "SANITIZED")] / EXPECTED_ROWS[ds] for ds, task in settings),
        "instfam_n": float(cache["instfam_n"]),
        "instfam_den": float(cache["instfam_den"]),
        "instfam_cov": cache["instfam_n"] / cache["instfam_den"],
    }
    fd_exact = True
    for ds in DATASETS:
        exact_fd, g3, single = fd_stats(cache["core"], cache["gold_family"], (cache["dataset"] == ds) & full)
        fd_exact &= exact_fd
        cells[f"fd_g3_{FD_CELL_SUFFIX[ds]}"] = g3
        cells[f"fd_single_{FD_CELL_SUFFIX[ds]}"] = single
    cells_ok = fd_exact and all(
        k in primary_cells and math.isclose(primary_cells[k], v, rel_tol=0.0, abs_tol=5e-7)
        for k, v in cells.items())
    add_check(rows, "controls.simple_check_cells_independent", 1, int(cells_ok), cells_ok)

    headline = {
        "l0_rate": cells["l0_rate"],
        "l0_hit": cells["l0_hit"],
        "l0_hit_san": cells["l0_hit_san"],
        "instfam_n": int(cells["instfam_n"]),
        "instfam_den": int(cells["instfam_den"]),
        f"fd_g3_{FD_CELL_SUFFIX['DF']}": cells[f"fd_g3_{FD_CELL_SUFFIX['DF']}"],
        f"fd_g3_{FD_CELL_SUFFIX['FP']}": cells[f"fd_g3_{FD_CELL_SUFFIX['FP']}"],
    }
    add_check(rows, "controls.headline_cells", EXPECTED_HEADLINE_CELLS, headline, headline == EXPECTED_HEADLINE_CELLS)

    # Permutation null: k, p = (k+1)/(B+1) and the null mean recomputed from the null rows (file row order).
    null = read_numeric(src.output("utility/permutation_null.csv"))
    summary = read_numeric(src.output("utility/permutation.csv"))
    perm_ok = True
    perm: dict[str, Any] = {}
    for r in summary.to_dict("records"):
        direction = str(r["direction"])
        vals = null[null["direction"].astype(str) == direction]["null_macro_f1"].astype(float).to_numpy()
        if len(vals) != EXPECTED_PERMUTATIONS:
            raise SystemExit(f"permutation null {direction}: {len(vals)} rows")
        observed = float(r["observed_macro_f1"])
        k = int(np.count_nonzero(vals >= observed))
        pval = (k + 1) / (len(vals) + 1)
        mean = float(vals.mean())
        perm_ok &= (int(r["B"]) == EXPECTED_PERMUTATIONS and k == int(r["k_ge_observed"]) == 0
                    and math.isclose(pval, float(r["p_value"]), abs_tol=1e-15)
                    and math.isclose(mean, float(r["null_mean"]), abs_tol=1e-12))
        perm[direction] = {"B": len(vals), "k": k, "p": pval, "null_mean": mean, "observed": observed}
    add_check(rows, "controls.permutation_B999_recomputed", 1, int(perm_ok), perm_ok)
    return {"details": {"scope_counts": scopes, "simple_check_cells": cells, "permutation": perm}}


def scaling_grouping(src: Sources, summ: pd.DataFrame, conf: dict[str, Any], cache: dict[str, Any],
                     reg: dict[str, Any]) -> tuple[int, list[str]]:
    """Recounts every grouping column of scaling_summary.csv from the input view and the conformance masks."""
    so = read_table(src.input("sample_order.csv.gz"), usecols=["dataset", "anchor_id", "sample_rank"])
    rank_of = dict(zip(so["anchor_id"], so["sample_rank"]))
    if len(rank_of) != EXPECTED_TOTAL or any(rid not in rank_of for rid in conf["row_id"].tolist()):
        raise SystemExit("sample_order.csv.gz does not cover the anchor table")
    ranks = np.fromiter((int(rank_of[rid]) for rid in conf["row_id"].tolist()), dtype=np.uint64,
                        count=EXPECTED_TOTAL)
    orders = {}
    for ds in DATASETS:
        idx = np.flatnonzero(conf["row_ds"] == ds)
        orders[ds] = idx[np.lexsort((idx.astype(np.int64), ranks[idx]))]

    proj = read_table(src.output("registry/view_projection.csv"))
    projection = {r["view_id"]: json.loads(r["projection_json"]) for r in proj.to_dict("records")}
    fam_idx = {x: i for i, x in enumerate(reg["family_coords"])}
    type_idx = {x: i for i, x in enumerate(reg["type_coords"])}
    labels = {"facet": cache["gold_facet"], "family": cache["gold_family"]}
    nlabels = {"facet": len(reg["facet_coords"]), "family": len(reg["family_coords"])}
    qpos = {v: i for i, v in enumerate(conf["qviews"])}
    label_codes = {t: np.unique(labels[t], return_inverse=True)[1].astype(np.int64) for t in labels}

    hits, misses = 0, []
    for r in summ.to_dict("records"):
        vid, task, ds, fraction = r["view_id"], r["task"], r["dataset"], float(r["fraction"])
        cols = []
        for block, coords in projection[vid].items():
            if block == "family::":
                cols.append(cache["family_counts"][:, [fam_idx[c] for c in coords]])
            elif block == "type::":
                cols.append(cache["type_counts"][:, [type_idx[c] for c in coords]])
            else:
                raise SystemExit(f"scaling view {vid} projects an unsupported block {block}")
        values = np.concatenate(cols, axis=1)
        dm = conf["row_ds"] == ds
        n = int(dm.sum())
        k = n if fraction == 1.0 else int(math.floor(n * fraction))
        sample = np.zeros(EXPECTED_TOTAL, dtype=bool)
        sample[orders[ds][:k]] = True
        q = conf["qmask"][qpos[vid]] & dm & sample
        idx = np.flatnonzero(q)
        if len(idx):
            _, inv, counts = np.unique(values[idx], axis=0, return_inverse=True, return_counts=True)
            inv = inv.reshape(-1)
            pairs = np.unique(np.column_stack([inv, label_codes[task][idx]]), axis=0)
            diversity = np.bincount(pairs[:, 0], minlength=len(counts))
        else:
            counts = diversity = np.zeros(0, dtype=np.int64)
        got = {
            "dataset_rows_total": n,
            "sampled_dataset_rows": int(np.count_nonzero(dm & sample)),
            "sampled_full_conform_rows": int(np.count_nonzero(dm & sample & conf["full"])),
            "sampled_query_conform_rows": int(len(idx)),
            "facet_dimension": len(reg["facet_coords"]),
            "label_universe_size": nlabels[task],
            "theoretical_competing_labels_per_candidate": max(0, nlabels[task] - 1),
            "observed_unique_fibers": int(len(counts)),
            "multi_label_fibers": int(np.count_nonzero(diversity >= 2)),
            "single_label_fibers": int(np.count_nonzero(diversity == 1)),
            "max_fiber_rows": int(counts.max()) if len(counts) else 0,
            "mean_fiber_rows": float(counts.mean()) if len(counts) else 0.0,
        }
        same = all(float(r[c]) == float(got[c]) for c in SCALING_GROUPING_COLUMNS)
        # Integer programs run on the single-label fibers; each ends UNSAFE, not UNSAFE or infeasible.
        solver_ok = (as_int(r["solver_fibers"]) == got["single_label_fibers"]
                     and as_int(r["unsafe_fibers"]) + as_int(r["nonunsafe_fibers"]) + as_int(r["infeasible_fibers"])
                     == as_int(r["solver_fibers"]))
        if same and solver_ok:
            hits += 1
        else:
            misses.append(f"{vid}|{ds}|{r['fraction']}")
    return hits, misses


def check_scaling(src: Sources, conf: dict[str, Any], cache: dict[str, Any], reg: dict[str, Any],
                  rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_path = src.output("scaling/scaling_runs.csv")
    summ_path = src.output("scaling/scaling_summary.csv")
    agg_path = src.output("scaling/scaling.csv")
    raw, summ, agg = read_table(raw_path), read_table(summ_path), read_table(agg_path)
    raw_n, summ_n, agg_n = read_numeric(raw_path), read_numeric(summ_path), read_numeric(agg_path)
    if (len(raw), len(summ), len(agg)) != EXPECTED_SCALING_SHAPE:
        raise SystemExit(f"scaling cardinalities {(len(raw), len(summ), len(agg))}")
    mismatches: list[str] = []

    def cell_key(r: dict[str, Any]) -> tuple[Any, ...]:
        return (float(r["fraction"]),) + tuple(str(r[k]) for k in SCALING_KEYS[1:])

    # Raw -> summary: three repeats per cell with identical non-timing columns (exact values) and the
    # median, minimum and maximum of every timing column.
    struct_cols = [c for c in raw.columns if c not in ("repeat",) + SCALING_KEYS + SCALING_TIMING]
    sindex = {cell_key(r): r for r in summ.to_dict("records")}
    sindex_n = {cell_key(r): r for r in summ_n.to_dict("records")}
    if len(sindex) != len(summ) or len(sindex_n) != len(summ):
        raise SystemExit("scaling_summary.csv repeats a cell")
    raw_n_groups = {cell_key(g.iloc[0].to_dict()): g for _, g in raw_n.groupby(list(SCALING_KEYS), sort=False)}
    groups = 0
    for _, g in raw.groupby(list(SCALING_KEYS), sort=False):
        groups += 1
        key = cell_key(g.iloc[0].to_dict())
        if len(g) != 3:
            raise SystemExit(f"scaling repeat count != 3: {key}")
        s, s_n, g_n = sindex.get(key), sindex_n.get(key), raw_n_groups.get(key)
        if s is None or s_n is None or g_n is None:
            raise SystemExit(f"scaling summary lookup failed: {key}")
        for c in struct_cols:
            if any(float(x) != float(s[c]) for x in g[c]):
                mismatches.append(f"raw->summary {key} {c}")
        if as_int(s["repeats"]) != len(g):
            mismatches.append(f"raw->summary {key} repeats")
        for c in SCALING_TIMING:
            vals = g_n[c].astype(float).tolist()
            if not math.isclose(statistics.median(vals), float(s_n[c + "_median"]), rel_tol=0, abs_tol=1e-15):
                mismatches.append(f"raw->summary {key} {c}_median")
            if min(vals) != float(s_n[c + "_min"]) or max(vals) != float(s_n[c + "_max"]):
                mismatches.append(f"raw->summary {key} {c}_min/max")
    if groups != len(summ):
        mismatches.append("raw->summary cell count")

    # Summary -> aggregate: per-fraction grouping sums of the summary cells.
    for frac, g in summ.groupby("fraction", sort=True):
        ar = agg[agg["fraction"].astype(float) == float(frac)]
        ar_n = agg_n[agg_n["fraction"].astype(float) == float(frac)]
        g_n = summ_n[summ_n["fraction"].astype(float) == float(frac)]
        if len(ar) != 1 or len(ar_n) != 1:
            raise SystemExit(f"scaling aggregate lookup failed: {frac}")
        ar, ar_n = ar.iloc[0], ar_n.iloc[0]
        first = {ds: g[g["dataset"] == ds].iloc[0] for ds in DATASETS}
        exact_ints = {
            "sampled_rows_total": sum(as_int(first[ds]["sampled_dataset_rows"]) for ds in DATASETS),
            "sampled_full_conform_rows_sum_over_datasets":
                sum(as_int(first[ds]["sampled_full_conform_rows"]) for ds in DATASETS),
            "sampled_query_rows_sum_over_pathb_cells": sum(as_int(x) for x in g["sampled_query_conform_rows"]),
            "pathb_cells": int(len(g)),
            "sum_view_dataset_fibers": sum(as_int(x) for x in g["observed_unique_fibers"]),
            "sum_solver_fibers": sum(as_int(x) for x in g["solver_fibers"]),
            "sum_solver_rival_checks": sum(as_int(x) for x in g["solver_rival_checks"]),
        }
        for col, val in exact_ints.items():
            if as_int(ar[col]) != val:
                mismatches.append(f"summary->aggregate {frac} {col}")
        for c in SCALING_TIMING:
            val = float(g_n[c + "_median"].astype(float).sum())
            if not math.isclose(val, float(ar_n["sum_" + c + "_median"]), rel_tol=0, abs_tol=1e-12):
                mismatches.append(f"summary->aggregate {frac} sum_{c}_median")
    infeasible = sum(as_int(x) for x in summ["infeasible_fibers"])

    def agg_row(fraction: float) -> pd.Series:
        return agg_n[agg_n["fraction"].astype(float) == fraction].iloc[0]

    a01, a10 = agg_row(0.1), agg_row(1.0)
    observational = {
        "rows_10pct": as_int(a01["sampled_rows_total"]),
        "rows_100pct": as_int(a10["sampled_rows_total"]),
        "time_10pct": float(a01["sum_total_seconds_median"]),
        "time_100pct": float(a10["sum_total_seconds_median"]),
        "time_ratio_100_over_10": float(a10["sum_total_seconds_median"]) / float(a01["sum_total_seconds_median"]),
        "grouping_ratio_100_over_10":
            float(a10["sum_grouping_seconds_median"]) / float(a01["sum_grouping_seconds_median"]),
        "milp_ratio_100_over_10": float(a10["sum_milp_seconds_median"]) / float(a01["sum_milp_seconds_median"]),
        "fibers_10pct": as_int(a01["sum_view_dataset_fibers"]),
        "fibers_100pct": as_int(a10["sum_view_dataset_fibers"]),
        "solver_fibers_10pct": as_int(a01["sum_solver_fibers"]),
        "solver_fibers_100pct": as_int(a10["sum_solver_fibers"]),
    }
    consistent = not mismatches
    add_check(rows, "scaling.raw_summary_aggregate_recomputation", 1, int(consistent), consistent)
    rows_got = [observational["rows_10pct"], observational["rows_100pct"]]
    add_check(rows, "scaling.rows_10_to_100", EXPECTED_SCALING_ROWS, rows_got, rows_got == EXPECTED_SCALING_ROWS)
    add_check(rows, "scaling.infeasible_fibers", 0, infeasible, infeasible == 0)
    add_check(rows, "scaling.scaling_observational", None, observational, None,
              "timing columns of the released files are those of the reference run (scaling/reference_times.json)")

    hits, misses = scaling_grouping(src, summ, conf, cache, reg)
    add_check(rows, "scaling.grouping_columns_from_inputs", len(summ), hits, hits == len(summ) and not misses,
              "fiber grouping of the sampled query rows recounted from the input view")
    return {"details": {"observational": observational, "consistency_mismatches": mismatches,
                        "grouping_mismatches": misses}}


def check_path_b_open(src: Sources, conf: dict[str, Any], cache: dict[str, Any], reg: dict[str, Any],
                      rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-checks every open single-label Path B fiber of the full table in
    integer arithmetic: its witness state is a nonzero nonnegative integer state on the dataset support whose
    exposed marginals equal the fiber value, and the rival ties with or exceeds the observed label there; an
    unbounded program needs a free supported facet that the carrier map sends to the rival. The open fibers of
    each view and dataset must be the non-UNSAFE solver fibers of the full-size scaling cell, whose grouping is
    recounted from the input view above. The query-conforming fiber size of every row is recounted from the input
    view; the public table must equal the full table without the DF rows of single-anchor fibers, and the summary
    must hold the counts and the SHA-256 of both tables."""
    fmap = read_table(src.output("registry/facet_map.csv"))
    t_map = dict(zip(fmap["facet"], fmap["term_type"]))
    w_map = dict(zip(fmap["facet"], fmap["family"]))
    support = {ds: list(v["S_members"]) for ds, v in read_json(src.output("registry/dataset_support.json"))["datasets"].items()}
    proj = read_table(src.output("registry/view_projection.csv"))
    projection = {r["view_id"]: json.loads(r["projection_json"]) for r in proj.to_dict("records")}
    task_of = {r["view_id"]: r["task"] for r in proj.to_dict("records")}
    image = {"family::": w_map, "type::": t_map}
    table = read_table(src.output("verdicts/path_b_open_witnesses.csv"))
    bad: list[str] = []
    per_cell: Counter = Counter()
    status: Counter = Counter()
    for r in table.to_dict("records"):
        vid, ds = str(r["view_id"]), str(r["dataset"])
        key = f"{vid}|{ds}|{r['group_index']}"
        per_cell[(vid, ds)] += 1
        status[str(r["program_status"])] += 1
        state = json.loads(r["witness_state"])
        value = json.loads(r["fiber_value"])
        ok = (all(isinstance(x, int) and x > 0 for x in state.values()) and set(state) <= set(support[ds])
              and sum(state.values()) >= 1)
        k, bound = 0, set()
        for block, coords in projection[vid].items():
            for coord in coords:
                ok &= sum(x for f, x in state.items() if image[block][f] == coord) == int(value[k])
                bound |= {f for f in support[ds] if image[block][f] == coord}
                k += 1
        label_of = (lambda f: f) if task_of[vid] == "facet" else (lambda f: w_map[f])
        obs = sum(x for f, x in state.items() if label_of(f) == r["observed_label"])
        riv = sum(x for f, x in state.items() if label_of(f) == r["rival"])
        gap = riv - obs
        ok &= gap >= 0 and gap == int(r["rival_minus_observed_at_state"]) and r["rival"] != r["observed_label"]
        free_to_rival = [f for f in support[ds] if f not in bound and label_of(f) == r["rival"]]
        if r["program_status"] == "optimal":
            ok &= str(r["rival_minus_observed_optimum"]) == str(gap)
        elif r["program_status"] == "unbounded":
            ok &= str(r["rival_minus_observed_optimum"]) == "inf" and len(free_to_rival) > 0
        else:
            ok = False
        if not ok:
            bad.append(key)
    # query-conforming fiber sizes recounted from the input view
    fam_idx = {x: i for i, x in enumerate(reg["family_coords"])}
    type_idx = {x: i for i, x in enumerate(reg["type_coords"])}
    qpos = {v: i for i, v in enumerate(conf["qviews"])}
    size_bad: list[str] = []
    sizes: dict[str, int] = {}
    for vid in sorted(set(table["view_id"].astype(str))):
        cols = []
        for block, coords in projection[vid].items():
            if block == "family::":
                cols.append(cache["family_counts"][:, [fam_idx[c] for c in coords]])
            elif block == "type::":
                cols.append(cache["type_counts"][:, [type_idx[c] for c in coords]])
            else:
                raise SystemExit(f"open Path B view {vid} projects an unsupported block {block}")
        values = np.concatenate(cols, axis=1).astype(np.int64)
        for ds in DATASETS:
            idx = np.flatnonzero(conf["qmask"][qpos[vid]] & (conf["row_ds"] == ds))
            fiber_count = Counter(map(tuple, values[idx].tolist()))
            for r in table[(table["view_id"].astype(str) == vid) & (table["dataset"].astype(str) == ds)].to_dict("records"):
                key = f"{vid}|{ds}|{r['group_index']}"
                sizes[key] = int(fiber_count.get(tuple(int(x) for x in json.loads(r["fiber_value"])), 0))
                if sizes[key] != as_int(r["query_conforming_fiber_size"]):
                    size_bad.append(key)
    single_df = {k for k, n in sizes.items() if k.split("|")[1] == "DF" and n == 1}
    full_text = src.output("verdicts/path_b_open_witnesses.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    public_path = src.output("verdicts/path_b_open_witnesses_public.csv")
    public_text = public_path.read_text(encoding="utf-8").splitlines(keepends=True)
    keyed = [f"{r['view_id']}|{r['dataset']}|{r['group_index']}" for r in table.to_dict("records")]
    expected_public = full_text[:1] + [line for k, line in zip(keyed, full_text[1:]) if k not in single_df]
    public_ok = public_text == expected_public
    summary = read_json(src.output("verdicts/path_b_open_summary.json"))
    public_df = sum(1 for line in public_text[1:] if next(csv.reader([line]))[1] == "DF")
    summary_got = {"full_rows": summary["full_table"]["rows"], "public_rows": summary["public_table"]["rows"],
                   "withheld_rows": summary["withheld_rows"]["rows"],
                   "rows_passing_exact_check": summary["full_table"]["rows_passing_exact_check"],
                   "full_sha256": summary["full_table"]["sha256"], "public_sha256": summary["public_table"]["sha256"],
                   "by_program_status": summary["full_table"]["by_program_status"]}
    summary_expected = {"full_rows": len(table), "public_rows": len(public_text) - 1, "withheld_rows": len(single_df),
                        "rows_passing_exact_check": len(table) - len(bad),
                        "full_sha256": sha256_file(src.output("verdicts/path_b_open_witnesses.csv")),
                        "public_sha256": sha256_file(public_path), "by_program_status": dict(sorted(status.items()))}
    summary_ok = summary_got == summary_expected and len(public_text) - 1 + len(single_df) == len(table)
    summ = read_table(src.output("scaling/scaling_summary.csv"))
    full = {(r["view_id"], r["dataset"]): as_int(r["nonunsafe_fibers"])
            for r in summ.to_dict("records") if float(r["fraction"]) == 1.0}
    counts_ok = all(per_cell.get(k, 0) == v for k, v in full.items()) and set(per_cell) <= set(full)
    add_check(rows, "verdicts.PathB_open_witness_states", len(table), len(table) - len(bad), not bad,
              "every open single-label Path B fiber re-checked in integer arithmetic")
    add_check(rows, "verdicts.PathB_open_equals_nonunsafe_solver_fibers", sum(full.values()), len(table),
              counts_ok, "per view and dataset, against scaling/scaling_summary.csv at fraction 1.0")
    add_check(rows, "verdicts.PathB_open_fiber_sizes_from_inputs", len(table), len(table) - len(size_bad),
              not size_bad, "query-conforming anchors of each open fiber recounted from the input view")
    add_check(rows, "verdicts.PathB_open_public_subset", len(table) - len(single_df), len(public_text) - 1,
              public_ok and public_df == sum(1 for k in keyed if k.split("|")[1] == "DF" and k not in single_df),
              "public table = full table without the DF rows of single-anchor fibers")
    add_check(rows, "verdicts.PathB_open_summary", summary_expected, summary_got, summary_ok,
              "counts and SHA-256 of both tables in verdicts/path_b_open_summary.json")
    return {"details": {"open_fibers": len(table), "by_status": dict(sorted(status.items())), "failed": bad,
                        "public_rows": len(public_text) - 1, "withheld_single_anchor_DF_rows": len(single_df),
                        "fiber_size_mismatches": size_bad}}


def check_certificate_fibers(src: Sources, conf: dict[str, Any], cache: dict[str, Any], reg: dict[str, Any],
                             rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every DF row of the two public certificate files names a view value that at least two DF anchors carry
    (recounted from the input view; for the released certificates also among the query-conforming anchors)."""
    proj = read_table(src.output("registry/view_projection.csv"))
    projection = {r["view_id"]: json.loads(r["projection_json"]) for r in proj.to_dict("records")}
    fam_idx = {x: i for i, x in enumerate(reg["family_coords"])}
    type_idx = {x: i for i, x in enumerate(reg["type_coords"])}
    qpos = {v: i for i, v in enumerate(conf["qviews"])}
    dm = conf["row_ds"] == "DF"
    sizes: dict[str, list[int]] = {}
    for rel in ("verdicts/path_b_certificates.csv", "support/support_certificates.csv"):
        table = read_table(src.output(rel))
        sizes[rel] = []
        for r in table[table["dataset"] == "DF"].to_dict("records"):
            vid = r["view_id"]
            cols = []
            for block, coords in projection[vid].items():
                if block == "family::":
                    cols.append(cache["family_counts"][:, [fam_idx[c] for c in coords]])
                elif block == "type::":
                    cols.append(cache["type_counts"][:, [type_idx[c] for c in coords]])
                else:
                    raise SystemExit(f"certificate view {vid} projects an unsupported block {block}")
            values = np.concatenate(cols, axis=1).astype(np.int64)
            hit = dm & np.all(values == np.asarray(json.loads(r["fiber_value"]), dtype=np.int64), axis=1)
            sizes[rel].append(int(hit.sum()))
            if rel == "verdicts/path_b_certificates.csv":
                sizes.setdefault("query_conforming", []).append(int((hit & conf["qmask"][qpos[vid]]).sum()))
    minimum = {k: min(v) if v else None for k, v in sizes.items()}
    ok = all(x is None or x >= 2 for x in minimum.values())
    add_check(rows, "verdicts.PathB_certificate_DF_fibers_shared", ">= 2", minimum, ok,
              "DF anchors carrying the view value of each DF row of the public certificate files")
    return {"details": {"minimum_DF_anchors_per_row": minimum}}


# ----------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, default=REPO / "data" / "hub")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    a = ap.parse_args()
    started = time.perf_counter()

    src = Sources(a.inputs, a.outputs, read_sums(SUMS_FILE))
    for name in INPUT_FILES:
        src.input(name)

    rows: list[dict[str, Any]] = []
    reg = check_registry(src, rows)
    inp = check_registry_inputs(src, reg, rows)
    conf = check_conformance(src, rows)
    milp = milp_counterexample(rows)
    cache = anchor_pass(src.input("anchors.csv.gz"), conf["row_ds"], conf["row_id"], reg["pi"], reg["w"],
                        reg["family_coords"], reg["type_coords"])
    print(f"[verify] anchor pass {time.perf_counter() - started:.1f}s", flush=True)
    rcon = check_conformance_recomputed(src, conf, conformance_pass(src.input("anchors.csv.gz"), conf["row_ds"],
                                                                    conf["row_id"], inp["pi"], inp["t"], inp["w"]), rows)
    print(f"[verify] conformance recomputed {time.perf_counter() - started:.1f}s", flush=True)
    ver = check_verdicts(src, conf, cache, rows)
    car = check_carrier(src, conf, cache, rows)
    sem = check_semantics(src, conf, cache, reg, rows)
    check_mutations(src, rows)
    uti = check_utility(src, rows)
    ctl = check_controls(src, cache, conf["full"], rows)
    sca = check_scaling(src, conf, cache, reg, rows)
    pbo = check_path_b_open(src, conf, cache, reg, rows)
    cfb = check_certificate_fibers(src, conf, cache, reg, rows)

    failed = [r["check"] for r in rows if r["exact_match"] == 0]
    out = a.outputs / "checks"
    write_checks_csv(out / "independent_checks.csv", rows)
    write_json(out / "independent_verification.json", {
        "step": "verify",
        "schema_version": SCHEMA_VERSION,
        "writer_id": WRITER_ID,
        "inputs_sha256": dict(sorted(src.hashes.items())),
        "counts": {
            "checks_total": len(rows),
            "checks_with_exact_comparison": sum(r["exact_match"] != "NA" for r in rows),
            "checks_observational": sum(r["exact_match"] == "NA" for r in rows),
            "checks_failed": len(failed),
        },
        "all_pass": not failed,
        "failed_checks": failed,
        "pipeline_library_imported": False,
        "settings": {
            "threads": THREAD_LIMIT,
            "float_parsing": {
                "identities_between_tables": "Python float() on the CSV strings (exact round trip)",
                "reported_quantities": "pandas.read_csv default type inference and float converter",
            },
            "milp_options": {"disp": False, "mip_rel_gap": 0.0},
            "tolerances": {"bound_slack": BOUND_SLACK, "simple_check_cells_abs": 5e-7,
                           "permutation_p_abs": 1e-15, "permutation_mean_abs": 1e-12,
                           "scaling_median_abs": 1e-15, "scaling_sum_abs": 1e-12},
        },
        "details": {
            "registry": reg["details"],
            "conformance": {**conf["details"], "recomputed": rcon["details"]},
            "verdicts": {"milp_counterexample": milp, **ver["details"], "path_b_open": pbo["details"],
                         "path_b_certificate_fibers": cfb["details"],
                         "carrier_equivalence": car["details"]},
            "semantics": sem["details"],
            "utility": uti["details"],
            "controls": ctl["details"],
            "scaling": sca["details"],
        },
    })
    print(f"[verify] checks={len(rows)} failed={len(failed)} done in {time.perf_counter() - started:.1f}s", flush=True)
    if failed:
        raise SystemExit("independent verification failed: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
