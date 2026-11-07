# SPDX-License-Identifier: MIT
"""Evaluator definition of the utility step: evaluation features, fitting and target rows, metric labels, scoring."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score

from .names import DATASETS, FAMILY_ORDER

# Scoring rule of every score-based probe.
SCORE_EPS = 1e-12            # a maximum score <= SCORE_EPS abstains
TIE_ATOL = 1e-12             # an absolute tie at a positive maximum abstains
MIN_LABEL_SUPPORT = 3        # a source label enters the inventory with at least this many source rows
ABSTAIN = "__ABSTAIN__"      # abstention label; it counts as an error

# Utility views use the four role blocks only.
EVAL_PREFIXES = ("core::", "facet::", "family::", "type::")
EXPECTED_EVAL_FEATURE_DIMS = 1181 + 26 + 5 + 3
EXPECTED_ROWS = {"DF": 289202, "FP": 342182}
EXPECTED_FULL_ROWS = {"DF": 274622, "FP": 291686}
EXPECTED_VIEWS = 32
CHUNK_ROWS = 50000

DIRECTIONS = ((DATASETS[0], DATASETS[1]), (DATASETS[1], DATASETS[0]))
ALL_SOURCE = "all_source"          # fit on every source row
REFIT = "fixed_dataset_refit"                 # fit on registry-conforming source rows only
ALL_TARGET = "all_supported_target"           # target rows whose gold label is in the source inventory
CONFORM_TARGET = "full_conform_target"        # ... restricted to registry-conforming target rows
PROTOCOL_SCOPES = ((ALL_SOURCE, ALL_TARGET), (REFIT, CONFORM_TARGET))
SHARED_PROTOCOL = "fit_row_shared_projection"

MAIN_VIEW_IDS = (
    "identity_admitted__family",
    "core_plus_type__family",
    "facet_only__family",
    "type3__family",
    "type_attr_part__family",
    "identity_admitted__facet",
    "family3__facet",
    "family3_plus_type__facet",
)
# Views with few coordinates, where a source-fiber lookup table is defined.
LOOKUP_VIEWS = frozenset({
    "facet_only__family",
    "type3__family",
    "type_attr_part__family",
    "family3__facet",
    "family3_plus_type__facet",
})

# Fixed experiment settings (written into utility/evaluator.json).
TOLERANCE_GRID = (1e-12, 1e-9, 1e-8)
TYPE_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
VOLUME_SEEDS = (0, 1, 2, 3, 4)
CALIBRATION_SCHEDULE = (
    ("DF", "FP", "facet", 20260819),
    ("DF", "FP", "family", 20260820),
    ("FP", "DF", "facet", 20260821),
    ("FP", "DF", "family", 20260822),
)
UNIFORM_DRAWS = 10000
BOUND_SLACK = 1e-12          # a probe exceeds a bound when F1 > bound + BOUND_SLACK
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS")
THREADS = 1


def settings() -> dict[str, Any]:
    return {
        "tolerance_grid": list(TOLERANCE_GRID),
        "type_weights": list(TYPE_WEIGHTS),
        "volume_seeds": list(VOLUME_SEEDS),
        "volume_sampler": "numpy.random.default_rng(seed).choice(pool, size=n, replace=False) per source label, "
                          "labels in source-inventory order",
        "calibration_schedule": [{"source": s, "target": t, "task": k, "seed": seed}
                                 for s, t, k, seed in CALIBRATION_SCHEDULE],
        "uniform_draws": UNIFORM_DRAWS,
        "uniform_sampler": "multinomial(n, [1/k]*k) of numpy.random.default_rng(seed) (PCG64; binomial by "
                           "inversion if n*p <= 30, else BTPE) with every a*b+c of the binomial algorithms rounded "
                           "once (fused multiply-add); one stream per schedule row, labels in declared order",
        "bound_slack": BOUND_SLACK,
        "threads": THREADS,
        "thread_environment": {k: str(THREADS) for k in THREAD_ENV},
        "cosine_arithmetic": "float32; prototype sums and norms in float32; target rows L2-normalised with a "
                             "float64 norm; score = one fused multiply-add per target nonzero in ascending "
                             "column order",
        "label_order": "inventory: decreasing source frequency, ties in first-occurrence order; tie breaks and "
                       "uniform-draw columns: names.FAMILY_ORDER (family) and the registry facet-map order (facet)",
    }


SEMANTICS = {
    "abstention": "error",
    "metric_labels": "source-supported AND target-present, fixed per evaluation scope",
    "nonpositive": "max score <= score_eps -> ABSTAIN",
    "positive_tie": "absolute tie within tie_atol at positive maximum -> ABSTAIN",
    "prediction_candidates": "all source-supported labels",
    "primary_metric": "Macro-F1 over fixed metric labels",
    "score_eps": SCORE_EPS,
    "source_support_min": MIN_LABEL_SUPPORT,
    "target_eval": "target rows whose gold label is source-supported",
    "tie_atol": TIE_ATOL,
}
COVERAGE_DEFINITIONS = {
    "class_coverage": "metric labels / all target-present labels",
    "decisive_score_coverage": "non-abstained evaluated rows / evaluated rows",
    "instance_coverage": "source-supported target rows / all target rows",
}


# ----------------------------------------------------------------------------
# Hashes (provenance columns only; no value depends on them)
# ----------------------------------------------------------------------------

def hash_strings(values: Sequence[str]) -> str:
    """SHA-256 of the canonical JSON list of the strings."""
    b = json.dumps([str(v) for v in values], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(b.encode("utf-8")).hexdigest()


def hash_rows(row_ids: Sequence[str]) -> str:
    """SHA-256 over a length-framed UTF-8 row-id sequence."""
    h = hashlib.sha256()
    for value in row_ids:
        b = str(value).encode("utf-8")
        h.update(len(b).to_bytes(8, "little")); h.update(b)
    return h.hexdigest()


def prediction_hash(row_ids: Sequence[str], preds: Sequence[str]) -> str:
    """SHA-256 over length-framed (row id, prediction) pairs."""
    if len(row_ids) != len(preds):
        raise SystemExit("prediction hash: length mismatch")
    h = hashlib.sha256()
    for rid, p in zip(row_ids, preds):
        for x in (str(rid), str(p)):
            b = x.encode("utf-8")
            h.update(len(b).to_bytes(8, "little")); h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------

def load_feature_cache(anchors: Path, full_conform: Path) -> dict[str, Any]:
    """Sparse float32 count matrix over the four role blocks (columns in first-seen order), labels and the
    full-registry-conformance mask, all in anchor release order."""
    feature_map: dict[str, int] = {}
    names: list[str] = []
    rr: list[np.ndarray] = []
    cc: list[np.ndarray] = []
    vv: list[np.ndarray] = []
    ds: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    g_facet: list[np.ndarray] = []
    g_family: list[np.ndarray] = []
    n = 0
    cols = ["dataset", "anchor_id", "full_feats_json", "gold_facet", "gold_family"]
    for ch in pd.read_csv(anchors, usecols=cols, dtype=str, keep_default_na=False, chunksize=CHUNK_ROWS):
        r: list[int] = []; c: list[int] = []; v: list[float] = []
        for i, text in enumerate(ch["full_feats_json"].tolist()):
            feats = json.loads(text)
            if not isinstance(feats, dict):
                raise SystemExit("full_feats_json is not an object")
            for key, value in feats.items():
                key = str(key)
                if not key.startswith(EVAL_PREFIXES):
                    continue
                j = feature_map.get(key)
                if j is None:
                    j = len(names); feature_map[key] = j; names.append(key)
                x = float(value)
                if not math.isfinite(x) or x < 0 or not x.is_integer():
                    raise SystemExit(f"invalid released count {key}={value!r}")
                if x != 0:
                    r.append(n + i); c.append(j); v.append(x)
        rr.append(np.asarray(r, dtype=np.int64)); cc.append(np.asarray(c, dtype=np.int64))
        vv.append(np.asarray(v, dtype=np.float32))
        ds.append(ch["dataset"].to_numpy(dtype=str)); ids.append(ch["anchor_id"].to_numpy(dtype=str))
        g_facet.append(ch["gold_facet"].to_numpy(dtype=str)); g_family.append(ch["gold_family"].to_numpy(dtype=str))
        n += len(ch)
    if len(names) != EXPECTED_EVAL_FEATURE_DIMS:
        raise SystemExit(f"evaluation feature dimensions {len(names)} != {EXPECTED_EVAL_FEATURE_DIMS}")
    X = sparse.csr_matrix(
        (np.concatenate(vv), (np.concatenate(rr).astype(np.int32), np.concatenate(cc).astype(np.int32))),
        shape=(n, len(names)), dtype=np.float32)
    X.sum_duplicates(); X.sort_indices()
    dataset = np.concatenate(ds); row_id = np.concatenate(ids)

    fc = pd.read_csv(full_conform, usecols=["dataset", "row_id", "full_registry_conform"],
                     dtype={"dataset": str, "row_id": str, "full_registry_conform": int}, keep_default_na=False)
    if len(fc) != n:
        raise SystemExit("full-conformance table row count differs from the anchors")
    if not np.array_equal(fc["row_id"].to_numpy(dtype=str), row_id):
        raise SystemExit("full-conformance row order differs from the anchors")
    if not np.array_equal(fc["dataset"].to_numpy(dtype=str), dataset):
        raise SystemExit("full-conformance dataset column differs from the anchors")
    full_mask = fc["full_registry_conform"].to_numpy(dtype=bool)
    for d, expected in EXPECTED_ROWS.items():
        if int(np.sum(dataset == d)) != expected:
            raise SystemExit(f"{d} row count differs from the declared benchmark")
        if int(np.sum((dataset == d) & full_mask)) != EXPECTED_FULL_ROWS[d]:
            raise SystemExit(f"{d} registry-conforming row count differs from the declared benchmark")
    return {
        "X": X,
        "features": np.asarray(names, dtype=object),
        "row_id": row_id,
        "dataset": dataset,
        "gold_facet": np.concatenate(g_facet),
        "gold_family": np.concatenate(g_family),
        "full_mask": full_mask,
    }


def load_views(path: Path) -> dict[str, dict[str, Any]]:
    """View projections: view id -> task, exposed blocks and the per-block coordinate selector."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for r in df.to_dict("records"):
        out[r["view_id"]] = {"task": r["task"], "exposed_blocks": r["exposed_blocks"],
                             "projection": json.loads(r["projection_json"])}
    if len(out) != EXPECTED_VIEWS or any(v not in out for v in MAIN_VIEW_IDS):
        raise SystemExit("view projection table does not hold the declared 32 views")
    return out


def facet_order(facet_map: Path) -> tuple[str, ...]:
    """Declared facet order: the row order of the registry facet map."""
    with open(facet_map, encoding="utf-8", newline="") as f:
        return tuple(r["facet"] for r in csv.DictReader(f))


def label_sort_key(task: str, facets: Sequence[str]):
    """Declared label order used wherever an order can change a value (tie breaks, draw columns)."""
    order = {x: i for i, x in enumerate(FAMILY_ORDER if task == "family" else facets)}

    def key(lab: str) -> int:
        if lab not in order:
            raise SystemExit(f"label {lab!r} is not in the declared {task} order")
        return order[lab]
    return key


# ----------------------------------------------------------------------------
# Views, rows and shared projections
# ----------------------------------------------------------------------------

def selected_columns(features: np.ndarray, projection: dict[str, Any]) -> np.ndarray:
    """Feature indices exposed by a projection ({block prefix: "ALL" | [coordinates]}), ascending."""
    selected: set[int] = set()
    for block, coords in projection.items():
        prefix = str(block)
        if coords == "ALL":
            selected.update(i for i, f in enumerate(features) if str(f).startswith(prefix))
        else:
            wanted = {prefix + str(x) for x in coords}
            selected.update(i for i, f in enumerate(features) if str(f) in wanted)
    arr = np.asarray(sorted(selected), dtype=np.int32)
    if len(arr) == 0:
        raise SystemExit(f"projection selects no feature: {projection}")
    return arr


def view_columns(cache: dict[str, Any], views: dict[str, dict[str, Any]], view_id: str) -> np.ndarray:
    return selected_columns(cache["features"], views[view_id]["projection"])


def source_inventory(cache: dict[str, Any], source: str, task: str) -> tuple[list[str], np.ndarray]:
    """Source label inventory (labels with >= MIN_LABEL_SUPPORT source rows, in decreasing frequency,
    ties in first-occurrence order) and all source row indices."""
    si_all = np.flatnonzero(cache["dataset"] == source)
    counts = Counter(cache["gold_" + task][si_all].tolist())
    return [k for k, v in counts.most_common() if int(v) >= MIN_LABEL_SUPPORT], si_all


def fitting_rows(cache: dict[str, Any], source: str, task: str, protocol: str
                 ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(fitting mask rows, fitting rows with an inventory label, inventory) for a training protocol."""
    labels, si_all = source_inventory(cache, source, task)
    if protocol == ALL_SOURCE:
        mask_rows = si_all
    elif protocol == REFIT:
        mask_rows = si_all[cache["full_mask"][si_all]]
    else:
        raise SystemExit(f"unknown training protocol {protocol}")
    y = cache["gold_" + task]
    fit = mask_rows[pd.Series(y[mask_rows]).isin(labels).to_numpy()]
    return mask_rows, fit, labels


def target_rows(cache: dict[str, Any], source: str, target: str, task: str, scope: str
                ) -> tuple[np.ndarray, list[str], list[str]]:
    """(evaluated target rows, source inventory, metric labels) for an evaluation scope."""
    labels, _ = source_inventory(cache, source, task)
    y = cache["gold_" + task]
    ti_all = np.flatnonzero(cache["dataset"] == target)
    ti = ti_all[pd.Series(y[ti_all]).isin(labels).to_numpy()]
    if scope == CONFORM_TARGET:
        ti = ti[cache["full_mask"][ti]]
    elif scope != ALL_TARGET:
        raise SystemExit(f"unknown evaluation scope {scope}")
    present = set(y[ti].tolist())
    return ti, labels, [lab for lab in labels if lab in present]


def metric_labels_for(cache: dict[str, Any], task: str, labels: list[str], rows: np.ndarray) -> list[str]:
    present = set(cache["gold_" + task][rows].tolist())
    return [lab for lab in labels if lab in present]


def shared_projection(cache: dict[str, Any], cols: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, str]:
    """Shared projection V^sh: the view columns with nonzero support among the fitting rows."""
    support = np.asarray(cache["X"][rows][:, cols].getnnz(axis=0) > 0).ravel()
    shared = cols[support]
    return shared, feature_hash(cache, shared)


def feature_hash(cache: dict[str, Any], cols: np.ndarray) -> str:
    return hash_strings([str(cache["features"][i]) for i in cols])


# ----------------------------------------------------------------------------
# Scoring rule and metrics
# ----------------------------------------------------------------------------

def score_predictions(scores: np.ndarray, labels: Sequence[str], score_eps: float = SCORE_EPS,
                      tie_atol: float = TIE_ATOL) -> tuple[np.ndarray, dict[str, Any]]:
    """Arg-max prediction; a nonpositive maximum or a positive tie within tie_atol abstains."""
    labels_arr = np.asarray(labels, dtype=object)
    if scores.shape[1] != len(labels_arr) or len(labels_arr) == 0:
        raise SystemExit("score matrix and label inventory disagree")
    argmax_idx = scores.argmax(axis=1)
    max_score = scores[np.arange(len(scores)), argmax_idx]
    tie_count = np.isclose(scores, max_score[:, None], rtol=0.0, atol=tie_atol).sum(axis=1)
    nonpositive = max_score <= score_eps
    positive_tie = (max_score > score_eps) & (tie_count > 1)
    abstain = nonpositive | positive_tie
    pred = labels_arr[argmax_idx].astype(object)
    pred[abstain] = ABSTAIN
    return pred, {"n_nonpositive": int(nonpositive.sum()), "n_positive_top_tie": int(positive_tie.sum()),
                  "n_abstain": int(abstain.sum()),
                  "abstain_rate": float(abstain.mean()) if len(abstain) else float("nan")}


def evaluate_prediction(y_true: np.ndarray, pred: np.ndarray, metric_labels: Sequence[str]) -> tuple[float, float]:
    """(Macro-F1 over the fixed metric labels, accuracy); abstentions count as errors."""
    f1 = f1_score(y_true, pred, labels=list(metric_labels), average="macro", zero_division=0)
    return float(f1), float(accuracy_score(y_true, pred))


def abstain_rate(pred: np.ndarray) -> float:
    return float(np.mean(pred == ABSTAIN)) if len(pred) else float("nan")
