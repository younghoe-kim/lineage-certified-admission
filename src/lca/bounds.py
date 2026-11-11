# SPDX-License-Identifier: MIT
"""Utility bounds of a view on the target (Lemma 2): accuracy ceiling 1-g3 and Macro-F1 bound beta."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Hashable, Sequence

import numpy as np
from scipy import sparse

from . import evaluation as ev
from .names import TASKS


def row_key(indices: np.ndarray, data: np.ndarray) -> bytes:
    """Exact identity of one sparse row: entry count, then int32 indices and float32 values (little endian)."""
    n = int(len(indices))
    return (n.to_bytes(4, "little", signed=False) + np.asarray(indices, dtype="<i4").tobytes(order="C")
            + np.asarray(data, dtype="<f4").tobytes(order="C"))


def sparse_signatures(Z: sparse.spmatrix) -> list[bytes]:
    """Fiber signature of every row of Z (rows with equal signature lie in the same fiber)."""
    Z = Z.tocsr()
    Z.sort_indices()
    return [row_key(Z.indices[Z.indptr[i]:Z.indptr[i + 1]], Z.data[Z.indptr[i]:Z.indptr[i + 1]])
            for i in range(Z.shape[0])]


def fiber_overlap(source_signatures: Sequence[Hashable], target_signatures: Sequence[Hashable]) -> float:
    """Fraction of target rows whose fiber is observed among the source fitting rows."""
    seen = set(source_signatures)
    return float(np.mean(np.fromiter((s in seen for s in target_signatures), bool, count=len(target_signatures))))


def utility_bounds(signatures: Sequence[Hashable], labels_true: Sequence[str], metric_labels: Sequence[str]
                   ) -> tuple[float, float, int]:
    """(beta, 1-g3, number of fibers) for any predictor that is a function of the view.

    1-g3 = (1/N) sum_F max_y n_{F,y}.
    beta = (1/|L|) sum_c max over fiber sets F_c of 2 sum t^c_F / (sum n_F + G_c); the maximum is attained
    by a prefix of the fibers sorted by decreasing t^c_F / n_F, so every prefix is evaluated."""
    by: dict[Hashable, Counter] = defaultdict(Counter)
    for sig, y in zip(signatures, labels_true):
        by[sig][str(y)] += 1
    N = len(signatures)
    if N == 0:
        raise SystemExit("utility bounds: empty target domain")
    counters = list(by.values())
    nF = np.fromiter((sum(c.values()) for c in counters), dtype=np.int64, count=len(counters))
    fiber_max = np.fromiter((max(c.values()) for c in counters), dtype=np.int64, count=len(counters))
    acc_ceiling = float(fiber_max.sum() / N)
    y_arr = np.asarray(labels_true, dtype=object)
    class_bounds: list[float] = []
    for cls in metric_labels:
        cls = str(cls)
        Gc = int(np.sum(y_arr == cls))
        tc = np.fromiter((int(c.get(cls, 0)) for c in counters), dtype=np.int64, count=len(counters))
        ratios = np.divide(tc, nF, out=np.zeros(len(nF), dtype=np.float64), where=nF > 0)
        order = np.argsort(-ratios, kind="stable")
        pred_mass = np.cumsum(nF[order], dtype=np.int64)
        true_pos = np.cumsum(tc[order], dtype=np.int64)
        den = pred_mass + Gc
        vals = np.divide(2.0 * true_pos, den, out=np.zeros(len(den), dtype=np.float64), where=den > 0)
        class_bounds.append(float(vals.max(initial=0.0)))
    beta = float(np.mean(class_bounds)) if class_bounds else 0.0
    return beta, acc_ceiling, len(by)


def view_bounds(cache: dict[str, Any], fit_rows: np.ndarray, target_idx: np.ndarray, shared_cols: np.ndarray,
                task: str, metric_labels: Sequence[str]) -> dict[str, Any]:
    """Bounds of the shared projection on the evaluated target rows, plus the source-fiber overlap."""
    ss = sparse_signatures(cache["X"][fit_rows][:, shared_cols])
    ts = sparse_signatures(cache["X"][target_idx][:, shared_cols])
    beta, acc, nfib = utility_bounds(ts, cache["gold_" + task][target_idx].tolist(), metric_labels)
    return {"beta": beta, "accuracy_ceiling": acc, "fiber_overlap": fiber_overlap(ss, ts), "fibers": nfib}


# ============================================================================
# Bound tables
# ============================================================================

BOUND_FIELDS = ["view_id", "task", "direction", "training_protocol", "evaluation_scope", "fitting_mask_rows",
                "source_rows_fit", "shared_support_hash", "shared_feature_count", "target_eval_rows", "metric_labels",
                "macro_f1_upper_bound_C2", "accuracy_ceiling", "fiber_overlap_o", "shared_projection_fibers"]
CHECK_FIELDS = ["view_id", "direction", "probe", "training_protocol", "evaluation_scope", "macro_f1",
                "C2_upper_bound", "margin_bound_minus_f1", "violation"]


def bound_row(cache: dict[str, Any], views: dict[str, dict[str, Any]], view_id: str, source: str, target: str,
              protocol: str, scope: str) -> dict[str, Any]:
    task = views[view_id]["task"]
    cols = ev.view_columns(cache, views, view_id)
    mask_rows, fit, _ = ev.fitting_rows(cache, source, task, protocol)
    shared, shared_hash = ev.shared_projection(cache, cols, fit)
    ti, _, metric = ev.target_rows(cache, source, target, task, scope)
    b = view_bounds(cache, fit, ti, shared, task, metric)
    return {"view_id": view_id, "task": task, "direction": f"{source}->{target}", "training_protocol": protocol,
            "evaluation_scope": scope, "fitting_mask_rows": len(mask_rows), "source_rows_fit": len(fit),
            "shared_support_hash": shared_hash, "shared_feature_count": len(shared), "target_eval_rows": len(ti),
            "metric_labels": len(metric), "macro_f1_upper_bound_C2": b["beta"],
            "accuracy_ceiling": b["accuracy_ceiling"], "fiber_overlap_o": b["fiber_overlap"],
            "shared_projection_fibers": b["fibers"]}


def bounds_table(cache: dict[str, Any], views: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Bounds of the shared projection for the main views (both protocols) and the 8-cell factorial (refit)."""
    rows = []
    for view_id in ev.MAIN_VIEW_IDS:
        for source, target in ev.DIRECTIONS:
            for protocol, scope in ev.PROTOCOL_SCOPES:
                rows.append(bound_row(cache, views, view_id, source, target, protocol, scope))
    for task in TASKS:
        for F in (0, 1):
            for M in (0, 1):
                for T in (0, 1):
                    for source, target in ev.DIRECTIONS:
                        rows.append(bound_row(cache, views, f"factorial_F{F}M{M}T{T}__{task}", source, target,
                                              ev.REFIT, ev.CONFORM_TARGET))
    return rows


def probe_bound_check_table(probe_rows: list[dict[str, Any]], factorial_rows: list[dict[str, Any]],
                            bound_rows: list[dict[str, Any]], slack: float) -> list[dict[str, Any]]:
    """Every scored probe against beta of the same view, direction, protocol and scope; a violation stops."""
    index = {(r["view_id"], r["direction"], r["training_protocol"], r["evaluation_scope"]): r for r in bound_rows}
    checks = []
    scored = [(r["view_id"], r["direction"], r["probe"], r["training_protocol"], r["evaluation_scope"], r["macro_f1"])
              for r in probe_rows if r["status"] == "OK" and r["macro_f1"] is not None]
    scored += [(r["view_id"], r["direction"], "cosine", ev.REFIT, ev.CONFORM_TARGET, r["macro_f1"])
               for r in factorial_rows]
    for view_id, direction, probe, protocol, scope, f1 in scored:
        b = index.get((view_id, direction, protocol, scope))
        if b is None:
            continue
        beta = float(b["macro_f1_upper_bound_C2"])
        violation = f1 > beta + slack
        if violation:
            raise SystemExit(f"probe exceeds the utility bound: {view_id} {direction} {probe} {protocol}")
        checks.append({"view_id": view_id, "direction": direction, "probe": probe, "training_protocol": protocol,
                       "evaluation_scope": scope, "macro_f1": f1, "C2_upper_bound": beta,
                       "margin_bound_minus_f1": beta - f1, "violation": int(violation)})
    return checks
