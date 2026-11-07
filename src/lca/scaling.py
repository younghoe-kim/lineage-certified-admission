# SPDX-License-Identifier: MIT
"""Scaling: Path B fiber grouping and exact integer-program time on nested hash samples of the rows."""
from __future__ import annotations

import csv
import gc
import gzip
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.optimize

from .names import DATASETS, FAMILY_ORDER

EXPECTED_ROWS = {"DF": 289202, "FP": 342182}
EXPECTED_TOTAL = 631384
EXPECTED_FACET_DIM = 26
EXPECTED_PATH_B_VIEWS = ("family3__facet", "family3_plus_type__facet", "type3__family", "type_attr_part__family")
CHUNK_ROWS = 20000

FRACTIONS = tuple(i / 10.0 for i in range(1, 11))
REPEATS = 3
SOLVER = "scipy.optimize.milp / HiGHS direct integer MILP"
MILP_OPTIONS = {"disp": False, "presolve": True, "mip_rel_gap": 0.0}
MILP_OPTIONS_NO_PRESOLVE = {"disp": False, "presolve": False, "mip_rel_gap": 0.0}

RAW_FIELDS = [
    "fraction", "repeat", "view_id", "task", "dataset",
    "dataset_rows_total", "sampled_dataset_rows", "sampled_full_conform_rows",
    "sampled_query_conform_rows", "facet_dimension", "label_universe_size",
    "theoretical_competing_labels_per_candidate", "observed_unique_fibers",
    "multi_label_fibers", "single_label_fibers", "max_fiber_rows", "mean_fiber_rows",
    "solver_fibers", "solver_rival_checks", "unsafe_fibers", "nonunsafe_fibers",
    "infeasible_fibers", "grouping_seconds", "milp_seconds", "total_seconds",
]
STRUCT_FIELDS = [f for f in RAW_FIELDS if f not in ("repeat", "grouping_seconds", "milp_seconds", "total_seconds")]
SUMMARY_FIELDS = STRUCT_FIELDS + [
    "repeats",
    "grouping_seconds_median", "grouping_seconds_min", "grouping_seconds_max",
    "milp_seconds_median", "milp_seconds_min", "milp_seconds_max",
    "total_seconds_median", "total_seconds_min", "total_seconds_max",
]
AGG_FIELDS = [
    "fraction", "sampled_rows_total", "sampled_full_conform_rows_sum_over_datasets",
    "sampled_query_rows_sum_over_pathb_cells", "pathb_cells",
    "sum_view_dataset_fibers", "sum_solver_fibers", "sum_solver_rival_checks",
    "sum_grouping_seconds_median", "sum_milp_seconds_median", "sum_total_seconds_median",
]


def _count(v: Any, key: str) -> int:
    x = float(v)
    if not math.isfinite(x) or x < 0 or not x.is_integer() or x > np.iinfo(np.uint16).max:
        raise SystemExit(f"invalid count at {key}: {v!r}")
    return int(x)


def load_model(anchors_path: Path, outputs: Path) -> dict[str, Any]:
    """Registry maps, dataset supports, query masks, labels and dense family/type counts."""
    pops = json.loads((outputs / "registry" / "populations.json").read_text(encoding="utf-8"))
    members = pops["block_coordinate_members"]
    facet_coords = list(members["facet::"])
    family_coords = [f for f in FAMILY_ORDER if f in set(members["family::"])]
    if set(family_coords) != set(members["family::"]):
        raise SystemExit("registry family coordinates outside the declared family order")
    type_coords = list(members["type::"])
    if len(facet_coords) != EXPECTED_FACET_DIM:
        raise SystemExit(f"facet dimension {len(facet_coords)} != {EXPECTED_FACET_DIM}")
    fidx = {x: i for i, x in enumerate(facet_coords)}
    famidx = {x: i for i, x in enumerate(family_coords)}
    tidx = {x: i for i, x in enumerate(type_coords)}

    fmap = pd.read_csv(outputs / "registry" / "facet_map.csv", dtype=str, keep_default_na=False)
    W = dict(zip(fmap["facet"], fmap["family"]))
    T = dict(zip(fmap["facet"], fmap["term_type"]))
    Wmat = np.zeros((len(facet_coords), len(family_coords)), dtype=np.int8)
    Tmat = np.zeros((len(facet_coords), len(type_coords)), dtype=np.int8)
    for j, fac in enumerate(facet_coords):
        Wmat[j, famidx[W[fac]]] = 1
        if T[fac] in tidx:
            Tmat[j, tidx[T[fac]]] = 1
        elif not (fac == "color" and T[fac] == "color"):
            # color has its own type, which is not a materialized type coordinate; its T row stays zero.
            raise SystemExit(f"facet {fac} maps to a type outside the type block: {T[fac]}")

    cert = pd.read_csv(outputs / "registry" / "certificate_spec.csv", dtype=str, keep_default_na=False)
    pb = cert[cert["certificate_path"] == "B"]
    views = [(str(v), str(t)) for v, t in zip(pb["view_id"], pb["task"])]
    if tuple(sorted(v for v, _ in views)) != tuple(sorted(EXPECTED_PATH_B_VIEWS)):
        raise SystemExit(f"Path B view set differs: {[v for v, _ in views]}")
    proj = pd.read_csv(outputs / "registry" / "view_projection.csv", dtype=str, keep_default_na=False)
    projection = {str(v): json.loads(p) for v, p in zip(proj["view_id"], proj["projection_json"])}

    sup = json.loads((outputs / "registry" / "dataset_support.json").read_text(encoding="utf-8"))
    support_ub = {}
    for ds in DATASETS:
        s = set(sup["datasets"][ds]["S_members"])
        support_ub[ds] = np.array([np.inf if fac in s else 0.0 for fac in facet_coords], dtype=np.float64)

    fc = pd.read_csv(outputs / "conformance" / "full_conform.csv.gz", usecols=["dataset", "row_id", "full_registry_conform"],
                     dtype={"dataset": str, "row_id": str, "full_registry_conform": int}, keep_default_na=False)
    if len(fc) != EXPECTED_TOTAL or fc["row_id"].nunique() != EXPECTED_TOTAL:
        raise SystemExit("conformance row identity differs from the anchor table")
    row_id = fc["row_id"].to_numpy(dtype=str)
    ds_arr = fc["dataset"].to_numpy(dtype=str)
    dataset_code = np.full(EXPECTED_TOTAL, 255, dtype=np.uint8)
    for k, ds in enumerate(DATASETS):
        dataset_code[ds_arr == ds] = k
    if np.any(dataset_code == 255):
        raise SystemExit("unknown dataset in conformance output")
    for k, ds in enumerate(DATASETS):
        if int(np.count_nonzero(dataset_code == k)) != EXPECTED_ROWS[ds]:
            raise SystemExit(f"{ds} row count differs from the released benchmark")
    full = fc["full_registry_conform"].to_numpy(dtype=np.uint8).astype(bool)

    qz = np.load(outputs / "conformance" / "query_masks.npz", allow_pickle=False)
    qviews = [str(x) for x in qz["view_ids"]]
    if not np.array_equal(qz["full_registry_conform"].astype(bool), full):
        raise SystemExit("query masks and conformance output disagree on full conformance")
    qmask = {v: qz["query_conform"][qviews.index(v)].astype(bool) for v, _ in views}

    family = np.zeros((EXPECTED_TOTAL, len(family_coords)), dtype=np.uint16)
    typ = np.zeros((EXPECTED_TOTAL, len(type_coords)), dtype=np.uint16)
    gold_facet = np.empty(EXPECTED_TOTAL, dtype=object)
    gold_family = np.empty(EXPECTED_TOTAL, dtype=object)
    pos = 0
    reader = pd.read_csv(anchors_path, usecols=["dataset", "anchor_id", "full_feats_json", "gold_facet", "gold_family"],
                         dtype=str, keep_default_na=False, chunksize=CHUNK_ROWS)
    for chunk in reader:
        for ds, aid, raw, gf, gy in zip(chunk["dataset"], chunk["anchor_id"], chunk["full_feats_json"],
                                        chunk["gold_facet"], chunk["gold_family"]):
            if pos >= EXPECTED_TOTAL or ds != ds_arr[pos] or aid != row_id[pos]:
                raise SystemExit(f"anchor and conformance row order differ at row {pos}")
            for key, v in json.loads(raw).items():
                if key.startswith("family::"):
                    c = _count(v, key)
                    if c:
                        coord = key[len("family::"):]
                        if coord not in famidx:
                            raise SystemExit(f"unknown family coordinate {coord}")
                        family[pos, famidx[coord]] = c
                elif key.startswith("type::"):
                    c = _count(v, key)
                    if c:
                        coord = key[len("type::"):]
                        if coord not in tidx:
                            raise SystemExit(f"unknown type coordinate {coord}")
                        typ[pos, tidx[coord]] = c
            gold_facet[pos] = gf
            gold_family[pos] = gy
            pos += 1
    if pos != EXPECTED_TOTAL:
        raise SystemExit(f"anchor rows {pos} != {EXPECTED_TOTAL}")

    fcode = pd.Categorical(gold_facet, categories=facet_coords).codes.astype(np.int16)
    ycode = pd.Categorical(gold_family, categories=family_coords).codes.astype(np.int16)
    if np.any(fcode < 0) or np.any(ycode < 0):
        raise SystemExit("gold label outside the registry coordinate universe")

    return {
        "facet_coords": facet_coords, "family_coords": family_coords, "type_coords": type_coords,
        "famidx": famidx, "tidx": tidx, "Wmat": Wmat, "Tmat": Tmat,
        "views": views, "projection": projection, "support_ub": support_ub,
        "row_id": row_id, "dataset_code": dataset_code, "full": full, "qmask": qmask,
        "family": family, "type": typ, "labels": {"facet": fcode, "family": ycode},
        "label_names": {"facet": facet_coords, "family": family_coords},
    }


def read_sample_rank(path: Path, ids: Sequence[str]) -> np.ndarray:
    """Per-anchor sample rank of the input view (an unsigned 64-bit hash rank), aligned to the given anchor ids."""
    rank = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rank[r["anchor_id"]] = int(r["sample_rank"])
    if len(rank) != len(ids) or any(x not in rank for x in ids):
        raise SystemExit("sample_order.csv.gz does not cover the anchor table")
    return np.fromiter((rank[x] for x in ids), dtype=np.uint64, count=len(ids))


def nested_orders(ranks: np.ndarray, dataset_code: np.ndarray) -> dict[str, np.ndarray]:
    """Row order per dataset by hash rank (row index breaks ties); prefixes give nested samples."""
    orders = {}
    for k, ds in enumerate(DATASETS):
        idx = np.flatnonzero(dataset_code == k)
        orders[ds] = idx[np.lexsort((idx.astype(np.int64), ranks[idx]))]
    return orders


def fraction_count(n: int, fraction: float) -> int:
    return n if fraction == 1.0 else int(math.floor(n * fraction))


def sample_mask(n_total: int, order: np.ndarray, fraction: float) -> np.ndarray:
    m = np.zeros(n_total, dtype=bool)
    m[order[: fraction_count(len(order), fraction)]] = True
    return m


def dense_projection(view_id: str, model: dict[str, Any]) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Exposed family/type counts of a Path B view and the (block, coordinate) of every column."""
    cols, coords = [], []
    for block, selector in model["projection"][view_id].items():
        if block == "family::":
            cols.append(model["family"][:, [model["famidx"][x] for x in selector]])
        elif block == "type::":
            cols.append(model["type"][:, [model["tidx"][x] for x in selector]])
        else:
            raise SystemExit(f"Path B projection of {view_id} uses an unsupported block {block}")
        coords += [(block, x) for x in selector]
    return np.concatenate(cols, axis=1).astype(np.int64), coords


def group_stats(keys: np.ndarray, mask: np.ndarray, labels: np.ndarray, nlabels: int) -> dict[str, Any]:
    """Exact lexicographic grouping of the masked rows into fibers with per-label counts."""
    idx = np.flatnonzero(mask)
    sub = np.ascontiguousarray(keys[idx])
    if len(sub) == 0:
        inv = np.empty(0, dtype=np.int64)
        u = np.empty((0, sub.shape[1]), dtype=sub.dtype)
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
        u = ss[boundary].copy()
        counts = np.bincount(inv, minlength=len(u)).astype(np.int64)
    mat = np.bincount(inv * nlabels + labels[idx].astype(np.int64), minlength=len(u) * nlabels).reshape(len(u), nlabels)
    first_pos = np.full(len(u), len(mask), dtype=np.int64)
    if len(u):
        np.minimum.at(first_pos, inv, idx)
    return {"idx": idx, "unique": u, "inv": inv, "counts": counts, "label_counts": mat,
            "diversity": np.count_nonzero(mat, axis=1), "first_pos": first_pos}


def solve_integer_milp(c: np.ndarray, Aeq: np.ndarray, beq: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                       require_nonzero: bool) -> tuple[str, int | None]:
    """Minimize c@x over integer x >= 0 with Aeq x = beq (and sum x >= 1); verify the optimum exactly."""
    nvar = len(c)
    constraints = [scipy.optimize.LinearConstraint(Aeq, beq, beq)]
    if require_nonzero:
        constraints.append(scipy.optimize.LinearConstraint(np.ones((1, nvar)), np.array([1.0]), np.array([np.inf])))
    kw = dict(c=c, integrality=np.ones(nvar, dtype=np.uint8), bounds=scipy.optimize.Bounds(lb, ub),
              constraints=constraints)
    res = scipy.optimize.milp(options=MILP_OPTIONS, **kw)
    if int(res.status) == 4:
        # Status 4 is "infeasible or unbounded"; the same program without presolve tells which.
        res = scipy.optimize.milp(options=MILP_OPTIONS_NO_PRESOLVE, **kw)
    if int(res.status) == 2:
        return "infeasible", None
    if int(res.status) == 3:
        return "unbounded", None
    if int(res.status) != 0 or res.x is None or res.fun is None:
        raise SystemExit(f"integer program not solved: status={res.status} {res.message}")
    x = np.asarray(res.x, dtype=np.float64)
    xr = np.rint(x).astype(np.int64)
    if np.max(np.abs(x - xr)) > 1e-7 or np.any(xr < 0) or np.any(xr > ub + 1e-9):
        raise SystemExit("integer program returned a non-integral or out-of-bounds solution")
    if not np.array_equal(np.rint(Aeq).astype(np.int64) @ xr, np.rint(beq).astype(np.int64)):
        raise SystemExit("integer program solution violates Ax=b")
    if require_nonzero and int(xr.sum()) < 1:
        raise SystemExit("integer program solution violates x != 0")
    obj = int(np.rint(c).astype(np.int64) @ xr)
    if abs(float(res.fun) - obj) > 1e-7:
        raise SystemExit("integer program objective differs from the exact integer objective")
    return "optimal", obj


def solve_fiber(view_id: str, task: str, value: np.ndarray, label: int, ds: str, coords: list[tuple[str, str]],
                model: dict[str, Any]) -> dict[str, Any]:
    """UNSAFE iff the observed label is the strict unique maximum over every integer facet vector of the fiber."""
    W = model["Wmat"].astype(np.float64)
    T = model["Tmat"].astype(np.float64)
    Aeq = np.vstack([W[:, model["famidx"][x]] if b == "family::" else T[:, model["tidx"][x]] for b, x in coords])
    beq = np.asarray([int(v) for v in value], dtype=np.float64)
    nvar = len(model["facet_coords"])
    lb = np.zeros(nvar, dtype=np.float64)
    ub = model["support_ub"][ds]
    coeff = np.eye(nvar, dtype=np.int8) if task == "facet" else model["Wmat"].astype(np.int8).T
    rivals = 0
    for rival in range(len(model["label_names"][task])):
        if rival == label:
            continue
        c = (coeff[label].astype(np.int16) - coeff[rival].astype(np.int16)).astype(np.float64)
        status, min_cmr = solve_integer_milp(c, Aeq, beq, lb, ub, require_nonzero=True)
        if status == "infeasible":
            return {"infeasible": True, "unsafe": False, "rivals": rivals}
        rivals += 1
        if status == "unbounded" or int(min_cmr) < 1:
            return {"infeasible": False, "unsafe": False, "rivals": rivals}
    return {"infeasible": False, "unsafe": True, "rivals": rivals}


def solve_view_fibers(view_id: str, task: str, stats: dict[str, Any], values: np.ndarray, ds: str,
                      coords: list[tuple[str, str]], model: dict[str, Any]) -> dict[str, int]:
    out = {"solver_fibers": 0, "solver_rival_checks": 0, "unsafe_fibers": 0, "nonunsafe_fibers": 0,
           "infeasible_fibers": 0}
    for g in np.flatnonzero(stats["diversity"] == 1):
        label_pos = np.flatnonzero(stats["label_counts"][g])
        val = values[int(stats["first_pos"][g])].astype(np.int64, copy=False)
        solved = solve_fiber(view_id, task, val, int(label_pos[0]), ds, coords, model)
        out["solver_fibers"] += 1
        out["solver_rival_checks"] += solved["rivals"]
        if solved["infeasible"]:
            out["infeasible_fibers"] += 1
        elif solved["unsafe"]:
            out["unsafe_fibers"] += 1
        else:
            out["nonunsafe_fibers"] += 1
    return out


def run_scaling(model: dict[str, Any], orders: dict[str, np.ndarray], fractions: Sequence[float] = FRACTIONS,
                repeats: int = REPEATS, log=print) -> tuple[list[dict], list[dict], list[dict]]:
    """Grouping and integer-program time for every Path B view, dataset and sample fraction."""
    n = EXPECTED_TOTAL
    raw_rows, summary_rows = [], []
    for view_id, task in model["views"]:
        values, coords = dense_projection(view_id, model)
        values = np.ascontiguousarray(values)
        labels = model["labels"][task]
        nlabels = len(model["label_names"][task])
        for k, ds in enumerate(DATASETS):
            cm = model["dataset_code"] == k
            qbase = model["qmask"][view_id] & cm
            for fraction in fractions:
                sample = sample_mask(n, orders[ds], fraction)
                q = qbase & sample
                base = {
                    "fraction": fraction, "view_id": view_id, "task": task, "dataset": ds,
                    "dataset_rows_total": int(cm.sum()),
                    "sampled_dataset_rows": int(np.count_nonzero(cm & sample)),
                    "sampled_full_conform_rows": int(np.count_nonzero(cm & sample & model["full"])),
                    "sampled_query_conform_rows": int(np.count_nonzero(q)),
                    "facet_dimension": len(model["facet_coords"]),
                    "label_universe_size": nlabels,
                    "theoretical_competing_labels_per_candidate": max(0, nlabels - 1),
                }
                reps, ref = [], None
                for repeat in range(1, repeats + 1):
                    t0 = time.perf_counter()
                    stats = group_stats(values, q, labels, nlabels)
                    t1 = time.perf_counter()
                    t2 = time.perf_counter()
                    solver = solve_view_fibers(view_id, task, stats, values, ds, coords, model)
                    t3 = time.perf_counter()
                    counts = stats["counts"]
                    struct = {
                        "observed_unique_fibers": int(len(stats["unique"])),
                        "multi_label_fibers": int(np.count_nonzero(stats["diversity"] >= 2)),
                        "single_label_fibers": int(np.count_nonzero(stats["diversity"] == 1)),
                        "max_fiber_rows": int(counts.max()) if len(counts) else 0,
                        "mean_fiber_rows": float(counts.mean()) if len(counts) else 0.0,
                        **solver,
                    }
                    if int(counts.sum()) != base["sampled_query_conform_rows"]:
                        raise SystemExit("grouped rows differ from the sampled query rows")
                    if ref is None:
                        ref = struct
                    elif struct != ref:
                        raise SystemExit(f"grouping or solver outcome changed across repeats: {view_id}/{ds}/{fraction}")
                    row = {**base, "repeat": repeat, **struct, "grouping_seconds": t1 - t0,
                           "milp_seconds": t3 - t2, "total_seconds": (t1 - t0) + (t3 - t2)}
                    raw_rows.append(row)
                    reps.append(row)
                    del stats
                    gc.collect()
                g = [r["grouping_seconds"] for r in reps]
                m = [r["milp_seconds"] for r in reps]
                t = [r["total_seconds"] for r in reps]
                summary_rows.append({
                    **{f: reps[0][f] for f in STRUCT_FIELDS}, "repeats": repeats,
                    "grouping_seconds_median": statistics.median(g), "grouping_seconds_min": min(g),
                    "grouping_seconds_max": max(g),
                    "milp_seconds_median": statistics.median(m), "milp_seconds_min": min(m), "milp_seconds_max": max(m),
                    "total_seconds_median": statistics.median(t), "total_seconds_min": min(t),
                    "total_seconds_max": max(t),
                })
                log(f"[scaling] {view_id:26s} {ds} f={fraction:.1f} q={base['sampled_query_conform_rows']:,} "
                    f"fibers={ref['observed_unique_fibers']:,} solver={ref['solver_fibers']:,} "
                    f"median={statistics.median(t):.4f}s")
                del sample, q
                gc.collect()

    agg_rows = []
    for fraction in fractions:
        rs = [r for r in summary_rows if r["fraction"] == fraction]
        if len(rs) != len(model["views"]) * len(DATASETS):
            raise SystemExit(f"aggregate cell count differs at fraction {fraction}")
        by_ds = {ds: next(r for r in rs if r["dataset"] == ds) for ds in DATASETS}
        agg_rows.append({
            "fraction": fraction,
            "sampled_rows_total": sum(by_ds[ds]["sampled_dataset_rows"] for ds in DATASETS),
            "sampled_full_conform_rows_sum_over_datasets": sum(by_ds[ds]["sampled_full_conform_rows"] for ds in DATASETS),
            "sampled_query_rows_sum_over_pathb_cells": sum(r["sampled_query_conform_rows"] for r in rs),
            "pathb_cells": len(rs),
            "sum_view_dataset_fibers": sum(r["observed_unique_fibers"] for r in rs),
            "sum_solver_fibers": sum(r["solver_fibers"] for r in rs),
            "sum_solver_rival_checks": sum(r["solver_rival_checks"] for r in rs),
            "sum_grouping_seconds_median": sum(r["grouping_seconds_median"] for r in rs),
            "sum_milp_seconds_median": sum(r["milp_seconds_median"] for r in rs),
            "sum_total_seconds_median": sum(r["total_seconds_median"] for r in rs),
        })

    for ds in DATASETS:
        hundred = [r for r in summary_rows if r["dataset"] == ds and r["fraction"] == 1.0]
        if fractions[-1] == 1.0 and any(r["sampled_dataset_rows"] != EXPECTED_ROWS[ds] for r in hundred):
            raise SystemExit(f"the 100% sample does not contain every {ds} row")
    if any(r["infeasible_fibers"] != 0 for r in summary_rows):
        raise SystemExit("an observed Path B fiber is infeasible")
    return raw_rows, summary_rows, agg_rows


def protocol(inputs_sha256: dict[str, str], thread_limit: int) -> dict[str, Any]:
    return {
        "step": "scaling",
        "schema_version": "1",
        "inputs_sha256": inputs_sha256,
        "design": {
            "fractions": list(FRACTIONS),
            "sampling": "nested samples: rows of each dataset in increasing sample_rank of the input view (row index breaks ties)",
            "count_rule": "floor(dataset_rows*fraction), except 100%=all rows",
            "timing_repeats": REPEATS,
            "headline_timing": f"median of {REPEATS} repeats",
            "path_b_views": list(EXPECTED_PATH_B_VIEWS),
            "solver": SOLVER,
            "solver_options": MILP_OPTIONS,
            "solver_options_on_status_4": MILP_OPTIONS_NO_PRESOLVE,
            "thread_limit": thread_limit,
            "projection_building_excluded_from_timed_region": True,
            "anchor_cache_building_excluded_from_timed_region": True,
        },
        "requirements": {
            "fraction_range": "10%-100%",
            "metrics": ["observed_unique_fibers", "competing_labels", f"facet_dimension={EXPECTED_FACET_DIM}",
                        "FULL_CONFORM_rows", "QUERY_CONFORM_rows", "grouping_time", "MILP_time"],
        },
    }
