# SPDX-License-Identifier: MIT
"""Support sensitivity of the verdicts (Remark 1): every registered view is audited again with the facet-basis
support S_i replaced by a counterfactual coordinate set. Only S_i changes. C_i, w(C_i), the registry maps,
FULL-CONFORM, the evidence, the labels, the view projections, the certificate paths and the solver stay fixed, and
QUERY-CONFORM is recomputed on each set. The sets are the released supports, their union, their intersection and
each released support with the facet color added."""
from __future__ import annotations

import copy
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import conformance as cmod
from . import registry as rmod
from . import verdicts as vd
from .names import DATASETS

REGIMES = ("released", "union", "intersection", "released_plus_color")
COUNTERFACTUAL = REGIMES[1:]
COLOR = "color"
# Admission thresholds shown in the paper (Problem 1); the step also reports the smallest admitting threshold.
EPSILONS = ("0", "0.01", "0.05")
CODE_TEXT = {int(vd.UNVERIFIED): "UNVERIFIED", int(vd.UNSAFE): "UNSAFE", int(vd.NONDETERMINED): "NONDETERMINED"}

REGIME_FIELDS = ["regime", "dataset", "support_size", "added_vs_released", "removed_vs_released", "members"]
VERDICT_FIELDS = ["regime", "view", "dataset", "basis", "path", "fibers", "query_rows", "unsafe_rows", "nondetermined_rows",
                  "unverified_observed_rows", "unverified_defect_rows", "m^U", "m^N", "m^V_obs", "m^V_defect",
                  "near_pure", "purity"]
TRANSITION_FIELDS = ["regime", "view", "dataset", "released_query_conform", "query_conform", "released_verdict",
                     "verdict", "rows", "changed_fibers"]
ADMISSION_FIELDS = ["regime", "view", "task", "path", "mU_DF", "mU_FP", "mV_DF", "mV_FP", "admitted_eps_0",
                    "admitted_eps_0.01", "admitted_eps_0.05", "smallest_admitting_eps"]
CERTIFICATE_FIELDS = ["regime"] + vd.PATH_B_FIELDS


def coordinate_sets(reg: dict[str, Any]) -> dict[str, dict[str, set[str]]]:
    """The four coordinate sets per dataset. Color must be a released facet coordinate outside both supports."""
    released = {ds: set(reg["support"]["datasets"][ds]["S_members"]) for ds in DATASETS}
    if COLOR not in reg["facet_coords"]:
        raise SystemExit("color is not a released facet coordinate")
    if any(COLOR in released[ds] for ds in DATASETS):
        raise SystemExit("color lies in a released support")
    union = set.union(*released.values())
    inter = set.intersection(*released.values())
    return {"released": released, "union": {ds: set(union) for ds in DATASETS},
            "intersection": {ds: set(inter) for ds in DATASETS},
            "released_plus_color": {ds: released[ds] | {COLOR} for ds in DATASETS}}


def color_facts(reg: dict[str, Any]) -> dict[str, Any]:
    """Why color is a boundary case: no active core term reaches it, and its type is not a released type coordinate."""
    reach = {ds: sum(1 for t in reg["support"]["datasets"][ds]["C_members"] if reg["Pi"].get(t) == COLOR)
             for ds in DATASETS}
    return {"facet": COLOR, "T": reg["T"].get(COLOR), "W": reg["W"].get(COLOR),
            "T_is_released_type_coordinate": reg["T"].get(COLOR) in reg["type_coords"],
            "active_core_terms_reaching_facet": reach}


def with_support(reg: dict[str, Any], sets: dict[str, set[str]]) -> dict[str, Any]:
    """Copy of the registry with S_i replaced (in the released facet order); everything else is shared."""
    out = copy.deepcopy(reg)
    for ds in DATASETS:
        members = [f for f in reg["facet_coords"] if f in sets[ds]]
        if set(members) != sets[ds]:
            raise SystemExit(f"coordinate set of {ds} leaves the released facet coordinates")
        out["support"]["datasets"][ds]["S_members"] = members
        out["support"]["datasets"][ds]["n_dataset_support"] = len(members)
    return out


def query_masks(reg: dict[str, Any], conf: dict[str, Any], anchors: Path, sets: dict[str, set[str]]) -> np.ndarray:
    """QUERY-CONFORM masks of all registered views with S_i replaced; C_i and w(C_i) stay as released."""
    supports = {}
    for ds in DATASETS:
        base = reg["support"]["datasets"][ds]
        supports[ds] = {"C_i": set(base["C_members"]), "S_i": set(sets[ds]), "w(C_i)": set(base["W_img_members"])}
    datasets_text = [DATASETS[int(x)] for x in conf["dataset_code"]]
    arrays = cmod.query_arrays(rmod.iter_anchor_chunks(anchors, cmod.QUERY_ANCHOR_COLUMNS),
                               [str(x) for x in conf["row_id"]], datasets_text, reg["Pi"], reg["T"], reg["W"], supports)
    return np.stack([cmod.compute_query_mask(c, conf["full"], arrays) for c in reg["cert"]])


def threshold(masses: dict[str, tuple[float, float]]) -> float | None:
    """Smallest epsilon that admits the view (Problem 1), or None when a dataset has UNSAFE mass."""
    if any(u > 0 for u, _ in masses.values()):
        return None
    return max(v for _, v in masses.values())


def admitted(masses: dict[str, tuple[float, float]], eps: str) -> bool:
    t = threshold(masses)
    return t is not None and Decimal(repr(t)) <= Decimal(eps)


def run(reg0: dict[str, Any], conf0: dict[str, Any], ev: dict[str, Any], anchors: Path,
        log: Callable[[str], None] = print) -> dict[str, Any]:
    """Audit every registered view on the four coordinate sets."""
    sets = coordinate_sets(reg0)
    regime_rows, verdict_rows, cert_rows, transition_rows, admission_rows = [], [], [], [], []
    caches, masks, counts = {}, {}, {}
    masses: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for name in REGIMES:
        reg = with_support(reg0, sets[name])
        qmask = query_masks(reg0, conf0, anchors, sets[name])
        if name == "released" and not np.array_equal(qmask, conf0["qmask"]):
            raise SystemExit("the released coordinate sets do not reproduce the QUERY-CONFORM masks")
        res = vd.run_verdicts(reg, dict(conf0, qmask=qmask), ev, log=lambda s: None)
        certs = vd.verify_path_b(reg, res["path_b_unsafe"])
        opened = vd.verify_path_b_open(reg, res["path_b_open"])
        if not all(int(r["pass"]) == 1 for r in certs + opened):
            raise SystemExit(f"{name}: a Path B certificate or open-fiber witness fails its exact check")
        caches[name], masks[name] = res["verdict_cache"], qmask
        counts[name] = {"path_b_certificates": len(certs), "path_b_open_fibers": len(opened),
                        "certificates_verified_exactly": len(certs), "open_fibers_verified_exactly": len(opened)}
        cert_rows += [{"regime": name, **r} for r in certs]
        for ds in DATASETS:
            members = sorted(sets[name][ds])
            regime_rows.append({"regime": name, "dataset": ds, "support_size": len(members),
                                "added_vs_released": sorted(sets[name][ds] - sets["released"][ds]),
                                "removed_vs_released": sorted(sets["released"][ds] - sets[name][ds]),
                                "members": members})
        for r in res["result_rows"]:
            m = res["metrics_by"][(r["view"], r["dataset"])]
            verdict_rows.append({"regime": name, **r, **{k: m[k] for k in (
                "query_rows", "unsafe_rows", "nondetermined_rows", "unverified_observed_rows", "unverified_defect_rows")}})
            masses.setdefault((name, r["view"]), {})[r["dataset"]] = (r["m^U"], r["m^V_obs"] + r["m^V_defect"])
        log(f"[support] {name}: certificates={len(certs)} open single-label fibers={len(opened)}")
    tasks = {c["view_id"]: (c["task"], c["certificate_path"]) for c in reg0["cert"]}
    for name in REGIMES:
        for view, (task, path) in tasks.items():
            m = masses[(name, view)]
            row = {"regime": name, "view": view, "task": task, "path": path,
                   "mU_DF": m["DF"][0], "mU_FP": m["FP"][0], "mV_DF": m["DF"][1], "mV_FP": m["FP"][1],
                   "smallest_admitting_eps": threshold(m)}
            for eps in EPSILONS:
                row[f"admitted_eps_{eps}"] = int(admitted(m, eps))
            admission_rows.append(row)
    view_values: dict[int, np.ndarray] = {}

    def values_of(i: int) -> np.ndarray:
        """Per-anchor view values of registered view i (evidence only, independent of the coordinate set)."""
        if i not in view_values:
            keys, dense = vd.view_inputs(reg0["cert"][i], reg0, ev)
            arr = np.asarray(keys if keys is not None else dense)
            view_values[i] = arr if arr.ndim == 2 else arr[:, None]
        return view_values[i]

    for name in COUNTERFACTUAL:
        for i, c in enumerate(reg0["cert"]):
            for d, ds in enumerate(DATASETS):
                sel = conf0["dataset_code"] == d
                rows_ds = np.flatnonzero(sel)
                key = np.stack([masks["released"][i][sel].astype(np.uint8), masks[name][i][sel].astype(np.uint8),
                                caches["released"][i][sel], caches[name][i][sel]], axis=1)
                uniq, inv, n = np.unique(key, axis=0, return_inverse=True, return_counts=True)
                inv = np.asarray(inv).ravel()
                for g, ((q0, q1, v0, v1), k) in enumerate(zip(uniq.tolist(), n.tolist())):
                    fibers = 0
                    if v0 != v1:  # fibers of the anchors whose verdict changes
                        fibers = int(len(np.unique(values_of(i)[rows_ds[inv == g]], axis=0)))
                    transition_rows.append({"regime": name, "view": c["view_id"], "dataset": ds,
                                            "released_query_conform": q0, "query_conform": q1,
                                            "released_verdict": CODE_TEXT[v0], "verdict": CODE_TEXT[v1], "rows": k,
                                            "changed_fibers": fibers})
    return {"sets": sets, "regime_rows": regime_rows, "verdict_rows": verdict_rows, "certificate_rows": cert_rows,
            "transition_rows": transition_rows, "admission_rows": admission_rows, "counts": counts,
            "color": color_facts(reg0)}
