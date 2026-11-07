# SPDX-License-Identifier: MIT
"""Declared event order: which UNVERIFIED mass could an order that the release declares turn into UNSAFE mass, and
would any admission decision change (Sec. III after Proposition 1; Sec. VI-A, Open fibers need a third verdict).

A declared event order that the released labels satisfy only removes states (z becomes fixed), so it can certify a
single-label fiber only when no state of the fiber lets another label strictly beat the observed one: a tie alone keeps
such a fiber open.
  Path A: every state of a fiber has the same carrier (Corollary 1), so every single-label tie fiber is tie-only.
  Path B: for each open single-label fiber the rival programs are solved for every rival; a state at which some rival
          strictly beats the observed label, re-checked in integer arithmetic, keeps the fiber open under every order;
          a fiber without one is tie-only.
  Path C: no certificate path, so no fiber can become UNSAFE.
UNSAFE and NONDETERMINED verdicts and query defects do not depend on z. Admission (Problem 1): admitted at eps iff
m^U = 0 and m^V <= eps in each dataset. A decision is fixed under every declared order when the view has UNSAFE mass
(rejected at every eps) or no tie-only rows (no verdict can change)."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable

import numpy as np

from . import certificates as cert
from . import verdicts as vd
from .support import EPSILONS, admitted, threshold

DATASETS = ("DF", "FP")
VIEW_FIELDS = ["view", "dataset", "task", "path", "query_rows", "unsafe_rows", "nondetermined_rows",
               "unverified_observed_rows", "unverified_defect_rows", "open_path_b_fibers", "strict_fibers",
               "strict_rows", "integer_checked_fibers", "tie_only_fibers", "tie_only_rows", "tie_only_mass"]
FIBER_FIELDS = ["view_id", "dataset", "group_index", "fiber_value", "observed_label", "class", "rival", "margin", "state",
                "rival_optima", "query_conforming_fiber_size"]
ADMISSION_FIELDS = (["view", "task", "path", "mU_DF", "mU_FP", "mV_DF", "mV_FP", "tie_only_DF", "tie_only_FP"]
                    + [f"admitted_eps_{e}" for e in EPSILONS]
                    + ["smallest_admitting_eps", "decision_fixed_under_every_order"])


def strict_state(a_eq: np.ndarray, b_eq: np.ndarray, ub: np.ndarray, coeff: np.ndarray, observed: int,
                 names: list[str]) -> dict[str, Any] | None:
    """First rival (in label order) with a fiber state at which it strictly beats the observed label, with the state
    re-checked in integer arithmetic; None when every rival can at most tie."""
    lb = np.zeros(a_eq.shape[1], dtype=np.float64)
    a_int = np.rint(a_eq).astype(np.int64)
    b_int = np.rint(b_eq).astype(np.int64)
    support = ~(ub == 0)
    co = np.asarray(coeff).astype(np.int64)
    for rival in range(len(names)):
        if rival == observed:
            continue
        c = co[observed].astype(np.int16) - co[rival].astype(np.int16)
        status, min_cmr, state = cert.solve_integer_program_state(c, a_eq, b_eq, lb, ub, require_nonzero=True)
        if status == "infeasible":
            raise SystemExit("an observed open fiber is infeasible")
        if status == "unbounded":
            free = [int(j) for j in cert.free_coordinates(a_eq, ub) if co[rival, j] == 1 and co[observed, j] == 0]
            if not free:
                raise SystemExit("unbounded rival program without a free supported coordinate mapped to the rival")
            _s, _o, base = cert.solve_integer_program_state(np.zeros(a_eq.shape[1]), a_eq, b_eq, lb, ub,
                                                            require_nonzero=True)
            if base is None:
                raise SystemExit("an unbounded rival program has no feasible state")
            x = np.rint(base).astype(np.int64)
            x[free[0]] += max(0, int(co[observed] @ x) - int(co[rival] @ x)) + 1
        elif -int(min_cmr) >= 1:
            x = np.rint(state).astype(np.int64)
        else:
            continue
        margin = int((co[rival] - co[observed]) @ x)
        ok = (np.array_equal(a_int @ x, b_int) and bool(np.all(x >= 0)) and bool(np.all(x[~support] == 0))
              and bool(x.any()) and margin >= 1)
        if not ok:
            raise SystemExit("a strictly winning rival state fails the integer check")
        return {"rival": names[rival], "margin": margin, "status": status, "state": x}
    return None


def rival_optima(a_eq: np.ndarray, b_eq: np.ndarray, ub: np.ndarray, coeff: np.ndarray, observed: int,
                 names: list[str]) -> dict[str, int]:
    """Exact optimum of (rival - observed) over the fiber for every rival (used for tie-only fibers, all <= 0)."""
    lb = np.zeros(a_eq.shape[1], dtype=np.float64)
    co = np.asarray(coeff).astype(np.int16)
    out = {}
    for rival in range(len(names)):
        if rival != observed:
            status, min_cmr, _ = cert.solve_integer_program_state(co[observed] - co[rival], a_eq, b_eq, lb, ub,
                                                                  require_nonzero=True)
            if status != "optimal":
                raise SystemExit("a tie-only fiber has a rival program that is not optimal")
            out[names[rival]] = -int(min_cmr)
    return out


def run(reg: dict[str, Any], conf: dict[str, Any], ev: dict[str, Any],
        log: Callable[[str], None] = print) -> dict[str, Any]:
    res = vd.run_verdicts(reg, conf, ev, log=lambda s: None)
    metrics = res["metrics_by"]
    spec = {c["view_id"]: c for c in reg["cert"]}
    per: dict[tuple[str, str], dict[str, int]] = {}
    for (vid, ds), m in metrics.items():
        path = spec[vid]["certificate_path"]
        per[(vid, ds)] = {"open": 0, "strict": 0, "strict_rows": 0, "checked": 0, "tie": 0,
                          "tie_rows": m["unverified_observed_rows"] if path == "A" else 0}
    fiber_rows: list[dict[str, Any]] = []
    withheld = 0
    for rec in res["path_b_open"]:
        vid, ds, task = rec["view_id"], rec["dataset"], rec["task"]
        labels, names = vd.labels_for(task, reg, ev)
        observed = names.index(rec["observed_label"])
        val = np.asarray(rec["fiber_value"], dtype=np.int64)
        a_eq, b_eq, ub = vd.path_b_system(vid, val, ds, reg, ev)
        coeff = np.eye(len(reg["facet_coords"]), dtype=np.int8) if task == "facet" else ev["Wmat"].astype(np.int8).T
        p = per[(vid, ds)]
        p["open"] += 1
        found = strict_state(a_eq, b_eq, ub, coeff, observed, names)
        row = {"view_id": vid, "dataset": ds, "group_index": int(rec["group_index"]),
               "fiber_value": [int(v) for v in val], "observed_label": rec["observed_label"],
               "query_conforming_fiber_size": int(rec["fiber_size"])}
        if found is None:
            opt = rival_optima(a_eq, b_eq, ub, coeff, observed, names)
            if max(opt.values()) != 0:
                raise SystemExit("a tie-only fiber has no rival that ties")
            row.update({"class": "tie_only", "rival_optima": opt})
        else:
            row.update({"class": "strict", "rival": found["rival"], "margin": found["margin"],
                        "state": {reg["facet_coords"][j]: int(v) for j, v in enumerate(found["state"]) if v}})
        if not (ds == "DF" and row["query_conforming_fiber_size"] == 1):
            fiber_rows.append(row)
        else:
            withheld += 1
        if found is None:
            p["tie"] += 1
            p["tie_rows"] += int(rec["fiber_size"])
        else:
            p["strict"] += 1
            p["checked"] += 1
            p["strict_rows"] += int(rec["fiber_size"])
    view_rows, admission_rows = [], []
    nrows = {ds: int(vd.dataset_mask(conf, ds).sum()) for ds in DATASETS}
    for vid, c in spec.items():
        masses, tie = {}, {}
        for ds in DATASETS:
            m, p = metrics[(vid, ds)], per[(vid, ds)]
            if c["certificate_path"] == "B" and p["strict_rows"] + p["tie_rows"] != m["unverified_observed_rows"]:
                raise SystemExit(f"open Path B fibers of {vid} on {ds} do not add up to its UNVERIFIED rows")
            tie[ds] = Decimal(p["tie_rows"]) / Decimal(nrows[ds])
            view_rows.append({"view": vid, "dataset": ds, "task": c["task"], "path": c["certificate_path"],
                              "query_rows": m["query_rows"], "unsafe_rows": m["unsafe_rows"],
                              "nondetermined_rows": m["nondetermined_rows"],
                              "unverified_observed_rows": m["unverified_observed_rows"],
                              "unverified_defect_rows": m["unverified_defect_rows"],
                              "open_path_b_fibers": p["open"], "strict_fibers": p["strict"],
                              "strict_rows": p["strict_rows"], "integer_checked_fibers": p["checked"],
                              "tie_only_fibers": p["tie"], "tie_only_rows": p["tie_rows"],
                              "tie_only_mass": p["tie_rows"] / nrows[ds]})
            masses[ds] = (float(m["mU"]), float(m["mV_obs"]) + float(m["mV_defect"]))
        t = threshold(masses)
        fixed = t is None or all(per[(vid, ds)]["tie_rows"] == 0 for ds in DATASETS)
        row = {"view": vid, "task": c["task"], "path": c["certificate_path"],
               "mU_DF": masses["DF"][0], "mU_FP": masses["FP"][0], "mV_DF": masses["DF"][1], "mV_FP": masses["FP"][1],
               "tie_only_DF": per[(vid, "DF")]["tie_rows"] / nrows["DF"],
               "tie_only_FP": per[(vid, "FP")]["tie_rows"] / nrows["FP"],
               "smallest_admitting_eps": t, "decision_fixed_under_every_order": int(fixed)}
        for eps in EPSILONS:
            row[f"admitted_eps_{eps}"] = int(admitted(masses, eps))
        admission_rows.append(row)
    counts = {
        "open_path_b_fibers": sum(p["open"] for p in per.values()),
        "strict_fibers": sum(p["strict"] for p in per.values()),
        "integer_checked_fibers": sum(p["checked"] for p in per.values()),
        "tie_only_path_b_fibers": sum(p["tie"] for p in per.values()),
        "views_with_tie_only_rows": sum(1 for r in admission_rows if r["tie_only_DF"] > 0 or r["tie_only_FP"] > 0),
        "views_with_tie_only_rows_and_no_unsafe_mass": sum(
            1 for r in admission_rows if (r["tie_only_DF"] > 0 or r["tie_only_FP"] > 0)
            and r["mU_DF"] == 0 and r["mU_FP"] == 0),
        "decisions_not_fixed": sum(1 for r in admission_rows if r["decision_fixed_under_every_order"] == 0),
        "views": len(admission_rows),
        "public_fiber_rows": len(fiber_rows),
        "withheld_fiber_rows": withheld,
    }
    log(f"[tie_order] open Path B fibers {counts['open_path_b_fibers']}: strict {counts['strict_fibers']}, "
        f"tie-only {counts['tie_only_path_b_fibers']}; decisions not fixed {counts['decisions_not_fixed']}")
    return {"verdict_rows": res["result_rows"], "view_rows": view_rows, "admission_rows": admission_rows,
            "fiber_rows": fiber_rows, "counts": counts}
