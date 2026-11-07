# SPDX-License-Identifier: MIT
"""Fiber certificates: Path A carrier maxima, Path B integer programs with exact checks, Path C witness pairs, solver-free checks of Path B certificates and open fibers."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import scipy.optimize

# Fixed HiGHS settings for every Path B integer program (threads are fixed to one by the step CLIs).
SOLVER_NAME = "scipy.optimize.milp (HiGHS)"
SOLVER_THREADS = 1
MILP_OPTIONS = {"disp": False, "presolve": True, "mip_rel_gap": 0.0}
# HiGHS may answer "infeasible or unbounded" (status 4) after presolve; the same program is then solved
# once more without presolve to decide which of the two holds.
MILP_OPTIONS_NO_PRESOLVE = {"disp": False, "presolve": False, "mip_rel_gap": 0.0}
SOLVER_MODE = "direct-HiGHS-MILP"
INTEGRALITY_TOL = 1e-7
BOUND_TOL = 1e-9
EXACT_TOL = 1e-12


def solver_settings() -> dict[str, Any]:
    return {"solver": SOLVER_NAME, "threads": SOLVER_THREADS, "options": dict(MILP_OPTIONS),
            "options_when_status_4": dict(MILP_OPTIONS_NO_PRESOLVE), "integrality_tolerance": INTEGRALITY_TOL,
            "integrality": "all variables integer, lower bound 0", "nonzero_condition": "sum(x) >= 1"}


# ----------------------------------------------------------------------------------------------------
# Path A: registered carrier with a strict unique maximum
# ----------------------------------------------------------------------------------------------------

def strict_unique_max(carrier: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise test that the carrier has a positive, strictly unique maximum; returns (mask, arg-max)."""
    c = np.asarray(carrier)
    mx = c.max(axis=1)
    strict = (mx > 0) & (np.count_nonzero(c == mx[:, None], axis=1) == 1)
    return strict, c.argmax(axis=1).astype(np.int16)


# ----------------------------------------------------------------------------------------------------
# Path C: witness pair (two rows of one fiber with different labels)
# ----------------------------------------------------------------------------------------------------

def witness_pair(positions: np.ndarray, labels: np.ndarray) -> tuple[int, int, int, int]:
    """First two distinct labels met in increasing row order; returned as (row_a, row_b, label_a, label_b)
    with label_a < label_b."""
    first_by_label: dict[int, int] = {}
    for pos in positions:
        lab = int(labels[pos])
        first_by_label.setdefault(lab, int(pos))
        if len(first_by_label) >= 2:
            break
    if len(first_by_label) < 2:
        raise SystemExit("witness pair requested for a single-label fiber")
    la, lb = sorted(first_by_label)[:2]
    return first_by_label[la], first_by_label[lb], la, lb


# ----------------------------------------------------------------------------------------------------
# Path B: nonnegative integer program solved by HiGHS and checked exactly
# ----------------------------------------------------------------------------------------------------

def solve_integer_program(c: np.ndarray, a_eq: np.ndarray, b_eq: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                          *, require_nonzero: bool) -> tuple[str, int | None]:
    """min c@x s.t. A x = b, lb <= x <= ub, x integer (and sum(x) >= 1). Returns (status, optimum) with status in
    {optimal, infeasible, unbounded}; every optimum is re-checked in exact integer arithmetic."""
    status, obj, _state = solve_integer_program_state(c, a_eq, b_eq, lb, ub, require_nonzero=require_nonzero)
    return status, obj


def solve_integer_program_state(c: np.ndarray, a_eq: np.ndarray, b_eq: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                                *, require_nonzero: bool) -> tuple[str, int | None, np.ndarray | None]:
    """As solve_integer_program, and also returns the optimal integer state (None unless status is optimal)."""
    c = np.asarray(c, dtype=np.float64)
    a_eq = np.asarray(a_eq, dtype=np.float64)
    b_eq = np.asarray(b_eq, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    nvar = len(c)
    if c.ndim != 1 or a_eq.ndim != 2 or a_eq.shape[1] != nvar or b_eq.shape != (a_eq.shape[0],):
        raise SystemExit("integer program shape mismatch")
    if lb.shape != (nvar,) or ub.shape != (nvar,):
        raise SystemExit("integer program bound shape mismatch")
    constraints: list[Any] = [scipy.optimize.LinearConstraint(a_eq, b_eq, b_eq)]
    if require_nonzero:
        constraints.append(scipy.optimize.LinearConstraint(np.ones((1, nvar)), np.asarray([1.0]), np.asarray([np.inf])))
    kw = dict(c=c, integrality=np.ones(nvar, dtype=np.uint8), bounds=scipy.optimize.Bounds(lb, ub),
              constraints=constraints)
    res = scipy.optimize.milp(options=dict(MILP_OPTIONS), **kw)
    if int(res.status) == 4:
        res = scipy.optimize.milp(options=dict(MILP_OPTIONS_NO_PRESOLVE), **kw)
    if int(res.status) == 2:
        return "infeasible", None, None
    if int(res.status) == 3:
        return "unbounded", None, None
    if int(res.status) != 0 or res.x is None or res.fun is None:
        raise SystemExit(f"HiGHS returned no accepted state: status={res.status}; message={res.message}")
    x = np.asarray(res.x, dtype=np.float64)
    xr = np.rint(x).astype(np.int64)
    if np.max(np.abs(x - xr)) > INTEGRALITY_TOL:
        raise SystemExit("HiGHS solution is not integral within tolerance")
    if np.any(xr < 0) or np.any(xr < lb - BOUND_TOL) or np.any(np.isfinite(ub) & (xr > ub + BOUND_TOL)):
        raise SystemExit("HiGHS solution violates its bounds")
    a_int = np.rint(a_eq).astype(np.int64)
    b_int = np.rint(b_eq).astype(np.int64)
    if np.max(np.abs(a_eq - a_int)) > EXACT_TOL or np.max(np.abs(b_eq - b_int)) > EXACT_TOL:
        raise SystemExit("Path B equality system is not integral")
    if not np.array_equal(a_int @ xr, b_int):
        raise SystemExit("rounded solution violates A x = b")
    if require_nonzero and int(xr.sum()) < 1:
        raise SystemExit("rounded solution violates x != 0")
    c_int = np.rint(c).astype(np.int64)
    if np.max(np.abs(c - c_int)) > EXACT_TOL:
        raise SystemExit("Path B objective is not integral")
    obj = int(c_int @ xr)
    if abs(float(res.fun) - obj) > INTEGRALITY_TOL:
        raise SystemExit(f"HiGHS objective {res.fun} differs from the exact objective {obj}")
    return "optimal", obj, xr


def solver_self_check() -> dict[str, Any]:
    """Three small programs with known answers: an optimum (-5), an unbounded and an infeasible program."""
    status, obj = solve_integer_program(np.asarray([0, -2, -3]), np.asarray([[1, 1, 0], [1, 0, 1]]), np.asarray([1, 1]),
                                        np.zeros(3), np.full(3, np.inf), require_nonzero=True)
    if status != "optimal" or obj != -5:
        raise SystemExit(f"solver check (optimum) failed: {status}, {obj}")
    status_u, _ = solve_integer_program(np.asarray([0, -1]), np.asarray([[1, 0]]), np.asarray([0]),
                                        np.zeros(2), np.full(2, np.inf), require_nonzero=True)
    if status_u != "unbounded":
        raise SystemExit(f"solver check (unbounded) failed: {status_u}")
    status_i, _ = solve_integer_program(np.asarray([1]), np.asarray([[1]]), np.asarray([1]),
                                        np.asarray([0.0]), np.asarray([0.0]), require_nonzero=True)
    if status_i != "infeasible":
        raise SystemExit(f"solver check (infeasible) failed: {status_i}")
    return {"optimum_objective": int(obj), "unbounded_detected": True, "infeasible_detected": True}


def path_b_fiber(a_eq: np.ndarray, b_eq: np.ndarray, ub: np.ndarray, label_coeff: np.ndarray, observed: int,
                 label_names: Sequence[str], *, record_open: bool = False) -> dict[str, Any]:
    """Decide whether the observed label is the strict unique maximum over the whole integer fiber
    {x >= 0 integer, A x = b, x within the dataset support, x != 0}. For each rival in label order the program
    min (candidate - rival)@x is solved; the scan stops at the first rival that is not strictly below.
    With record_open, a fiber that stays open also returns an integer state of the fiber at which that rival ties
    with or exceeds the candidate: the optimal state of its program, or, for an unbounded program, a feasible
    state plus enough events on a free supported coordinate that the carrier map sends to the rival."""
    lb = np.zeros(a_eq.shape[1], dtype=np.float64)
    rivals: list[dict[str, Any]] = []
    for rival in range(len(label_names)):
        if rival == observed:
            continue
        c = label_coeff[observed].astype(np.int16) - label_coeff[rival].astype(np.int16)
        status, min_cmr, state = solve_integer_program_state(c, a_eq, b_eq, lb, ub, require_nonzero=True)
        if status == "infeasible":
            return {"infeasible": True, "unsafe": False, "rivals": rivals}
        if status == "unbounded":
            opt: int | str = "inf"
            strict = False
        else:
            opt = -int(min_cmr)
            strict = int(min_cmr) >= 1
        rivals.append({"rival": label_names[rival], "rival_minus_candidate_optimum": opt,
                       "strictly_below": bool(strict), "solver_mode": SOLVER_MODE})
        if not strict:
            out: dict[str, Any] = {"infeasible": False, "unsafe": False, "rivals": rivals}
            if record_open:
                out["open"] = _open_state(status, state, a_eq, b_eq, lb, ub, label_coeff, observed, rival, label_names)
            return out
    return {"infeasible": False, "unsafe": True, "rivals": rivals}


def free_coordinates(a_eq: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Supported coordinates that no exposed marginal constrains (zero column of A, unbounded above)."""
    return np.flatnonzero((np.count_nonzero(np.asarray(a_eq), axis=0) == 0) & np.isinf(np.asarray(ub)))


def _open_state(status: str, state: np.ndarray | None, a_eq: np.ndarray, b_eq: np.ndarray, lb: np.ndarray,
                ub: np.ndarray, label_coeff: np.ndarray, observed: int, rival: int,
                label_names: Sequence[str]) -> dict[str, Any]:
    coeff = np.asarray(label_coeff).astype(np.int64)
    free_to_rival = [int(j) for j in free_coordinates(a_eq, ub) if coeff[rival, j] == 1]
    if status == "unbounded":
        if not free_to_rival:
            raise SystemExit("unbounded rival program without a free supported coordinate mapped to the rival")
        _s, _o, base = solve_integer_program_state(np.zeros(a_eq.shape[1]), a_eq, b_eq, lb, ub, require_nonzero=True)
        if base is None:
            raise SystemExit("an unbounded rival program has no feasible state")
        x = base.astype(np.int64).copy()
        x[free_to_rival[0]] += max(0, int(coeff[observed] @ x) - int(coeff[rival] @ x))
    else:
        x = np.asarray(state, dtype=np.int64)
    return {"rival": label_names[rival], "program_status": status, "state": [int(v) for v in x],
            "free_coordinates_to_rival": len(free_to_rival)}


# ----------------------------------------------------------------------------------------------------
# Solver-independent verification of Path B certificates
# ----------------------------------------------------------------------------------------------------

def enumerate_fiber_states(support_facets: Sequence[str], requirements: dict[str, dict[str, int]],
                           maps: dict[str, dict[str, str]]) -> tuple[list[tuple[int, ...]], list[str]]:
    """All nonzero nonnegative integer states on the support facets whose selected marginals equal the
    requirements ({map name: {image coordinate: count}}). Facets constrained by no selected marginal are returned
    as free coordinates (the state set is then infinite and no state is listed)."""
    def bound_maps(fac: str) -> list[str]:
        return [m for m in requirements if maps[m][fac] in requirements[m]]

    free = [fac for fac in support_facets if not bound_maps(fac)]
    if free:
        return [], free
    rem = {m: dict(r) for m, r in requirements.items()}
    state = [0] * len(support_facets)
    out: list[tuple[int, ...]] = []

    def rec(i: int) -> None:
        if i == len(support_facets):
            if all(v == 0 for r in rem.values() for v in r.values()) and sum(state) >= 1:
                out.append(tuple(state))
            return
        fac = support_facets[i]
        ms = bound_maps(fac)
        for x in range(min(rem[m][maps[m][fac]] for m in ms) + 1):
            state[i] = x
            for m in ms:
                rem[m][maps[m][fac]] -= x
            if all(v >= 0 for r in rem.values() for v in r.values()):
                rec(i + 1)
            for m in ms:
                rem[m][maps[m][fac]] += x
        state[i] = 0

    rec(0)
    for s in out:
        obs = {m: {k: 0 for k in r} for m, r in requirements.items()}
        for fac, x in zip(support_facets, s):
            for m in requirements:
                if maps[m][fac] in obs[m]:
                    obs[m][maps[m][fac]] += x
        if obs != requirements or sum(s) < 1:
            raise SystemExit("enumeration produced an infeasible state")
    if len(out) != len(set(out)):
        raise SystemExit("enumeration produced duplicate states")
    return sorted(out), []


def state_set_sha256(states: Sequence[tuple[int, ...]]) -> str:
    """SHA-256 of the enumerated integer state set (one comma-separated state per line)."""
    return hashlib.sha256("\n".join(",".join(str(int(x)) for x in s) for s in states).encode("ascii")).hexdigest()


def verify_path_b_certificate(*, view_id: str, dataset: str, group_index: int, task: str,
                              fiber_value: Sequence[int], requirements: dict[str, dict[str, int]],
                              certified_label: str, reported_optima: dict[str, int], support_facets: Sequence[str],
                              label_names: Sequence[str], maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Recompute every rival-minus-candidate optimum of one UNSAFE Path B fiber without a solver, either by
    exhaustive enumeration (bounded fiber) or by the free-coordinate argument (unbounded fiber), and compare
    with the optima reported by the integer programs."""
    if certified_label not in label_names:
        raise SystemExit(f"certified label outside the label universe: {certified_label}")

    def label_of(fac: str) -> str:
        return fac if task == "facet" else maps["W"][fac]

    states, free = enumerate_fiber_states(support_facets, requirements, maps)
    row: dict[str, Any] = {"view_id": view_id, "dataset": dataset, "group_index": int(group_index),
                           "fiber_value": json.dumps([int(x) for x in fiber_value], separators=(",", ":")),
                           "certified_label": certified_label}
    rivals = [lab for lab in label_names if lab != certified_label]
    if free:
        # Every selected marginal is zero, so all constrained coordinates are zero; every free coordinate maps
        # to the certified label. Any nonzero state therefore has candidate >= 1 and every rival = 0, with
        # candidate = 1 attained, so each rival-minus-candidate optimum is exactly -1.
        if any(v != 0 for r in requirements.values() for v in r.values()):
            raise SystemExit(f"unbounded fiber with a nonzero selected marginal: {view_id}|{dataset}")
        if any(label_of(fac) != certified_label for fac in free):
            raise SystemExit(f"a free coordinate maps outside the certified label: {view_id}|{dataset}")
        exact = {rival: -1 for rival in rivals}
        if exact != reported_optima:
            raise SystemExit(f"free-coordinate optima differ from the reported optima: {view_id}|{dataset}")
        row.update({"verification_method": "analytic_unbounded_category_only",
                    "integer_states_checked": "ALL_INFINITE_BY_PROOF", "state_set_sha256": "",
                    "free_coordinate_count": len(free), "minimum_strict_margin": 1, "maximum_strict_margin": "",
                    "rival_optima_exact_match": 1, "pass": 1})
        return row
    if not states:
        raise SystemExit(f"bounded fiber has no state: {view_id}|{dataset}")
    exact = {rival: -10**18 for rival in rivals}
    min_margin, max_margin = 10**18, -10**18
    for s in states:
        counts = {lab: 0 for lab in label_names}
        for fac, x in zip(support_facets, s):
            counts[label_of(fac)] += int(x)
        cand = counts[certified_label]
        margin = cand - max(counts[r] for r in rivals)
        min_margin, max_margin = min(min_margin, margin), max(max_margin, margin)
        for r in rivals:
            exact[r] = max(exact[r], counts[r] - cand)
    if exact != reported_optima:
        raise SystemExit(f"enumerated optima differ from the reported optima: {view_id}|{dataset}|{list(fiber_value)}")
    if min_margin < 1:
        raise SystemExit(f"candidate is not a strict unique maximum in every state: {view_id}|{dataset}")
    row.update({"verification_method": "exhaustive_integer_enumeration", "integer_states_checked": len(states),
                "state_set_sha256": state_set_sha256(states), "free_coordinate_count": 0,
                "minimum_strict_margin": int(min_margin), "maximum_strict_margin": int(max_margin),
                "rival_optima_exact_match": 1, "pass": 1})
    return row


def verify_open_witness(*, view_id: str, dataset: str, group_index: int, task: str, fiber_value: Sequence[int],
                        requirements: dict[str, dict[str, int]], observed_label: str, rival: str,
                        program_status: str, reported_optimum: int | str, state: dict[str, int],
                        support_facets: Sequence[str], label_names: Sequence[str],
                        maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Check one open Path B fiber in integer arithmetic: the state is a nonzero nonnegative integer state on the
    dataset support whose selected marginals equal the fiber value, and the rival ties with or exceeds the observed
    label there; an unbounded program also needs a free supported coordinate that the carrier map sends to the rival."""
    if observed_label not in label_names or rival not in label_names or rival == observed_label:
        raise SystemExit(f"open witness with an invalid label pair: {view_id}|{dataset}|{group_index}")

    def label_of(fac: str) -> str:
        return fac if task == "facet" else maps["W"][fac]

    support = set(support_facets)
    if any(f not in support for f in state) or any((not isinstance(v, int)) or v < 0 for v in state.values()):
        raise SystemExit(f"open witness state outside the support or not a nonnegative integer: {view_id}|{dataset}")
    if sum(state.values()) < 1:
        raise SystemExit(f"open witness state is zero: {view_id}|{dataset}")
    obs = {m: {k: 0 for k in r} for m, r in requirements.items()}
    for fac, x in state.items():
        for m in requirements:
            if maps[m][fac] in obs[m]:
                obs[m][maps[m][fac]] += x
    if obs != requirements:
        raise SystemExit(f"open witness state lies outside its fiber: {view_id}|{dataset}|{group_index}")
    counts = {lab: 0 for lab in label_names}
    for fac, x in state.items():
        counts[label_of(fac)] += x
    gap = counts[rival] - counts[observed_label]
    if gap < 0:
        raise SystemExit(f"open witness state keeps the observed label strictly above the rival: {view_id}|{dataset}")
    free = [f for f in support_facets if not any(maps[m][f] in requirements[m] for m in requirements)]
    free_to_rival = [f for f in free if label_of(f) == rival]
    if program_status == "unbounded":
        if reported_optimum != "inf" or not free_to_rival:
            raise SystemExit(f"unbounded open fiber without a free coordinate of the rival: {view_id}|{dataset}")
    elif program_status == "optimal":
        if reported_optimum != gap:
            raise SystemExit(f"open witness objective differs from the reported optimum: {view_id}|{dataset}")
    else:
        raise SystemExit(f"unknown program status {program_status}")
    return {"view_id": view_id, "dataset": dataset, "group_index": int(group_index),
            "fiber_value": json.dumps([int(x) for x in fiber_value], separators=(",", ":")),
            "observed_label": observed_label, "rival": rival, "program_status": program_status,
            "rival_minus_observed_optimum": reported_optimum,
            "witness_state": json.dumps({f: int(state[f]) for f in support_facets if state.get(f)}, ensure_ascii=False,
                                        separators=(",", ":")),
            "rival_minus_observed_at_state": int(gap), "free_coordinates_to_rival": len(free_to_rival), "pass": 1}


def reported_optima(rivals: Sequence[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rivals:
        v = r["rival_minus_candidate_optimum"]
        if not isinstance(v, int) or isinstance(v, bool):
            raise SystemExit(f"UNSAFE certificate with a non-integer rival optimum: {v!r}")
        if not r.get("strictly_below"):
            raise SystemExit("UNSAFE certificate with a rival that is not strictly below")
        out[str(r["rival"])] = int(v)
    return out
