# SPDX-License-Identifier: MIT
"""Independent check of the support step: recomputes the four coordinate sets from the registry outputs, the
accounting of every regime (masses, transitions, admission decisions), the reproduction of the verdict step on the
released supports, and every Path B certificate of every regime by enumerating the basis-block values of its fiber
in integer arithmetic. Like verify_outputs.py, it does not import the pipeline library."""
from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("DF", "FP")
REGIMES = ("released", "union", "intersection", "released_plus_color")
COLOR = "color"
EPSILONS = ("0", "0.01", "0.05")
VERDICT_COPY = ("basis", "path", "fibers", "m^U", "m^N", "m^V_obs", "m^V_defect", "near_pure", "purity")
CHECK_FIELDS = ["check", "primary_value", "independent_value", "exact_match", "note"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def text(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":")) if isinstance(v, (list, dict)) else str(v)


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, name: str, primary: Any, independent: Any, ok: bool, note: str = "") -> None:
        self.rows.append({"check": name, "primary_value": text(primary), "independent_value": text(independent),
                          "exact_match": int(bool(ok)), "note": note})


def enumerate_values(bound: list[str], constraints: list[tuple[set[str], int]]) -> list[dict[str, int]]:
    """All nonnegative integer assignments to the bound facets whose sums over each constraint set equal its
    target (every bound facet lies in at least one constraint, so the set is finite)."""
    member = {f: [k for k, (s, _) in enumerate(constraints) if f in s] for f in bound}
    targets = [t for _, t in constraints]
    partial = [0] * len(constraints)
    out: list[dict[str, int]] = []
    state: dict[str, int] = {}

    def rec(i: int) -> None:
        if i == len(bound):
            if partial == targets:
                out.append(dict(state))
            return
        f = bound[i]
        cap = min(targets[k] - partial[k] for k in member[f])
        for x in range(cap + 1):
            for k in member[f]:
                partial[k] += x
            state[f] = x
            rec(i + 1)
            for k in member[f]:
                partial[k] -= x
        state.pop(f, None)

    rec(0)
    return out


def check_certificate(row: dict[str, str], support: list[str], projection: dict[str, list[str]],
                      images: dict[str, dict[str, str]], label_of: dict[str, str], labels: list[str]
                      ) -> tuple[bool, dict[str, Any]]:
    value = json.loads(row["fiber_value"])
    constraints: list[tuple[set[str], int]] = []
    k = 0
    for block, coords in projection.items():
        for coord in coords:
            constraints.append(({f for f in support if images[block][f] == coord}, int(value[k])))
            k += 1
    if k != len(value):
        return False, {"reason": "fiber value length differs from the projection"}
    bound = sorted(set().union(*(s for s, _ in constraints))) if constraints else []
    free = [f for f in support if f not in bound]
    target = row["certified_label"]
    if any(label_of[f] != target for f in free):
        return False, {"reason": "a free supported facet feeds a rival"}
    margins = []
    for b in enumerate_values(bound, constraints):
        total = sum(b.values())
        if total == 0 and not free:
            continue
        carrier = {lab: 0 for lab in labels}
        for f, x in b.items():
            carrier[label_of[f]] += x
        extra = 0 if total >= 1 else 1  # with free facets, the smallest admissible state adds one event to them
        margins.append(carrier[target] + extra - max(v for lab, v in carrier.items() if lab != target))
    if not margins:
        return False, {"reason": "empty fiber"}
    ok = min(margins) >= 1
    info: dict[str, Any] = {"min_margin": min(margins), "free": len(free)}
    if free:
        ok &= row["integer_states_checked"] == "ALL_INFINITE_BY_PROOF" and int(row["free_coordinate_count"]) == len(free)
        ok &= int(row["minimum_strict_margin"]) == min(margins)
    else:
        info["states"] = len(margins)
        ok &= int(row["integer_states_checked"]) == len(margins) and int(row["free_coordinate_count"]) == 0
        ok &= int(row["minimum_strict_margin"]) == min(margins) and int(row["maximum_strict_margin"]) == max(margins)
    return ok and row["pass"] == "1", info


def threshold(m: dict[str, tuple[float, float]]) -> float | None:
    if any(u > 0 for u, _ in m.values()):
        return None
    return max(v for _, v in m.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    a = ap.parse_args()
    out = a.outputs
    ck = Checks()

    # ------------------------------------------------------------------ coordinate sets and color
    sup = read_json(out / "registry/dataset_support.json")["datasets"]
    fmap = read_csv(out / "registry/facet_map.csv")
    T = {r["facet"]: r["term_type"] for r in fmap}
    W = {r["facet"]: r["family"] for r in fmap}
    pi = {r["core_term_uid"]: r["core_facet"] for r in read_csv(out / "registry/term_to_facet.csv")}
    pops = read_json(out / "registry/populations.json")
    members = pops["block_coordinate_members"]
    facet_coords = [c for c in members["facet::"]]
    released = {ds: set(sup[ds]["S_members"]) for ds in DATASETS}
    union = set.union(*released.values())
    inter = set.intersection(*released.values())
    sets = {"released": released, "union": {ds: union for ds in DATASETS},
            "intersection": {ds: inter for ds in DATASETS},
            "released_plus_color": {ds: released[ds] | {COLOR} for ds in DATASETS}}
    ok_color = COLOR in facet_coords and all(COLOR not in released[ds] for ds in DATASETS)
    ck.add("support.color_is_a_released_facet_outside_both_supports", True, ok_color, ok_color)
    regimes = read_csv(out / "support/support_regimes.csv")
    for r in regimes:
        mine = sorted(sets[r["regime"]][r["dataset"]])
        ok = (json.loads(r["members"]) == mine and int(r["support_size"]) == len(mine)
              and json.loads(r["added_vs_released"]) == sorted(set(mine) - released[r["dataset"]])
              and json.loads(r["removed_vs_released"]) == sorted(released[r["dataset"]] - set(mine)))
        ck.add(f"support.coordinate_set.{r['regime']}.{r['dataset']}", r["support_size"], len(mine), ok)
    ck.add("support.regime_rows", 8, len(regimes), len(regimes) == 8)
    doc = read_json(out / "support/support.json")
    reach = {ds: sum(1 for t in sup[ds]["C_members"] if pi.get(t) == COLOR) for ds in DATASETS}
    color = doc["color"]
    ok = (color["active_core_terms_reaching_facet"] == reach and color["T"] == T[COLOR] and color["W"] == W[COLOR]
          and color["T_is_released_type_coordinate"] == (T[COLOR] in members["type::"]))
    ck.add("support.color_facts", color, {"active_core_terms_reaching_facet": reach, "T": T[COLOR], "W": W[COLOR]}, ok)

    # ------------------------------------------------------------------ released regime = verdict step
    sv = read_csv(out / "support/support_verdicts.csv")
    frozen = {(r["view"], r["dataset"]): r for r in read_csv(out / "verdicts/verdicts.csv")}
    rel = [r for r in sv if r["regime"] == "released"]
    ok = len(rel) == len(frozen) and all(all(r[k] == frozen[(r["view"], r["dataset"])][k] for k in VERDICT_COPY)
                                         for r in rel)
    ck.add("support.released_rows_equal_verdict_step", len(frozen), len(rel), ok)
    certs = read_csv(out / "support/support_certificates.csv")
    frozen_c = read_csv(out / "verdicts/path_b_certificates.csv")
    mine_c = [{k: v for k, v in r.items() if k != "regime"} for r in certs if r["regime"] == "released"]
    key = lambda r: (r["view_id"], r["dataset"], r["group_index"])  # noqa: E731
    ok = sorted(mine_c, key=key) == sorted(frozen_c, key=key)
    ck.add("support.released_certificates_equal_verdict_step", len(frozen_c), len(mine_c), ok)

    # ------------------------------------------------------------------ accounting of every regime
    n = {ds: int(pops["dataset_rows"][ds]) for ds in DATASETS}
    vrows = {(r["regime"], r["view"], r["dataset"]): r for r in sv}
    views = sorted({r["view"] for r in sv})
    bad = []
    for (reg, view, ds), r in vrows.items():
        cnt = [int(r[k]) for k in ("unsafe_rows", "nondetermined_rows", "unverified_observed_rows", "unverified_defect_rows")]
        masses = [float(r[k]) for k in ("m^U", "m^N", "m^V_obs", "m^V_defect")]
        if sum(cnt) != n[ds] or any(m != c / n[ds] for m, c in zip(masses, cnt)) \
                or int(r["query_rows"]) != n[ds] - cnt[3]:
            bad.append(f"{reg}|{view}|{ds}")
    ck.add("support.masses_and_counts", len(vrows), len(vrows) - len(bad), not bad, ";".join(bad[:5]))
    ck.add("support.verdict_rows", len(REGIMES) * len(views) * 2, len(vrows), len(vrows) == len(REGIMES) * len(views) * 2)

    tr = read_csv(out / "support/support_transitions.csv")
    code = {"UNSAFE": "unsafe_rows", "NONDETERMINED": "nondetermined_rows"}
    bad = []
    for reg in REGIMES[1:]:
        for view in views:
            for ds in DATASETS:
                rows = [r for r in tr if r["regime"] == reg and r["view"] == view and r["dataset"] == ds]
                tot = sum(int(r["rows"]) for r in rows)
                new, old = vrows[(reg, view, ds)], vrows[("released", view, ds)]
                ok = tot == n[ds]
                for lab, col in code.items():
                    ok &= sum(int(r["rows"]) for r in rows if r["verdict"] == lab) == int(new[col])
                    ok &= sum(int(r["rows"]) for r in rows if r["released_verdict"] == lab) == int(old[col])
                ok &= sum(int(r["rows"]) for r in rows if r["query_conform"] == "1") == int(new["query_rows"])
                ok &= sum(int(r["rows"]) for r in rows if r["released_query_conform"] == "1") == int(old["query_rows"])
                for r in rows:
                    changed = r["verdict"] != r["released_verdict"]
                    f = int(r["changed_fibers"])
                    ok &= (1 <= f <= int(r["rows"])) if changed else f == 0
                if not ok:
                    bad.append(f"{reg}|{view}|{ds}")
    ck.add("support.transitions", len(REGIMES[1:]) * len(views) * 2, len(REGIMES[1:]) * len(views) * 2 - len(bad),
           not bad, ";".join(bad[:5]))

    adm = read_csv(out / "support/support_admission.csv")
    bad = []
    for r in adm:
        m = {ds: (float(vrows[(r["regime"], r["view"], ds)]["m^U"]),
                  float(vrows[(r["regime"], r["view"], ds)]["m^V_obs"]) + float(vrows[(r["regime"], r["view"], ds)]["m^V_defect"]))
             for ds in DATASETS}
        t = threshold(m)
        ok = (float(r["mU_DF"]) == m["DF"][0] and float(r["mU_FP"]) == m["FP"][0]
              and float(r["mV_DF"]) == m["DF"][1] and float(r["mV_FP"]) == m["FP"][1]
              and (r["smallest_admitting_eps"] == "" if t is None else float(r["smallest_admitting_eps"]) == t))
        for eps in EPSILONS:
            ok &= int(r[f"admitted_eps_{eps}"]) == int(t is not None and Decimal(repr(t)) <= Decimal(eps))
        if not ok:
            bad.append(f"{r['regime']}|{r['view']}")
    ck.add("support.admission_decisions", len(adm), len(adm) - len(bad), not bad and len(adm) == len(REGIMES) * len(views),
           ";".join(bad[:5]))

    # ------------------------------------------------------------------ every Path B certificate, enumerated
    proj = {r["view_id"]: json.loads(r["projection_json"]) for r in read_csv(out / "registry/view_projection.csv")}
    images = {"family::": W, "type::": T}
    labels = {"family": list(members["family::"]), "facet": facet_coords}
    bad, states, analytic = [], 0, 0
    per_regime = {reg: 0 for reg in REGIMES}
    for r in certs:
        task = r["view_id"].rsplit("__", 1)[1]
        label_of = W if task == "family" else {f: f for f in T}
        names = labels[task]
        support = sorted(sets[r["regime"]][r["dataset"]])
        ok, info = check_certificate(r, support, proj[r["view_id"]], images, label_of, names)
        per_regime[r["regime"]] += 1
        if "states" in info:
            states += info["states"]
        else:
            analytic += 1
        if not ok:
            bad.append(f"{r['regime']}|{r['view_id']}|{r['dataset']}|{r['group_index']}")
    ck.add("support.certificates_enumerated", len(certs), len(certs) - len(bad), not bad, ";".join(bad[:5]))
    counts = {reg: doc["counts"][reg]["path_b_certificates"] for reg in REGIMES}
    ck.add("support.certificate_counts", counts, per_regime, counts == per_regime)
    ck.add("support.released_reproduces_flags", True, doc["released_reproduces_verdict_step"],
           all(doc["released_reproduces_verdict_step"].values()))

    failed = [r["check"] for r in ck.rows if r["exact_match"] == 0]
    dst = out / "checks"
    dst.mkdir(parents=True, exist_ok=True)
    with (dst / "support_checks.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CHECK_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(ck.rows)
    summary = {"step": "verify_support", "schema_version": "1", "checks_total": len(ck.rows), "checks_failed": len(failed),
               "failed_checks": failed, "all_pass": not failed, "pipeline_library_imported": False,
               "certificates_checked": len(certs), "basis_block_values_enumerated": states,
               "certificates_with_free_coordinates": analytic}
    (dst / "support_verification.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[verify_support] checks={len(ck.rows)} failed={len(failed)} certificates={len(certs)} states={states}")
    if failed:
        raise SystemExit("support verification failed: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
