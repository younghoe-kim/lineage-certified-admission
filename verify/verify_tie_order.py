# SPDX-License-Identifier: MIT
"""Independent check of the declared-order step from the shipped outputs alone (no pipeline code): every strictly
winning rival state of outputs/tie_order/tie_order_fibers_public.csv is rechecked in integer arithmetic against the
registry maps, the dataset support and the view projection; for every tie-only fiber, all states of the fiber are
enumerated and each rival's optimum is recomputed and matched with the recorded one; the counts are matched with
tie_order.json."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import combinations_with_replacement, product
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    out = ap.parse_args().outputs
    fmap = read_csv(out / "registry/facet_map.csv")
    images = {"family::": {r["facet"]: r["family"] for r in fmap}, "type::": {r["facet"]: r["term_type"] for r in fmap}}
    support = {ds: set(v["S_members"]) for ds, v in
               json.loads((out / "registry/dataset_support.json").read_text())["datasets"].items()}
    proj = {r["view_id"]: json.loads(r["projection_json"]) for r in read_csv(out / "registry/view_projection.csv")}
    task = {r["view_id"]: r["task"] for r in read_csv(out / "registry/view_projection.csv")}
    rows = read_csv(out / "tie_order/tie_order_fibers_public.csv")
    counts = json.loads((out / "tie_order/tie_order.json").read_text())["counts"]
    failed, strict, tie, enumerated, recomputed = [], 0, 0, 0, 0

    def carrier(state: dict[str, int], label: str, t: str) -> int:
        return state.get(label, 0) if t == "facet" else sum(n for f, n in state.items() if images["family::"][f] == label)

    def fiber_states(view: str, value: list[int], supp: set[str]):
        """Every state of the fiber of a type-count value: a multiset of supported facets per exposed type column."""
        cols = proj[view].get("type::")
        if set(proj[view]) != {"type::"} or not isinstance(cols, list):
            raise SystemExit(f"tie-only fiber of {view}: only type-count projections are enumerated here")
        if any(images["type::"][f] not in cols for f in supp):
            raise SystemExit(f"tie-only fiber of {view}: a supported facet is read by no exposed column")
        per_col = [combinations_with_replacement(sorted(f for f in supp if images["type::"][f] == c), int(k))
                   for c, k in zip(cols, value)]
        for parts in product(*[list(p) for p in per_col]):
            yield Counter(f for part in parts for f in part)

    for r in rows:
        key = f"{r['view_id']}|{r['dataset']}|{r['group_index']}"
        if r["class"] == "strict":
            strict += 1
            x = {f: int(n) for f, n in json.loads(r["state"]).items()}
            value, k, ok = json.loads(r["fiber_value"]), 0, True
            ok &= all(n > 0 for n in x.values()) and set(x) <= support[r["dataset"]] and bool(x)
            for block, coords in proj[r["view_id"]].items():
                for c in coords:
                    ok &= sum(n for f, n in x.items() if images[block][f] == c) == int(value[k])
                    k += 1
            t = task[r["view_id"]]
            margin = carrier(x, r["rival"], t) - carrier(x, r["observed_label"], t)
            ok &= k == len(value) and margin == int(r["margin"]) and margin >= 1
        else:
            tie += 1
            opt = json.loads(r["rival_optima"])
            ok = r["class"] == "tie_only" and max(opt.values()) == 0 and r["observed_label"] not in opt
            t, best, n_states = task[r["view_id"]], {rv: None for rv in opt}, 0
            for x in fiber_states(r["view_id"], json.loads(r["fiber_value"]), support[r["dataset"]]):
                n_states += 1
                own = carrier(x, r["observed_label"], t)
                for rv in best:
                    m = carrier(x, rv, t) - own
                    best[rv] = m if best[rv] is None or m > best[rv] else best[rv]
            enumerated += n_states
            match = n_states > 0 and all(best[rv] == int(v) for rv, v in opt.items())
            recomputed += match
            ok &= match
        if not ok:
            failed.append(key)
    ok_counts = (counts["public_fiber_rows"] == len(rows)
                 and counts["public_fiber_rows"] + counts["withheld_fiber_rows"] == counts["open_path_b_fibers"]
                 and tie == counts["tie_only_path_b_fibers"])
    if not ok_counts:
        failed.append("counts")
    summary = {"check": "tie_order", "pipeline_library_imported": False, "rows": len(rows),
               "strict_states_checked": strict, "tie_only_rows": tie, "tie_only_optima_recomputed": recomputed,
               "tie_only_states_enumerated": enumerated, "counts_match_tie_order_json": ok_counts,
               "failed": failed, "checks_failed": len(failed), "all_pass": not failed}
    dst = out / "checks"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "tie_order_verification.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[verify_tie_order] rows={len(rows)} strict_states={strict} tie_only={tie} "
          f"tie_only_optima_recomputed={recomputed} (states enumerated={enumerated}) failed={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
