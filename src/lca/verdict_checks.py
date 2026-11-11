# SPDX-License-Identifier: MIT
"""Two aggregate checks on the per-anchor verdicts: probe accuracy within each verdict class of the evaluated
targets, and row-by-row verdict agreement of the eight block combinations on registry-conforming anchors."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from . import evaluation as ev
from .names import DATASETS, TASKS

FIELDS = ["scope", "quantity", "value"]
UNVERIFIED, UNSAFE, NONDETERMINED = 0, 1, 2
PREDICTION_VIEW = "identity_admitted__family"
CLASS_VIEW = "core_plus_type__family"   # verdict classes of the targets come from this view


def _one(rows: list[dict[str, str]], **conds: str) -> dict[str, str]:
    hit = [r for r in rows if all(r[k] == v for k, v in conds.items())]
    if len(hit) != 1:
        raise SystemExit(f"selector {conds} matched {len(hit)} rows")
    return hit[0]


def check_rows(cache: dict[str, Any], verdicts: dict[str, np.ndarray],
               matrix: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Rows (scope, quantity, value). `verdicts` maps a view id to its per-anchor verdict codes in anchor order;
    `matrix` is utility/cosine_determination_matrix.csv read as strings."""
    ds = cache["dataset"]
    full = cache["full_mask"]
    fam = cache["gold_family"]
    counts = Counter(zip(ds.tolist(), fam.tolist()))
    candidates = {d: {f for (dd, f), c in counts.items() if dd == d and c >= ev.MIN_LABEL_SUPPORT}
                  for d in DATASETS}
    v = verdicts[CLASS_VIEW]
    rows: list[dict[str, Any]] = []
    for src, tgt in ev.DIRECTIONS:
        t = full & (ds == tgt) & np.isin(fam, sorted(candidates[src]))
        n = int(t.sum())
        u = int(((v == UNSAFE) & t).sum())
        s = int(((v == NONDETERMINED) & t).sum())
        w = n - u - s
        d = f"{src}->{tgt}"
        m = _one(matrix, prediction_view_id=PREDICTION_VIEW, determination_view_id=CLASS_VIEW, direction=d,
                 training_protocol=ev.REFIT)
        if int(m["target_eval_rows"]) != n:
            raise SystemExit(f"target count {n} differs from the probe's {m['target_eval_rows']}")
        cu, cs, cw = (int(m[k]) for k in ("correct_on_unsafe", "correct_on_nondetermined", "correct_on_unverified"))
        for q, val in (("targets", n), ("targets_unsafe", u), ("targets_nondetermined", s), ("targets_unverified", w),
                       ("correct_unsafe", cu), ("correct_nondetermined", cs), ("correct_unverified", cw),
                       ("acc_unsafe", cu / u if u else None), ("acc_unverified", cw / w if w else None),
                       ("base_rate_unsafe", u / n)):
            rows.append({"scope": d, "quantity": q, "value": val})
    for task in TASKS:
        cells = [f"factorial_F{f}M{m_}T{t_}__{task}" for f in (0, 1) for m_ in (0, 1) for t_ in (0, 1)]
        for d in DATASETS:
            mask = full & (ds == d)
            base = verdicts[cells[0]][mask]
            diff = sum(int((verdicts[c][mask] != base).sum()) for c in cells)
            rows.append({"scope": f"{task}|{d}", "quantity": "factorial_verdict_mismatches_on_conforming",
                         "value": diff})
            rows.append({"scope": f"{task}|{d}", "quantity": "conforming_anchors", "value": int(mask.sum())})
    return rows
