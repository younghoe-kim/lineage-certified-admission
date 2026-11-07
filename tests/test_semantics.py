# SPDX-License-Identifier: MIT
"""The rules that tie verdicts to certificate paths (plain asserts).

UNSAFE needs a certificate of Path A or B; NONDETERMINED holds exactly on fibers with two observed labels, on every
path; Path C issues no UNSAFE; the two definite verdicts exclude each other; everything else is UNVERIFIED, defects
included. The toy checks run the verdict functions; the output checks read the shipped aggregate outputs, and the
independent verifier recounts the same rules on every anchor (outputs/checks/independent_checks.csv)."""
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca import verdicts as vd  # noqa: E402

RULES = ("defect_rows_unverified", "nondetermined_iff_witness_pair", "unsafe_only_on_paths_a_b",
         "definite_verdicts_exclusive", "verdict_constant_on_fiber")


def _conf(n: int, full=None) -> dict:
    return {"row_id": np.array([f"r{i}" for i in range(n)], dtype=object),
            "full": np.ones(n, dtype=bool) if full is None else full}


def test_path_a_rules():
    # fibers: 1 strict, one label -> UNSAFE; 2 tie, two labels -> NONDETERMINED; 3 tie, one label -> UNVERIFIED;
    # row 6 is a query defect (outside QUERY-CONFORM) of fiber 1 -> UNVERIFIED
    keys = np.array([[1], [1], [2], [2], [3], [3], [1]], dtype=np.int32)
    labels = np.array([0, 0, 0, 1, 1, 1, 0], dtype=np.int16)
    strict = np.array([True, True, False, False, False, False, True])
    q = np.array([True] * 6 + [False])
    every = np.ones(7, dtype=bool)
    v, _, m, _, _ = vd.audit_path_a("toy", "facet::", "", q, keys, labels, ["a", "b"], every, {}, _conf(7),
                                    strict_label=(strict, np.zeros(7, dtype=np.int64)))
    assert v.tolist() == [vd.UNSAFE, vd.UNSAFE, vd.NONDETERMINED, vd.NONDETERMINED, vd.UNVERIFIED, vd.UNVERIFIED,
                          vd.UNVERIFIED]
    assert m["certificate_label_conflicts"] == 0 and m["unverified_defect_rows"] == 1


def test_path_a_conflict_is_counted():
    # a strict carrier on a fiber that holds two labels is a certificate-label conflict, which the audit reports
    keys = np.array([[1], [1]], dtype=np.int32)
    labels = np.array([0, 1], dtype=np.int16)
    every = np.ones(2, dtype=bool)
    _, _, m, _, _ = vd.audit_path_a("toy", "facet::", "", every, keys, labels, ["a", "b"], every, {}, _conf(2),
                                    strict_label=(np.ones(2, dtype=bool), np.zeros(2, dtype=np.int64)))
    assert m["certificate_label_conflicts"] == 1


def test_path_c_rules():
    keys = np.array([[1], [1], [2], [2], [3]], dtype=np.int32)
    labels = np.array([0, 1, 1, 1, 0], dtype=np.int16)
    q = np.array([True, True, True, True, False])
    every = np.ones(5, dtype=bool)
    v, unsafe_label, m, w, _ = vd.audit_path_c("toy", q, keys, labels, ["a", "b"], every, _conf(5))
    assert v.tolist() == [vd.NONDETERMINED, vd.NONDETERMINED, vd.UNVERIFIED, vd.UNVERIFIED, vd.UNVERIFIED]
    assert not np.any(v == vd.UNSAFE) and np.all(unsafe_label == -1) and m["unsafe_rows"] == 0
    assert [x["kind"] for x in w] == ["NONDETERMINED"] and set(w[0]["label_pair"]) == {"a", "b"}


def _shipped(rel: str) -> Path | None:
    p = REPO / "outputs" / rel
    return p if p.is_file() else None


def test_shipped_outputs():
    vr, spec_path = _shipped("verdicts/verdicts.csv"), _shipped("registry/certificate_spec.csv")
    if vr is None or spec_path is None:
        print("shipped outputs not present; output checks skipped")
        return
    with spec_path.open(encoding="utf-8", newline="") as fh:
        spec = {r["view_id"]: r for r in csv.DictReader(fh)}
    with vr.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 * len(spec) and {r["view"] for r in rows} == set(spec)
    for r in rows:
        assert r["path"] == spec[r["view"]]["certificate_path"]
        mu, mn = float(r["m^U"]), float(r["m^N"])
        total = mu + mn + float(r["m^V_obs"]) + float(r["m^V_defect"])
        assert abs(total - 1.0) < 1e-12, r["view"]
        if r["path"] == "C":
            assert mu == 0.0, r["view"]
        # NONDETERMINED mass iff some query-conforming fiber holds two labels, i.e. purity below one
        purity = float(r["purity"])
        assert (mn > 0) == (not math.isnan(purity) and purity < 1.0), r["view"]
    ic = _shipped("checks/independent_checks.csv")
    if ic is not None:
        with ic.open(encoding="utf-8", newline="") as fh:
            checks = {r["check"]: r for r in csv.DictReader(fh)}
        for rule in RULES:
            r = checks[f"semantics.{rule}"]
            assert r["exact_match"] == "1" and r["independent_value"] == "0", rule


if __name__ == "__main__":
    test_path_a_rules()
    test_path_a_conflict_is_counted()
    test_path_c_rules()
    test_shipped_outputs()
    print("ok")
