# SPDX-License-Identifier: MIT
"""Independent check of the SEN12MS study (paper Sec. VI-C).

This file imports nothing from the pipeline library (src/lca) or the pipeline step. It re-reads the pinned class
fractions and released 17-class and 10-class single labels of data/external/sen12ms/sources.json, the fractions with
its own restricted unpickler, merges the 17 class counts with the grouping of data/external/sen12ms/igbp_simplified_map.json, and decides
every fiber by computing the optimum of each rival's program directly: on the fiber of a merged value v, a rival r of
label f reaches at most v[h(r)] (all of its group's mass) and f at least v[g(f)] if f is the only supported class of its
group, else 0. For every certified fiber whose 17 counts vary, it builds a second state of the fiber (one pixel moved
within the largest merged group of two or more supported classes) and checks that the label still wins there. It
recounts every quantity of outputs/external/sen12ms_pathb.json and writes outputs/checks/sen12ms_verification.json."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import pickle
import warnings
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "data" / "external" / "sen12ms"
OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
      ("numpy", "ndarray"), ("numpy", "dtype")}


class ArraysOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in OK:
            raise pickle.UnpicklingError(f"refused {module}.{name}")
        try:
            return getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            return getattr(importlib.import_module(module.replace("numpy.core", "numpy._core")), name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sen12ms", type=Path, default=HERE / "files")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    a = ap.parse_args()
    src = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
    pkl = next(e for e in src["files"] if e["path"].endswith(".pkl"))
    txt = next(e for e in src["files"] if e["path"] == "single_label_IGBPfull_ClsNum.txt")
    txt10 = next(e for e in src["files"] if e["path"] == "single_label_IGBPsimple_ClsNum.txt")
    raw = (a.sen12ms / pkl["path"]).read_bytes()
    lines = (a.sen12ms / txt["path"]).read_bytes()
    lines10 = (a.sen12ms / txt10["path"]).read_bytes()
    if (hashlib.sha256(raw).hexdigest() != pkl["sha256"] or hashlib.sha256(lines).hexdigest() != txt["sha256"]
            or hashlib.sha256(lines10).hexdigest() != txt10["sha256"]):
        sys.exit("SHA-256 mismatch for the class fractions or the released labels")
    released10 = {}
    for line in lines10.decode("ascii").splitlines():
        key, cls = line.split(": ")
        released10.setdefault(key, []).append(int(cls) - 1)
    released = {}
    for line in lines.decode("ascii").splitlines():
        key, cls = line.split(": ")
        released.setdefault(key, []).append(int(cls) - 1)
    declared = json.loads((HERE / "igbp_simplified_map.json").read_text())
    groups = [[k - 1 for k in g] for g in declared["groups_1based"]]
    owner = {k: gi for gi, g in enumerate(groups) for k in g}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.VisibleDeprecationWarning)   # dtype arguments of the old pickle
        table = ArraysOnly(io.BytesIO(raw)).load()

    counts = {}
    exact = 0
    for key, frac in table.items():
        f = np.asarray(frac, dtype=np.float64) * 65536
        c = np.rint(f).astype(np.int64)
        if np.abs(f - c).max() < 1e-6 and int(c.sum()) == 65536:
            exact += 1
        counts[key] = c
    n = len(counts)
    one_label = sorted(released) == sorted(counts) and all(len(v) == 1 for v in released.values())
    label_of = {key: released[key][0] for key in counts} if one_label else {}
    strict_max = sum(1 for key, c in counts.items() if one_label and (c == c.max()).sum() == 1
                     and int(np.argmax(c)) == label_of[key])
    ten_ok = 0
    if sorted(released10) == sorted(counts) and all(len(v) == 1 for v in released10.values()):
        for key, c in counts.items():
            merged = [int(c[g].sum()) for g in groups]
            top = max(merged)
            ten_ok += merged.count(top) == 1 and merged.index(top) == released10[key][0]
    support = {k for c in counts.values() for k in np.flatnonzero(c)}
    fibers: dict[tuple, list] = defaultdict(list)
    for key, c in counts.items():
        merged = tuple(int(c[g].sum()) for g in groups)
        label = label_of.get(key, int(np.argmax(c)))
        if c[label] != c.max():
            sys.exit("a label does not attain the carrier maximum")
        fibers[merged].append((key, label))

    shared = [gi for gi, g in enumerate(groups) if len([k for k in g if k in support]) >= 2]
    patches = {"UNSAFE": 0, "NONDETERMINED": 0, "UNVERIFIED": 0}
    fib = {"UNSAFE": 0, "NONDETERMINED": 0, "UNVERIFIED": 0}
    vary = {"patches": 0, "fibers": 0}
    vary_by_class: dict[int, int] = defaultdict(int)
    second_ok = 0
    whole = []
    for merged, rows in fibers.items():
        labels = [lab for _, lab in rows]
        if len(set(labels)) > 1:
            v = "NONDETERMINED"
            if max(merged) == 65536:
                gi = merged.index(65536)
                tally = defaultdict(int)
                for lab in labels:
                    tally[lab + 1] += 1
                whole.append((gi, len(rows), dict(sorted(tally.items()))))
        else:
            f = labels[0]
            gf = owner[f]
            alone = [k for k in groups[gf] if k in support] == [f]
            low_f = merged[gf] if alone else 0
            best = max(merged[owner[r]] - low_f for r in support if r != f)
            v = "UNSAFE" if best <= -1 else "UNVERIFIED"
            loose = [gi for gi in shared if merged[gi] > 0]
            if v == "UNSAFE" and loose:
                vary["patches"] += len(rows)
                vary["fibers"] += 1
                vary_by_class[f + 1] += len(rows)
                x = counts[rows[0][0]].copy()
                h = max(loose, key=lambda gi: (merged[gi], -gi))
                mem = [k for k in groups[h] if k in support]
                donor = max(mem, key=lambda k: (x[k], -k))
                taker = next(k for k in reversed(mem) if k != donor)
                x[donor] -= 1
                x[taker] += 1
                second_ok += (tuple(int(x[g].sum()) for g in groups) == merged and x.min() >= 0
                              and not np.array_equal(x, counts[rows[0][0]])
                              and all(x[f] > x[r] for r in range(len(x)) if r != f))
        fib[v] += 1
        patches[v] += len(rows)
    whole.sort(key=lambda t: (-t[1], t[0]))

    rep = json.loads((a.outputs / "external" / "sen12ms_pathb.json").read_text(encoding="utf-8"))
    rw = rep["nondetermined_whole_group_fibers"]
    pairs_ok = len(rw) == len(whole) > 0
    for e in rw if pairs_ok else []:
        (p1, l1), (p2, l2) = ((w["patch"], w["label_1based"] - 1) for w in e["witness_pair"])
        m1 = tuple(int(counts[p1][g].sum()) for g in groups) if p1 in counts else None
        m2 = tuple(int(counts[p2][g].sum()) for g in groups) if p2 in counts else None
        pairs_ok &= (m1 is not None and m1 == m2 and max(m1) == 65536 and groups[m1.index(65536)] == [k - 1 for k in e["merged_group_1based"]]
                     and label_of.get(p1) == l1 != l2 == label_of.get(p2))
    checks = [
        ("patches", rep["patches"] == n == 180662),
        ("exact pixel counts", rep["query_conform"]["exact_counts"] == exact == n),
        ("support = all 17 classes", rep["support_1based"] == [k + 1 for k in sorted(support)] and len(support) == 17),
        ("query-conforming patches", rep["query_conform"]["patches"] == n),
        ("fibers", rep["fibers"] == len(fibers)),
        ("fibers by verdict", rep["fibers_by_verdict"] == fib),
        ("patches by verdict", rep["patches_by_verdict"] == patches),
        ("verdicts partition the patches", sum(patches.values()) == n),
        ("no certificate on a NONDETERMINED fiber", rep["checks"]["nondetermined_fibers_with_a_certificate"] == 0),
        ("integer programs agree on every fiber that is not NONDETERMINED", rep["checks"]["milp_agree"] == rep["checks"]["milp_fibers"] == rep["fibers"] - rep["fibers_by_verdict"]["NONDETERMINED"] > 0),
        ("one released label per patch, for the patches of the class fractions",
         one_label and rep["released_label"]["patches"] == n),
        ("released label = strict arg-max of the 17 counts on every patch",
         strict_max == n == rep["released_label"]["equals_strict_argmax"]),
        ("released 10-class label = strict arg-max of the merged counts on every patch",
         ten_ok == n == rep["released_label_10"]["equals_strict_argmax_of_view"] == rep["released_label_10"]["patches"]),
        ("certified patches and fibers whose 17 counts vary",
         rep["certified_counts"]["vary_in_fiber"] == vary
         and rep["certified_counts"]["fixed_by_view"] == {"patches": patches["UNSAFE"] - vary["patches"],
                                                          "fibers": fib["UNSAFE"] - vary["fibers"]}),
        ("a second state of every certified fiber whose counts vary, at which the label still wins",
         second_ok == vary["fibers"] == rep["checks"]["second_states"] > 0),
        ("certified patches whose counts vary, per class",
         {e["class_1based"]: e["unsafe_counts_vary"] for e in rep["per_label"] if e["unsafe_counts_vary"]} == dict(vary_by_class)),
        ("NONDETERMINED fibers of patches wholly in one merged group, with their witness pairs",
         pairs_ok and [(groups.index([k - 1 for k in e["merged_group_1based"]]), e["patches"],
                        {int(c): t for c, t in e["labels_1based"].items()}) for e in rw] == whole),
    ]
    failed = [name for name, ok in checks if not ok]
    out = {"checks_total": len(checks), "checks_failed": len(failed), "failed_checks": failed,
           "all_pass": not failed, "pipeline_library_imported": "lca" in sys.modules,
           "patches_by_verdict": patches, "fibers_by_verdict": fib, "certified_counts_vary": vary}
    p = a.outputs / "checks" / "sen12ms_verification.json"
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"verify_sen12ms: {len(checks) - len(failed)}/{len(checks)} checks pass")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
