# SPDX-License-Identifier: MIT
"""SEN12MS study (paper Sec. VI-C): Path B and witness pairs on a third-party release.

The basis block of a patch is its pixel count of the 17 IGBP classes (fraction x 65,536, an exact integer). The view is
the image of that block under the published 17-to-10 merge, and the protected query is the released 17-class single
label (labels/single_label_IGBPfull_ClsNum.txt), which the step checks is the strict arg-max of the block on every patch
(the loader's single-label rule). With one exposed image of a total function every rival program is totally unimodular
and bounded (Theorem 1(a)), so a single-label value is certifiable iff its label is the only supported class of its
merged group and that group's count is the strict maximum of the view. The step

* checks the query-specific conformance of every patch (exact counts, the declared merge, the released label attains
  the carrier maximum), and that the released 10-class label (labels/single_label_IGBPsimple_ClsNum.txt) is the strict
  arg-max of the view on every patch (the loader's rule for that label),
* groups the patches into fibers of the view and gives each fiber a verdict: NONDETERMINED if two observed patches
  carry different labels, UNSAFE if the certificate holds, UNVERIFIED otherwise,
* checks every certificate, every witness pair and, for every UNVERIFIED fiber, a stored state of the fiber at which a
  rival ties with or exceeds the label, in integer arithmetic,
* for every UNSAFE fiber whose 17 counts vary (a merged group of two or more supported classes has a positive count),
  checks a second state of the fiber at which the label still wins,
* records the NONDETERMINED fibers of patches that lie wholly in one merged group, each with a witness pair of
  released patch keys,
* solves the rival integer programs (HiGHS through SciPy) of every fiber that is not NONDETERMINED and compares the
  verdicts with the closed form.

It writes outputs/external/sen12ms_pathb.json (aggregate counts and the released keys of the listed witness pairs).
Usage: pipeline/07d_sen12ms.py --sen12ms data/external/sen12ms/files --outputs outputs
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import pickle
import warnings
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "data" / "external" / "sen12ms"
DENOM = 65536
LABEL_FILE = "single_label_IGBPfull_ClsNum.txt"
LABEL_FILE_10 = "single_label_IGBPsimple_ClsNum.txt"
VERDICTS = ("UNSAFE", "NONDETERMINED", "UNVERIFIED")
LOADER_LINES = ("cls1 = sum(lc[0:5])", "cls2 = sum(lc[5:7])", "cls3 = sum(lc[7:9])", "cls6 = lc[11] + lc[13]",
                "lc = np.asarray([cls1, cls2, cls3, lc[9], lc[10], cls6, lc[12], lc[14], lc[15], lc[16]])",
                "np.argmax(lc")
ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
           ("numpy", "ndarray"), ("numpy", "dtype")}


class _Unpickler(pickle.Unpickler):
    """Admits NumPy array reconstruction only."""

    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        try:
            return getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            return getattr(importlib.import_module(module.replace("numpy.core", "numpy._core")), name)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def certifiable(v: np.ndarray, groups: list[list[int]], gid: np.ndarray, support: set[int], f: int) -> bool:
    g = gid[f]
    alone = [k for k in groups[g] if k in support] == [f]
    return alone and all(v[g] > v[h] for h in range(len(groups)) if h != g)


def rival_state(b: np.ndarray, v: np.ndarray, groups: list[list[int]], gid: np.ndarray, f: int) -> tuple[np.ndarray, int]:
    """A state of b's fiber at which a rival ties with or exceeds f, for a value that is not certifiable."""
    x = b.copy()
    g = gid[f]
    if len(groups[g]) > 1:                       # move the mass of f's group to a sibling
        r = next(k for k in groups[g] if k != f)
        x[groups[g]] = 0
        x[r] = v[g]
        return x, r
    h = max((k for k in range(len(groups)) if k != g), key=lambda k: v[k])   # a group at least as large as f's
    r = groups[h][0]
    x[groups[h]] = 0
    x[r] = v[h]
    return x, r


def second_state(b: np.ndarray, v: np.ndarray, groups: list[list[int]], support: set[int]) -> np.ndarray:
    """Another state of b's fiber: one pixel moves between two supported classes of a merged group with a positive count."""
    h = next(g for g, mem in enumerate(groups) if v[g] > 0 and sum(k in support for k in mem) >= 2)
    x = b.copy()
    a = next(k for k in groups[h] if x[k] > 0)
    x[a] -= 1
    x[next(k for k in groups[h] if k != a and k in support)] += 1
    return x


def read_labels(p: Path) -> dict[str, int]:
    """The released single labels: one 'patch key: class (1-based)' line per patch."""
    out: dict[str, int] = {}
    for line in p.read_text(encoding="ascii").splitlines():
        key, _, cls = line.rpartition(":")
        if key.strip() in out:
            raise SystemExit(f"two released labels for {key.strip()}")
        out[key.strip()] = int(cls)
    return out


def milp_certifiable(v: np.ndarray, W: np.ndarray, f: int, support: list[int]) -> bool:
    """Program (1) of the paper for every rival: certifiable iff every optimum is at most -1."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    n = W.shape[1]
    ub = np.array([np.inf if k in support else 0.0 for k in range(n)])
    cons = [LinearConstraint(W.astype(float), v.astype(float), v.astype(float)),
            LinearConstraint(np.ones((1, n)), 1, np.inf)]
    for r in support:
        if r == f:
            continue
        c = np.zeros(n)
        c[r], c[f] = -1.0, 1.0
        res = milp(c, constraints=cons, integrality=np.ones(n), bounds=Bounds(np.zeros(n), ub))
        if res.status != 0 or round(-res.fun) >= 0:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sen12ms", type=Path, default=HERE / "files")
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    a = ap.parse_args()

    src = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
    for e in src["files"]:
        if sha256(a.sen12ms / e["path"]) != e["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {e['path']}")
    loader = re.sub(r"\s+", " ", (a.sen12ms / "dataset.py").read_text(encoding="utf-8"))
    missing = [s for s in LOADER_LINES if re.sub(r"\s+", " ", s) not in loader]
    if missing:
        raise SystemExit(f"the loader does not declare the expected merge and label rule: {missing}")
    declared = json.loads((HERE / "igbp_simplified_map.json").read_text(encoding="utf-8"))
    groups = [[k - 1 for k in g] for g in declared["groups_1based"]]
    if groups != [[0, 1, 2, 3, 4], [5, 6], [7, 8], [9], [10], [11, 13], [12], [14], [15], [16]]:
        raise SystemExit("the declared map differs from the loader's merge")

    with open(a.sen12ms / "IGBP_probability_labels.pkl", "rb") as fh, warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.VisibleDeprecationWarning)   # dtype arguments of the old pickle
        d = _Unpickler(fh).load()
    ids = sorted(d)
    P = np.stack([np.asarray(d[i], dtype=np.float64) for i in ids])
    B = np.rint(P * DENOM).astype(np.int64)
    n, m = B.shape
    W = np.zeros((len(groups), m), dtype=np.int64)
    gid = np.zeros(m, dtype=int)
    for g, mem in enumerate(groups):
        W[g, mem] = 1
        gid[mem] = g
    V = B @ W.T
    released = read_labels(a.sen12ms / LABEL_FILE)
    if set(released) != set(ids):
        raise SystemExit("the released labels and the class fractions cover different patches")
    lab = np.array([released[i] - 1 for i in ids])
    if lab.min() < 0 or lab.max() >= m:
        raise SystemExit("a released label is not one of the 17 classes")
    argmax = int((lab == B.argmax(1)).sum())
    released10 = read_labels(a.sen12ms / LABEL_FILE_10)
    if set(released10) != set(ids):
        raise SystemExit("the released 10-class labels and the class fractions cover different patches")
    lab10 = np.array([released10[i] - 1 for i in ids])
    if lab10.min() < 0 or lab10.max() >= len(groups):
        raise SystemExit("a released 10-class label is not one of the 10 merged classes")
    view_strict = np.sort(V, 1)[:, -1] > np.sort(V, 1)[:, -2]
    argmax10 = int(((lab10 == V.argmax(1)) & view_strict).sum())
    support = sorted(int(k) for k in np.flatnonzero(B.sum(0) > 0))   # classes with a count in some patch
    sup = set(support)
    multi = [g for g, mem in enumerate(groups) if sum(k in sup for k in mem) >= 2]

    # query-specific conformance: exact counts, the declared merge, the released label attains the carrier maximum
    exact = (np.abs(P * DENOM - B).max(1) < 1e-6) & (B.sum(1) == DENOM) & (B.min(1) >= 0)
    edge = np.all(V == B @ W.T, axis=1)
    attains = B[np.arange(n), lab] == B.max(1)
    qc = exact & edge & attains
    srt = np.sort(B, 1)
    ties17 = int((srt[:, -1] == srt[:, -2]).sum())
    vs = np.sort(V, 1)
    ties10 = int((vs[:, -1] == vs[:, -2]).sum())

    # fibers of the view over the query-conforming patches
    keys, inv = np.unique(V[qc], axis=0, return_inverse=True)
    rows_q = np.flatnonzero(qc)
    members: list[list[int]] = [[] for _ in range(len(keys))]
    for i, k in zip(rows_q, inv.ravel()):
        members[k].append(int(i))
    verdict = np.array([""] * n, dtype=object)
    vary = np.zeros(n, dtype=bool)
    fiber_verdict: list[str] = []
    whole: list[dict] = []
    cert_checked = rival_checked = witness_checked = overlap = second_checked = 0
    for k, rows in enumerate(members):
        v = keys[k]
        labels = sorted({int(lab[i]) for i in rows})
        if len(labels) > 1:
            i, j = next((i, j) for i in rows for j in rows if lab[i] != lab[j])
            assert np.array_equal(V[i], V[j]) and lab[i] != lab[j]
            witness_checked += 1
            overlap += sum(certifiable(v, groups, gid, sup, f) for f in labels)
            fiber_verdict.append("NONDETERMINED")
            if v.max() == DENOM:                     # every pixel of these patches lies in one merged group
                tally = collections.Counter(int(lab[r]) for r in rows)
                whole.append({"merged_group_1based": declared["groups_1based"][int(v.argmax())], "patches": len(rows),
                              "labels_1based": {str(c + 1): tally[c] for c in sorted(tally)},
                              "witness_pair": [{"patch": ids[r], "label_1based": int(lab[r]) + 1} for r in (i, j)]})
        elif certifiable(v, groups, gid, sup, labels[0]):
            f = labels[0]
            g = gid[f]
            # every state x of the fiber has x_f = v[g] and x_r <= v[gid[r]] < v[g] for every supported rival r
            assert all(v[gid[r]] < v[g] for r in support if r != f)
            cert_checked += 1
            fiber_verdict.append("UNSAFE")
            if any(v[h] > 0 for h in multi):         # the view leaves the 17 counts free: a second state still elects f
                x = second_state(B[rows[0]], v, groups, sup)
                assert (np.array_equal(W @ x, v) and x.min() >= 0 and not np.array_equal(x, B[rows[0]])
                        and set(np.flatnonzero(x)) <= sup and all(x[f] > x[r] for r in range(m) if r != f))
                second_checked += 1
                vary[rows] = True
        else:
            f = labels[0]
            x, r = rival_state(B[rows[0]], v, groups, gid, f)
            assert np.array_equal(W @ x, v) and x.min() >= 0 and x[r] >= x[f] and set(np.flatnonzero(x)) <= sup
            rival_checked += 1
            fiber_verdict.append("UNVERIFIED")
        verdict[rows] = fiber_verdict[-1]
    if overlap:
        raise SystemExit("a NONDETERMINED fiber carries a certificate")

    single = np.array([len(groups[g]) == 1 for g in range(len(groups))])
    strict = vs[:, -1] > vs[:, -2]
    unmerged_max = strict & single[V.argmax(1)]
    if not np.array_equal(unmerged_max[qc], verdict[qc] == "UNSAFE"):
        raise SystemExit("the certified patches are not those whose merged maximum is an unmerged class")

    checked = [k for k, fv in enumerate(fiber_verdict) if fv != "NONDETERMINED"]
    agree = sum(milp_certifiable(keys[k], W, int(lab[members[k][0]]), support) == (fiber_verdict[k] == "UNSAFE")
                for k in checked)

    by_patch = {s: int((verdict == s).sum()) for s in VERDICTS}
    by_fiber = {s: fiber_verdict.count(s) for s in VERDICTS}
    unsafe_fibers = by_fiber["UNSAFE"]
    certified_counts = {"vary_in_fiber": {"patches": int(vary.sum()), "fibers": second_checked},
                        "fixed_by_view": {"patches": by_patch["UNSAFE"] - int(vary.sum()),
                                          "fibers": unsafe_fibers - second_checked}}
    whole.sort(key=lambda e: (-e["patches"], e["merged_group_1based"]))
    per_label = []
    for k in range(m):
        mask = lab == k
        per_label.append({"class_1based": k + 1, "name": declared["classes_1based"][str(k + 1)],
                          "merged_group_size": len(groups[gid[k]]), "patches": int(mask.sum()),
                          **{s.lower(): int((mask & (verdict == s)).sum()) for s in VERDICTS},
                          "unsafe_counts_vary": int((mask & vary).sum())})
    out = {
        "source": {"repository": src["repository"], "commit": src["commit"],
                   "files_sha256": {e["path"]: e["sha256"] for e in src["files"]}},
        "map": {"groups_1based": declared["groups_1based"], "authority": declared["authority"]},
        "query": "the released 17-class single label (labels/" + LABEL_FILE + "), the strict arg-max of the basis block",
        "view": "the 10 merged counts",
        "released_label": {"file": LABEL_FILE, "patches": len(released), "equals_strict_argmax": argmax},
        "released_label_10": {"file": LABEL_FILE_10, "patches": len(released10), "equals_strict_argmax_of_view": argmax10},
        "patches": n,
        "pixel_denominator": DENOM,
        "support_1based": [k + 1 for k in support],
        "classes": m,
        "query_conform": {"patches": int(qc.sum()), "exact_counts": int(exact.sum()), "map_edge_holds": int(edge.sum()),
                          "label_attains_carrier_max": int(attains.sum())},
        "ties": {"carrier_17": ties17, "view_10": ties10},
        "fibers": len(members),
        "fibers_by_verdict": by_fiber,
        "patches_by_verdict": by_patch,
        "certified_counts": certified_counts,
        "nondetermined_whole_group_fibers": whole,
        "checks": {"certificates": cert_checked, "witness_pairs": witness_checked, "rival_states": rival_checked,
                   "second_states": second_checked,
                   "nondetermined_fibers_with_a_certificate": overlap,
                   "certified_equals_unmerged_strict_merged_max": True,
                   "milp_fibers": len(checked), "milp_agree": agree},
        "per_label": per_label,
        "scope": "The released 17-class label of a third-party release, with its class fractions, published merge and "
                 "official loader at the pinned commit; the output holds aggregate counts and the released keys of the "
                 "listed witness pairs.",
    }
    if agree != len(checked) or sum(by_patch.values()) != int(qc.sum()) or sum(by_fiber.values()) != len(members):
        raise SystemExit("the verdict counts do not add up or an integer program disagrees")
    if argmax != n or not np.all(vary <= (verdict == "UNSAFE")):
        raise SystemExit("a released label is not the strict arg-max, or a varying fiber is not certified")
    p = a.outputs / "external" / "sen12ms_pathb.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"SEN12MS: {by_patch} of {n} patches in {len(members)} fibers; query-conforming {int(qc.sum())}; "
          f"integer programs agree on {agree}/{len(checked)} fibers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
