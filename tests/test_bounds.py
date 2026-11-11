# SPDX-License-Identifier: MIT
"""Toy checks of the utility bounds (beta, 1-g3) and of the parameter-free registry rule."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

from lca import bounds, probes  # noqa: E402
from lca.evaluation import ABSTAIN, evaluate_prediction  # noqa: E402


def _toy_target(seed: int, n_rows: int = 40, n_fibers: int = 5, labels=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    fib = rng.integers(0, n_fibers, size=n_rows)
    y = np.asarray([labels[i] for i in rng.integers(0, len(labels), size=n_rows)], dtype=object)
    return [f"F{f}" for f in fib], y, list(labels)


def test_accuracy_ceiling_and_beta_dominate_every_fiber_function():
    for seed in range(6):
        sig, y, labels = _toy_target(seed)
        beta, acc_ceiling, n_fib = bounds.utility_bounds(sig, y.tolist(), labels)
        fibers = sorted(set(sig))
        assert n_fib == len(fibers)
        best_acc = 0.0
        # every deterministic predictor that is a function of the fiber (abstention allowed)
        for choice in itertools.product(labels + [ABSTAIN], repeat=len(fibers)):
            table = dict(zip(fibers, choice))
            pred = np.asarray([table[s] for s in sig], dtype=object)
            f1, acc = evaluate_prediction(y, pred, labels)
            assert f1 <= beta + 1e-12, (seed, f1, beta)
            best_acc = max(best_acc, acc)
        assert abs(best_acc - acc_ceiling) < 1e-15, (seed, best_acc, acc_ceiling)


def test_beta_prefix_rule_equals_subset_maximum():
    for seed in range(6):
        sig, y, labels = _toy_target(seed, n_rows=30, n_fibers=6)
        beta, _, _ = bounds.utility_bounds(sig, y.tolist(), labels)
        fibers = sorted(set(sig))
        per_class = []
        for c in labels:
            G = int(np.sum(y == c))
            n = {f: sum(1 for s in sig if s == f) for f in fibers}
            t = {f: sum(1 for s, lab in zip(sig, y) if s == f and lab == c) for f in fibers}
            best = 0.0
            for r in range(1, len(fibers) + 1):
                for sub in itertools.combinations(fibers, r):
                    den = sum(n[f] for f in sub) + G
                    best = max(best, 2.0 * sum(t[f] for f in sub) / den if den else 0.0)
            per_class.append(best)
        assert abs(beta - float(np.mean(per_class))) < 1e-12, (seed, beta, per_class)


def test_signatures_identify_equal_sparse_rows():
    Z = sparse.csr_matrix(np.asarray([[1, 0, 2], [1, 0, 2], [0, 0, 0], [2, 0, 1], [0, 0, 0]], dtype=np.float32))
    s = bounds.sparse_signatures(Z)
    assert s[0] == s[1] and s[2] == s[4] and len({s[0], s[2], s[3]}) == 3
    assert bounds.fiber_overlap(s[:2], s) == 0.4


def _toy_registry():
    # basis facet:: coordinates mapped to families by W; one coordinate is left unmapped
    features = np.asarray(["facet::f1", "facet::f2", "facet::f3", "facet::f4", "type::t"], dtype=object)
    W = {"f1": "A", "f2": "A", "f3": "B"}
    labels = ["A", "B"]
    X = sparse.csr_matrix(np.asarray([
        [1, 0, 0, 0, 3],   # A
        [1, 1, 3, 0, 0],   # B (3 > 2)
        [0, 1, 1, 0, 0],   # tie -> abstain
        [0, 0, 0, 0, 1],   # no basis evidence -> abstain
        [0, 0, 2, 0, 0],   # B
        [0, 0, 0, 5, 0],   # evidence only on the unmapped coordinate -> abstain
    ], dtype=np.float32))
    return features, W, labels, X


def test_registry_rule_strict_argmax_and_two_computations():
    features, W, labels, X = _toy_registry()
    cols = np.asarray([0, 1, 2, 3], dtype=np.int32)
    M, names, unmapped = probes.mapping_matrix(features, cols, "facet::", "family", "W", labels, {}, W)
    assert names == ["facet::f1", "facet::f2", "facet::f3", "facet::f4"] and unmapped == ["facet::f4"]
    assert M.tolist() == [[1, 0], [1, 0], [0, 1], [0, 0]]
    rows = np.arange(X.shape[0])
    assert probes.unmapped_evidence(X, rows, cols, M) == 1
    scores = probes.rule_scores(X, rows, cols, M)
    assert scores.tolist() == [[1, 0], [2, 3], [1, 1], [0, 0], [0, 2], [0, 0]]
    keep = np.arange(5)
    assert np.array_equal(probes.rule_scores(X, keep, cols[:3], M[:3]),
                          probes.rule_scores_by_accumulation(X, keep, cols[:3], M[:3]))
    pred, decided = probes.strict_argmax(scores, labels)
    assert pred.tolist() == ["A", "B", ABSTAIN, ABSTAIN, "B", ABSTAIN]
    assert decided.tolist() == [True, True, False, False, True, False]


def test_registry_rule_decides_exactly_the_determined_rows():
    # A row is determined when its registered image has a unique positive maximum; on such rows a label rule
    # that is a function of that image (gold = arg-max family) makes every decision correct.
    features, W, labels, X = _toy_registry()
    cols = np.asarray([0, 1, 2], dtype=np.int32)
    M, _, _ = probes.mapping_matrix(features, cols, "facet::", "family", "W", labels, {}, W)
    rows = np.arange(X.shape[0])
    scores = probes.rule_scores(X, rows, cols, M)
    pred, decided = probes.strict_argmax(scores, labels)
    gold = np.asarray(["A", "B", "A", "B", "B", "A"], dtype=object)
    determined = np.asarray([(s.max() > 0) and (s == s.max()).sum() == 1 for s in scores])
    assert np.array_equal(decided, determined)
    correct = int(np.count_nonzero(decided & (pred == gold)))
    assert correct == int(np.count_nonzero(decided)) == 3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
