# SPDX-License-Identifier: MIT
"""Toy checks of the declared-order step: a strictly winning rival state keeps a fiber open under every order, and a
fiber whose rivals can at most tie is tie-only (plain asserts)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lca import tie_order as to  # noqa: E402


def test_strict_rival_state():
    # Facet task with two facets; the view exposes their sum (= 2) and the observed label is f0.
    # The state (0, 2) lets f1 beat f0 by 2, so no declared order can certify the fiber.
    coeff = np.eye(2, dtype=np.int8)
    found = to.strict_state(np.array([[1.0, 1.0]]), np.array([2.0]), np.full(2, np.inf), coeff, 0, ["f0", "f1"])
    assert found is not None and found["rival"] == "f1" and found["margin"] == 2


def test_tie_only_fiber():
    # The view exposes both counts (1, 1): every state ties f0 and f1, so only the event order decides the label.
    coeff = np.eye(2, dtype=np.int8)
    assert to.strict_state(np.eye(2), np.array([1.0, 1.0]), np.full(2, np.inf), coeff, 0, ["f0", "f1"]) is None


def test_unbounded_rival():
    # Family task: facets f0, f1 map to family A and f2 to family B; the view exposes the A-marginal (= 1).
    # f2 is free in the support, so B can exceed A by any amount; outside the support B cannot even tie.
    a_eq = np.array([[1.0, 1.0, 0.0]])
    coeff = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.int8)
    found = to.strict_state(a_eq, np.array([1.0]), np.full(3, np.inf), coeff, 0, ["A", "B"])
    assert found is not None and found["rival"] == "B" and found["margin"] >= 1 and found["status"] == "unbounded"
    assert to.strict_state(a_eq, np.array([1.0]), np.array([np.inf, np.inf, 0.0]), coeff, 0, ["A", "B"]) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
