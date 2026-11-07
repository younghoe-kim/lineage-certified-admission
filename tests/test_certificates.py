# SPDX-License-Identifier: MIT
"""Toy checks of the Path A certificate, of Path B open fibers and of a Path C witness pair (plain asserts)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lca import certificates as cert  # noqa: E402
from lca import verdicts as vd  # noqa: E402


def test_path_a_strict_unique_max():
    carrier = np.array([[0, 3, 1],    # strict unique maximum at label 1
                        [2, 2, 0],    # tie: no certificate
                        [0, 0, 0],    # empty carrier: no certificate
                        [5, 0, 4]])   # strict unique maximum at label 0
    strict, label = cert.strict_unique_max(carrier)
    assert strict.tolist() == [True, False, False, True]
    assert int(label[0]) == 1 and int(label[3]) == 0


def test_path_a_fiber_verdicts():
    # Two fibers (keys 7 and 9). Fiber 7: carrier strict on both rows -> UNSAFE.
    # Fiber 9: carrier tied, two labels -> NONDETERMINED.
    keys = np.array([[7], [7], [9], [9]], dtype=np.int32)
    carrier = np.array([[0, 2], [0, 1], [1, 1], [2, 2]])
    labels = np.array([1, 1, 0, 1], dtype=np.int16)
    everything = np.ones(4, dtype=bool)
    conf = {"row_id": np.array(["r0", "r1", "r2", "r3"], dtype=object), "full": everything}
    v, unsafe_label, metrics, witnesses, _ = vd.audit_path_a(
        "toy", "facet::", "", everything, keys, labels, ["a", "b"], everything, {"facet": carrier}, conf)
    assert v.tolist() == [vd.UNSAFE, vd.UNSAFE, vd.NONDETERMINED, vd.NONDETERMINED]
    assert unsafe_label.tolist() == [1, 1, -1, -1]
    assert metrics["certificate_label_conflicts"] == 0
    nondet = [w for w in witnesses if w["kind"] == "NONDETERMINED"]
    assert nondet == [{"kind": "NONDETERMINED", "group_index": 1, "row_pair": ["r2", "r3"], "label_pair": ["a", "b"]}]


def test_witness_pair():
    positions = np.array([3, 5, 8, 11])
    labels = np.zeros(12, dtype=np.int16)
    labels[[3, 5]] = 2
    labels[8] = 0
    labels[11] = 1
    a, b, la, lb = cert.witness_pair(positions, labels)
    # first two distinct labels met: 2 (row 3) and 0 (row 8); returned in label order
    assert (a, b, la, lb) == (8, 3, 0, 2)


def test_path_b_open_fiber():
    # Facets f0, f1 map by W to family A and f2 to family B; the view exposes the A-marginal only.
    # Fiber value A = 1 with observed label A: f2 is free and maps to B, so the rival program is unbounded and the
    # recorded state ties B with A. Fiber value A = 1 with f2 outside the support: A is certified (optimum -1).
    w = {"f0": "A", "f1": "A", "f2": "B"}
    a_eq = np.array([[1.0, 1.0, 0.0]])
    coeff = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.int8)
    open_res = cert.path_b_fiber(a_eq, np.array([1.0]), np.full(3, np.inf), coeff, 0, ["A", "B"], record_open=True)
    assert not open_res["unsafe"] and open_res["open"]["program_status"] == "unbounded"
    state = dict(zip(["f0", "f1", "f2"], open_res["open"]["state"]))
    row = cert.verify_open_witness(
        view_id="toy", dataset="D", group_index=0, task="family", fiber_value=[1], requirements={"W": {"A": 1}},
        observed_label="A", rival="B", program_status="unbounded", reported_optimum="inf",
        state={k: v for k, v in state.items() if v}, support_facets=["f0", "f1", "f2"], label_names=["A", "B"],
        maps={"W": w})
    assert row["pass"] == 1 and row["rival_minus_observed_at_state"] == 0 and row["free_coordinates_to_rival"] == 1
    closed = cert.path_b_fiber(a_eq, np.array([1.0]), np.array([np.inf, np.inf, 0.0]), coeff, 0, ["A", "B"],
                               record_open=True)
    assert closed["unsafe"] and closed["rivals"][0]["rival_minus_candidate_optimum"] == -1 and "open" not in closed


if __name__ == "__main__":
    test_path_a_strict_unique_max()
    test_path_a_fiber_verdicts()
    test_witness_pair()
    test_path_b_open_fiber()
    print("ok")
