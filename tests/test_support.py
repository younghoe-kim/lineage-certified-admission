# SPDX-License-Identifier: MIT
"""Checks of the support step on toy inputs: coordinate sets, admission thresholds, the floor display of bounds,
and the enumeration used by the independent support verifier."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca import support  # noqa: E402
from lca.paper.common import PaperError, floor_frac  # noqa: E402

_spec = importlib.util.spec_from_file_location("verify_support", REPO / "verify" / "verify_support.py")
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)


def toy_registry():
    return {"support": {"datasets": {"DF": {"S_members": ["a", "b"]}, "FP": {"S_members": ["b", "c"]}}},
            "facet_coords": ["a", "b", "c", "color"]}


def test_coordinate_sets():
    s = support.coordinate_sets(toy_registry())
    assert s["union"]["DF"] == s["union"]["FP"] == {"a", "b", "c"}
    assert s["intersection"]["DF"] == {"b"}
    assert s["released_plus_color"]["FP"] == {"b", "c", "color"}
    reg = toy_registry()
    reg["support"]["datasets"]["DF"]["S_members"].append("color")
    try:
        support.coordinate_sets(reg)
        raise AssertionError("color inside a released support must be refused")
    except SystemExit:
        pass


def test_threshold_and_admission():
    assert support.threshold({"DF": (0.0, 0.04), "FP": (0.0, 0.01)}) == 0.04
    assert support.threshold({"DF": (0.001, 0.0), "FP": (0.0, 0.0)}) is None
    m = {"DF": (0.0, 0.05), "FP": (0.0, 0.0)}
    assert support.admitted(m, "0.05") and not support.admitted(m, "0.01")


def test_floor_display():
    assert floor_frac("0.14809708", 3) == "0.148"
    assert floor_frac("0.2019999", 3) == "0.201"
    assert floor_frac("0.0", 3) == "0.000"
    try:
        floor_frac("1.0", 3)
        raise AssertionError("a value of 1 has no floor display below 1")
    except PaperError:
        pass


def test_enumeration_and_certificate():
    vals = vs.enumerate_values(["x", "y", "z"], [({"x", "y"}, 2), ({"z"}, 1)])
    assert len(vals) == 3 and all(v["x"] + v["y"] == 2 and v["z"] == 1 for v in vals)
    T = {"c1": "category", "c2": "category", "a1": "attribute"}
    W = {"c1": "TAX", "c2": "TAX", "a1": "STR"}
    row = {"fiber_value": "[0,2]", "certified_label": "TAX", "integer_states_checked": "3", "free_coordinate_count": "0",
           "minimum_strict_margin": "2", "maximum_strict_margin": "2", "pass": "1"}
    ok, info = vs.check_certificate(row, ["a1", "c1", "c2"], {"type::": ["attribute", "category"]},
                                    {"type::": T}, W, ["TAX", "STR"])
    assert ok and info["states"] == 3
    # a free facet that feeds a rival voids the certificate (color added to a support)
    T2, W2 = dict(T, color="color"), dict(W, color="SURF")
    ok, info = vs.check_certificate(row, ["a1", "c1", "c2", "color"], {"type::": ["attribute", "category"]},
                                    {"type::": T2}, W2, ["TAX", "STR", "SURF"])
    assert not ok and info["reason"].startswith("a free supported facet")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
