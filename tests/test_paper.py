# SPDX-License-Identifier: MIT
"""Checks of the display formats of the paper cells, the plain-text rendering and the aggregate-only reader."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lca.paper import numbers  # noqa: E402
from lca.paper.common import Outputs, PaperError, count, frac, pct, plain, sci, secs, signed  # noqa: E402


def test_display_formats():
    assert frac("0.62627160254770", 4) == "0.6263"
    assert frac("0.99996", 4) == "1.0000"
    assert frac("0.00005", 4) == "0.0001"            # half up, not half even
    assert signed("-0.12345", 4) == "$-$0.1235"
    assert signed("0.0", 3) == "$+$0.000"
    assert pct("0.62627", 1) == "62.6\\%"
    assert count("631384.0") == "631,384"
    assert secs("20.02899591700043", 1) == "20.0"
    assert sci("2.922421401476407e-06") == "$2.9{\\cdot}10^{-6}$"


def test_plain_text():
    assert plain("62.6\\%") == "62.6%"
    assert plain("$-$0.120") == "-0.120"


def test_restricted_outputs_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "verdicts").mkdir()
        (root / "verdicts" / "fig1_instance.json").write_text("{}", encoding="utf-8")
        try:
            Outputs(root).json("verdicts/fig1_instance.json")
        except PaperError:
            pass
        else:
            raise AssertionError("a restricted output was read")


def test_gold_value_sets():
    from lca import external as ext
    norm = ext.normalize_token
    assert ext.gold_value_set('["Strategy", " Role-Playing "]', True, norm) == {"strategy", "role-playing"}
    assert ext.gold_value_set('"Strategy"', True, norm) == {"strategy"}
    assert ext.gold_value_set("Banks", False, norm) == {"banks"}
    assert len(ext.gold_files()) == 6 and all(f.endswith(".csv") and "/normalization/" in f for f in ext.gold_files())


def test_unit_tick_labels():
    from lca.paper.figures import unit_label
    assert [unit_label(v) for v in (0.0, 0.2, 0.25, 0.5, 1.0)] == ["0", "0.2", "0.25", "0.5", "1"]


def test_numbers_tex_round_trip():
    cell = numbers.Cell("nanchor", "631,384", "631384", ("outputs/registry/populations.json",), "n_anchor_total")
    text = numbers.numbers_tex([cell])
    assert numbers.parse_numbers_tex(text) == {"nanchor": "631,384"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
