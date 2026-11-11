# SPDX-License-Identifier: MIT
"""Unit test of the registry maps, label-rule check, certificate rule and materialized map on a toy registry."""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lca import registry as reg  # noqa: E402
from lca.names import FAMILY_ORDER  # noqa: E402


def raises(fn, *args):
    try:
        fn(*args)
    except SystemExit:
        return True
    return False


def toy_registry():
    core = [
        {"core_term_uid": "hub:core:term/category/a", "term_type": "category", "facet": "category/upperbody"},
        {"core_term_uid": "hub:core:term/category/b", "term_type": "category", "facet": "category/upperbody"},
        {"core_term_uid": "hub:core:term/attribute/c", "term_type": "attribute", "facet": "attribute/nickname"},
        {"core_term_uid": "hub:core:term/attribute/d", "term_type": "attribute", "facet": "attribute/waistline"},
    ]
    edges = [{"src_uid": r["core_term_uid"], "edge_type": "IN_FACET", "dst_uid": "hub:facet/" + r["facet"]}
             for r in core]
    edges.append({"src_uid": "hub:core:term/category/a", "edge_type": "HAS_ATTRIBUTE",
                  "dst_uid": "hub:core:term/attribute/d"})
    return core, edges


def test_maps():
    core, edges = toy_registry()
    m = reg.build_registry_maps(core, edges)
    assert m["facets"] == ["attribute/nickname", "attribute/waistline", "category/upperbody"]
    assert m["Pi"]["hub:core:term/category/b"] == "category/upperbody"
    assert m["T"] == {"attribute/nickname": "attribute", "attribute/waistline": "attribute",
                      "category/upperbody": "category"}
    assert m["W"] == {"attribute/nickname": "tag_terms", "attribute/waistline": "attribute_expression",
                      "category/upperbody": "taxonomic_category"}
    assert all(m["w"][u] == m["W"][m["Pi"][u]] for u in m["Pi"])
    assert m["relation_counts"] == {"HAS_ATTRIBUTE": 1, "IN_FACET": 4}
    assert [r["match"] for r in m["term_to_facet_rows"]] == [1, 1, 1, 1]


def test_registry_guards():
    core, edges = toy_registry()
    wrong_facet = [dict(e, dst_uid="hub:facet/category/others") if e["src_uid"].endswith("/a")
                   and e["edge_type"] == "IN_FACET" else e for e in edges]
    assert raises(reg.build_registry_maps, core, wrong_facet)
    wrong_prefix = [dict(e, dst_uid=e["dst_uid"].replace("hub:facet/", "facet/")) for e in edges]
    assert raises(reg.build_registry_maps, core, wrong_prefix)
    two_types = [dict(r, term_type="part") if r["core_term_uid"].endswith("/b") else r for r in core]
    assert raises(reg.build_registry_maps, two_types, edges)


def test_label_rule_check():
    source = inspect.getsource(reg.norm_str) + "\n" + inspect.getsource(reg.derive_family)
    reg.verify_label_rule(source)
    assert raises(reg.verify_label_rule, source.replace('return "tag_terms"', 'return "other_expression"'))


def test_certificates():
    c = reg.certificate_for_view("family", ("core::", "type::"))
    assert (c["basis"], c["certificate_path"], c["required_maps"]) == ("core::", "A", "Pi|W")
    c = reg.certificate_for_view("facet", ("family::", "type::"))
    assert (c["basis"], c["certificate_path"], c["required_maps"], c["carrier_edges"]) == ("facet::", "B", "T|W", "")
    assert reg.certificate_for_view("family", ("rep_core::",))["certificate_path"] == "C"
    views = reg.declared_views()
    cert_rows, view_rows = reg.build_certificate_spec(views)
    assert len(views) == len(cert_rows) == len(view_rows) == 32
    assert [r["view_id"] for r in cert_rows] == sorted(r["view_id"] for r in cert_rows)


def test_orders():
    assert sorted(reversed(FAMILY_ORDER), key=reg.family_sort_key) == list(FAMILY_ORDER)
    assert reg.ordered_coords("inst_family::", ["material_expression", "tag_terms", "decorative_expression"]) == \
        ["tag_terms", "decorative_expression", "material_expression"]


def test_materialized_map():
    m = reg._new_moments()
    rows = [({"color": 1, "category/upperbody": 1}, {"attribute_expression": 1, "taxonomic_category": 1}),
            ({"category/upperbody": 2}, {"taxonomic_category": 2}),
            ({"color": 1}, {"attribute_expression": 1})]
    for x, y in rows:
        reg._add_moments(m, x, y)
    facets, families = ["category/upperbody", "color"], ["attribute_expression", "taxonomic_category"]
    M, rank = reg.solve_materialized_map(m, facets, families)
    assert rank == 2 and M == {"category/upperbody": "taxonomic_category", "color": "attribute_expression"}
    assert set(reg.residual_sumsq(m, facets, families, M).values()) == {0}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
