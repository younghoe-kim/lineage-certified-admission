# SPDX-License-Identifier: MIT
"""Number cells of the paper: every computed number is bound to the aggregate output(s) it is read from.

build(outputs) returns the cells in a fixed order; write() produces numbers.tex (one \\val definition per cell,
followed by a comment naming the output and the selector) and number_ledger.csv. The build stops when a selector
matches zero or several rows or when a consistency check of the outputs fails."""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from math import comb
from pathlib import Path

from ..conformance import ROW_CLASSES
from .common import (DATASETS, DIRECTIONS, Outputs, PaperError, count, dec, floor_frac, frac, ival, one, pct, ratio, sci,
                     secs, select, signed)

LEDGER_FIELDS = ["cell_id", "display", "raw_value", "output", "output_sha256", "selector"]
DS_KEY = (("DF", "df"), ("FP", "fp"))
REFIT, ALL_SOURCE = "fixed_dataset_refit", "all_source"
VIEW_KEYS = {
    "iafam": "identity_admitted__family",
    "ctfam": "core_plus_type__family",
    "fofam": "facet_only__family",
    "tthree": "type3__family",
    "ttwo": "type_attr_part__family",
    "iafac": "identity_admitted__facet",
    "fthree": "family3__facet",
    "fthreet": "family3_plus_type__facet",
    "lzfac": "l0_facet__facet",
    "lzfam": "l0_family__family",
    "isfam": "inst_sem__family",
    "ifam": "inst_family__family",
    "rcfam": "rep_core__family",
    "rcfac": "rep_core__facet",
}
# Row classes of the FULL-CONFORM failures, in the declared order of the conformance step.
NO_REP_CORE, UNSUPPORTED_COLOR, CATEGORY_UNIT = ROW_CLASSES
ROW_CLASS_TEXT = {NO_REP_CORE: "key=no_rep_core (empty rep_core:: block)",
                  UNSUPPORTED_COLOR: "the unsupported-color row class (color counts with a rep_core:: block)",
                  CATEGORY_UNIT: "key=unsupported_category_unit (one category event)"}
EIGHT_CELLS = [f"F{f}M{m}T{t}" for f in (0, 1) for m in (0, 1) for t in (0, 1)]
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
# Counts below ten that the running text spells out; the word cell is "<id>w" with the same raw value.
WORD_CELLS = ("niafam", "pbxenum", "pbxanalytic", "extqueries", "extdomq", "extdombound", "extdomall", "volseeds",
              "screps", "rccatdf", "factverdictmm", "labelfalse", "pbxfthreet", "rulecells", "rulefullcells",
              "rtcpu", "ncatfp", "ruleerriafamdffp", "extgoldmulti", "xcoord", "xcoordrows", "attrfptargets",
              "nsmallfac", "smallfaclo", "smallfachi", "sgaindf", "sgainfp", "pbsupchg", "pbcolor", "pbcolorqcviews",
              "ruleerrtot", "kgwnrrsim", "kgwnrrinv")
# The Fig. 1 anchor of the released benchmark: its nonzero facet coordinates and their cell ids.
FIG1_FACETS = {"attribute/garment detail": "figgarment", "attribute/textile material type": "figtextile",
               "category/wholebody": "figwhole"}
FIG1_FAMILIES = {"structural_expression": "figstr", "surface_expression": "figsurf", "taxonomic_category": "figtax"}


@dataclass(frozen=True)
class Cell:
    cell_id: str
    display: str
    raw: str
    outputs: tuple[str, ...]
    selector: str


class Cells:
    def __init__(self, out: Outputs):
        self.out = out
        self.items: list[Cell] = []
        self._ids: set[str] = set()
        self.refit_fit: dict[tuple[str, str], set[tuple[int, int]]] = {}

    def add(self, cid: str, display: str, raw, rels, selector: str) -> None:
        if cid in self._ids:
            raise PaperError(f"duplicate cell {cid}")
        if not re.fullmatch(r"[a-z][a-z0-9]*", cid):
            raise PaperError(f"cell id must be lowercase alphanumeric: {cid}")
        rels = (rels,) if isinstance(rels, str) else tuple(rels)
        names = tuple(r if r.startswith("src/") else "outputs/" + r for r in rels)
        self._ids.add(cid)
        self.items.append(Cell(cid, display, str(raw), names, selector))


def _sum_dec(values) -> Decimal:
    return sum((dec(v) for v in values), Decimal(0))


def build(out: Outputs) -> list[Cell]:
    c = Cells(out)
    _registry(c, out)
    _conformance(c, out)
    _verdicts(c, out)
    _fig1(c, out)
    _probes(c, out)
    _bounds(c, out)
    _controls(c, out)
    _scaling(c, out)
    _external(c, out)
    _external_gold(c, out)
    _mutations(c, out)
    _verdict_checks(c, out)
    _registry_rule(c, out)
    _spreads(c, out)
    _evidence(c, out)
    _support(c, out)
    _build_record(c, out)
    _rerun(c, out)
    _independent(c, out)
    _kg(c, out)
    _sen12ms(c, out)
    _tie_order(c, out)
    _text_versions(c)
    _word_versions(c)
    return c.items


# ---------------------------------------------------------------- registry and populations
def _registry(c: Cells, out: Outputs) -> None:
    src = "registry/registry.yaml"
    reg = out.yaml(src)
    ac = reg["authority_counts"]
    for key, cid in (("n_authority_terms", "nauthterms"), ("n_authority_facets", "nauthfacets"),
                     ("n_in_facet_edges", "ninfacet"), ("n_in_facet_matches_core_facet", "ninfacetmatch"),
                     ("n_ontology_edges", "nedges")):
        c.add(cid, count(ac[key]), ac[key], src, f"authority_counts.{key}")
    for rel, cid in [("HAS_ATTRIBUTE", "relhasattr"), ("ALIAS_OF", "relalias"), ("IN_FACET", "relinfacet"),
                     ("EVIDENCE_ASSOCIATED_WITH", "relevid"), ("SUBCATEGORY_OF", "relsubcat"),
                     ("HAS_PART", "relhaspart")]:
        v = ac["relation_counts"][rel]
        c.add(cid, count(v), v, src, f"authority_counts.relation_counts.{rel}")

    sr = out.source_release()
    src_sr = "src/lca/paper/source_release.json"
    for key, cid in (("term table", "hashterms"), ("relation table", "hashedges")):
        h = sr["source_files_sha256"][key]
        c.add(cid, h[:8], h, src_sr, f"source_files_sha256[{key}] (first 8 hex digits)")

    src = "registry/populations.json"
    pop = out.json(src)
    c.add("nanchor", count(pop["n_anchor_total"]), pop["n_anchor_total"], src, "n_anchor_total")
    c.add("ndf", count(pop["dataset_rows"]["DF"]), pop["dataset_rows"]["DF"], src, "dataset_rows.DF")
    c.add("nfp", count(pop["dataset_rows"]["FP"]), pop["dataset_rows"]["FP"], src, "dataset_rows.FP")
    c.add("ncols", count(pop["n_feature_columns"]), pop["n_feature_columns"], src, "n_feature_columns")
    for blk, cid in [("core::", "bcore"), ("rep_core::", "brepcore"), ("facet::", "bfacet"),
                     ("inst_family::", "binstfam"), ("family::", "bfamily"), ("type::", "btype"),
                     ("inst_sem::", "binstsem")]:
        v = pop["block_coordinate_counts"][blk]
        c.add(cid, count(v), v, src, f"block_coordinate_counts.{blk}")
    c.add("nactivefacets", count(pop["n_active_core_facets"]), pop["n_active_core_facets"], src,
          "n_active_core_facets")
    h = sr["source_files_sha256"]["anchor table"]
    c.add("hashanchor", h[:8], h, src_sr, "source_files_sha256[anchor table] (first 8 hex digits)")

    src = "registry/dataset_support.json"
    sup = out.json(src)["datasets"]
    C = {d: set(sup[d]["C_members"]) for d in DATASETS}
    S = {d: set(sup[d]["S_members"]) for d in DATASETS}
    c.add("cdf", count(len(C["DF"])), len(C["DF"]), src, "datasets.DF.C_members")
    c.add("cfp", count(len(C["FP"])), len(C["FP"]), src, "datasets.FP.C_members")
    c.add("cshared", count(len(C["DF"] & C["FP"])), len(C["DF"] & C["FP"]), src, "|C_DF & C_FP|")
    c.add("cunion", count(len(C["DF"] | C["FP"])), len(C["DF"] | C["FP"]), src, "|C_DF | C_FP|")
    c.add("sdf", count(len(S["DF"])), len(S["DF"]), src, "datasets.DF.S_members")
    c.add("sfp", count(len(S["FP"])), len(S["FP"]), src, "datasets.FP.S_members")
    c.add("sinter", count(len(S["DF"] & S["FP"])), len(S["DF"] & S["FP"]), src, "|S_DF & S_FP|")


# ---------------------------------------------------------------- conformance and drift anatomy
def _conformance(c: Cells, out: Outputs) -> None:
    pop = out.json("registry/populations.json")
    src = "conformance/conformance_summary.csv"
    cs = out.csv(src)
    for d, k in DS_KEY:
        r = one(cs, dataset=d, view_id="core_plus_type__family")
        c.add("fcfail" + k, count(r["query_defect_rows"]), r["query_defect_rows"], src,
              f"dataset={d}, view_id=core_plus_type__family, query_defect_rows")
        c.add("fcok" + k, count(r["query_conform_rows"]), r["query_conform_rows"], src,
              f"dataset={d}, view_id=core_plus_type__family, query_conform_rows")
        c.add("qoutct" + k, count(r["outside_Drel_rows"]), r["outside_Drel_rows"], src,
              f"dataset={d}, view_id=core_plus_type__family, outside_Drel_rows")
        r = one(cs, dataset=d, view_id="facet_only__family")
        c.add("wfail" + k, count(r["W_fail_rows"]), r["W_fail_rows"], src,
              f"dataset={d}, view_id=facet_only__family, W_fail_rows")
        c.add("qdefw" + k, count(r["query_defect_rows"]), r["query_defect_rows"], src,
              f"dataset={d}, view_id=facet_only__family, query_defect_rows")
        c.add("qw" + k, count(r["query_conform_rows"]), r["query_conform_rows"], src,
              f"dataset={d}, view_id=facet_only__family, query_conform_rows")
        c.add("qout" + k, count(r["outside_Drel_rows"]), r["outside_Drel_rows"], src,
              f"dataset={d}, view_id=facet_only__family, outside_Drel_rows")
    falses = sum(ival(r["gold_is_carrier_max_false_rows"]) for r in cs)
    c.add("labelfalse", count(falses), falses, src, "sum of gold_is_carrier_max_false_rows over view x dataset rows")
    nview = len({r["view_id"] for r in cs})
    c.add("nviews", count(nview), nview, src, "distinct view_id")
    fdf = ival(one(cs, dataset="DF", view_id="core_plus_type__family")["query_defect_rows"])
    ffp = ival(one(cs, dataset="FP", view_id="core_plus_type__family")["query_defect_rows"])
    nall = ival(pop["n_anchor_total"])
    both = (src, "registry/populations.json")
    c.add("ndrift", count(fdf + ffp), fdf + ffp, src, "FULL-CONFORM failures DF + FP (core_plus_type__family query_defect_rows)")
    c.add("driftshare", pct(Decimal(fdf + ffp) / nall), Decimal(fdf + ffp) / nall, both,
          "FULL-CONFORM failures / n_anchor_total")
    ndf, nfp = ival(pop["dataset_rows"]["DF"]), ival(pop["dataset_rows"]["FP"])
    c.add("driftdfpct", pct(Decimal(fdf) / ndf), Decimal(fdf) / ndf, both, "DF failures / dataset_rows.DF")
    c.add("driftfppct", pct(Decimal(ffp) / nfp), Decimal(ffp) / nfp, both, "FP failures / dataset_rows.FP")
    c.add("driftsharefr", frac(Decimal(fdf + ffp) / nall, 4), Decimal(fdf + ffp) / nall, both,
          "FULL-CONFORM failures / n_anchor_total [fraction]")
    c.add("driftdffr", frac(Decimal(fdf) / ndf, 4), Decimal(fdf) / ndf, both, "DF failures / dataset_rows.DF [fraction]")
    c.add("driftfpfr", frac(Decimal(ffp) / nfp, 4), Decimal(ffp) / nfp, both, "FP failures / dataset_rows.FP [fraction]")

    src = "conformance/drift_anatomy.csv"
    da = out.csv(src)
    for d, k in DS_KEY:
        r = one(da, dataset=d, quantity="unsupported_facet_mass", key="total")
        c.add("unsup" + k, count(r["value"]), r["value"], src, f"dataset={d}, quantity=unsupported_facet_mass, key=total")
        r = one(da, dataset=d, quantity="rows_with_deficit", key="")
        c.add("deficit" + k, count(r["value"]), r["value"], src, f"dataset={d}, quantity=rows_with_deficit")
    for d, facet, cid in [("DF", "color", "unsupcolordf"), ("DF", "category/lowerbody", "unsuplbdf"),
                          ("DF", "category/upperbody", "unsupubdf"), ("FP", "category/upperbody", "unsupubfp"),
                          ("FP", "category/wholebody", "unsupwbfp"), ("FP", "category/lowerbody", "unsuplbfp"),
                          ("FP", "color", "unsupcolorfp"), ("DF", "category/wholebody", "unsupwbdf")]:
        v = sum(ival(r["value"]) for r in select(da, dataset=d, quantity="unsupported_facet_mass", key=facet))
        c.add(cid, count(v), v, src, f"dataset={d}, quantity=unsupported_facet_mass, key={facet} (0 if the key does not occur)")
    for d, cls, cid in [("DF", NO_REP_CORE, "rcnorepdf"), ("DF", UNSUPPORTED_COLOR, "rcghostdf"),
                        ("DF", CATEGORY_UNIT, "rccatdf"), ("FP", NO_REP_CORE, "rcnorepfp")]:
        r = one(da, dataset=d, quantity="row_class", key=cls)
        c.add(cid, count(r["value"]), r["value"], src, f"dataset={d}, quantity=row_class, {ROW_CLASS_TEXT[cls]}")
    v = sum(ival(r["value"]) for r in select(da, dataset="FP", quantity="row_class", key=UNSUPPORTED_COLOR))
    c.add("rcghostfp", count(v), v, src,
          f"dataset=FP, quantity=row_class, {ROW_CLASS_TEXT[UNSUPPORTED_COLOR]} (0 if the key does not occur)")
    # DF anchors with color counts are exactly the W-only defects; they split by row class
    v = ival(one(da, dataset="DF", quantity="color_rows", key="total")["value"])
    wf = ival(one(cs, dataset="DF", view_id="facet_only__family")["W_fail_rows"])
    qd = ival(one(cs, dataset="DF", view_id="facet_only__family")["query_defect_rows"])
    if not v == wf == qd:
        raise PaperError(f"DF color anchors {v}, W failures {wf} and W-only defects {qd} differ")
    if select(da, dataset="DF", quantity="color_rows_by_row_class", key="registry_conforming"):
        raise PaperError("a registry-conforming DF anchor carries color counts")
    r = one(da, dataset="DF", quantity="color_rows_by_row_class", key=UNSUPPORTED_COLOR)
    if r["value"] != one(da, dataset="DF", quantity="row_class", key=UNSUPPORTED_COLOR)["value"]:
        raise PaperError("the color row class is not the set of color anchors with a representative core")
    r = one(da, dataset="DF", quantity="color_rows_by_row_class", key=NO_REP_CORE)
    c.add("colornorepdf", count(r["value"]), r["value"], src,
          f"dataset=DF, quantity=color_rows_by_row_class, {ROW_CLASS_TEXT[NO_REP_CORE]}")
    if one(da, dataset="ALL", quantity="excess_rows_equal_pi_failures")["value"] != "1":
        raise PaperError("drift anatomy does not reproduce the FULL-CONFORM Pi failures")

    # release-build provenance: the build record of the source release (the controls step reports the same values)
    brec = out.source_release()["build_record"]
    ctl = out.json("utility/controls.json")["build_report"]
    if (brec.get("conforms") is not True or ctl.get("conforms") is not True
            or ival(ctl["dropped_core_entries"]) != ival(brec["dropped_core_term_entries"])):
        raise PaperError("the build record and the controls step disagree")
    c.add("builddropped", count(brec["dropped_core_term_entries"]), brec["dropped_core_term_entries"],
          "src/lca/paper/source_release.json", "build_record.dropped_core_term_entries")
    sr = out.source_release()["release_builder"]
    if ival(sr["instance_anchors_matched"]) != nall:
        raise PaperError("matched instance anchors differ from n_anchor_total")
    src_sr = "src/lca/paper/source_release.json"
    c.add("bridgetotal", count(sr["instance_anchors_total"]), sr["instance_anchors_total"], src_sr,
          "release_builder.instance_anchors_total")
    c.add("bridgematched", count(sr["instance_anchors_matched"]), sr["instance_anchors_matched"], src_sr,
          "release_builder.instance_anchors_matched")
    # DF provenance: images of the benchmark, one-to-one matches, and the lower-cased path keys that collide in pairs
    rows = out.json("registry/populations.json")["dataset_rows"]
    df_img, df_m = ival(sr["df_benchmark_images"]), ival(sr["df_matched"])
    coll, pairs = ival(sr["df_casefold_collision_images"]), ival(sr["df_casefold_collision_pairs"])
    if (df_m != ival(rows["DF"]) or ival(sr["fp_matched"]) != ival(rows["FP"]) or df_img - coll != df_m
            or coll != 2 * pairs
            or ival(sr["df_instance_anchors_before_rekeying"]) != df_img + ival(sr["df_inshop_images"])
            or ival(sr["fp_instance_anchors_before_rekeying"]) != ival(sr["fp_matched"])
            or ival(sr["instance_anchors_total"]) != nall + ival(sr["df_inshop_images"]) + coll):
        raise PaperError("the release-builder record does not decompose into the released dataset rows")
    c.add("dfsource", count(df_img), df_img, src_sr, "release_builder.df_benchmark_images")
    c.add("dfexcluded", count(coll), coll, src_sr, "release_builder.df_casefold_collision_images")
    c.add("dfpairs", count(pairs), pairs, src_sr, "release_builder.df_casefold_collision_pairs")


# ---------------------------------------------------------------- verdicts, decomposition, Path B
def _verdicts(c: Cells, out: Outputs) -> None:
    pop = out.json("registry/populations.json")
    src = "verdicts/verdicts.csv"
    vr = out.csv(src)
    for key, view in VIEW_KEYS.items():
        for d, k in DS_KEY:
            r = one(vr, view=view, dataset=d)
            mu, ms = dec(r["m^U"]), dec(r["m^N"])
            mv = dec(r["m^V_obs"]) + dec(r["m^V_defect"])
            if abs(mu + ms + mv - 1) > Decimal("1e-12"):
                raise PaperError(f"masses of {view} {d} do not sum to one")
            sel = f"view={view}, dataset={d}"
            c.add(f"u{key}{k}", sci(mu) if 0 < mu < Decimal("0.00005") else frac(mu, 4), mu, src, sel + ", m^U")
            c.add(f"s{key}{k}", frac(ms, 4), ms, src, sel + ", m^N")
            c.add(f"v{key}{k}", frac(mv, 4), mv, src, sel + ", m^V_obs + m^V_defect")
            c.add(f"vo{key}{k}", frac(r["m^V_obs"], 4), r["m^V_obs"], src, sel + ", m^V_obs")
            c.add(f"vd{key}{k}", frac(r["m^V_defect"], 4), r["m^V_defect"], src, sel + ", m^V_defect")
            c.add(f"fib{key}{k}", count(r["fibers"]), r["fibers"], src, sel + ", fibers")
            c.add(f"pur{key}{k}", frac(r["purity"], 4), r["purity"], src, sel + ", purity")
            c.add(f"np{key}{k}", frac(r["near_pure"], 4), r["near_pure"], src, sel + ", near_pure")
    worst = {}
    for r in vr:
        mu = dec(r["m^U"])
        mv = dec(r["m^V_obs"]) + dec(r["m^V_defect"])
        u, v = worst.get(r["view"], (Decimal(0), Decimal(0)))
        worst[r["view"]] = (max(u, mu), max(v, mv))
    pos = [v for u, v in worst.values() if u > 0]
    if not pos or min(pos) <= Decimal("0.05"):
        raise PaperError("a view with positive m^U is expected to have m^V above 0.05 in some dataset")
    c.add("mvposmin", floor_frac(min(pos), 3), min(pos), src,
          "min over views with positive m^U in some dataset of max over datasets of m^V [rounded down]")
    for d, k in DS_KEY:
        r = one(vr, view="identity_admitted__family", dataset=d)
        c.add(f"headpct{k}", pct(r["m^U"]), r["m^U"], src, f"view=identity_admitted__family, dataset={d}, m^U")
        r = one(vr, view="core_plus_type__family", dataset=d)
        c.add(f"ctpct{k}", pct(r["m^U"]), r["m^U"], src, f"view=core_plus_type__family, dataset={d}, m^U")
        r = one(vr, view="identity_admitted__facet", dataset=d)
        c.add(f"iafacpct{k}", pct(r["m^U"]), r["m^U"], src, f"view=identity_admitted__facet, dataset={d}, m^U")
        r = one(vr, view="family3__facet", dataset=d)
        mv = dec(r["m^V_obs"]) + dec(r["m^V_defect"])
        c.add(f"fthreevpct{k}", pct(mv, 2), mv, src, f"view=family3__facet, dataset={d}, m^V_obs + m^V_defect")
        c.add(f"fthreedefpct{k}", pct(r["m^V_defect"], 2), r["m^V_defect"], src,
              f"view=family3__facet, dataset={d}, m^V_defect")
    # every family view without family:: that keeps core:: or facet:: reaches at least the core+type mass
    keep = ["identity_admitted__family", "core_plus_type__family", "facet_only__family"] + \
           [f"factorial_F{f}M0T{t}__family" for f in (0, 1) for t in (0, 1)]
    for d in DATASETS:
        floor = dec(one(vr, view="core_plus_type__family", dataset=d)["m^U"])
        if min(dec(one(vr, view=v, dataset=d)["m^U"]) for v in keep) < floor:
            raise PaperError(f"a family view that keeps core:: or facet:: is below the core+type mass in {d}")
    # identity-admissible family schemas (core and role blocks other than the carrier)
    ia = ["factorial_F0M0T0__family", "factorial_F0M0T1__family", "factorial_F1M0T0__family",
          "factorial_F1M0T1__family", "facet_only__family", "type3__family", "type_attr_part__family"]
    if not all(min(dec(one(vr, view=v, dataset=d)["m^U"]) for d in DATASETS) > 0 for v in ia):
        raise PaperError("an identity-admissible family schema has zero UNSAFE mass in a dataset")
    c.add("niafam", count(len(ia)), len(ia), src,
          "identity-admissible family schemas with m^U > 0 in both datasets: " + ",".join(ia))

    src = "verdicts/decomposition.csv"
    dc = out.csv(src)
    for d, k in DS_KEY:
        r = one(dc, dataset=d)
        if r["full_conform_rowwise_equal"] != "1":
            raise PaperError("the decomposition is not row-wise equal on registry-conforming anchors")
        c.add("decall" + k, count(r["primary_unsafe_rows"]), r["primary_unsafe_rows"], src,
              f"dataset={d}, primary_unsafe_rows")
        c.add("decfull" + k, count(r["primary_unsafe_full_rows"]), r["primary_unsafe_full_rows"], src,
              f"dataset={d}, primary_unsafe_full_rows")
        c.add("decout" + k, count(r["primary_unsafe_outside_full_rows"]), r["primary_unsafe_outside_full_rows"], src,
              f"dataset={d}, primary_unsafe_outside_full_rows")
        n = ival(pop["dataset_rows"][d])
        m = Decimal(ival(r["primary_unsafe_outside_full_rows"])) / n
        both = (src, "registry/populations.json")
        c.add("decoutm" + k, frac(m, 4), m, both, f"dataset={d}, primary_unsafe_outside_full_rows / dataset_rows")
        c.add("decoutpct" + k, pct(m), m, both, f"dataset={d}, primary_unsafe_outside_full_rows / dataset_rows")

    # carrier equivalence: on query-conforming anchors the identity-admitted view of each task is UNSAFE exactly where
    # the direct-carrier view is; the anchors that only the direct-carrier view certifies are query defects
    src = "verdicts/carrier_equivalence.csv"
    ce = out.csv(src)
    if len(ce) != 4:
        raise PaperError("the carrier equivalence has a row per task and dataset")
    for r in ce:
        only = ival(r["direct_only_rows"])
        if (ival(r["mismatch_on_identity_query_conform_rows"]) != 0 or ival(r["identity_only_rows"]) != 0
                or ival(r["direct_only_identity_query_defect_rows"]) != only
                or ival(r["direct_only_outside_full_conform_rows"]) != only):
            raise PaperError(f"carrier equivalence fails for {r['task']} {r['dataset']}")
    cs = out.csv("conformance/conformance_summary.csv")
    for d, _ in DS_KEY:
        r = one(ce, task="family", dataset=d)
        if (r["identity_unsafe_rows"] != one(dc, dataset=d)["primary_unsafe_rows"]
                or r["identity_query_conform_rows"]
                != one(cs, dataset=d, view_id="facet_only__family")["query_conform_rows"]):
            raise PaperError("the carrier equivalence and the decomposition count different anchors")
    fam_fp, fam_df = one(ce, task="family", dataset="FP"), one(ce, task="family", dataset="DF")
    if ival(fam_fp["direct_only_rows"]) != 0 or fam_fp["direct_unsafe_rows"] != fam_fp["identity_unsafe_rows"]:
        raise PaperError("in FP the identity-admitted and direct-carrier family UNSAFE sets differ")
    if fam_df["direct_only_with_color_rows"] != fam_df["direct_only_rows"]:
        raise PaperError("a DF anchor that only the direct-carrier family view certifies carries no color count")
    c.add("carronlydf", count(fam_df["direct_only_rows"]), fam_df["direct_only_rows"], src,
          "task=family, dataset=DF, direct_only_rows (= direct_only_with_color_rows = query defects)")
    for d, k in DS_KEY:
        r = one(ce, task="facet", dataset=d)
        c.add("carronlyfac" + k, count(r["direct_only_rows"]), r["direct_only_rows"], src,
              f"task=facet, dataset={d}, direct_only_rows (= direct_only_outside_full_conform_rows)")

    # family-3 on DF: UNVERIFIED mass = drift defects + observed open anchors
    r = one(vr, view="family3__facet", dataset="DF")
    n = Decimal(ival(pop["dataset_rows"]["DF"]))
    opened, defects = dec(r["m^V_obs"]) * n, dec(r["m^V_defect"]) * n
    if (abs(opened - opened.to_integral_value()) > Decimal("1e-6") or abs(defects - defects.to_integral_value()) > Decimal("1e-6")
            or int(defects.to_integral_value()) != ival(one(cs, dataset="DF", view_id="facet_only__family")["query_defect_rows"])):
        raise PaperError("the DF UNVERIFIED mass of family-3 does not split into drift defects and open anchors")
    c.add("fthreeopendf", count(int(opened.to_integral_value())), int(opened.to_integral_value()),
          ("verdicts/verdicts.csv", "registry/populations.json"), "view=family3__facet, dataset=DF, m^V_obs x dataset_rows")

    # Path B fibers and the exact checks of the Path B certificates
    pb = [r for r in vr if r["path"] == "B"]
    fibs = [ival(r["fibers"]) for r in pb]
    c.add("fibmin", count(min(fibs)), min(fibs), "verdicts/verdicts.csv", "min fibers over Path B view x dataset")
    c.add("fibmax", count(max(fibs)), max(fibs), "verdicts/verdicts.csv", "max fibers over Path B view x dataset")
    src = "scaling/scaling.csv"
    agg = out.csv(src)
    full = one(agg, fraction="1.0")
    c.add("pbsolved", count(full["sum_solver_fibers"]), full["sum_solver_fibers"], src, "fraction=1.0, sum_solver_fibers")
    c.add("pbrivals", count(full["sum_solver_rival_checks"]), full["sum_solver_rival_checks"], src,
          "fraction=1.0, sum_solver_rival_checks")
    c.add("pbfibers", count(full["sum_view_dataset_fibers"]), full["sum_view_dataset_fibers"], src,
          "fraction=1.0, sum_view_dataset_fibers")
    src = "verdicts/path_b_certificates.csv"
    pbx = out.csv(src)
    if not all(r["pass"] == "1" for r in pbx):
        raise PaperError("a Path B certificate check failed")
    enum = [r for r in pbx if r["verification_method"] == "exhaustive_integer_enumeration"]
    ana = [r for r in pbx if r["verification_method"] != "exhaustive_integer_enumeration"]
    c.add("pbunsafe", count(len(pbx)), len(pbx), src, "rows (all Path B UNSAFE fibers)")
    pbf = [r for r in pbx if r["view_id"] == "type3__family" and r["dataset"] == "FP"]
    c.add("pbtypefpcert", count(len(pbf)), len(pbf), src, "rows with view_id=type3__family, dataset=FP")
    # open single-label Path B fibers: counts from the summary and the public subset
    src_open = "verdicts/path_b_open_summary.json"
    pbs = out.json(src_open)
    full_t, pub_t, held_t = pbs["full_table"], pbs["public_table"], pbs["withheld_rows"]
    n_open = int(full_t["rows"])
    if (int(full_t["rows_passing_exact_check"]) != n_open or n_open + len(pbx) != ival(full["sum_solver_fibers"])
            or int(pub_t["rows"]) + int(held_t["rows"]) != n_open):
        raise PaperError("the open and the certified Path B fibers do not account for every single-label fiber")
    if set(full_t["by_program_status"]) - {"optimal", "unbounded"} or sum(full_t["by_program_status"].values()) != n_open:
        raise PaperError("an open Path B fiber has an unexpected program status")
    pbo = out.csv("verdicts/path_b_open_witnesses_public.csv")
    unb = [r for r in pbo if r["program_status"] == "unbounded"]
    if (len(pbo) != int(pub_t["rows"]) or not all(r["pass"] == "1" for r in pbo)
            or not all(r["free_coordinates_to_rival"] != "0" for r in unb)
            or any(r["dataset"] == "DF" and r["query_conforming_fiber_size"] == "1" for r in pbo)):
        raise PaperError("the public open-witness table disagrees with its summary or its rule")
    c.add("pbopen", count(n_open), n_open, src_open, "full_table.rows (single-label Path B fibers that stay open)")
    c.add("pbxenum", count(len(enum)), len(enum), src, "rows with verification_method=exhaustive_integer_enumeration")
    c.add("pbxanalytic", count(len(ana)), len(ana), src, "rows with verification_method=analytic_unbounded_category_only")
    ft = [r for r in pbx if r["view_id"] == "family3_plus_type__facet"]
    c.add("pbxfthreet", count(len(ft)), len(ft), src, "rows with view_id=family3_plus_type__facet")
    st = sum(ival(r["integer_states_checked"]) for r in enum)
    c.add("pbxstates", count(st), st, src, "sum of integer_states_checked over enumerated fibers")
    # every UNSAFE Path B fiber of the full-size scaling run is covered
    p14s = out.csv("scaling/scaling_summary.csv")
    u100 = sum(ival(r["unsafe_fibers"]) for r in select(p14s, fraction="1.0"))
    if u100 != len(pbx):
        raise PaperError(f"{len(pbx)} certificates checked but {u100} UNSAFE Path B fibers at full size")
    fv: dict[str, tuple[int, Decimal]] = {}
    for r in select(p14s, fraction="1.0"):
        f_, t_ = fv.get(r["view_id"], (0, Decimal(0)))
        fv[r["view_id"]] = (f_ + ival(r["single_label_fibers"]), t_ + dec(r["milp_seconds_median"]))
    top = max(fv, key=lambda k: fv[k][0])
    if top != "family3_plus_type__facet" or max(fv, key=lambda k: fv[k][1]) != top:
        raise PaperError("the family-3+type view is expected to carry the most Path B work")
    if sum(f for f, _ in fv.values()) != ival(full["sum_solver_fibers"]):
        raise PaperError("per-view single-label fibers do not add up")
    src3 = "scaling/scaling_summary.csv"
    c.add("pbtopfib", count(fv[top][0]), fv[top][0], src3,
          "fraction=1.0, view_id=family3_plus_type__facet, single_label_fibers (both datasets)")
    c.add("pbtopms", f"{(fv[top][1] * 1000).quantize(Decimal('1'))}", fv[top][1], src3,
          "fraction=1.0, view_id=family3_plus_type__facet, milp_seconds_median x1000 (both datasets)")
    tot = sum((t for _, t in fv.values()), Decimal(0))
    c.add("pbtotms", f"{(tot * 1000).quantize(Decimal('1'))}", tot, src3,
          "fraction=1.0, sum over views of milp_seconds_median x1000")
    # worked example of the extended audit details: the FP type view at values (0,2,0) and (1,2,0)
    reg = out.csv("registry/facet_map.csv")
    sfp = set(out.json("registry/dataset_support.json")["datasets"]["FP"]["S_members"])
    ncat = sum(1 for r in reg if r["facet"] in sfp and r["term_type"] == "category")
    natt = sum(1 for r in reg if r["facet"] in sfp and r["term_type"] == "attribute")
    if any(r["family"] != "taxonomic_category" for r in reg if r["term_type"] == "category"):
        raise PaperError("a category facet is mapped outside the taxonomic family")
    ex = one(pbx, view_id="type3__family", dataset="FP", fiber_value="[0,2,0]")
    if ival(ex["integer_states_checked"]) != comb(ncat + 1, 2):
        raise PaperError("worked example: enumerated states differ from the multiset count")
    ex2 = one(pbx, view_id="type3__family", dataset="FP", fiber_value="[1,2,0]")
    if ival(ex2["integer_states_checked"]) != natt * comb(ncat + 1, 2):
        raise PaperError("worked example: enumerated states of (1,2,0) differ from the count")
    src2 = ("registry/facet_map.csv", "registry/dataset_support.json")
    c.add("ncatfp", count(ncat), ncat, src2, "category facets in S_FP")
    c.add("nattfp", count(natt), natt, src2, "attribute facets in S_FP")
    c.add("pbxexstates", count(ex["integer_states_checked"]), ex["integer_states_checked"], src,
          "view_id=type3__family, dataset=FP, fiber_value=[0,2,0], integer_states_checked")
    c.add("pbxexstatestwo", count(ex2["integer_states_checked"]), ex2["integer_states_checked"], src,
          "view_id=type3__family, dataset=FP, fiber_value=[1,2,0], integer_states_checked")
    c.add("pbxexmargin", count(ex["minimum_strict_margin"]), ex["minimum_strict_margin"], src,
          "view_id=type3__family, dataset=FP, fiber_value=[0,2,0], minimum_strict_margin")
    c.add("pbxexmargintwo", count(ex2["minimum_strict_margin"]), ex2["minimum_strict_margin"], src,
          "view_id=type3__family, dataset=FP, fiber_value=[1,2,0], minimum_strict_margin")


# ---------------------------------------------------------------- Fig. 1 example
def _fig1(c: Cells, out: Outputs) -> None:
    src = "verdicts/fig1_example.json"
    f1 = out.json(src)
    sel = f1["selected"]
    if f1.get("selection_uses_label") is not False:
        raise PaperError("the Fig. 1 selection must not read a label")
    if set(sel["facet_nonzero"]) != set(FIG1_FACETS):
        raise PaperError(f"the Fig. 1 anchor has facets {sorted(sel['facet_nonzero'])}, expected {sorted(FIG1_FACETS)}")
    if set(sel["declared_W_family"]) != set(FIG1_FAMILIES):
        raise PaperError(f"the Fig. 1 anchor has families {sorted(sel['declared_W_family'])}")
    c.add("figcand", count(f1["candidate_rows"]), f1["candidate_rows"], src, "candidate_rows")
    for key, cid in FIG1_FACETS.items():
        v = ival(sel["facet_nonzero"][key])
        c.add(cid, NUMBER_WORDS[v] if v < 10 else str(v), v, src, f"selected.facet_nonzero[{key}] (spelled out)")
    for key, cid in FIG1_FAMILIES.items():
        v = ival(sel["declared_W_family"][key])
        c.add(cid, str(v), v, src, f"selected.declared_W_family[{key}]")


# ---------------------------------------------------------------- probes and calibration
def _probes(c: Cells, out: Outputs) -> None:
    src = "utility/probes.csv"
    pr = [r for r in out.csv(src) if r["status"] == "OK"]

    def probe(view, direction, name, protocol, scope):
        return one(pr, view_id=view, direction=direction, probe=name, training_protocol=protocol,
                   evaluation_scope=scope)

    for key in ("iafam", "ctfam", "fofam", "tthree", "ttwo", "iafac", "fthree", "fthreet"):
        view = VIEW_KEYS[key]
        for dk, dname in DIRECTIONS.items():
            for pname, pk in (("SrcMaj", "sm"), ("lookup", "lk"), ("cosine", "cos")):
                hit = select(pr, view_id=view, direction=dname, probe=pname, training_protocol=REFIT,
                             evaluation_scope="full_conform_target")
                if hit:
                    r = one(hit)
                    c.add(f"{pk}r{key}{dk}", frac(r["macro_f1"], 4), r["macro_f1"], src,
                          f"{view} {dname} {pname} {REFIT} full_conform_target, macro_f1")
            hit = select(pr, view_id=view, direction=dname, probe="cosine", training_protocol=ALL_SOURCE,
                         evaluation_scope="all_supported_target")
            if hit:
                r = one(hit)
                c.add(f"cosh{key}{dk}", frac(r["macro_f1"], 4), r["macro_f1"], src,
                      f"{view} {dname} cosine {ALL_SOURCE} all_supported_target, macro_f1")
    r = probe("identity_admitted__family", "FP->DF", "cosine", ALL_SOURCE, "full_conform_target")
    c.add("xsrc", frac(r["macro_f1"], 3), r["macro_f1"], src,
          f"identity_admitted__family FP->DF cosine {ALL_SOURCE} full_conform_target, macro_f1")
    r = probe("identity_admitted__family", "FP->DF", "SrcMaj", ALL_SOURCE, "full_conform_target")
    c.add("tsmhiafamfpdf", frac(r["macro_f1"], 3), r["macro_f1"], src,
          f"identity_admitted__family FP->DF SrcMaj {ALL_SOURCE} full_conform_target, macro_f1 [text]")
    r = probe("identity_admitted__family", "FP->DF", "cosine", REFIT, "full_conform_target")
    c.add("xrefit", frac(r["macro_f1"], 3), r["macro_f1"], src,
          f"identity_admitted__family FP->DF cosine {REFIT} full_conform_target, macro_f1")
    # shared coordinates of the fit: all source anchors (and the volume-matched draws) versus the refit
    r = probe("identity_admitted__family", "FP->DF", "cosine", ALL_SOURCE, "full_conform_target")
    c.add("shfhiafamfpdf", count(ival(r["shared_feature_count"])), ival(r["shared_feature_count"]), src,
          f"identity_admitted__family FP->DF cosine {ALL_SOURCE} full_conform_target, shared_feature_count")
    r = probe("identity_admitted__family", "FP->DF", "cosine", REFIT, "full_conform_target")
    c.add("shfriafamfpdf", count(ival(r["shared_feature_count"])), ival(r["shared_feature_count"]), src,
          f"identity_admitted__family FP->DF cosine {REFIT} full_conform_target, shared_feature_count")
    # the same-target comparison (all source anchors vs registry-conforming source anchors) in four settings
    for key in ("iafam", "iafac"):
        view = VIEW_KEYS[key]
        for dk, dname in DIRECTIONS.items():
            r = probe(view, dname, "cosine", ALL_SOURCE, "full_conform_target")
            c.add(f"stsrc{key}{dk}", frac(r["macro_f1"], 3), r["macro_f1"], src,
                  f"{view} {dname} cosine {ALL_SOURCE} full_conform_target, macro_f1")
            r = probe(view, dname, "cosine", REFIT, "full_conform_target")
            c.add(f"strefit{key}{dk}", frac(r["macro_f1"], 3), r["macro_f1"], src,
                  f"{view} {dname} cosine {REFIT} full_conform_target, macro_f1 [text, 3 digits]")
    # tolerance sensitivity: identical predictions for every view and direction at every tolerance setting
    tsrc = "utility/tolerance_sensitivity.csv"
    tol = out.csv(tsrc)
    tg: dict[tuple[str, str], set[str]] = {}
    for x in tol:
        tg.setdefault((x["view_id"], x["direction"]), set()).add(x["prediction_hash"])
    if not all(len(h) == 1 for h in tg.values()):
        raise PaperError("a prediction changes with the tolerance")
    ntols = len({x["score_eps"] for x in tol})
    c.add("tolpairs", count(len(tg)), len(tg), tsrc, "distinct (view_id, direction) with identical prediction_hash")
    c.add("tolsettingsw", NUMBER_WORDS[ntols], ntols, tsrc, "distinct score_eps settings [spelled out]")
    # the refit fits on the registry-conforming source anchors with a candidate label, for every probe and view
    refit_fit: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for x in pr:
        if x["training_protocol"] == REFIT:
            refit_fit.setdefault((x["direction"], x["task"]), set()).add(
                (ival(x["fitting_mask_rows"]), ival(x["source_rows_fit"])))
    for key, vals in refit_fit.items():
        if len(vals) != 1:
            raise PaperError(f"refit fitting counts differ across probes for {key}: {sorted(vals)}")
    for dk, dname in DIRECTIONS.items():
        r = probe("type3__family", dname, "lookup", REFIT, "full_conform_target")
        c.add(f"lktthreetxt{dk}", frac(r["macro_f1"], 3), r["macro_f1"], src, f"type3__family {dname} lookup {REFIT}, macro_f1")
        r = probe("type3__family", dname, "cosine", REFIT, "full_conform_target")
        c.add(f"costthreetxt{dk}", frac(r["macro_f1"], 3), r["macro_f1"], src, f"type3__family {dname} cosine {REFIT}, macro_f1")
        r = probe("family3__facet", dname, "cosine", ALL_SOURCE, "all_supported_target")
        c.add(f"cosfthreetxt{dk}", frac(r["macro_f1"], 3), r["macro_f1"], src,
              f"family3__facet {dname} cosine {ALL_SOURCE}, macro_f1")
        r = probe("family3__facet", dname, "lookup", ALL_SOURCE, "all_supported_target")
        c.add(f"lkfthreetxt{dk}", frac(r["macro_f1"], 3), r["macro_f1"], src,
              f"family3__facet {dname} lookup {ALL_SOURCE}, macro_f1")
    for dname in DIRECTIONS.values():
        r = probe("type_attr_part__family", dname, "cosine", REFIT, "full_conform_target")
        if dec(r["abstain_rate"]) != 1:
            raise PaperError("cosine on type{attribute, part} is expected to abstain on every target")
    for dk, dname in DIRECTIONS.items():
        for task in ("facet", "family"):
            view = {"facet": "family3__facet", "family": "type3__family"}[task]
            r = probe(view, dname, "SrcMaj", ALL_SOURCE, "all_supported_target")
            rr = probe(view, dname, "SrcMaj", REFIT, "full_conform_target")
            t = task[:3]
            c.add(f"cand{t}{dk}", count(r["source_label_candidates"]), r["source_label_candidates"], src,
                  f"{task} {dname} source_label_candidates")
            c.add(f"metric{t}{dk}", count(r["metric_labels"]), r["metric_labels"], src, f"{task} {dname} metric_labels")
            c.add(f"neval{t}{dk}", count(r["target_eval_rows"]), r["target_eval_rows"], src,
                  f"{task} {dname} {ALL_SOURCE} all_supported_target, target_eval_rows")
            c.add(f"nevalr{t}{dk}", count(rr["target_eval_rows"]), rr["target_eval_rows"], src,
                  f"{task} {dname} {REFIT} full_conform_target, target_eval_rows")
            share = Decimal(ival(rr["target_eval_rows"])) / ival(r["target_eval_rows"])
            c.add(f"confshare{t}{dk}", frac(share, 4), share, src, f"{task} {dname} refit / all-source target_eval_rows")
    for key, view in (("iafam", "identity_admitted__family"), ("iafac", "identity_admitted__facet")):
        r = probe(view, "DF->FP", "cosine", REFIT, "full_conform_target")
        c.add(f"dim{key}", count(r["feature_count"]), r["feature_count"], src, f"{view} DF->FP cosine feature_count")

    src = "utility/calibration.csv"
    cal = out.csv(src)
    for dk, (s_, t_) in (("dffp", ("DF", "FP")), ("fpdf", ("FP", "DF"))):
        for task in ("facet", "family"):
            r = one(cal, source=s_, target=t_, task=task)
            c.add(f"uni{task[:3]}{dk}", frac(r["uniform_macro_f1"], 4), r["uniform_macro_f1"], src,
                  f"{s_}->{t_} {task} uniform_macro_f1")
            c.add(f"smh{task[:3]}{dk}", frac(r["srcmaj_macro_f1"], 4), r["srcmaj_macro_f1"], src,
                  f"{s_}->{t_} {task} srcmaj_macro_f1")
    if len({r["n_draws"] for r in cal}) != 1:
        raise PaperError("calibration settings differ in the number of draws")
    c.add("unidraws", count(cal[0]["n_draws"]), cal[0]["n_draws"], src, "n_draws")
    sdmax = max(dec(r["uniform_draw_sd"]) for r in cal)
    c.add("unisdmax", frac(sdmax, 5), sdmax, src, "max uniform_draw_sd over settings")

    c.refit_fit = refit_fit


# ---------------------------------------------------------------- bounds, determination bridge, combinations
def _bounds(c: Cells, out: Outputs) -> None:
    src = "utility/bounds.csv"
    bd = out.csv(src)
    for key in ("iafam", "ctfam", "fofam", "tthree", "ttwo", "iafac", "fthree", "fthreet"):
        view = VIEW_KEYS[key]
        for dk, dname in DIRECTIONS.items():
            r = one(bd, view_id=view, direction=dname, training_protocol=REFIT)
            c.add(f"ctwo{key}{dk}", frac(r["macro_f1_upper_bound_C2"], 4), r["macro_f1_upper_bound_C2"], src,
                  f"{view} {dname} {REFIT}, macro_f1_upper_bound_C2")
    r = one(bd, view_id="factorial_F1M1T1__facet", direction="FP->DF", training_protocol=REFIT)
    c.add("ceilcarrier", frac(r["accuracy_ceiling"], 3), r["accuracy_ceiling"], src,
          f"factorial_F1M1T1__facet FP->DF {REFIT}, accuracy_ceiling")
    r = one(bd, view_id="identity_admitted__facet", direction="FP->DF", training_protocol=REFIT)
    c.add("ceiladm", frac(r["accuracy_ceiling"], 3), r["accuracy_ceiling"], src,
          f"identity_admitted__facet FP->DF {REFIT}, accuracy_ceiling")
    src = "utility/probe_bound_check.csv"
    pbc = [r for r in out.csv(src) if r["probe"] in ("cosine", "SrcMaj")]
    if sum(ival(r["violation"]) for r in pbc):
        raise PaperError("a probe score exceeds its F1 bound")
    c.add("nboundchecks", count(len(pbc)), len(pbc), src,
          "rows (probe in cosine, SrcMaj; view, direction, protocol) checked against C2")
    src = "utility/support_check.csv"
    vst = out.csv(src)
    if not all(r["support_recompute_equal"] == "1" for r in vst):
        raise PaperError("the separate support computation disagrees")
    c.add("nvstchecks", count(len(vst)), len(vst), src, "rows with support_recompute_equal=1")

    src = "utility/cosine_determination_matrix.csv"
    cd = out.csv(src)
    for dk, dname in DIRECTIONS.items():
        r = one(cd, prediction_view_id="identity_admitted__family", determination_view_id="core_plus_type__family",
                direction=dname, training_protocol=REFIT)
        if r["correct_on_nondetermined"] != "0":
            raise PaperError("correct predictions on NONDETERMINED targets are expected to be zero under the refit")
        c.add(f"detutil{dk}", pct(r["fraction_correct_on_unsafe"]), r["fraction_correct_on_unsafe"], src,
              f"{dname} {REFIT} identity_admitted__family x core_plus_type__family, fraction_correct_on_unsafe")
        c.add(f"detutilrest{dk}", count(r["correct_on_unverified"]), r["correct_on_unverified"], src,
              f"{dname} {REFIT} identity_admitted__family x core_plus_type__family, correct_on_unverified")

    src = "utility/factorial_8cell.csv"
    fx = out.csv(src)
    widths = []
    crm = []
    for dk, dname in DIRECTIONS.items():
        for task in ("facet", "family"):
            vals = {cc: dec(one(fx, task=task, direction=dname, cell_id=cc)["macro_f1"]) for cc in EIGHT_CELLS}
            t = task[:3]
            for cc in EIGHT_CELLS:
                c.add(f"ex{t}{dk}{cc.lower()}", frac(vals[cc], 4), vals[cc], src, f"{task} {dname} cell_id={cc}, macro_f1")
            w = max(vals.values()) - min(vals.values())
            widths.append(w)
            c.add(f"width{t}{dk}", frac(w, 4), w, src, f"{task} {dname} max - min macro_f1 over eight cells")
            on, off = ("F1M1T1", "F0M1T1") if task == "facet" else ("F1M1T1", "F1M0T1")
            delta = vals[off] - vals[on]
            crm.append(delta)
            c.add(f"crm{t}{dk}", signed(delta, 4), delta, src, f"{task} {dname} {off} - {on}")
            ia_cell = "F0M1T1" if task == "facet" else "F1M0T1"
            delta = vals[ia_cell] - vals["F0M0T0"]
            c.add(f"cia{t}{dk}", signed(delta, 4), delta, src, f"{task} {dname} {ia_cell} - F0M0T0")
    c.add("crmlo", frac(-max(crm), 3), -max(crm), src, "smallest carrier-removal decrease over four settings")
    c.add("crmhi", frac(-min(crm), 3), -min(crm), src, "largest carrier-removal decrease over four settings")
    c.add("widthlo", frac(min(widths), 3), min(widths), src, "min eight-cell width over four settings")
    c.add("widthhi", frac(max(widths), 3), max(widths), src, "max eight-cell width over four settings")
    r1 = dec(one(fx, task="facet", direction="FP->DF", cell_id="F1M1T1")["macro_f1"])
    r0 = dec(one(fx, task="facet", direction="FP->DF", cell_id="F0M1T1")["macro_f1"])
    c.add("crmtxton", frac(r1, 3), r1, src, "facet FP->DF cell_id=F1M1T1, macro_f1")
    c.add("crmtxtoff", frac(r0, 3), r0, src, "facet FP->DF cell_id=F0M1T1, macro_f1")

    src = "utility/type_weight.csv"
    tw = out.csv(src)
    names = {Decimal("0.25"): "q", Decimal("0.5"): "h", Decimal("1"): "one", Decimal("2"): "two", Decimal("4"): "four"}
    tws = []
    for r in tw:
        tws.append(dec(r["macro_f1"]))
        c.add("tw" + names[dec(r["type_weight"])], frac(r["macro_f1"], 4), r["macro_f1"], src,
              f"type_weight={r['type_weight']}, macro_f1")
    c.add("twlo", frac(min(tws), 3), min(tws), src, "min macro_f1 over type weights")
    c.add("twhi", frac(max(tws), 3), max(tws), src, "max macro_f1 over type weights")

    # volume-matched random fits (diagnostic output)
    src = "utility/volume_control.csv"
    vc = out.csv(src)
    vv = [dec(r["macro_f1"]) for r in vc]
    c.add("volseeds", count(len(vv)), len(vv), src, "volume-matched seeds")
    c.add("vollo", frac(min(vv), 3), min(vv), src, "min macro_f1 over volume-matched seeds")
    c.add("volhi", frac(max(vv), 3), max(vv), src, "max macro_f1 over volume-matched seeds")
    if {ival(r["source_rows_fit"]) for r in vc} != {next(iter(c.refit_fit[("FP->DF", "family")]))[1]}:
        raise PaperError("volume-matched fits differ in size from the FP family refit")
    c.add("volfit", count(vc[0]["source_rows_fit"]), vc[0]["source_rows_fit"], src,
          "source_rows_fit (equals the FP family refit fit)")


# ---------------------------------------------------------------- partition identity, simpler checks, null
def _controls(c: Cells, out: Outputs) -> None:
    src = "utility/partition_identity.csv"
    pi_ = out.csv(src)
    for d, k in DS_KEY:
        rows = select(pi_, dataset=d, scope="FULL_CONFORM")
        if len(rows) != 8 or not all(r["partition_equal"] == "1" for r in rows):
            raise PaperError("the eight-cell partition identity fails on registry-conforming anchors")
        c.add("fibfc" + k, count(rows[0]["core_fibers"]), rows[0]["core_fibers"], src,
              f"dataset={d}, scope=FULL_CONFORM, core_fibers")
        rows = select(pi_, dataset=d, scope="DRIFT")
        r0 = one(rows, cell_id="F0M0T0")
        c.add("fibdrcore" + k, count(r0["core_fibers"]), r0["core_fibers"], src, f"dataset={d}, scope=DRIFT, core_fibers")
        mx = max(ival(r["cell_fibers"]) for r in rows)
        c.add("fibdrmax" + k, count(mx), mx, src, f"dataset={d}, scope=DRIFT, max cell_fibers")
        eq = sum(ival(r["partition_equal"]) for r in rows)
        c.add("fibdreq" + k, count(eq), eq, src, f"dataset={d}, scope=DRIFT, cells with partition_equal=1")

    src = "utility/simple_checks_values.csv"
    mc = out.csv(src)
    for cid, key, fmt in (("lzrate", "l0_rate", "int"), ("lzhit", "l0_hit", "int"), ("lzhitsan", "l0_hit_san", "int"),
                          ("fdgthreedf", "fd_g3_df", "int"), ("fdgthreefp", "fd_g3_fp", "int"),
                          ("fdsingledf", "fd_single_df", "f4"), ("fdsinglefp", "fd_single_fp", "f4"),
                          ("instfamcov", "instfam_cov", "f4"), ("instfamn", "instfam_n", "count")):
        r = one(mc, cell_id=key)
        disp = {"int": lambda x: str(ival(x)), "f4": lambda x: frac(x, 4), "count": count}[fmt](r["value"])
        c.add(cid, disp, r["value"], src, f"cell_id={key}")
    src = "utility/simple_checks_details.csv"
    bdl = out.csv(src)
    g3all = [dec(r["value"]) for r in bdl if r["metric"] == "g3_row_deletion_error" and r["setting"].endswith("|ALL")]
    if len(g3all) != 4:
        raise PaperError("expected four ALL-scope g3 rows")
    c.add("fdgthreeallmax", frac(max(g3all), 4), max(g3all), src,
          "max g3_row_deletion_error over the four ALL-scope settings")
    for task, cid in (("facet", "nmifac"), ("family", "nmifam")):
        r = one(bdl, baseline="pooled_correlation_screen", setting=f"ALL|{task}", metric="NMI")
        c.add(cid, frac(r["value"], 4), r["value"], src, f"baseline=pooled_correlation_screen, setting=ALL|{task}, NMI")
    within = [dec(r["value"]) for r in bdl if r["metric"] == "within_dataset_NMI"]
    if len(within) != 4 or any(x != 0 for x in within):
        raise PaperError("the within-dataset NMI is expected to be zero")
    c.add("nmiwithin", str(int(max(within))), max(within), src, "max within_dataset_NMI over four dataset-task cells")

    src = "utility/permutation.csv"
    nl = out.csv(src)
    for dk, dname in DIRECTIONS.items():
        r = one(nl, direction=dname)
        c.add(f"nullp{dk}", frac(r["p_value"], 3), r["p_value"], src, f"direction={dname}, p_value")
        c.add(f"nullmax{dk}", frac(r["null_max"], 3), r["null_max"], src, f"direction={dname}, null_max")
        c.add(f"nullk{dk}", count(r["k_ge_observed"]), r["k_ge_observed"], src, f"direction={dname}, k_ge_observed")
    c.add("nullb", count(nl[0]["B"]), nl[0]["B"], src, "B")


# ---------------------------------------------------------------- scaling and reference run times
def _scaling(c: Cells, out: Outputs) -> None:
    src = "scaling/scaling.csv"
    agg = out.csv(src)
    lo, hi, nine = one(agg, fraction="0.1"), one(agg, fraction="1.0"), one(agg, fraction="0.9")
    c.add("scrowslo", count(lo["sampled_rows_total"]), lo["sampled_rows_total"], src, "fraction=0.1, sampled_rows_total")
    c.add("scrowshi", count(hi["sampled_rows_total"]), hi["sampled_rows_total"], src, "fraction=1.0, sampled_rows_total")
    c.add("sctlo", secs(lo["sum_total_seconds_median"], 2), lo["sum_total_seconds_median"], src,
          "fraction=0.1, sum_total_seconds_median (reference run)")
    c.add("scthi", secs(hi["sum_total_seconds_median"], 2), hi["sum_total_seconds_median"], src,
          "fraction=1.0, sum_total_seconds_median (reference run)")
    for cid, col, what in (("scgratio", "sum_grouping_seconds_median", "grouping"),
                           ("scsratio", "sum_milp_seconds_median", "integer-program"),
                           ("sctratio", "sum_total_seconds_median", "total")):
        c.add(cid, ratio(hi[col], lo[col], 2), dec(hi[col]) / dec(lo[col]), src,
              f"{what} median ratio fraction=1.0 / fraction=0.1, {col} (reference run)")
    c.add("scsfiblo", count(lo["sum_solver_fibers"]), lo["sum_solver_fibers"], src, "fraction=0.1, sum_solver_fibers")
    c.add("scfiblo", count(lo["sum_view_dataset_fibers"]), lo["sum_view_dataset_fibers"], src,
          "fraction=0.1, sum_view_dataset_fibers")
    c.add("scrivlo", count(lo["sum_solver_rival_checks"]), lo["sum_solver_rival_checks"], src,
          "fraction=0.1, sum_solver_rival_checks")
    c.add("scrivnine", count(nine["sum_solver_rival_checks"]), nine["sum_solver_rival_checks"], src,
          "fraction=0.9, sum_solver_rival_checks")
    per = dec(hi["sum_milp_seconds_median"]) / ival(hi["sum_solver_rival_checks"]) * 1000
    c.add("scmsper", secs(per, 2), per, src,
          "fraction=1.0, sum_milp_seconds_median / sum_solver_rival_checks, in ms (reference run)")
    p14s = out.csv("scaling/scaling_summary.csv")
    reps = {ival(r["repeats"]) for r in p14s}
    if len(reps) != 1:
        raise PaperError("repeat counts differ across scaling rows")
    c.add("screps", count(next(iter(reps))), next(iter(reps)), "scaling/scaling_summary.csv", "repeats")

    src = "scaling/reference_times.json"
    rt = out.json(src)
    sec = rt["seconds"]
    for cid, key in (("rtfc", "full_conform"), ("rtqc", "query_conform"), ("rtaudit", "verdicts_and_mutations")):
        c.add(cid, secs(sec[key], 1), dec(sec[key]), src, f"seconds.{key}")
    tot = _sum_dec(sec[k] for k in ("full_conform", "query_conform", "verdicts_and_mutations"))
    if abs(tot - dec(rt["total_seconds"])) > Decimal("1e-6"):
        raise PaperError("reference step times do not add up to total_seconds")
    c.add("rttotal", secs(tot, 1), tot, src, "sum of seconds.full_conform, seconds.query_conform, "
          "seconds.verdicts_and_mutations")
    c.add("rtcpu", count(rt["machine"]["cpu_count"]), rt["machine"]["cpu_count"], src, "machine.cpu_count")
    for k, cid in (("python", "verpython"), ("scipy", "verscipy"), ("numpy", "vernumpy"), ("pandas", "verpandas")):
        c.add(cid, rt["software"][k], rt["software"][k], src, f"software.{k}")


# ---------------------------------------------------------------- external benchmark
def _external(c: Cells, out: Outputs) -> None:
    src = "external/query_results.csv"
    eq = out.csv(src)
    tot = {"nonempty_rows": 0, "L0_direct_rows": 0, "L1_exact_derived_rows": 0, "unverified_ambiguous_rows": 0,
           "unverified_no_match_rows": 0}
    for i, r in enumerate(sorted(eq, key=lambda r: r["query_id"]), 1):
        q = "q" + "abcdef"[i - 1]
        for col, cid in (("nonempty_rows", "ne"), ("L0_direct_rows", "lz"), ("L1_exact_derived_rows", "lone"),
                         ("unverified_no_match_rows", "unv")):
            c.add(f"ext{q}{cid}", count(r[col]), r[col], src, f"query_id={r['query_id']}, {col}")
        c.add(f"ext{q}lzr", frac(r["L0_direct_rate_nonempty"], 4), r["L0_direct_rate_nonempty"], src,
              f"query_id={r['query_id']}, L0_direct_rate_nonempty")
        c.add(f"ext{q}loner", frac(r["L1_exact_derived_rate_nonempty"], 4), r["L1_exact_derived_rate_nonempty"], src,
              f"query_id={r['query_id']}, L1_exact_derived_rate_nonempty")
        c.add(f"ext{q}cols", count(r["source_columns"]), r["source_columns"], src, f"query_id={r['query_id']}, source_columns")
        for k in tot:
            tot[k] += ival(r[k])
    c.add("extne", count(tot["nonempty_rows"]), tot["nonempty_rows"], src, "sum of nonempty_rows")
    c.add("extlz", count(tot["L0_direct_rows"]), tot["L0_direct_rows"], src, "sum of L0_direct_rows")
    c.add("extlone", count(tot["L1_exact_derived_rows"]), tot["L1_exact_derived_rows"], src, "sum of L1_exact_derived_rows")
    c.add("extamb", count(tot["unverified_ambiguous_rows"]), tot["unverified_ambiguous_rows"], src,
          "sum of unverified_ambiguous_rows")
    c.add("extunv", count(tot["unverified_no_match_rows"]), tot["unverified_no_match_rows"], src,
          "sum of unverified_no_match_rows")
    l0r = Decimal(tot["L0_direct_rows"]) / tot["nonempty_rows"]
    l1r = Decimal(tot["L1_exact_derived_rows"]) / tot["nonempty_rows"]
    c.add("extlzr", pct(l0r), l0r, src, "sum of L0_direct_rows / sum of nonempty_rows")
    c.add("extloner", pct(l1r), l1r, src, "sum of L1_exact_derived_rows / sum of nonempty_rows")
    c.add("extqueries", count(len(eq)), len(eq), src, "registered queries")
    thr_rows = {r["nontrivial_L1_min_rows"] for r in eq}
    thr_rate = {r["nontrivial_L1_min_rate"] for r in eq}
    if len(thr_rows) != 1 or len(thr_rate) != 1:
        raise PaperError("external thresholds differ across queries")
    c.add("extminrows", count(next(iter(thr_rows))), next(iter(thr_rows)), src, "nontrivial_L1_min_rows")
    c.add("extminrate", frac(next(iter(thr_rate)), 2), next(iter(thr_rate)), src, "nontrivial_L1_min_rate")
    src = "external/domain_results.csv"
    ed = out.csv(src)
    q = sum(ival(r["domain_qualifies"]) for r in ed)
    bound = sum(1 for r in ed if ival(r["registered_queries"]) > 0)
    c.add("extdomq", count(q), q, src, "domains qualifying for the breadth gate")
    c.add("extdombound", count(bound), bound, src, "domains with a registered taxonomy transform")
    c.add("extdomall", count(len(ed)), len(ed), src, "domains in the benchmark")
    src = "external/carrier_registry.csv"
    cr = out.csv(src)
    c.add("extcarriers", count(len(cr)), len(cr), src, "rows")
    src = "external/independent_checks.csv"
    ic = out.csv(src)
    if sum(ival(r["exact_match"]) for r in ic) != len(ic):
        raise PaperError("the second implementation of the external audit disagrees")
    c.add("extchecks", count(len(ic)), len(ic), src, "rows with exact_match=1")
    src = "external/protocol.json"
    commit = out.json(src)["madi_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PaperError("the external benchmark commit is not a full commit id")
    c.add("extcommit", commit[:7], commit, src, "madi_commit (first 7 hex digits)")


def _external_gold(c: Cells, out: Outputs) -> None:
    src = "external/gold_check.csv"
    rows = out.csv(src)
    if any(ival(r["independent_match"]) != 1 for r in rows):
        raise PaperError("the second implementation of the gold comparison disagrees")
    for cid, k in (("extgoldcells", "gold_cells"), ("extgoldlz", "L0_direct"), ("extgoldlone", "L1_exact_derived"),
                   ("extgoldcls", "classified"), ("extgoldcontra", "contradictions"),
                   ("extgoldstrict", "complete_set_equal"), ("extgoldmulti", "component_only"),
                   ("extgoldunv", "unverified_no_match"), ("extgoldunvnorm", "unverified_gold_normalization")):
        v = sum(ival(r[k]) for r in rows)
        c.add(cid, count(v), v, src, f"sum of {k}")


# ---------------------------------------------------------------- mutation checks
def _mutations(c: Cells, out: Outputs) -> None:
    nview = len({r["view_id"] for r in out.csv("conformance/conformance_summary.csv")})
    nall = ival(out.json("registry/populations.json")["n_anchor_total"])
    src = "mutations/mutations.json"
    doc = out.json(src)
    mu = doc["mutations"]
    if not all(mu[m]["pass"] for m in ("M1", "M2", "M3", "M4", "M5")):
        raise PaperError("a mutation check failed")
    m2, m5 = mu["M2"], mu["M5"]
    c.add("mtwoids", count(m2["selected_count"]), m2["selected_count"], src, "mutations.M2.selected_count")
    c.add("mtworows", count(m2["rows_containing_selected_uid"]), m2["rows_containing_selected_uid"], src,
          "mutations.M2.rows_containing_selected_uid")
    c.add("mtwonew", count(m2["newly_failed_previously_full_conform_rows_audited"]),
          m2["newly_failed_previously_full_conform_rows_audited"], src,
          "mutations.M2.newly_failed_previously_full_conform_rows_audited")
    if m2["newly_failed_previously_full_conform_rows_audited"] != m2["newly_failed_previously_full_conform_rows_expected"]:
        raise PaperError("M2 finds a different set of new FULL-CONFORM failures")
    c.add("mfivepairs", count(m5["invalid_baseline_query_rows_across_view_datasets"]),
          m5["invalid_baseline_query_rows_across_view_datasets"], src,
          "mutations.M5.invalid_baseline_query_rows_across_view_datasets")
    if m5["unsafe_certificates_on_invalid_rows"] != 0 or m5["recomputed_certificate_label_conflicts"] != 0:
        raise PaperError("the M5 fail-closed check failed")
    conf = [ival(x["certificate_label_conflicts"]) for x in m5["failure_capable_negative_control"]["checks"]]
    c.add("mfivestalelo", count(min(conf)), min(conf), src,
          "M5 stale-mask control: min certificate_label_conflicts over four settings")
    c.add("mfivestalehi", count(max(conf)), max(conf), src,
          "M5 stale-mask control: max certificate_label_conflicts over four settings")
    src = "mutations/m5_label_domain_check.csv"
    m5a = out.csv(src)
    if not all(r["pass"] == "1" and r["nonvacuous"] == "1" for r in m5a):
        raise PaperError("the M5 label-domain check failed")
    rows = [ival(r["rows_leaving_label_domain"]) for r in m5a]
    c.add("mfivearowslo", count(min(rows)), min(rows), src, "min rows_leaving_label_domain")
    c.add("mfivearowshi", count(max(rows)), max(rows), src, "max rows_leaving_label_domain")
    chk = doc["settings"]["m5_label_domain_check"]
    vchk = out.csv("mutations/m5_view_checks.csv")
    if (chk["bits_checked"] != nview * nall or chk["baseline_component_mismatches"] != 0
            or chk["label_rule_vs_audit_mismatches"] != 0 or len(vchk) != 2 * nview
            or not all(r["pass"] == "1" for r in vchk)):
        raise PaperError("the M5 label-domain comparison does not cover every view and anchor bit for bit")
    c.add("mfiveabits", count(chk["bits_checked"]), chk["bits_checked"], "mutations/mutations.json",
          "settings.m5_label_domain_check.bits_checked")


# ---------------------------------------------------------------- verdict-class checks
def _verdict_checks(c: Cells, out: Outputs) -> None:
    src = "utility/verdict_checks.csv"
    vc = out.csv(src)
    for dk, dname in DIRECTIONS.items():
        for q, cid, fmt in (("acc_unsafe", "accu", "f3"), ("acc_unverified", "accv", "f3"),
                            ("base_rate_unsafe", "baseu", "f3"), ("targets_unsafe", "ntu", "count"),
                            ("targets_unverified", "ntv", "count")):
            r = one(vc, scope=dname, quantity=q)
            disp = frac(r["value"], 3) if fmt == "f3" else count(r["value"])
            c.add(cid + dk, disp, r["value"], src, f"scope={dname}, quantity={q}")
    mm = sum(ival(r["value"]) for r in vc if r["quantity"] == "factorial_verdict_mismatches_on_conforming")
    c.add("factverdictmm", count(mm), mm, src,
          "sum of factorial_verdict_mismatches_on_conforming over both tasks and datasets")
    src = "registry/facet_map.csv"
    rm = out.csv(src)
    cats = [r for r in rm if r["facet"].startswith("category/")]
    if not cats or any(r["family"] != "taxonomic_category" for r in cats):
        raise PaperError("W is expected to send every category facet to the taxonomic family")
    c.add("ncatfacets", count(len(cats)), len(cats), src, "category facets, all mapped to TAX by W")


# ---------------------------------------------------------------- registry rule
def _registry_rule(c: Cells, out: Outputs) -> None:
    src = "utility/registry_rule.csv"
    rr = out.csv(src)
    dec_tot, err_tot, ncells = 0, 0, 0
    for key in ("iafam", "ctfam", "fofam", "iafac"):
        view = VIEW_KEYS[key]
        for dk, dname in DIRECTIONS.items():
            r = one(rr, view_id=view, direction=dname)
            if r["pass"] != "1" or r["rule_le_beta"] != "1" or r["learned_parameters"] != "0":
                raise PaperError(f"registry-rule cell {view} {dname} did not pass")
            ncells += 1
            d, cc = ival(r["decided_rows"]), ival(r["correct_decided_rows"])
            sel = f"view_id={view}, direction={dname}"
            c.add(f"rule{key}{dk}", frac(r["macro_f1"], 4), r["macro_f1"], src, f"{sel}, macro_f1")
            c.add(f"rulecov{key}{dk}", pct(r["coverage"]), r["coverage"], src, f"{sel}, coverage")
            c.add(f"rulecovfr{key}{dk}", frac(r["coverage"], 4), r["coverage"], src, f"{sel}, coverage [fraction]")
            c.add(f"ruledec{key}{dk}", count(d), d, src, f"{sel}, decided_rows")
            c.add(f"rulecorr{key}{dk}", count(cc), cc, src, f"{sel}, correct_decided_rows")
            c.add(f"ruleerr{key}{dk}", count(d - cc), d - cc, src, f"{sel}, decided_rows - correct_decided_rows")
            prec = Decimal(cc) / Decimal(d)
            c.add(f"ruleprec{key}{dk}", "100\\%" if cc == d else pct(prec, 3), prec, src,
                  f"{sel}, correct_decided_rows / decided_rows")
            c.add(f"ruleprecfr{key}{dk}", frac(prec, 4), prec, src,
                  f"{sel}, correct_decided_rows / decided_rows [fraction]")
            if cc != d:
                # one digit unless rounding would print 100.0% for a rule with errors
                shown = pct(prec) if not pct(prec).startswith("100.0") else pct(prec, 3)
                c.add(f"ruleprect{key}{dk}", shown, prec, src,
                      f"{sel}, correct_decided_rows / decided_rows [text]")
            c.add(f"ruletxt{key}{dk}", frac(r["macro_f1"], 3), r["macro_f1"], src, f"{sel}, macro_f1 [text, 3 digits]")
            c.add(f"ruletgt{key}{dk}", count(r["target_eval_rows"]), r["target_eval_rows"], src, f"{sel}, target_eval_rows")
            c.add(f"ruleout{key}{dk}", count(r["predicted_outside_source_inventory"]),
                  r["predicted_outside_source_inventory"], src, f"{sel}, predicted_outside_source_inventory")
            c.add(f"ruledu{key}{dk}", frac(r["fraction_decided_that_is_unsafe"], 4), r["fraction_decided_that_is_unsafe"],
                  src, f"{sel}, fraction_decided_that_is_unsafe")
            c.add(f"ruleud{key}{dk}", frac(r["fraction_unsafe_decided"], 4), r["fraction_unsafe_decided"], src,
                  f"{sel}, fraction_unsafe_decided")
            c.add(f"ruleandu{key}{dk}", count(r["decided_and_unsafe_rows"]), r["decided_and_unsafe_rows"], src,
                  f"{sel}, decided_and_unsafe_rows")
            nonu = d - ival(r["decided_and_unsafe_rows"])
            c.add(f"rulenonu{key}{dk}", count(nonu), nonu, src, f"{sel}, decided_rows - decided_and_unsafe_rows")
            c.add(f"ruleunsafe{key}{dk}", count(r["target_unsafe_rows"]), r["target_unsafe_rows"], src,
                  f"{sel}, target_unsafe_rows")
            if key == "iafam":
                dec_tot += d
                err_tot += d - cc
    c.add("rulecells", count(ncells), ncells, src, "registry-rule cells (four Path A views x two directions)")
    c.add("ruledectot", count(dec_tot), dec_tot, src, "identity_admitted__family decided_rows, both directions")
    c.add("ruleerrtot", count(err_tot), err_tot, src,
          "identity_admitted__family decided_rows - correct_decided_rows, both directions")
    by_id = {x.cell_id: x for x in c.items}
    tgt_tot = ival(by_id["ruletgtiafamdffp"].raw) + ival(by_id["ruletgtiafamfpdf"].raw)
    c.add("ruletgttot", count(tgt_tot), tgt_tot, src, "identity_admitted__family target_eval_rows, both directions")
    src = "utility/registry_rule_fullview.csv"
    fv = out.csv(src)
    if len(fv) != 8 or not all(r["pass"] == "1" and dec(r["decided_equals_unsafe_rate"]) == 1
                               and r["decided_rows"] == r["decided_label_correct_rows"] for r in fv):
        raise PaperError("the full-view registry rule does not reproduce the UNSAFE verdicts")
    c.add("rulefullcells", count(len(fv)), len(fv), src,
          "view x dataset cells where decided equals UNSAFE and every decision is correct")
    qrows = sum(ival(r["query_conform_rows"]) for r in fv)
    c.add("rulefullrows", count(qrows), qrows, src, "sum of query_conform_rows over the eight cells")
    src = "utility/lookup_coverage.csv"
    lc = out.csv(src)
    for dk, dname in DIRECTIONS.items():
        r = one(lc, view_id="facet_only__family", direction=dname)
        places = 2 if dec(r["coverage"]) < Decimal("0.01") else 1
        c.add(f"lkcovfofam{dk}", pct(r["coverage"], places), r["coverage"], src,
              f"view_id=facet_only__family, direction={dname}, coverage")
    # the comparative sentences on the registry rule hold only if these orderings hold (the DF->FP lead is
    # qualified label by label in _evidence)
    for dname in DIRECTIONS.values():
        r = one(rr, view_id="identity_admitted__family", direction=dname)
        if not dec(r["macro_f1"]) > dec(r["cosine_refit_f1"]):
            raise PaperError("the identity-admitted family rule does not exceed the refit cosine probe")
        rf = one(rr, view_id="identity_admitted__facet", direction=dname)
        if not dec(rf["precision_on_decisions"]) < Decimal("0.9"):
            raise PaperError("the facet rule precision is not below .9")
    core = [dec(r["precision_on_decisions"]) for r in rr if r["basis"] == "core::"]
    facet = [dec(r["precision_on_decisions"]) for r in rr if r["basis"] == "facet::"]
    if len(core) != 4 or len(facet) != 4 or not max(core) < min(facet):
        raise PaperError("the core-basis rule precision does not stay below the facet-basis precision")
    c.add("ruleprecoremax", frac(max(core), 3), max(core), "utility/registry_rule.csv",
          "max of precision_on_decisions over basis=core:: (both directions)")


# ---------------------------------------------------------------- spreads and tie masses
def _spreads(c: Cells, out: Outputs) -> None:
    src = "utility/factorial_8cell.csv"
    fx = out.csv(src)
    exw, gaps = [], []
    for dname in DIRECTIONS.values():
        for task in ("facet", "family"):
            vals = {cc: dec(one(fx, task=task, direction=dname, cell_id=cc)["macro_f1"]) for cc in EIGHT_CELLS}
            exposes = (lambda cc: cc[1] == "1") if task == "facet" else (lambda cc: cc[3] == "1")
            best = max(vals, key=lambda cc: vals[cc])
            if not exposes(best):
                raise PaperError(f"the best combination of {task} {dname} does not expose the carrier")
            ex = [v for cc, v in vals.items() if not exposes(cc)]
            exw.append(max(ex) - min(ex))
            gap = min(v for cc, v in vals.items() if exposes(cc)) - max(ex)
            if gap <= 0:
                raise PaperError(f"a combination without the carrier reaches one with it ({task} {dname})")
            gaps.append(gap)
    c.add("crmexlo", frac(min(exw), 3), min(exw), src,
          "min over settings of max - min among combinations without the carrier")
    c.add("crmexhi", frac(max(exw), 3), max(exw), src,
          "max over settings of max - min among combinations without the carrier")
    c.add("crsepgaplo", floor_frac(min(gaps), 3), min(gaps), src,
          "min over settings of (min among combinations with the carrier - max among those without it) [rounded down]")
    src = "verdicts/verdicts.csv"
    vr = out.csv(src)
    for view, t in (("l0_facet__facet", "fac"), ("l0_family__family", "fam")):
        for d, k in DS_KEY:
            r = one(vr, view=view, dataset=d)
            tie = dec(r["m^N"]) + dec(r["m^V_obs"])
            c.add(f"ztie{t}{k}", frac(tie, 3), tie, src, f"view={view}, dataset={d}, m^N + m^V_obs")
    h = out.source_release()["source_files_sha256"]["generator specification"]
    c.add("hashgen", h[:8], h, "src/lca/paper/source_release.json",
          "source_files_sha256[generator specification] (first 8 hex digits)")


# ---------------------------------------------------------------- evidence step
def _evidence(c: Cells, out: Outputs) -> None:
    pr = [r for r in out.csv("utility/probes.csv") if r["status"] == "OK"]
    sel = dict(view_id="identity_admitted__family", direction="FP->DF", probe="cosine",
               evaluation_scope="full_conform_target")
    allsrc, refit = one(pr, training_protocol=ALL_SOURCE, **sel), one(pr, training_protocol=REFIT, **sel)

    # refit separation: rows versus shared coordinates (Sec. VI-B)
    src = "evidence/refit_separation.csv"
    case = {r["case"]: r for r in out.csv(src)}
    if set(case) != {"A", "B", "C", "D"}:
        raise PaperError("refit separation needs the cases A-D")
    if case["A"]["macro_f1"] != allsrc["macro_f1"] or case["C"]["macro_f1"] != refit["macro_f1"]:
        raise PaperError("refit separation does not reproduce the all-source and refit scores")
    if (case["A"]["shared_feature_count"] != allsrc["shared_feature_count"]
            or case["C"]["shared_feature_count"] != refit["shared_feature_count"]):
        raise PaperError("refit separation uses other shared coordinates than the utility step")
    dropped, carriers = ival(case["A"]["dropped_coordinates"]), ival(case["A"]["rows_carrying_dropped"])
    if (ival(case["A"]["shared_feature_count"]) - ival(case["C"]["shared_feature_count"]) != dropped
            or ival(case["A"]["drifted_rows_carrying_dropped"]) != carriers or carriers < 1):
        raise PaperError("the dropped shared coordinates are not carried by drifted rows only")
    if frac(case["B"]["macro_f1"], 3) != frac(case["A"]["macro_f1"], 3) or \
            frac(case["D"]["macro_f1"], 3) != frac(case["C"]["macro_f1"], 3):
        raise PaperError("the score moves with the shared coordinates, not only with the fitted rows")
    c.add("xnocoord", frac(case["B"]["macro_f1"], 3), case["B"]["macro_f1"], src,
          "case=B (all source rows, refit coordinates), macro_f1")
    c.add("xcoord", count(dropped), dropped, src, "case=A, dropped_coordinates")
    c.add("xcoordrows", count(carriers), carriers, src, "case=A, rows_carrying_dropped (all drifted)")

    # where the correct family-3 predictions fall (extended audit details, Section C)
    src = "evidence/verdict_class_composition.csv"
    comp = out.csv(src)
    for dk, dname in DIRECTIONS.items():
        r = one(comp, view_id="family3__facet", direction=dname, training_protocol=ALL_SOURCE,
                evaluation_scope="all_supported_target")
        p = one(pr, view_id="family3__facet", direction=dname, probe="cosine", training_protocol=ALL_SOURCE,
                evaluation_scope="all_supported_target")
        if r["macro_f1"] != p["macro_f1"] or ival(r["targets_unsafe"]) != 0:
            raise PaperError("the family-3 composition does not match the reported cosine score")
        c.add(f"fthreendcorr{dk}", frac(r["fraction_correct_on_nondetermined"], 3),
              r["fraction_correct_on_nondetermined"], src,
              f"view_id=family3__facet, direction={dname}, {ALL_SOURCE}, fraction_correct_on_nondetermined")

    # coordinates of a target that the shared core basis drops (extended audit details, Section C)
    src = "evidence/shared_basis_drop.csv"
    drop = out.csv(src)
    for dk, dname in DIRECTIONS.items():
        r = one(drop, basis="core::", direction=dname)
        if not dec(r["mean_fraction_coordinates_dropped"]) > Decimal("0.5"):
            raise PaperError("the shared core basis does not drop most coordinates of a target")
        c.add(f"coredrop{dk}", frac(r["mean_fraction_coordinates_dropped"], 3), r["mean_fraction_coordinates_dropped"],
              src, f"basis=core::, direction={dname}, mean_fraction_coordinates_dropped")

    # 999 random draws with the label counts of the refit (diagnostic output)
    src = "evidence/volume_null.csv"
    null = out.csv(src)
    vals = [dec(r["macro_f1"]) for r in null]
    control = {r["seed"]: r["macro_f1"] for r in out.csv("utility/volume_control.csv")}
    if any(one(null, seed=s)["macro_f1"] != v for s, v in control.items()):
        raise PaperError("the first draws of the volume null differ from the volume control")
    if {r["source_rows_fit"] for r in null} != {refit["source_rows_fit"]} or not dec(refit["macro_f1"]) < min(vals):
        raise PaperError("the volume null draws differ in size from the refit or reach the refit score")
    c.add("volnulldraws", count(len(vals)), len(vals), src, "rows (random draws)")
    c.add("volnulllo", frac(min(vals), 3), min(vals), src, "min macro_f1 over the draws")
    c.add("volnullhi", frac(max(vals), 3), max(vals), src, "max macro_f1 over the draws")

    # Macro-F1 label by label (Sec. VI-B; extended audit details, Section C)
    src = "evidence/metric_label_sensitivity.csv"
    ms = out.csv(src)
    src_l = "evidence/metric_label_f1.csv"
    ml = out.csv(src_l)
    rule = one(ms, view_id="identity_admitted__family", direction="DF->FP", predictor="registry_rule")
    cos = one(ms, view_id="identity_admitted__family", direction="DF->FP", predictor="cosine")
    if not (dec(rule["macro_f1"]) > dec(cos["macro_f1"])
            and dec(cos["macro_f1_on_labels_with_min_targets"]) > dec(rule["macro_f1_on_labels_with_min_targets"])
            and ival(rule["labels_with_min_targets"]) == ival(rule["metric_labels"]) - 1):
        raise PaperError("the DF->FP rule lead is not carried by one small metric label")
    for dname in DIRECTIONS.values():
        a = one(ms, view_id="identity_admitted__family", direction=dname, predictor="registry_rule")
        if a["macro_f1"] != one(out.csv("utility/registry_rule.csv"), view_id="identity_admitted__family",
                                direction=dname)["macro_f1"]:
            raise PaperError("label-by-label rule scores differ from the registry-rule table")
    fp = one(ms, view_id="identity_admitted__family", direction="FP->DF", predictor="registry_rule")
    fc = one(ms, view_id="identity_admitted__family", direction="FP->DF", predictor="cosine")
    if not (ival(fp["labels_below_small_targets"]) == 0
            and dec(fp["macro_f1_on_labels_with_min_targets"]) > dec(fc["macro_f1_on_labels_with_min_targets"])):
        raise PaperError("the FP->DF rule lead does not hold on labels with enough targets")
    c.add("rulethreeiafamdffp", frac(rule["macro_f1_on_labels_with_min_targets"], 3),
          rule["macro_f1_on_labels_with_min_targets"], src,
          "identity_admitted__family DF->FP registry_rule, macro_f1_on_labels_with_min_targets")
    c.add("costhreeiafamdffp", frac(cos["macro_f1_on_labels_with_min_targets"], 3),
          cos["macro_f1_on_labels_with_min_targets"], src,
          "identity_admitted__family DF->FP cosine, macro_f1_on_labels_with_min_targets")
    ra = one(ml, view_id="identity_admitted__family", direction="DF->FP", predictor="registry_rule",
             metric_label="attribute_expression")
    ca = one(ml, view_id="identity_admitted__family", direction="DF->FP", predictor="cosine",
             metric_label="attribute_expression")
    if not (ival(ra["targets"]) == ival(ra["predicted"]) == ival(ra["correct"]) == 1 == ival(ca["correct"])):
        raise PaperError("the rule is expected to assign ATTR to the one ATTR target alone")
    c.add("attrfptargets", count(ra["targets"]), ra["targets"], src_l,
          "identity_admitted__family DF->FP registry_rule attribute_expression, targets")
    c.add("cosattrpred", count(ca["predicted"]), ca["predicted"], src_l,
          "identity_admitted__family DF->FP cosine attribute_expression, predicted")
    small = {(r["view_id"], r["direction"]) for r in ms if ival(r["labels_below_small_targets"]) > 0}
    expected = {("identity_admitted__facet", "FP->DF")} | {(v, "DF->FP") for v in
                                                          ("identity_admitted__family", "facet_only__family",
                                                           "core_plus_type__family")}
    if small != expected:
        raise PaperError(f"metric labels with few targets occur in other settings: {sorted(small)}")
    r = one(ms, view_id="identity_admitted__facet", direction="FP->DF", predictor="cosine")
    sizes = sorted(int(x.split("=")[1]) for x in r["small_label_targets"].split("|"))
    c.add("nsmallfac", count(r["labels_below_small_targets"]), r["labels_below_small_targets"], src,
          "identity_admitted__facet FP->DF, labels_below_small_targets (fewer than ten targets)")
    c.add("smallfaclo", count(min(sizes)), min(sizes), src, "identity_admitted__facet FP->DF, fewest targets of a label")
    c.add("smallfachi", count(max(sizes)), max(sizes), src,
          "identity_admitted__facet FP->DF, most targets among the labels with fewer than ten")
    rf = one(ms, view_id="identity_admitted__facet", direction="FP->DF", predictor="registry_rule")
    if not (ival(rf["metric_labels"]) - ival(rf["labels_with_min_targets"]) == 1 == ival(r["metric_labels"])
            - ival(r["labels_with_min_targets"]) and min(sizes) < max(sizes)
            and dec(rf["macro_f1_on_labels_with_min_targets"]) > dec(r["macro_f1_on_labels_with_min_targets"])):
        raise PaperError("facet FP->DF: expected one excluded label and the rule ahead on the others")
    c.add("rulemintiafacfpdf", frac(rf["macro_f1_on_labels_with_min_targets"], 3),
          rf["macro_f1_on_labels_with_min_targets"], src,
          "identity_admitted__facet FP->DF registry_rule, macro_f1_on_labels_with_min_targets (without the label with fewest targets)")
    c.add("cosmintiafacfpdf", frac(r["macro_f1_on_labels_with_min_targets"], 3),
          r["macro_f1_on_labels_with_min_targets"], src,
          "identity_admitted__facet FP->DF cosine, macro_f1_on_labels_with_min_targets (without the label with fewest targets)")

    # the type block of the drifted anchors (extended audit details, Section D)
    src = "evidence/drift_type_mechanism.csv"
    dm = {r["dataset"]: r for r in out.csv(src)}
    da = out.csv("conformance/drift_anatomy.csv")
    tot = 0
    for d in DATASETS:
        r = dm[d]
        dr = ival(r["drifted_rows"])
        exc = ival(one(da, dataset=d, quantity="unsupported_facet_mass", key="total")["value"])
        if not (ival(r["type_equals_core_types_plus_excess_as_attribute"]) == dr == ival(r["T_fail_rows"])
                == ival(r["T_fail_drifted_rows"]) and ival(r["excess_events"]) == exc
                == ival(r["type_attribute_surplus_events"])
                and ival(r["W_fail_rows"]) == ival(r["W_fail_rows_with_color_excess"])
                and ival(r["W_fail_rows_without_color_excess"]) == 0):
            raise PaperError(f"the type block of the drifted anchors is not explained in {d}")
        tot += ival(r["type_equals_core_types_plus_excess_as_attribute"])
    by_id = {x.cell_id: x for x in c.items}
    if (tot != ival(by_id["ndrift"].raw) or ival(dm["DF"]["W_fail_rows"]) != ival(by_id["wfaildf"].raw)
            or ival(dm["FP"]["W_fail_rows"]) != 0):
        raise PaperError("the drift mechanism does not cover the FULL-CONFORM failures")
    c.add("ntypeattrdrift", count(tot), tot, src, "sum of type_equals_core_types_plus_excess_as_attribute")


# ---------------------------------------------------------------- support step (Remark 1; extended audit details, Section E)
def _support(c: Cells, out: Outputs) -> None:
    by_id = {x.cell_id: x for x in c.items}
    src = "support/support_regimes.csv"
    rg = out.csv(src)
    gain = {d: json.loads(one(rg, regime="union", dataset=d)["added_vs_released"]) for d in DATASETS}
    if gain["FP"] != ["attribute/nickname"]:
        raise PaperError("the union is expected to add nickname to the FP support")
    if (ival(one(rg, regime="union", dataset="DF")["support_size"]) != ival(by_id["nactivefacets"].raw)
            or ival(one(rg, regime="intersection", dataset="DF")["support_size"]) != ival(by_id["sinter"].raw)):
        raise PaperError("the union or the intersection differs from the active facets or the shared support")
    c.add("sgaindf", count(len(gain["DF"])), len(gain["DF"]), src, "regime=union, dataset=DF, len(added_vs_released)")
    c.add("sgainfp", count(len(gain["FP"])), len(gain["FP"]), src, "regime=union, dataset=FP, len(added_vs_released)")

    src = "support/support_transitions.csv"
    tr = out.csv(src)
    spec = {r["view_id"]: r for r in out.csv("registry/certificate_spec.csv")}

    def rows(**eq: str) -> int:
        return sum(ival(r["rows"]) for r in select(tr, **eq))

    un = select(tr, regime="union")
    if any(r["released_query_conform"] != r["query_conform"] for r in un):
        raise PaperError("an anchor changes its QUERY-CONFORM flag on the union")
    chg = sum(ival(r["rows"]) for r in un if r["released_verdict"] != r["verdict"])
    c.add("pbsupchg", count(chg), chg, src, "regime=union, rows whose verdict changes, all views")
    for d, k in DS_KEY:
        lost = {}
        for r in select(tr, regime="intersection", dataset=d, released_query_conform="1", query_conform="0"):
            lost[r["view"]] = lost.get(r["view"], 0) + ival(r["rows"])
        v = max(lost.values())
        c.add("pbintdef" + k, count(v), v, src,
              f"regime=intersection, dataset={d}, max over views of rows with released_query_conform=1, query_conform=0")
    art = [r for r in select(tr, regime="intersection", released_verdict="NONDETERMINED", verdict="UNSAFE")]
    if {(r["view"], r["dataset"]) for r in art} != {("family3_plus_type__facet", "DF")}:
        raise PaperError("nondetermined anchors become UNSAFE outside the DF family-3+type view")
    n_art, f_art = sum(ival(r["rows"]) for r in art), sum(ival(r["changed_fibers"]) for r in art)
    c.add("pbintartrows", count(n_art), n_art, src, "regime=intersection, NONDETERMINED -> UNSAFE rows")
    c.add("pbintartfib", count(f_art), f_art, src, "regime=intersection, NONDETERMINED -> UNSAFE changed_fibers")
    col = select(tr, regime="released_plus_color")
    qviews = sorted({r["view"] for r in col if r["released_query_conform"] != r["query_conform"]})
    wf = ival(by_id["wfaildf"].raw)
    for v in qviews:
        s = spec[v]
        moved = [r for r in col if r["view"] == v and r["released_query_conform"] != r["query_conform"]]
        if (s["task"] != "facet" or s["certificate_path"] != "A" or s["required_maps"] != ""
                or {(r["dataset"], r["released_query_conform"]) for r in moved} != {("DF", "0")}
                or sum(ival(r["rows"]) for r in moved) != wf):
            raise PaperError(f"color changes QUERY-CONFORM in an unexpected way on {v}")
    c.add("pbcolorqcviews", count(len(qviews)), len(qviews), src,
          "regime=released_plus_color, views with a QUERY-CONFORM change")
    fam_b = {v for v, s in spec.items() if s["task"] == "family" and s["certificate_path"] == "B"}
    pairs = sum(ival(r["rows"]) for r in col if r["view"] in fam_b and r["released_verdict"] == "UNSAFE"
                and r["verdict"] == "UNVERIFIED")
    if any(r["view"] in fam_b and r["verdict"] == "UNSAFE" for r in col):
        raise PaperError("a family-task Path B anchor stays UNSAFE once color is added")
    c.add("pbcolorpairs", count(pairs), pairs, src,
          "regime=released_plus_color, family-task Path B views, UNSAFE -> UNVERIFIED rows")

    src = "support/support_certificates.csv"
    ce = out.csv(src)
    key = lambda r: (r["view_id"], r["dataset"], r["fiber_value"], r["certified_label"])  # noqa: E731
    rel = {key(r) for r in select(ce, regime="released")}
    uni = {key(r) for r in select(ce, regime="union")}
    voided = rel - uni
    if uni - rel or len(voided) != 1:
        raise PaperError("the union is expected to void exactly one released certificate")
    (vview, vds, vval, _), = voided
    if (vview, vds) != ("family3_plus_type__facet", "FP"):
        raise PaperError("the voided union certificate is expected on the FP family-3+type view")
    c.add("pbsupvoidval", "$(" + ",".join(str(x) for x in json.loads(vval)) + ")$", vval, src,
          "fiber_value of the released certificate that the union voids")
    fam = {k for k in rel if spec[k[0]]["task"] == "family"}
    colc = {key(r) for r in select(ce, regime="released_plus_color")}
    if any(spec[k[0]]["task"] == "family" for k in colc) or colc != rel - fam:
        raise PaperError("color is expected to void exactly the family-task certificates")
    c.add("pbcolor", count(len(fam)), len(fam), src, "released family-task certificates voided by color")

    src = "support/support_admission.csv"
    ad = out.csv(src)
    r = one(ad, regime="intersection", view="family3__facet")
    if one(ad, regime="released", view="family3__facet")["admitted_eps_0.05"] != "1" or r["admitted_eps_0.05"] != "0":
        raise PaperError("family-3 is expected to be admitted at .05 on the released support only")
    c.add("pbintfthreemvdf", frac(r["mV_DF"], 3), r["mV_DF"], src, "regime=intersection, view=family3__facet, mV_DF")
    c.add("pbintfthreemvfp", frac(r["mV_FP"], 3), r["mV_FP"], src, "regime=intersection, view=family3__facet, mV_FP")
    first = []
    for reg in ("union", "released_plus_color"):
        for v in sorted({x["view"] for x in ad}):
            t0 = one(ad, regime="released", view=v)["smallest_admitting_eps"]
            t1 = one(ad, regime=reg, view=v)["smallest_admitting_eps"]
            if t0 == t1:
                continue
            first.append(dec(t1) if t0 == "" else dec(t0) if t1 == "" else min(dec(t0), dec(t1)))
            for eps in ("0", "0.01", "0.05"):
                if one(ad, regime="released", view=v)[f"admitted_eps_{eps}"] != one(ad, regime=reg, view=v)[f"admitted_eps_{eps}"]:
                    raise PaperError(f"an admission decision changes at eps={eps} on {reg}")
    if not first:
        raise PaperError("no admission decision changes on the union or with color")
    eps_min = min(first)
    c.add("pbsupepsmin", floor_frac(eps_min, 3), eps_min, src,
          "smallest eps at which a decision differs from the released support on the union or with color [rounded down]")
    voided_views = {k[0] for k in fam}
    mv = min(dec(one(ad, regime="released_plus_color", view=v)[f"mV_{d}"]) for v in voided_views for d in DATASETS)
    c.add("pbcolorminmv", floor_frac(mv, 3), mv, src,
          "min over the views whose family-task certificates color voids and datasets of mV [rounded down]")


# ---------------------------------------------------------------- build record of the release (extended audit details, Section D)
def _build_record(c: Cells, out: Outputs) -> None:
    by_id = {x.cell_id: x for x in c.items}
    br = out.source_release()["build_record"]
    src = "src/lca/paper/source_release.json"
    core, rep, ev_, lost = (ival(br[k]) for k in ("dropped_core_block_entries", "dropped_rep_core_block_entries",
                                                    "core_block_events_dropped", "anchors_losing_a_core_entry"))
    excess = ival(by_id["unsupdf"].raw) + ival(by_id["unsupfp"].raw)
    norep = ival(by_id["rcnorepdf"].raw) + ival(by_id["rcnorepfp"].raw)
    if (core + rep != ival(by_id["builddropped"].raw) or ev_ != excess or rep != norep
            or lost != ival(by_id["ndrift"].raw) or br["conforms"] is not True):
        raise PaperError("the build record does not decompose into the released drift")
    c.add("dropcoreentries", count(core), core, src, "build_record.dropped_core_block_entries")
    c.add("droprepentries", count(rep), rep, src, "build_record.dropped_rep_core_block_entries")
    c.add("dropcoreevents", count(ev_), ev_, src, "build_record.core_block_events_dropped")
    k = "build_report_sha256"
    c.add("hashbuildreport", br[k][:8], br[k], src, f"build_record.{k} (first 8 hex digits)")


# ---------------------------------------------------------------- reruns (extended audit details, Section D)
def _rerun(c: Cells, out: Outputs) -> None:
    src = "checks/rerun_equivalence.json"
    rr = out.json(src)
    ab = rr["A_vs_B"]
    if ab["identical"] != ab["compared"] or ab["compared"] < 1:
        raise PaperError("a rerun output differs")
    c.add("nrerunsame", count(ab["compared"]), ab["compared"], src, "A_vs_B.compared (all identical)")
    c.add("verpythonrerun", rr["runs"]["B"]["python"], rr["runs"]["B"]["python"], src, "runs.B.python")
    sen = rr["sen12ms_A_vs_B"]
    if sen["identical"] != sen["compared"] or sen["compared"] < 1 or sen["differing"]:
        raise PaperError("a rerun output of the SEN12MS step differs")
    c.add("nrerunsen", count(sen["compared"]), sen["compared"], src, "sen12ms_A_vs_B.compared (all identical)")
    for key, cid in (("python", "verpythonpin"), ("numpy", "vernumpypin"), ("pandas", "verpandaspin"),
                     ("scipy", "verscipypin")):
        v = rr["runs"]["A"][key]
        c.add(cid, v, v, src, f"runs.A.{key} (pinned environment, environment/requirements.lock.txt)")


# ---------------------------------------------------------------- independent checks (extended audit details, Section D)
def _independent(c: Cells, out: Outputs) -> None:
    src = "checks/independent_verification.json"
    iv = out.json(src)
    n = iv["counts"]
    if (iv["all_pass"] is not True or n["checks_failed"] != 0 or iv["failed_checks"]
            or iv["pipeline_library_imported"] is not False
            or n["checks_with_exact_comparison"] + n["checks_observational"] != n["checks_total"]):
        raise PaperError("the independent recomputation of the verdicts does not pass")
    c.add("indchecks", count(n["checks_total"]), n["checks_total"], src, "counts.checks_total (checks_failed = 0)")
    c.add("indexact", count(n["checks_with_exact_comparison"]), n["checks_with_exact_comparison"], src,
          "counts.checks_with_exact_comparison")
    src = "checks/support_verification.json"
    sv = out.json(src)
    certs = out.csv("support/support_certificates.csv")
    if (sv["all_pass"] is not True or sv["checks_failed"] != 0 or sv["failed_checks"]
            or sv["pipeline_library_imported"] is not False or sv["certificates_checked"] != len(certs)
            or any(r["pass"] != "1" for r in certs)):
        raise PaperError("the independent check of the support step does not pass on every certificate")
    c.add("supcerts", count(sv["certificates_checked"]), sv["certificates_checked"], src,
          "certificates_checked (= rows of support/support_certificates.csv, all regimes)")
    c.add("supstates", count(sv["basis_block_values_enumerated"]), sv["basis_block_values_enumerated"], src,
          "basis_block_values_enumerated")
    src = "checks/data_access_checks.json"
    acc = out.json(src)["checks"]
    early = len(acc["registry_label_columns_loaded"]) + len(acc["conformance_label_columns_loaded"])
    if early != 0 or acc["query_labels_loaded_after_full_conform_written"] is not True:
        raise PaperError("a label column is loaded before FULL-CONFORM is written")
    c.add("nlabelearly", count(early), early, src,
          "label columns loaded by the registry and FULL-CONFORM steps (labels first loaded after FULL-CONFORM)")


# ---------------------------------------------------------------- link-prediction study (Sec. VI-C, extended audit details Section C)
KG_SELF_REFLEXIVE = {"_derivationally_related_form": "kgwnrrdrf", "_verb_group": "kgwnrrvg", "_similar_to": "kgwnrrsim"}


def _kg(c: Cells, out: Outputs) -> None:
    src = "kg/release_summary.csv"
    rs = out.csv(src)
    kv = out.json("checks/kg_verification.json")
    audit = out.json("kg/kg_audit.json")
    if (kv["all_pass"] is not True or kv["checks_failed"] != 0 or kv["pipeline_library_imported"] is not False
            or audit["nondetermined_issued"] is not False or audit["model_scores_read"] is not False):
        raise PaperError("the independent check of the link-prediction study does not pass")
    c.add("kgchecks", count(kv["checks_total"]), kv["checks_total"], "checks/kg_verification.json",
          "checks_total (checks_failed = 0; the verifier imports no pipeline code)")
    if len(rs) != 4 or any(ival(r["identity_overlap_with_train"]) != 0 or
                           ival(r["l1"]) + ival(r["unverified"]) != ival(r["held_out"]) for r in rs):
        raise PaperError("a held-out triple of WN18 or WN18RR occurs in training, or the verdicts do not add up")
    for rel, split, n_id, u_id in (("WN18", "test", "kgwnn", "kgwnu"), ("WN18", "valid", "kgwnvn", "kgwnvu"),
                                   ("WN18RR", "test", "kgwnrrn", "kgwnrru"), ("WN18RR", "valid", "kgwnrrvn", "kgwnrrvu")):
        r = one(rs, release=rel, split=split)
        sel = f"release={rel}, split={split}"
        c.add(n_id, count(r["held_out"]), r["held_out"], src, sel + ", held_out")
        c.add(u_id, count(r["l1"]), r["l1"], src, sel + ", l1 (identity_overlap_with_train = 0)")
    for rel in ("WN18RR",):
        for split in ("test", "valid"):
            if ival(one(rs, release=rel, split=split)["symmetrized_increment"]) != 0:
                raise PaperError("reading the Domain row both ways changes a WN18RR count")
    for split, cid in (("test", "kgwndomrev"), ("valid", "kgwndomrevv")):
        r = one(rs, release="WN18", split=split)
        c.add(cid, count(r["symmetrized_increment"]), r["symmetrized_increment"], src,
              f"release=WN18, split={split}, symmetrized_increment (Domain row read in both directions)")
    r = one(rs, release="WN18", split="test")
    c.add("kgwnakr", count(r["l1_symmetrized_domain"]), r["l1_symmetrized_domain"], src,
          "release=WN18, split=test, l1_symmetrized_domain")
    src = "kg/relation_breakdown.csv"
    rb = select(out.csv(src), release="WN18RR", split="test")
    certified = {r["relation"]: ival(r["l1"]) for r in rb if ival(r["l1"]) > 0}
    if set(certified) != set(KG_SELF_REFLEXIVE) or sum(certified.values()) != ival(one(rs, release="WN18RR", split="test")["l1"]):
        raise PaperError("the WN18RR test triples are not all certified through self-reflexive pointers")
    if any(r["declared"] != "0" for r in rb if r["relation"] == "_also_see"):
        raise PaperError("also-see is not outside the registry")
    for rel, cid in KG_SELF_REFLEXIVE.items():
        c.add(cid, count(certified[rel]), certified[rel], src, f"release=WN18RR, split=test, relation={rel}, l1")
    src = "kg/wn18rr_filter_check.json"
    fc = out.json(src)
    if (fc["reproduces_all_splits"] is not True or fc["each_removed_relation_has_its_inverse_kept"] is not True
            or fc["removed_relation_count"] != fc["inverse_pairs_in_wn18"]):
        raise PaperError("the released construction script does not remove one relation of each inverse pair")
    c.add("kgwnrrinv", count(fc["removed_relation_count"]), fc["removed_relation_count"], src,
          "removed_relation_count (= inverse_pairs_in_wn18; the filter reproduces all three WN18RR splits)")
    src = "kg/raw_authority.json"
    ra = out.json(src)
    total_l1 = sum(ival(r["l1"]) for r in rs)
    if not (ra["certified_triples"] == ra["certified_targets_that_are_wordnet_pointers"]
            == ra["certified_witnesses_that_are_wordnet_pointers"] == ra["certified_pairs_on_one_synset_pair"]
            == total_l1):
        raise PaperError("a certified triple and its witness are not WordNet pointers between the same two synsets")
    c.add("kgcertn", count(ra["certified_triples"]), ra["certified_triples"], (src, "kg/release_summary.csv"),
          "certified_triples (= certified_pairs_on_one_synset_pair = sum of l1 over the four splits)")
    c.add("kgfound", count(ra["wn18_triples_that_are_wordnet_pointers"]), ra["wn18_triples_that_are_wordnet_pointers"],
          src, "wn18_triples_that_are_wordnet_pointers")
    c.add("kgtotal", count(ra["wn18_triples"]), ra["wn18_triples"], src, "wn18_triples")
    c.add("kgwnver", ra["wordnet_version"], ra["wordnet_version"], src, "wordnet_version")
    dp = ra["definitions_file_pos"]
    if (dp["certified_pairs_matching"] + dp["others"] != ra["certified_triples"]
            or dp["others_all_on_shared_offsets"] is not True):
        raise PaperError("the definitions-file parts of speech do not account for every certified pair")
    c.add("kgposmatch", count(dp["certified_pairs_matching"]), dp["certified_pairs_matching"], src,
          "definitions_file_pos.certified_pairs_matching")
    c.add("kgposother", count(dp["others"]), dp["others"], src, "definitions_file_pos.others")


# ---------------------------------------------------------------- derived display versions
def _text_versions(c: Cells) -> None:
    """Three-digit text displays of four-digit table cells (same raw value)."""
    extra = []
    for x in list(c.items):
        if re.fullmatch(r"0?\.\d{4}|1\.0000", x.display):
            extra.append(("t" + x.cell_id, frac(x.raw, 3), x.raw, x.outputs, x.selector + " [text, 3 digits]"))
        elif re.fullmatch(r"\$[-+]\$0?\.\d{4}", x.display):
            extra.append(("t" + x.cell_id, signed(x.raw, 3), x.raw, x.outputs, x.selector + " [text, 3 digits]"))
    for cid, disp, raw, outs, sel in extra:
        c.items.append(Cell(cid, disp, raw, outs, sel))
        if cid in c._ids:
            raise PaperError(f"duplicate cell {cid}")
        c._ids.add(cid)


def _word_versions(c: Cells) -> None:
    by_id = {x.cell_id: x for x in c.items}
    for cid in WORD_CELLS:
        x = by_id[cid]
        if not re.fullmatch(r"\d", x.display):
            raise PaperError(f"cell {cid} is not a count below ten: {x.display}")
        c.items.append(Cell(cid + "w", NUMBER_WORDS[int(x.display)], x.raw, x.outputs, x.selector + " [text, spelled out]"))
        c._ids.add(cid + "w")


# ---------------------------------------------------------------- writers
def numbers_tex(cells: list[Cell]) -> str:
    lines = []
    for x in cells:
        lines.append("\\expandafter\\def\\csname val@%s\\endcsname{%s}%% %s | %s"
                     % (x.cell_id, x.display, " + ".join(x.outputs), x.selector))
    return "\n".join(lines) + "\n"


def ledger_rows(cells: list[Cell], used: dict[str, str]) -> list[dict[str, str]]:
    return [{"cell_id": x.cell_id, "display": x.display, "raw_value": x.raw, "output": ";".join(x.outputs),
             "output_sha256": ";".join(used[o] for o in x.outputs), "selector": x.selector} for x in cells]


def ledger_csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=LEDGER_FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def read_ledger(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_numbers_tex(text: str) -> dict[str, str]:
    """cell id -> display string of a numbers.tex file (comments ignored)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"\\expandafter\\def\\csname val@([a-z0-9]+)\\endcsname\{(.*)\}%", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


# ---------------------------------------------------------------- SEN12MS: Path B and witness pairs (Sec. VI-C)
def _sen12ms(c: Cells, out: Outputs) -> None:
    src = "external/sen12ms_pathb.json"
    r = out.json(src)
    sv = out.json("checks/sen12ms_verification.json")
    if sv["all_pass"] is not True or sv["checks_failed"] != 0 or sv["pipeline_library_imported"] is not False:
        raise PaperError("the independent check of the SEN12MS study does not pass")
    n, pv, fv, ck = r["patches"], r["patches_by_verdict"], r["fibers_by_verdict"], r["checks"]
    if (sum(pv.values()) != n or sum(fv.values()) != r["fibers"] or r["query_conform"]["patches"] != n
            or r["ties"]["carrier_17"] != 0 or len(r["support_1based"]) != r["classes"]
            or ck["nondetermined_fibers_with_a_certificate"] != 0 or not ck["milp_agree"] == ck["milp_fibers"] == r["fibers"] - fv["NONDETERMINED"]
            or ck["certified_equals_unmerged_strict_merged_max"] is not True
            or pv != sv["patches_by_verdict"] or fv != sv["fibers_by_verdict"]):
        raise PaperError("the SEN12MS verdicts do not add up or disagree with the independent check")
    rl, cc, wg = r["released_label"], r["certified_counts"], r["nondetermined_whole_group_fibers"]
    r10 = r["released_label_10"]
    if (rl["patches"] != n or rl["equals_strict_argmax"] != n
            or r10["patches"] != n or r10["equals_strict_argmax_of_view"] != n
            or cc["vary_in_fiber"]["patches"] + cc["fixed_by_view"]["patches"] != pv["UNSAFE"]
            or cc["vary_in_fiber"]["fibers"] + cc["fixed_by_view"]["fibers"] != fv["UNSAFE"]
            or cc["vary_in_fiber"]["fibers"] != ck["second_states"] or cc["vary_in_fiber"] != sv["certified_counts_vary"]
            or not wg or sum(e["patches"] for e in wg) > pv["NONDETERMINED"]):
        raise PaperError("the SEN12MS released labels, varying counts or whole-group fibers do not add up")
    c.add("senchecks", count(sv["checks_total"]), sv["checks_total"], "checks/sen12ms_verification.json",
          "checks_total (checks_failed = 0; the check imports no pipeline code)")
    c.add("senn", count(n), n, src, "patches")
    c.add("senclasses", count(r["classes"]), r["classes"], src, "classes (classes of the full scheme)")
    c.add("sengroups", count(len(r["map"]["groups_1based"])), len(r["map"]["groups_1based"]), src,
          "map.groups_1based (merged classes)")
    c.add("sensupport", count(len(r["support_1based"])), len(r["support_1based"]), src,
          "support_1based (classes with a count in some patch)")
    c.add("senqc", count(r["query_conform"]["patches"]), r["query_conform"]["patches"], src, "query_conform.patches")
    c.add("senfib", count(r["fibers"]), r["fibers"], src, "fibers")
    c.add("senunsafefib", count(fv["UNSAFE"]), fv["UNSAFE"], src, "fibers_by_verdict.UNSAFE")
    for key, cid in (("UNSAFE", "sencert"), ("NONDETERMINED", "sennondet"), ("UNVERIFIED", "senunver")):
        c.add(cid, count(pv[key]), pv[key], src, f"patches_by_verdict.{key}")
        mass = Decimal(pv[key]) / Decimal(n)
        c.add(cid + "r", frac(mass, 3), mass, src, f"patches_by_verdict.{key} / patches [text, 3 digits]")
    c.add("senmilp", count(ck["milp_fibers"]), ck["milp_fibers"], src,
          "checks.milp_fibers (every fiber that is not NONDETERMINED; all agree)")
    c.add("senlabmatch", count(rl["equals_strict_argmax"]), rl["equals_strict_argmax"], src,
          "released_label.equals_strict_argmax (released 17-class label = strict arg-max of the counts)")
    c.add("senlabten", count(r10["equals_strict_argmax_of_view"]), r10["equals_strict_argmax_of_view"], src,
          "released_label_10.equals_strict_argmax_of_view (released 10-class label = strict arg-max of the view)")
    c.add("senvary", count(cc["vary_in_fiber"]["patches"]), cc["vary_in_fiber"]["patches"], src,
          "certified_counts.vary_in_fiber.patches (UNSAFE patches whose 17 counts vary across the fiber)")
    c.add("senvaryfib", count(cc["vary_in_fiber"]["fibers"]), cc["vary_in_fiber"]["fibers"], src,
          "certified_counts.vary_in_fiber.fibers (each with a second state checked)")
    whole = sum(e["patches"] for e in wg)
    c.add("senndwhole", count(whole), whole, src,
          "nondetermined_whole_group_fibers, sum of patches (patches wholly in one merged group)")
    c.add("senndwholefib", count(len(wg)), len(wg), src, "nondetermined_whole_group_fibers (fibers)")
    share = Decimal(whole) / Decimal(pv["NONDETERMINED"])
    c.add("senndwholer", frac(share, 3), share, src,
          "nondetermined_whole_group_fibers patches / patches_by_verdict.NONDETERMINED [text, 3 digits]")


# ---------------------------------------------------------------- declared event order (Sec. VI-A, Open fibers need a third verdict)
def _tie_order(c: Cells, out: Outputs) -> None:
    src = "tie_order/tie_order.json"
    t = out.json(src)
    n = t["counts"]
    views = out.csv("tie_order/tie_order_views.csv")
    ad = out.csv("tie_order/tie_order_admission.csv")
    by_id = {x.cell_id: x for x in c.items}
    path_b = [r for r in views if r["path"] == "B"]
    if (t["recomputed_verdicts_reproduce_verdict_step"] is not True
            or n["open_path_b_fibers"] != ival(by_id["pbopen"].raw)
            or n["strict_fibers"] + n["tie_only_path_b_fibers"] != n["open_path_b_fibers"]
            or n["integer_checked_fibers"] != n["strict_fibers"]
            or sum(ival(r["open_path_b_fibers"]) for r in path_b) != n["open_path_b_fibers"]
            or any(ival(r["strict_rows"]) + ival(r["tie_only_rows"]) != ival(r["unverified_observed_rows"]) for r in path_b)
            or any(r["path"] == "C" and ival(r["tie_only_rows"]) != 0 for r in views)
            or len(ad) != n["views"]):
        raise PaperError("the declared-order step does not add up")
    c.add("topbopen", count(n["open_path_b_fibers"]), n["open_path_b_fibers"], src,
          "counts.open_path_b_fibers (= pbopen)")
    c.add("topbstrict", count(n["strict_fibers"]), n["strict_fibers"], src,
          "counts.strict_fibers (a strictly winning rival state, re-checked in integer arithmetic)")
    c.add("topbtie", count(n["tie_only_path_b_fibers"]), n["tie_only_path_b_fibers"], src,
          "counts.tie_only_path_b_fibers")
    c.add("totieviews", count(n["views_with_tie_only_rows"]), n["views_with_tie_only_rows"], src,
          "counts.views_with_tie_only_rows")
    c.add("todecchg", count(n["decisions_not_fixed"]), n["decisions_not_fixed"], src,
          "counts.decisions_not_fixed (admission decisions that a declared order could change, any eps)")
    fib = out.csv("tie_order/tie_order_fibers_public.csv")
    vt = out.json("checks/tie_order_verification.json")
    if (len(fib) != n["public_fiber_rows"] or n["public_fiber_rows"] + n["withheld_fiber_rows"] != n["open_path_b_fibers"]
            or vt["all_pass"] is not True or vt["pipeline_library_imported"] is not False or vt["rows"] != len(fib)):
        raise PaperError("the public declared-order rows do not add up or fail their independent check")

