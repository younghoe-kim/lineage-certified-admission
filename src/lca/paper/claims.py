# SPDX-License-Identifier: MIT
"""Claims of the paper: statement (filled from the number cells), where the paper states it, the cells that state
it, and a status recomputed from the aggregate outputs. CLAIMS.md maps each claim to its command and cells."""
from __future__ import annotations

import csv
import io
import math
from decimal import Decimal
from typing import Callable

from .common import DATASETS, DIRECTIONS, Outputs, dec, ival, one, plain, select
from .numbers import Cell

FIELDS = ["claim_id", "status", "statement", "manuscript_locations", "number_cells"]


# ---------------------------------------------------------------- checks (True when the claim holds)
def _head(out: Outputs) -> bool:
    vr = out.csv("verdicts/verdicts.csv")
    return all(dec(one(vr, view="identity_admitted__family", dataset=d)["m^U"]) > 0 for d in DATASETS)


def _decomp(out: Outputs) -> bool:
    ok = True
    for r in out.csv("verdicts/decomposition.csv"):
        ok &= r["full_conform_rowwise_equal"] == "1"
        ok &= ival(r["primary_unsafe_rows"]) == ival(r["primary_unsafe_full_rows"]) + ival(r["primary_unsafe_outside_full_rows"])
        ok &= ival(r["primary_unsafe_full_rows"]) == ival(r["alternative_unsafe_rows"])
        ok &= ival(r["alternative_unsafe_outside_full_rows"]) == 0
    return ok


def _drift(out: Outputs) -> bool:
    cs = out.csv("conformance/conformance_summary.csv")
    da = out.csv("conformance/drift_anatomy.csv")
    ok = out.source_release()["build_record"]["conforms"] is True
    carrier = [r for r in out.csv("utility/simple_checks_details.csv") if r["metric"] == "gold_carrier_present_rate"]
    ok &= len(carrier) == 4 and all(dec(r["value"]) == 1 for r in carrier)  # identity check passes on every anchor
    ind = {r["check"]: r for r in out.csv("checks/independent_checks.csv")}
    ok &= ind["conformance.FULL_CONFORM_recomputed_from_inputs"]["exact_match"] == "1"
    ok &= one(da, dataset="ALL", quantity="excess_rows_equal_pi_failures")["value"] == "1"
    for d in DATASETS:
        fails = ival(one(cs, dataset=d, view_id="core_plus_type__family")["query_defect_rows"])
        ok &= ival(one(da, dataset=d, quantity="rows_with_excess", key="")["value"]) == fails > 0
        ok &= ival(one(da, dataset=d, quantity="rows_with_deficit", key="")["value"]) == 0
    return ok


def _facet(out: Outputs) -> bool:
    vr = out.csv("verdicts/verdicts.csv")
    spec = one(out.csv("registry/certificate_spec.csv"), view_id="identity_admitted__facet")
    ok = "Pi" in spec["required_maps"].split("|")
    for d in DATASETS:
        ok &= dec(one(vr, view="identity_admitted__facet", dataset=d)["m^U"]) > 0
        r = one(vr, view="family3__facet", dataset=d)
        ok &= dec(r["m^U"]) == 0 and dec(r["m^V_obs"]) + dec(r["m^V_defect"]) <= Decimal("0.05")
    return ok


def _fiber(out: Outputs) -> bool:
    pi_ = out.csv("utility/partition_identity.csv")
    ok = True
    for d in DATASETS:
        full = select(pi_, dataset=d, scope="FULL_CONFORM")
        ok &= len(full) == 8 and all(r["partition_equal"] == "1" for r in full)
        drift = select(pi_, dataset=d, scope="DRIFT")
        ok &= max(ival(r["cell_fibers"]) for r in drift) > ival(one(drift, cell_id="F0M0T0")["core_fibers"])
    return ok


def _l3(out: Outputs) -> bool:
    comp = out.csv("evidence/verdict_class_composition.csv")
    ok = True
    for dname in DIRECTIONS.values():
        r = one(comp, view_id="family3__facet", direction=dname, training_protocol="all_source",
                evaluation_scope="all_supported_target")
        ok &= ival(r["targets_unsafe"]) == 0 and dec(r["fraction_correct_on_nondetermined"]) > Decimal("0.9")
    return ok


def _support(out: Outputs) -> bool:
    chk = out.json("checks/support_verification.json")
    tr = out.csv("support/support_transitions.csv")
    ad = out.csv("support/support_admission.csv")
    union_changes = sum(ival(r["rows"]) for r in tr if r["regime"] == "union" and r["released_verdict"] != r["verdict"])
    union_qc = all(r["released_query_conform"] == r["query_conform"] for r in tr if r["regime"] == "union")
    same = all(one(ad, regime=reg, view=r["view"])[f"admitted_eps_{e}"] == r[f"admitted_eps_{e}"]
               for r in ad if r["regime"] == "released" for reg in ("union", "released_plus_color")
               for e in ("0", "0.01", "0.05"))
    f3 = one(ad, regime="intersection", view="family3__facet")["admitted_eps_0.05"] == "0"
    return chk["all_pass"] is True and union_changes == 1 and union_qc and same and f3


def _tie_order(out: Outputs) -> bool:
    t = out.json("tie_order/tie_order.json")
    n = t["counts"]
    views = out.csv("tie_order/tie_order_views.csv")
    ad = out.csv("tie_order/tie_order_admission.csv")
    rr = out.json("checks/rerun_equivalence.json").get("tie_order_A_vs_B", {})
    vt = out.json("checks/tie_order_verification.json")
    tie_views = {r["view"] for r in views if r["path"] == "B" and ival(r["tie_only_fibers"]) > 0}
    vr = {(r["view"], r["dataset"]): r for r in out.csv("verdicts/verdicts.csv")}
    same_masses = all(
        float(r[f"mU_{d}"]) == float(vr[(r["view"], d)]["m^U"])
        and float(r[f"mV_{d}"]) == float(vr[(r["view"], d)]["m^V_obs"]) + float(vr[(r["view"], d)]["m^V_defect"])
        and (r["path"] != "A" or float(r[f"tie_only_{d}"]) == float(vr[(r["view"], d)]["m^V_obs"]))
        for r in ad for d in DATASETS) and len(ad) * len(DATASETS) == len(vr)
    with_tie = [r for r in ad if dec(r["tie_only_DF"]) > 0 or dec(r["tie_only_FP"]) > 0]
    return (t["recomputed_verdicts_reproduce_verdict_step"] is True and same_masses and vt["all_pass"] is True
            and vt["tie_only_optima_recomputed"] == vt["tie_only_rows"] == n["tie_only_path_b_fibers"] > 0
            and n["strict_fibers"] == n["integer_checked_fibers"]
            and n["strict_fibers"] + n["tie_only_path_b_fibers"] == n["open_path_b_fibers"]
            and tie_views <= {"type3__family"}
            and all(dec(r["mU_DF"]) > 0 or dec(r["mU_FP"]) > 0 for r in with_tie)
            and all(r["decision_fixed_under_every_order"] == "1" for r in ad) and n["decisions_not_fixed"] == 0
            and rr.get("compared", 0) >= 1 and rr.get("identical") == rr.get("compared") and not rr.get("differing"))


def _build(out: Outputs) -> bool:
    br = out.source_release()["build_record"]
    cs = out.csv("conformance/conformance_summary.csv")
    fails = sum(ival(one(cs, dataset=d, view_id="core_plus_type__family")["query_defect_rows"]) for d in DATASETS)
    da = out.csv("conformance/drift_anatomy.csv")
    excess = sum(ival(one(da, dataset=d, quantity="unsupported_facet_mass", key="total")["value"]) for d in DATASETS)
    return (br["conforms"] is True and ival(br["anchors_losing_a_core_entry"]) == fails
            and ival(br["core_block_events_dropped"]) == excess
            and ival(br["dropped_core_block_entries"]) + ival(br["dropped_rep_core_block_entries"])
            == ival(out.json("utility/controls.json")["build_report"]["dropped_core_entries"]))


def _driftmech(out: Outputs) -> bool:
    dm = out.csv("evidence/drift_type_mechanism.csv")
    return all(ival(r["type_equals_core_types_plus_excess_as_attribute"]) == ival(r["drifted_rows"])
               == ival(r["T_fail_rows"]) and ival(r["W_fail_rows"]) == ival(r["W_fail_rows_with_color_excess"])
               for r in dm) and ival(one(dm, dataset="FP")["W_fail_rows"]) == 0


def _rerun(out: Outputs) -> bool:
    rr = out.json("checks/rerun_equivalence.json")
    sen = rr["sen12ms_A_vs_B"]
    return (rr["A_vs_B"]["identical"] == rr["A_vs_B"]["compared"] > 0 and not rr["A_vs_B"]["differing"]
            and sen["identical"] == sen["compared"] > 0 and not sen["differing"])


def _bound(out: Outputs) -> bool:
    pbc = out.csv("utility/probe_bound_check.csv")
    rr = out.csv("utility/registry_rule.csv")
    return (all(r["violation"] == "0" for r in pbc)
            and all(r["rule_le_beta"] == "1" and dec(r["macro_f1"]) <= dec(r["beta_upper_bound"]) for r in rr))


def _vst(out: Outputs) -> bool:
    vst = out.csv("utility/support_check.csv")
    return bool(vst) and all(r["support_recompute_equal"] == "1" for r in vst)


def _tol(out: Outputs) -> bool:
    tol = out.csv("utility/tolerance_sensitivity.csv")
    probes = [r for r in out.csv("utility/probes.csv") if r["status"] == "OK" and r["probe"] == "cosine"
              and r["training_protocol"] == "all_source" and r["evaluation_scope"] == "all_supported_target"]
    reported = {(r["view_id"], r["direction"]): (r["prediction_hash"], r["shared_support_hash"]) for r in probes}
    groups: dict[tuple[str, str], set[str]] = {}
    for r in tol:
        groups.setdefault((r["view_id"], r["direction"]), set()).add(r["prediction_hash"])
    return (len({r["score_eps"] for r in tol}) >= 2 and set(groups) == set(reported)
            and all(len(h) == 1 for h in groups.values())
            and all((r["prediction_hash"], r["view_feature_hash"]) == reported[(r["view_id"], r["direction"])]
                    for r in tol))


def _null(out: Outputs) -> bool:
    nl = out.csv("utility/permutation.csv")
    null = out.csv("utility/permutation_null.csv")
    ok = True
    for dname in DIRECTIONS.values():
        r = one(nl, direction=dname)
        b, k = ival(r["B"]), ival(r["k_ge_observed"])
        ok &= k == 0 and dec(r["p_value"]) == Decimal(k + 1) / Decimal(b + 1)
        ok &= len(select(null, direction=dname)) == b and dec(r["null_max"]) < dec(r["observed_macro_f1"])
    return ok


def _simple(out: Outputs) -> bool:
    mc = out.csv("utility/simple_checks_values.csv")
    bdl = out.csv("utility/simple_checks_details.csv")
    val = {r["cell_id"]: dec(r["value"]) for r in mc}
    pooled = [dec(one(bdl, baseline="pooled_correlation_screen", setting=f"ALL|{t}", metric="NMI")["value"])
              for t in ("facet", "family")]
    within = [dec(r["value"]) for r in bdl if r["metric"] == "within_dataset_NMI"]
    probes = [r for r in out.csv("utility/probes.csv") if r["status"] == "OK"]
    family3 = [dec(one(probes, view_id="family3__facet", direction=d, probe="cosine", training_protocol="all_source",
                       evaluation_scope="all_supported_target")["macro_f1"]) for d in DIRECTIONS.values()]
    verdicts = out.csv("verdicts/verdicts.csv")
    family3_unsafe = [dec(one(verdicts, view="family3__facet", dataset=d)["m^U"]) for d in DATASETS]
    return (val["l0_hit"] == 1 and val["l0_hit_san"] == 0 and val["fd_g3_df"] == 0 and val["fd_g3_fp"] == 0
            and all(x > 0 for x in pooled) and len(within) == 4 and all(x == 0 for x in within)
            and out.source_release()["build_record"]["conforms"] is True
            and all(x > 0 for x in family3) and all(x == 0 for x in family3_unsafe) and val["instfam_cov"] > 0)


def _rule(out: Outputs) -> bool:
    rr = out.csv("utility/registry_rule.csv")
    fv = out.csv("utility/registry_rule_fullview.csv")
    ok = len(fv) == 8 and all(r["pass"] == "1" and dec(r["decided_equals_unsafe_rate"]) == 1
                              and r["decided_rows"] == r["decided_label_correct_rows"] == r["unsafe_rows"] for r in fv)
    ok &= len(rr) == 8 and all(r["learned_parameters"] == "0" for r in rr)
    ia = [one(rr, view_id="identity_admitted__family", direction=d) for d in DIRECTIONS.values()]
    decided = sum(int(r["decided_rows"]) for r in ia)
    errors = sum(int(r["decided_rows"]) - int(r["correct_decided_rows"]) for r in ia)
    ok &= all(r["evaluation_scope"] == "full_conform_target" for r in ia)
    # on the core basis the rule stays below the facet-basis precision and below .9; its minimum is the
    # identity-admitted facet view in DF->FP
    core = [dec(r["precision_on_decisions"]) for r in rr if r["basis"] == "core::"]
    facet = [dec(r["precision_on_decisions"]) for r in rr if r["basis"] == "facet::"]
    low = dec(one(rr, view_id="identity_admitted__facet", direction=DIRECTIONS["dffp"])["precision_on_decisions"])
    ok &= len(core) == 4 and len(facet) == 4 and max(core) < min(facet) and max(core) <= Decimal("0.9")
    ok &= low == min(core)
    # fewer than one error in ten thousand decisions on the shared projection
    return ok and decided > 0 and errors * 10000 < decided


def _mut(out: Outputs) -> bool:
    res = out.csv("mutations/mutation_results.csv")
    doc = out.json("mutations/mutations.json")
    mu = doc["mutations"]
    m1, m2, m3, m4, m5 = (mu[k] for k in ("M1", "M2", "M3", "M4", "M5"))
    stale = m5["failure_capable_negative_control"]
    dom = out.csv("mutations/m5_label_domain_check.csv")
    views = out.csv("mutations/m5_view_checks.csv")
    return (len(res) == 5 and all(r["pass"] == "1" for r in res) and all(m["pass"] is True for m in mu.values())
            and m1["rowwise_verdict_equals_l0_facet"] is True
            and m2["newly_failed_previously_full_conform_rows_audited"] == m2["newly_failed_previously_full_conform_rows_expected"]
            and m3["within_dataset_constant"] is True and m3["path_c_partition_unchanged"] is True
            and m4["injective_partition_unchanged"] is True and m4["binary_partition_changed"] is True
            and m5["unsafe_certificates_on_invalid_rows"] == 0 and m5["recomputed_certificate_label_conflicts"] == 0
            and stale["pass"] is True and all(c["expected_failure_observed"] is True for c in stale["checks"])
            and all(r["pass"] == "1" and r["nonvacuous"] == "1" for r in dom)
            and all(r["pass"] == "1" for r in views)
            and doc["settings"]["m5_label_domain_check"]["baseline_component_mismatches"] == 0
            and doc["settings"]["m5_label_domain_check"]["label_rule_vs_audit_mismatches"] == 0)


def _pbx(out: Outputs) -> bool:
    pbx = out.csv("verdicts/path_b_certificates.csv")
    summary = out.csv("scaling/scaling_summary.csv")
    unsafe = sum(ival(r["unsafe_fibers"]) for r in select(summary, fraction="1.0"))
    pbs = out.json("verdicts/path_b_open_summary.json")
    full_t, pub_t, held_t = pbs["full_table"], pbs["public_table"], pbs["withheld_rows"]
    pbo = out.csv("verdicts/path_b_open_witnesses_public.csv")
    solved = ival(one(out.csv("scaling/scaling.csv"), fraction="1.0")["sum_solver_fibers"])
    ind = {r["check"]: r for r in out.csv("checks/independent_checks.csv")}
    pb_checks = ("verdicts.PathB_open_witness_states", "verdicts.PathB_open_fiber_sizes_from_inputs",
                 "verdicts.PathB_open_public_subset", "verdicts.PathB_open_summary")
    return (bool(pbx) and len(pbx) == unsafe and len(pbx) + full_t["rows"] == solved
            and full_t["rows"] == pub_t["rows"] + held_t["rows"] and full_t["rows_passing_exact_check"] == full_t["rows"]
            and len(pbo) == pub_t["rows"]
            and all(r["pass"] == "1" and r["rival_optima_exact_match"] == "1" for r in pbx)
            and all(r["pass"] == "1" and int(r["rival_minus_observed_at_state"]) >= 0 for r in pbo)
            and not any(r["dataset"] == "DF" and r["query_conforming_fiber_size"] == "1" for r in pbo)
            and all(ind[c]["exact_match"] == "1" for c in pb_checks))


def _breadth(out: Outputs) -> bool:
    ed = out.csv("external/domain_results.csv")
    ic = out.csv("external/independent_checks.csv")
    qualifying = [r for r in ed if r["domain_qualifies"] == "1"]
    return (bool(qualifying) and all(ival(r["L1_exact_derived_rows"]) > 0 for r in qualifying)
            and all(r["exact_match"] == "1" for r in ic))


def _gold(out: Outputs) -> bool:
    rows = out.csv("external/gold_check.csv")

    def s(k: str) -> int:
        return sum(ival(r[k]) for r in rows)

    return (s("gold_cells") > 0 and s("contradictions") == 0 and s("consistent") == s("classified")
            and s("complete_set_equal") + s("component_only") == s("consistent")
            and s("unverified_gold_normalization") == s("unverified_no_match") + s("unverified_ambiguous")
            and all(r["independent_match"] == "1" for r in rows))


def _partscore(out: Outputs) -> bool:
    vc = out.csv("utility/verdict_checks.csv")
    mm = sum(ival(r["value"]) for r in vc if r["quantity"] == "factorial_verdict_mismatches_on_conforming")
    fx = out.csv("utility/factorial_8cell.csv")
    widths = [max(dec(r["macro_f1"]) for r in s) - min(dec(r["macro_f1"]) for r in s)
              for s in (select(fx, task=t, direction=d) for t in ("facet", "family") for d in DIRECTIONS.values())]
    tw = out.csv("utility/type_weight.csv")
    bd = out.csv("utility/bounds.csv")
    beta = [dec(one(bd, view_id=v, direction="DF->FP", training_protocol="fixed_dataset_refit")
                ["macro_f1_upper_bound_C2"]) for v in ("identity_admitted__family", "core_plus_type__family")]
    return (_fiber(out) and mm == 0 and len(widths) == 4 and min(widths) > 0 and len(tw) == 5
            and len({r["shared_support_hash"] for r in tw}) == 1 and len({r["macro_f1"] for r in tw}) > 1
            and beta[0] != beta[1])


def _tie(out: Outputs) -> bool:
    vr = out.csv("verdicts/verdicts.csv")
    spec = one(out.csv("registry/certificate_spec.csv"), view_id="l0_facet__facet")
    return (spec["certificate_path"] == "A" and spec["required_maps"] == ""
            and all(dec(one(vr, view="l0_facet__facet", dataset=d)["m^V_obs"]) > 0 for d in DATASETS))


def _labelmax(out: Outputs) -> bool:
    cs = out.csv("conformance/conformance_summary.csv")
    return bool(cs) and all(ival(r["gold_is_carrier_max_false_rows"]) == 0 for r in cs)


def _cost(out: Outputs) -> bool:
    rt = out.json("scaling/reference_times.json")
    return (rt["machine"]["processes"] == 1
            and abs(sum(float(v) for v in rt["seconds"].values()) - float(rt["total_seconds"])) < 1e-6)


def _scale(out: Outputs) -> bool:
    agg = out.csv("scaling/scaling.csv")
    lo, hi = one(agg, fraction="0.1"), one(agg, fraction="1.0")
    milp = math.log(float(hi["sum_milp_seconds_median"]) / float(lo["sum_milp_seconds_median"]))
    fibers = math.log(ival(hi["sum_solver_fibers"]) / ival(lo["sum_solver_fibers"]))
    rows = math.log(ival(hi["sampled_rows_total"]) / ival(lo["sampled_rows_total"]))
    return abs(milp - fibers) < abs(milp - rows)


def _supdef(out: Outputs) -> bool:
    tr = out.csv("support/support_transitions.csv")
    lost = [r for r in tr if r["regime"] == "intersection" and r["released_verdict"] == "UNSAFE" and r["verdict"] != "UNSAFE"]
    return bool(lost) and all(r["released_query_conform"] == "1" and r["query_conform"] == "0" for r in lost)


def _pairs(out: Outputs) -> bool:
    tr = out.csv("support/support_transitions.csv")
    spec = out.csv("registry/certificate_spec.csv")
    fam_b = {s["view_id"] for s in spec if s["task"] == "family" and s["certificate_path"] == "B"}
    col = [r for r in tr if r["regime"] == "released_plus_color" and r["view"] in fam_b]
    certified = [r for r in col if r["released_verdict"] == "UNSAFE"]
    return (bool(certified) and all(r["verdict"] == "UNVERIFIED" for r in certified)
            and all(r["released_query_conform"] == r["query_conform"] for r in col))


def _indep(out: Outputs) -> bool:
    iv = out.json("checks/independent_verification.json")
    sv = out.json("checks/support_verification.json")
    certs = out.csv("support/support_certificates.csv")
    ind = {r["check"]: r for r in out.csv("checks/independent_checks.csv")}
    recomputed = ("registry.Pi_from_input_relations", "registry.T_from_input_terms", "registry.W_equals_stated_map",
                  "conformance.FULL_CONFORM_recomputed_from_inputs", "conformance.QUERY_CONFORM_recomputed_from_inputs")
    return (iv["all_pass"] is True and iv["counts"]["checks_failed"] == 0 and sv["all_pass"] is True
            and sv["certificates_checked"] == len(certs) and all(r["pass"] == "1" for r in certs)
            and all(ind[c]["exact_match"] == "1" for c in recomputed))


def _carrierkeep(out: Outputs) -> bool:
    ce = out.csv("verdicts/carrier_equivalence.csv")
    ok = len(ce) == 4
    for r in ce:
        only = ival(r["direct_only_rows"])
        ok &= (ival(r["mismatch_on_identity_query_conform_rows"]) == 0 and ival(r["identity_only_rows"]) == 0
               and ival(r["direct_only_identity_query_defect_rows"]) == only)
    fp = one(ce, task="family", dataset="FP")
    return ok and ival(fp["direct_only_rows"]) == 0 and fp["direct_unsafe_rows"] == fp["identity_unsafe_rows"]


def _carrierorder(out: Outputs) -> bool:
    fx = out.csv("utility/factorial_8cell.csv")
    ok = True
    for dname in DIRECTIONS.values():
        for task in ("facet", "family"):
            rows = select(fx, task=task, direction=dname)
            pos = 1 if task == "facet" else 3
            with_c = [dec(r["macro_f1"]) for r in rows if r["cell_id"][pos] == "1"]
            without = [dec(r["macro_f1"]) for r in rows if r["cell_id"][pos] == "0"]
            ok &= len(with_c) == len(without) == 4 and min(with_c) > max(without)
    return ok


def _access(out: Outputs) -> bool:
    acc = out.json("checks/data_access_checks.json")["checks"]
    return (acc["registry_label_columns_loaded"] == [] and acc["conformance_label_columns_loaded"] == []
            and acc["query_labels_loaded_after_full_conform_written"] is True)


def _reject(out: Outputs) -> bool:
    vr = out.csv("verdicts/verdicts.csv")
    ia = ["factorial_F0M0T0__family", "factorial_F0M0T1__family", "factorial_F1M0T0__family",
          "factorial_F1M0T1__family", "facet_only__family", "type3__family", "type_attr_part__family"]
    return all(dec(one(vr, view=v, dataset=d)["m^U"]) > 0 for v in ia for d in DATASETS)


def _kg(out: Outputs) -> bool:
    rs = out.csv("kg/release_summary.csv")
    fc = out.json("kg/wn18rr_filter_check.json")
    ra = out.json("kg/raw_authority.json")
    kv = out.json("checks/kg_verification.json")
    audit = out.json("kg/kg_audit.json")
    rb = select(out.csv("kg/relation_breakdown.csv"), release="WN18RR", split="test")
    self_reflexive = {"_derivationally_related_form", "_verb_group", "_similar_to"}
    total = sum(ival(r["l1"]) for r in rs)
    return (len(rs) == 4 and all(ival(r["identity_overlap_with_train"]) == 0 for r in rs)
            and all(ival(r["symmetrized_increment"]) == 0 for r in rs if r["release"] == "WN18RR")
            and {r["relation"] for r in rb if ival(r["l1"]) > 0} == self_reflexive
            and fc["reproduces_all_splits"] is True and fc["each_removed_relation_has_its_inverse_kept"] is True
            and ra["certified_triples"] == ra["certified_targets_that_are_wordnet_pointers"]
            == ra["certified_witnesses_that_are_wordnet_pointers"] == ra["certified_pairs_on_one_synset_pair"]
            == total > 0 and ra["definitions_file_pos"]["others_all_on_shared_offsets"] is True
            and kv["all_pass"] is True and kv["pipeline_library_imported"] is False
            and audit["nondetermined_issued"] is False and audit["model_scores_read"] is False)


def _sen12ms(out: Outputs) -> bool:
    r = out.json("external/sen12ms_pathb.json")
    sv = out.json("checks/sen12ms_verification.json")
    pv, fv, ck = r["patches_by_verdict"], r["fibers_by_verdict"], r["checks"]
    return (r["query_conform"]["patches"] == r["patches"] == sum(pv.values())
            and sum(fv.values()) == r["fibers"] and r["ties"]["carrier_17"] == 0
            and len(r["support_1based"]) == r["classes"]
            and ck["certificates"] == fv["UNSAFE"] and ck["witness_pairs"] == fv["NONDETERMINED"]
            and ck["rival_states"] == fv["UNVERIFIED"] and ck["nondetermined_fibers_with_a_certificate"] == 0
            and ck["certified_equals_unmerged_strict_merged_max"] is True
            and ck["milp_agree"] == ck["milp_fibers"] == r["fibers"] - fv["NONDETERMINED"] > 0
            and sv["all_pass"] is True and sv["pipeline_library_imported"] is False
            and sv["patches_by_verdict"] == pv and sv["fibers_by_verdict"] == fv
            and r["released_label"]["equals_strict_argmax"] == r["released_label"]["patches"] == r["patches"]
            and r["released_label_10"]["equals_strict_argmax_of_view"] == r["released_label_10"]["patches"] == r["patches"]
            and r["certified_counts"]["vary_in_fiber"]["fibers"] == ck["second_states"] > 0
            and r["certified_counts"]["vary_in_fiber"] == sv["certified_counts_vary"]
            and r["certified_counts"]["vary_in_fiber"]["patches"] + r["certified_counts"]["fixed_by_view"]["patches"] == pv["UNSAFE"]
            and all(len({w["label_1based"] for w in e["witness_pair"]}) == 2
                    for e in r["nondetermined_whole_group_fibers"]))


# claim id -> (statement template over cell ids, manuscript locations, number cells, check)
CLAIMS: dict[str, tuple[str, str, str, Callable[[Outputs], bool]]] = {
    "CL-HEAD": ("The identity-admitted family view is UNSAFE on {tuiafamdf} of DF and {tuiafamfp} of FP anchors "
                "under the registry.",
                "abstract; Sec. I (Example 1); Table III L1; Table IV; Sec. VI-A (Exclusion leaves the label determined, Answer to RQ1); Sec. VII",
                "tuiafamdf tuiafamfp uiafamdf uiafamfp", _head),
    "CL-DECOMP": ("The facet-basis certificate adds exactly the drifted anchors on which W still holds to the "
                  "core-basis certificate.", "Sec. III (before Definition 2); Sec. VI-A (Exclusion leaves the label determined, "
                  "Conformance is query-specific, Answer to RQ1)", "decoutdf decoutfp", _decomp),
    "CL-DRIFT": ("{ndrift} anchors fail FULL-CONFORM and pass identity and row-count checks; every drifted anchor lies "
                 "on the excess side.", "Sec. I (Challenges, Results, Contributions); Table III L2; Table V; Sec. VI-A; Sec. VII",
                 "ndrift fcfaildf fcfailfp", _drift),
    "CL-FACET": ("The facet-task identity-admitted view is UNSAFE through Pi; family-3 is UNSAFE-free and admitted "
                 "at 0.05.", "Sec. I (Results, Contributions); Table III L1, L3; Table IV; Sec. VI-A; Sec. VII", "uiafacdf uiafacfp ufthreedf ufthreefp", _facet),
    "CL-FIBER": ("On registry-conforming anchors all eight block combinations induce the core partition; on drifted "
                 "anchors role blocks split it further.", "Sec. VI-B (Scores vary under a fixed partition); Fig. 2a",
                 "fibfcdf fibfcfp fibdrmaxdf fibdrmaxfp", _fiber),
    "CL-PARTSCORE": ("On registry-conforming anchors the eight block combinations share one partition and agree in "
                     "verdict ({factverdictmmw} mismatches), yet their conforming-fit Macro-F1 differs by "
                     "{widthlo}-{widthhi} per setting. Positive type-block weights keep every fiber and the F1 bound while "
                     "Macro-F1 ranges over {twlo}-{twhi}. The identity-admitted family view and core+type share one "
                     "partition but have F1 bounds {tctwoiafamdffp} and {tctwoctfamdffp} (DF->FP).",
                     "Sec. I (Results, Contributions); Fig. 2a, 2b; Sec. VI-B (Scores vary under a fixed partition, "
                     "Answer to RQ2); Sec. VII", "factverdictmm widthlo widthhi twlo twhi tctwoiafamdffp tctwoctfamdffp",
                     _partscore),
    "CL-TIE": ("The direct-carrier view of the facet task (Path A, no required map) is UNVERIFIED on "
               "{tvolzfacdf} of DF and {tvolzfacfp} of FP mass through single-label tie fibers.",
               "Sec. VI-A (Open fibers need a third verdict, Answer to RQ1)", "tvolzfacdf tvolzfacfp", _tie),
    "CL-LABELMAX": ("On every anchor and for both tasks the released label maximizes the materialized carrier "
                    "({labelfalsew} violating anchors).", "Sec. V (Tasks)", "labelfalse", _labelmax),
    "CL-L3": ("Nondetermined targets make up {fthreendcorrdffp}/{fthreendcorrfpdf} (DF->FP/FP->DF) of the correct "
              "cosine predictions on the family-3 view, which has no UNSAFE mass.",
              "Sec. VI-A (Prediction is not determination)", "fthreendcorrdffp fthreendcorrfpdf", _l3),
    "CL-SUPPORT": ("On the union of the supports {pbsupchgw} anchor changes its verdict; adding color voids all "
                   "{pbcolorw} family-task Path B certificates; both keep every admission decision for eps below "
                   "{pbsupepsmin}; on the intersection family-3 is rejected at 0.05.",
                   "Remark 1; Sec. IV (Proposition 2); Sec. VI-A (Path B certifies labels beyond forward derivation: union and color); extended audit details Section E and Table S3 (intersection)",
                   "pbsupchg pbcolor pbsupepsmin pbintfthreemvdf pbintfthreemvfp pbcolorpairs", _support),
    "CL-TIEORDER": ("A declared event order that the released labels satisfy only removes states, so it can certify "
                    "only fibers that a tie keeps open: the single-label tie fibers of Path A and {topbtie} of the "
                    "{topbopen} open Path B fibers. Each of the other {topbstrict} has a state, checked in integer "
                    "arithmetic, at which another label strictly wins. All {totieviews} views with such fibers already "
                    "have UNSAFE mass, so no admission decision changes under any declared order ({todecchg} can change).",
                    "Sec. III (after Proposition 1); Sec. VI-A (Open fibers need a third verdict)",
                    "topbopen topbstrict topbtie totieviews todecchg", _tie_order),
    "CL-BUILD": ("The build reports conforms: true while dropping {builddropped} core-term entries: {dropcoreentries} "
                 "core entries hold the {dropcoreevents} excess facet events and {droprepentries} representative-core "
                 "entries one per anchor of that row class; FULL-CONFORM fails on as many anchors as lost a core entry "
                 "({ndrift}).", "Table VI; Sec. VI-A (Drift passes build checks); extended audit details Section D",
                 "builddropped dropcoreentries dropcoreevents droprepentries ndrift", _build),
    "CL-DRIFTMECH": ("On all {ntypeattrdrift} drifted anchors the type block equals the type counts of the cleaned "
                     "core block plus the facet excess as attribute events; T fails on all of them and W only on the "
                     "{wfaildf} DF anchors with color counts.", "Table V; Sec. VI-A (Conformance is query-specific); "
                     "extended audit details Section D",
                     "ntypeattrdrift wfaildf", _driftmech),
    "CL-RERUN": ("All {nrerunsame} non-timing outputs of the two full runs are byte-identical between the pinned "
                 "environment and Python {verpythonrerun}, and so are the {nrerunsen} outputs of the SEN12MS step and "
                 "its independent check.",
                 "extended audit details Section D", "nrerunsame verpythonrerun nrerunsen", _rerun),
    "CL-BOUND": ("Every one of the {nboundchecks} cosine and SrcMaj probe scores stays within its F1 bound beta.",
                 "Sec. V (Probes, Lemma 2); Table IV beta column; Sec. VI-B (Answer to RQ2); "
                 "extended audit details Fig. S2", "nboundchecks", _bound),
    "CL-VST": ("A separate support computation agrees with the shared projection in every check.",
               "artifact only", "nvstchecks", _vst),
    "CL-TOL": ("All-source cosine predictions on the shared projection are identical at {tolsettingsw} tolerance "
               "settings for all {tolpairs} view-direction pairs and equal those reported.",
               "extended audit details Section C (All views and bounds)", "tolpairs tolsettingsw", _tol),
    "CL-NULL": ("No permuted family-3 score reaches the observed one (k = {nullkdffp}, n_pi = {nullb}, p = {nullpdffp} "
                "in each direction).", "Sec. I (Results); Table III L3; Sec. VI-A", "nullb nullpdffp nullpfpdf nullkdffp nullkfpdf", _null),
    "CL-SIMPLE": ("Each simpler check misses one of the audit findings or flags information without determination.",
                  "Sec. I; Table VI; Sec. VI-A (Simpler checks merge what the audit separates, Answer to RQ1)", "lzhit lzhitsan fdgthreedf nmifac nmifam", _simple),
    "CL-MUT": ("All five auditor mutations behave as specified.",
               "Sec. VI-A (Seeded mutations move verdicts as predicted); Sec. VI-C (Cost); extended audit details Section A.3",
               "mtwoids mtwonew mfivepairs", _mut),
    "CL-PBX": ("Every Path B UNSAFE certificate is verified exactly, by enumeration or by argument, and each of the "
               "{pbopen} open single-label Path B fibers has an integer state of its fiber, checked exactly, at "
               "which a rival ties with or exceeds the observed label.",
               "abstract; Sec. I (Results, Contributions); Sec. III (Proposition 1); Sec. IV (Path B program, Exact checking); "
               "Table III L1'; Sec. VI-A (Path B certifies labels beyond forward derivation, Answer to RQ1); Sec. VII; extended audit details Section A.2 and Table S1",
               "pbunsafe pbxstates pbopen", _pbx),
    "CL-BREADTH": ("On MaDI-Bench, registered taxonomy maps determine the protected value of {extlone} source cells in "
                   "{extdomq} of the {extdombound} taxonomy-bound domains.",
                   "Sec. I (Contributions); Sec. V (External integration benchmark); Fig. 3a; "
                   "Sec. VI-C (Integration benchmark, Answer to RQ3); Sec. VI-D (Scope and corroboration)",
                   "extdomq extdombound extlone extqblone", _breadth),
    "CL-GOLD": ("None of the {extgoldcls} L0 or L1 cells that the MaDI-Bench normalization gold covers contradicts the "
                "gold value; {extgoldstrict} equal the complete gold value, and every one of the {extgoldunv} "
                "UNVERIFIED gold cells is a normalized cell.",
                "Sec. VI-C (Integration benchmark); Sec. VI-D (Scope and corroboration); extended audit details Section C",
                "extgoldcls extgoldcontra extgoldcells extgoldlz extgoldlone extgoldstrict extgoldmulti extgoldunv "
                "extgoldunvnorm", _gold),
    "CL-SUPDEF": ("On the intersection of the supports every UNSAFE anchor that loses its verdict becomes a support "
                  "defect.",
                  "Remark 1; extended audit details Section E",
                  "pbintdefdf pbintdeffp", _supdef),
    "CL-PAIRS": ("Adding color to each support leaves every released view value and label unchanged and turns all "
                 "{pbcolorpairs} certified view-anchor pairs of the family-task Path B views UNVERIFIED, so the released "
                 "view-label pairs do not decide determination.",
                 "Sec. IV (Proposition 2); extended audit details Section E", "pbcolorpairs", _pairs),
    "CL-INDEP": ("The independent verifier, which imports no pipeline code, recomputes both conformance layers of "
                 "every anchor from the input view, rebuilds the fibers of all {nviews} views, tallies the verdicts and "
                 "checks the rules that tie them to certificate paths, rechecks every open Path B fiber and passes all "
                 "{indchecks} checks ({indexact} exact). The support check verifies all {supcerts} certificates by "
                 "enumerating {supstates} basis-block values.",
                 "Sec. V (Data availability); extended audit details Section D", "nviews indchecks indexact supcerts supstates", _indep),
    "CL-CARRIERKEEP": ("On its query-conforming anchors, the identity-admitted view is UNSAFE exactly where the "
                       "direct-carrier view of its task is; in FP the two family sets are equal ({decallfp} anchors). "
                       "The anchors that only the direct-carrier view certifies are query defects: {carronlydf} DF family "
                       "anchors with color counts and {carronlyfacdf} DF and {carronlyfacfp} FP facet anchors that fail "
                       "FULL-CONFORM.",
                       "Sec. VI-A (Exclusion leaves the label determined); extended audit details Section C (All views and bounds)",
                       "decallfp carronlydf carronlyfacdf carronlyfacfp", _carrierkeep),
    "CL-CARRIERORDER": ("In all four settings every block combination that exposes the carrier scores above every one that "
                        "excludes it, by at least {crsepgaplo} in Macro-F1.",
                        "Table III; Sec. VI-B (Scores vary under a fixed partition)", "crsepgaplo", _carrierorder),
    "CL-ACCESS": ("The registry step and the FULL-CONFORM pass load no label column ({nlabelearly} label columns); "
                  "the conformance step loads labels only after FULL-CONFORM has written its output and stops otherwise.",
                  "Definition 2(a); Sec. IV (after Algorithm 1); extended audit details Section D", "nlabelearly", _access),
    "CL-REJECT": ("All {niafamw} audited identity-admissible family-task views have positive UNSAFE mass in both datasets.",
                  "Sec. I (Results); Sec. V (Views); Sec. VI-A (Exclusion leaves the label determined, Answer to RQ1)",
                  "niafam niafamw", _reject),
    "CL-KG": ("WN18 and WN18RR keep no held-out triple in training, yet under the WordNet table of reflexive pointers "
              "training triples determine {kgwnu} of the {kgwnn} WN18 and {kgwnrru} of the {kgwnrrn} WN18RR test triples "
              "({kgwnvu}/{kgwnvn} and {kgwnrrvu}/{kgwnrrvn} validation triples). The released WN18RR construction "
              "script removes one relation of each of the {kgwnrrinvw} inverse pairs and reproduces all three splits "
              "from WN18, so its certified test triples use self-reflexive pointers only. Each of the {kgcertn} "
              "certified triples and its training witness are WordNet {kgwnver} pointers in opposite directions "
              "between the same two synsets, with one part of speech per synset; an independent check confirms every "
              "count ({kgchecks} checks).",
              "abstract; Sec. I (Results, Contributions); Sec. II; Sec. VI-C (Link prediction, Answer to RQ3); "
              "Sec. VI-D (Scope and corroboration); Sec. VII; extended audit details Section C",
              "kgwnu kgwnn kgwnrru kgwnrrn kgwnvu kgwnvn kgwnrrvu kgwnrrvn kgwnrrinv kgcertn kgwnver kgwndomrev "
              "kgwndomrevv kgwnakr kgposmatch kgposother kgchecks", _kg),
    "CL-SEN12MS": ("On SEN12MS, with the published 17-to-10 class merge as the view and the released 17-class label as "
                   "the protected query, all {senqc} of the {senn} patches are query-conforming. Path B certifies "
                   "{sencert} patches ({sencertr}), exactly those whose merged maximum is an unmerged class; for "
                   "{senvary} of them ({senvaryfib} fibers) the 17 counts vary across the fiber, and a second state of "
                   "each such fiber is checked in integer arithmetic. Witness pairs make {sennondet} patches "
                   "({sennondetr}) NONDETERMINED, {senndwhole} of them in fibers that lie in one merged group. The rival "
                   "integer programs decide all {senmilp} fibers that are not NONDETERMINED as the closed form does; an "
                   "independent check confirms every count ({senchecks} checks).",
                   "abstract; Sec. I (Results, Contributions); Sec. VI-C (Path B on third-party labels, Answer to RQ3); "
                   "Sec. VI-D (Scope and corroboration); Sec. VII",
                   "senn senclasses sengroups sensupport senqc senfib sencert sencertr sennondet sennondetr senunver senunverr senmilp senchecks "
                   "senlabmatch senlabten senvary senvaryfib senndwhole senndwholefib senndwholer",
                   _sen12ms),
    "CL-RULE": ("The registry rule, which learns no parameters and reads no target label, labels exactly the "
                "query-conforming UNSAFE anchors of the four Path A views on the full basis block, all correctly "
                "({rulefullrows} view-anchor pairs). On the shared projection of the identity-admitted family view it "
                "labels {trulecovfriafamdffp}/{trulecovfriafamfpdf} (DF->FP/FP->DF) of the evaluated registry-conforming "
                "targets with {ruleerrtot} errors in {ruledectot} decisions; on the two core-basis views its precision "
                "is {truleprecfriafacdffp}-{ruleprecoremax}.",
                "abstract; Sec. I (Example 1, Results, Contributions); Sec. V (Probes); "
                "Sec. VI-B (Determined rows need no model, Answer to RQ2); Sec. VII",
                "trulecovfriafamdffp trulecovfriafamfpdf ruleerrtot ruledectot rulefullcells rulefullrows "
                "truleprecfriafacdffp ruleprecoremax", _rule),
    "CL-COST": ("In a single-process reference run, conformance, verdicts and mutation checks take {rttotal} s: FULL-CONFORM {rtfc} s, "
                "QUERY-CONFORM of all {nviews} views {rtqc} s, verdicts and mutation checks {rtaudit} s.",
                "Sec. I (Results); Sec. VI-C (Cost, Answer to RQ3)", "rttotal rtfc rtqc rtaudit", _cost),
    "CL-SCALE": ("On this workload Path B integer-program time tracks the single-label fibers more closely than the "
                 "rows.", "Sec. VI-C (Cost); Fig. 3b; extended audit details Section C (Cost per view) and Fig. S3", "scsratio scgratio pbsolved", _scale),
}

# Evidence links of CLAIMS.md, in order.
GROUPS = (
    ("Exposure and drift", ("CL-LABELMAX", "CL-HEAD", "CL-TIE", "CL-CARRIERKEEP", "CL-REJECT", "CL-FACET", "CL-DRIFT", "CL-BUILD")),
    ("Mechanism", ("CL-FIBER", "CL-DECOMP", "CL-DRIFTMECH")),
    ("Certification", ("CL-PBX", "CL-MUT", "CL-SUPPORT", "CL-TIEORDER", "CL-SUPDEF", "CL-PAIRS")),
    ("Execution without training", ("CL-BOUND", "CL-ACCESS")),
    ("Label recovery and scores", ("CL-RULE", "CL-PARTSCORE", "CL-CARRIERORDER", "CL-NULL", "CL-L3",
                                "CL-VST", "CL-TOL", "CL-SIMPLE")),
    ("Breadth and cost", ("CL-BREADTH", "CL-GOLD", "CL-KG", "CL-SEN12MS", "CL-COST", "CL-SCALE")),
    ("Reproduction", ("CL-RERUN", "CL-INDEP")),
)
# Scope notes printed under a group of CLAIMS.md (templates over cell ids).
NOTES = {
    "Certification": (
        "Evidence of CL-PBX: outputs/verdicts/path_b_certificates.csv (all certificates), "
        "outputs/verdicts/path_b_open_witnesses.csv (all {pbopen} open fibers), "
        "outputs/verdicts/path_b_open_witnesses_public.csv (the subset that make check rechecks from the outputs "
        "alone) and outputs/verdicts/path_b_open_summary.json (counts and hashes). The independent verifier rechecks "
        "every open fiber from the input view (outputs/checks/independent_checks.csv)."),
    "Breadth and cost": (
        "Scope of CL-KG. It concerns WN18 and WN18RR only. A split withholds each held-out triple by its identity "
        "and admits the rest of the graph, the analogue of identity-based admission for splits. WN18 merges synsets "
        "that share an offset across parts of speech, so each certified pair is matched with one part of speech per "
        "synset. The WN18RR test count equals the one published by Akrami et al. (SIGMOD 2020). Reading the Domain row "
        "in both directions adds {kgwndomrev} WN18 test triples ({kgwnakr} in all). Held-out triples without a "
        "training witness are UNVERIFIED. "
        "Scope of CL-SEN12MS. The view is the merged counts that the official loader computes for the 10-class label, "
        "and the protected query is the released 17-class label; counts, map and labels come from the third-party "
        "release. With one exposed image of a total function, Theorem 1(a) applies and the certified patches have a "
        "closed form. The rival states of UNVERIFIED fibers are states of the fiber, not observed patches."),
}
# Rows of the main-results table at the top of CLAIMS.md: (result, cell template, claim).
MAIN_RESULTS = (
    ("Identity-admitted family view, UNSAFE mass (DF/FP)", "{tuiafamdf}/{tuiafamfp}", "CL-HEAD"),
    ("FP: identity-admitted and direct-carrier family UNSAFE sets equal", "{decallfp} anchors", "CL-CARRIERKEEP"),
    ("Anchors that fail FULL-CONFORM and pass identity and row-count checks", "{ndrift}", "CL-DRIFT"),
    ("Registry rule on the identity-admitted family view: errors in decisions", "{ruleerrtot} in {ruledectot}", "CL-RULE"),
    ("Type view, UNSAFE mass certified through Path B (DF/FP)", "{tutthreedf}/{tutthreefp}", "CL-PBX"),
    ("Path B certificates verified exactly; open single-label fibers", "{pbunsafe}; {pbopen}", "CL-PBX"),
    ("MaDI-Bench L1 source cells", "{extlone} of {extne}", "CL-BREADTH"),
    ("WN18 / WN18RR test triples determined (L1)", "{kgwnu}/{kgwnn}; {kgwnrru}/{kgwnrrn}", "CL-KG"),
    ("SEN12MS patches: Path B certified; NONDETERMINED (released 17-class label)", "{sencertr}; {sennondetr}", "CL-SEN12MS"),
    ("Independent verifier checks (all pass)", "{indchecks}", "CL-INDEP"),
    ("Non-timing outputs byte-identical across two full runs", "{nrerunsame}", "CL-RERUN"),
)
CONTROLS_FILES = ("utility/partition_identity.csv", "utility/simple_checks.csv", "utility/simple_checks_details.csv",
                  "utility/simple_checks_values.csv", "utility/permutation.csv", "utility/permutation_null.csv",
                  "utility/controls.json")


def evaluate(out: Outputs, cells: list[Cell]) -> list[dict[str, str]]:
    """One row per claim; the statement is filled with the plain-text displays of its cells."""
    by_id = {c.cell_id: c for c in cells}
    rows = []
    for cid, (template, loc, numbers, check) in CLAIMS.items():
        missing = [x for x in numbers.split() if x not in by_id]
        if missing:
            raise SystemExit(f"claim {cid} names unknown cells {missing}")
        statement = template.format(**{k: plain(v.display) for k, v in by_id.items()})
        rows.append({"claim_id": cid, "status": "PASS" if check(out) else "FAIL", "statement": statement,
                     "manuscript_locations": loc, "number_cells": numbers})
    if {cid for _, ids in GROUPS for cid in ids} != set(CLAIMS) or len([c for _, ids in GROUPS for c in ids]) != len(CLAIMS):
        raise SystemExit("CLAIMS.md groups do not cover the claims exactly once")
    if any(cid not in CLAIMS for _, _, cid in MAIN_RESULTS):
        raise SystemExit("a main-results row names an unknown claim")
    return rows


def claims_csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def read_claims(path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def make_target(name: str) -> str:
    """Make target that writes a ledger output."""
    if not name.startswith("outputs/"):
        return "declared in " + name
    rel = name[len("outputs/"):]
    if rel in CONTROLS_FILES:
        return "controls"
    if rel == "checks/data_access_checks.json":
        return "conformance"
    if rel == "checks/rerun_equivalence.json":
        return "declared in outputs/checks/rerun_equivalence.json (record of two full reruns)"
    if rel == "external/sen12ms_pathb.json":
        return "sen12ms"
    if rel.startswith("checks/"):
        return "verify"
    return rel.split("/", 1)[0]


def claims_md(rows: list[dict[str, str]], cells: list[Cell]) -> str:
    by_id = {c.cell_id: c for c in cells}
    by_claim = {r["claim_id"]: r for r in rows}
    disp = {k: plain(v.display) for k, v in by_id.items()}
    lines = ["# Claims", "",
             "`make check` recomputes the status of every claim and every number cell from `outputs/`. "
             "`paper/claims.csv` lists the cells of each claim, and `paper/number_ledger.csv` the output file and "
             "SHA-256 of each cell.", "",
             "## Main results", "",
             "| Result | Value | Claim | Output(s) |", "|---|---|---|---|"]
    for label, template, cid in MAIN_RESULTS:
        numbers = by_claim[cid]["number_cells"].split()
        outs = sorted({o for x in numbers for o in by_id[x].outputs})
        lines.append(f"| {label} | {template.format(**disp)} | {cid} ({by_claim[cid]['status']}) | "
                     + ", ".join(f"`{o}`" for o in outs) + " |")
    lines += ["", "## All claims", "", "| Claim | Statement | Paper | Command |", "|---|---|---|---|"]
    for group, ids in GROUPS:
        for cid in ids:
            r = by_claim[cid]
            targets = []
            for x in r["number_cells"].split():
                for o in by_id[x].outputs:
                    t = make_target(o)
                    t = f"`make {t}`" if not t.startswith("declared") else t
                    if t not in targets:
                        targets.append(t)
            lines.append(f"| {cid} ({r['status']}) | {r['statement']} | {r['manuscript_locations']} | "
                         f"{', '.join(targets)} |")
    notes = [NOTES[g].format(**disp) for g, _ in GROUPS if g in NOTES]
    if notes:
        lines += ["", "## Scope notes", ""]
        for n in notes:
            lines += [n, ""]
    return "\n".join(lines).rstrip("\n") + "\n"
