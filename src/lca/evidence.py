# SPDX-License-Identifier: MIT
"""Evidence step: analyses that sharpen claims of Sections VI-A and VI-B with the evaluator of the utility step.

refit_separation         the FP->DF identity-admitted family refit decomposed into rows and shared coordinates:
                         fits on all source anchors or on the registry-conforming ones, each with the shared
                         coordinates of either fit
verdict_class_composition where the correct cosine predictions on the family-3 views fall among their own verdicts
shared_basis_drop        the share of a target's basis coordinates that the shared projection of the fit drops,
                         for the core and the facet basis of the registry rule
volume_null              999 fits on random source draws with the per-label sizes of the registry-conforming fit;
                         its first five draws are the volume control of the utility step
metric_labels            per-label F1 of the registry rule and the cosine probe on the four Path A rule views (refit,
                         registry-conforming targets), and Macro-F1 over the metric labels with at least three targets
drift_type_mechanism     per dataset, how the copied role blocks of the drifted anchors relate to the cleaned core
                         block: type block = type counts of core.Pi plus the facet excess counted as attribute events"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from . import evaluation as ev
from . import probes

NULL_DRAWS = 999
COMPOSITION_VIEWS = ("family3__facet", "family3_plus_type__facet")
BASES = ("core::", "facet::")

SEPARATION_FIELDS = ["case", "fit_rows", "coordinates", "view_id", "direction", "source_rows_fit", "shared_feature_count",
                     "target_eval_rows", "macro_f1", "accuracy", "dropped_coordinates", "rows_carrying_dropped",
                     "drifted_rows_carrying_dropped"]
COMPOSITION_FIELDS = ["view_id", "direction", "training_protocol", "evaluation_scope", "target_eval_rows", "macro_f1",
                      "accuracy", "macro_f1_on_nondetermined_targets", "targets_unsafe", "targets_nondetermined",
                      "targets_unverified", "correct_predictions", "correct_on_unsafe", "correct_on_nondetermined",
                      "correct_on_unverified", "fraction_correct_on_nondetermined"]
DROP_FIELDS = ["basis", "direction", "training_protocol", "evaluation_scope", "basis_coordinates", "shared_coordinates",
               "targets_with_basis_evidence", "mean_fraction_coordinates_dropped", "median_fraction_coordinates_dropped",
               "share_targets_losing_most_coordinates", "share_targets_losing_all_coordinates",
               "fraction_basis_events_dropped"]
NULL_FIELDS = ["seed", "view_id", "direction", "source_rows_fit", "shared_feature_count", "target_eval_rows",
               "macro_f1", "accuracy"]
CODE = {"UNVERIFIED": 0, "UNSAFE": 1, "NONDETERMINED": 2}
MIN_TARGETS = 3   # a metric label with fewer targets is reported separately (Macro-F1 weighs every label equally)
SMALL_TARGETS = 10
LABEL_F1_FIELDS = ["view_id", "task", "direction", "predictor", "metric_label", "targets", "predicted", "correct", "f1"]
LABEL_SUMMARY_FIELDS = ["view_id", "task", "direction", "predictor", "metric_labels", "macro_f1",
                        "labels_with_min_targets", "macro_f1_on_labels_with_min_targets", "labels_below_small_targets",
                        "small_label_targets"]
DRIFT_FIELDS = ["dataset", "rows", "drifted_rows", "type_equals_core_types_plus_excess_as_attribute", "excess_events",
                "excess_events_on_category_facets", "excess_events_on_color", "type_attribute_surplus_events",
                "T_fail_rows", "T_fail_drifted_rows", "W_fail_rows", "W_fail_rows_with_color_excess",
                "W_fail_rows_without_color_excess"]


def _direction(source: str, target: str) -> str:
    return f"{source}->{target}"


def load_verdicts(out: Path, cache: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-anchor verdict codes of every registered view (restricted output of the verdict step)."""
    with (out / "verdicts" / "anchor_verdicts_index.csv").open(encoding="utf-8", newline="") as fh:
        index = list(csv.DictReader(fh))
    with np.load(out / "verdicts" / "anchor_verdicts.npz", allow_pickle=False) as z:
        code = np.asarray(z["verdict_code"])
    n = len(cache["row_id"])
    if code.shape != (len(index), n) or any(int(r["row_count"]) != n for r in index):
        raise SystemExit("verdict array and anchors differ in shape")
    with np.load(out / "conformance" / "query_masks.npz", allow_pickle=False) as z:
        if not np.array_equal(np.asarray(z["dataset_code"]), np.where(cache["dataset"] == "DF", 0, 1)):
            raise SystemExit("verdict row order differs from the anchors")
    return {r["view_id"]: code[int(r["array_index"])] for r in index}


def refit_separation(cache: dict[str, Any], views: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    source, target = probes.VOLUME_DIRECTION
    task, view_id = "family", probes.PREDICTION_VIEW
    cols = ev.view_columns(cache, views, view_id)
    _, fit_all, labels = ev.fitting_rows(cache, source, task, ev.ALL_SOURCE)
    _, fit_ref, labels_ref = ev.fitting_rows(cache, source, task, ev.REFIT)
    if list(labels) != list(labels_ref):
        raise SystemExit("the two fits have different label inventories")
    sh_all, _ = ev.shared_projection(cache, cols, fit_all)
    sh_ref, _ = ev.shared_projection(cache, cols, fit_ref)
    if len(np.setdiff1d(sh_ref, sh_all)):
        raise SystemExit("the refit support is not contained in the all-source support")
    dropped = np.setdiff1d(sh_all, sh_ref)
    X = cache["X"].tocsr()
    carry = np.asarray(X[fit_all][:, dropped].getnnz(axis=1) > 0).ravel() if len(dropped) else np.zeros(len(fit_all), bool)
    carriers = fit_all[carry]
    drifted = int(np.count_nonzero(~cache["full_mask"][carriers]))
    ti, _, metric = ev.target_rows(cache, source, target, task, ev.CONFORM_TARGET)
    y = cache["gold_" + task][ti]
    rows = []
    for case, fit_name, fit, coord_name, shared in (
            ("A", "all_source", fit_all, "all_source_support", sh_all),
            ("B", "all_source", fit_all, "refit_support", sh_ref),
            ("C", "registry_conforming", fit_ref, "refit_support", sh_ref),
            ("D", "registry_conforming", fit_ref, "all_source_support", sh_all)):
        pred, _ = probes.cosine_predict(cache, fit, ti, labels, shared, task)
        f1, acc = ev.evaluate_prediction(y, pred, metric)
        rows.append({"case": case, "fit_rows": fit_name, "coordinates": coord_name, "view_id": view_id,
                     "direction": _direction(source, target), "source_rows_fit": len(fit),
                     "shared_feature_count": len(shared), "target_eval_rows": len(ti), "macro_f1": f1, "accuracy": acc,
                     "dropped_coordinates": len(dropped), "rows_carrying_dropped": len(carriers),
                     "drifted_rows_carrying_dropped": drifted})
    return rows


def verdict_class_composition(cache: dict[str, Any], views: dict[str, dict[str, Any]],
                              verdicts: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for view_id in COMPOSITION_VIEWS:
        task = views[view_id]["task"]
        cols = ev.view_columns(cache, views, view_id)
        for source, target in ev.DIRECTIONS:
            for protocol, scope in ((ev.ALL_SOURCE, ev.ALL_TARGET), (ev.REFIT, ev.CONFORM_TARGET)):
                _, fit, labels = ev.fitting_rows(cache, source, task, protocol)
                shared, _ = ev.shared_projection(cache, cols, fit)
                ti, _, metric = ev.target_rows(cache, source, target, task, scope)
                pred, _ = probes.predict("cosine", cache, fit, ti, labels, shared, task)
                gold = cache["gold_" + task][ti]
                f1, acc = ev.evaluate_prediction(gold, pred, metric)
                v = verdicts[view_id][ti]
                ok = pred == gold
                cls = {k: v == c for k, c in CODE.items()}
                f1_nondet, _ = ev.evaluate_prediction(gold[cls["NONDETERMINED"]], pred[cls["NONDETERMINED"]], metric)
                n_ok = int(np.count_nonzero(ok))
                rows.append({"view_id": view_id, "direction": _direction(source, target), "training_protocol": protocol,
                             "evaluation_scope": scope, "target_eval_rows": len(ti), "macro_f1": f1, "accuracy": acc,
                             "macro_f1_on_nondetermined_targets": f1_nondet,
                             "targets_unsafe": int(np.count_nonzero(cls["UNSAFE"])),
                             "targets_nondetermined": int(np.count_nonzero(cls["NONDETERMINED"])),
                             "targets_unverified": int(np.count_nonzero(cls["UNVERIFIED"])),
                             "correct_predictions": n_ok,
                             "correct_on_unsafe": int(np.count_nonzero(ok & cls["UNSAFE"])),
                             "correct_on_nondetermined": int(np.count_nonzero(ok & cls["NONDETERMINED"])),
                             "correct_on_unverified": int(np.count_nonzero(ok & cls["UNVERIFIED"])),
                             "fraction_correct_on_nondetermined": (int(np.count_nonzero(ok & cls["NONDETERMINED"])) / n_ok
                                                                    if n_ok else 0.0)})
    return rows


def shared_basis_drop(cache: dict[str, Any]) -> list[dict[str, Any]]:
    X = cache["X"].tocsr()
    rows = []
    for basis in BASES:
        cols = np.asarray([i for i, f in enumerate(cache["features"]) if str(f).startswith(basis)], dtype=np.int64)
        for source, target in ev.DIRECTIONS:
            _, fit, _ = ev.fitting_rows(cache, source, "family", ev.REFIT)
            shared, _ = ev.shared_projection(cache, cols, fit)
            ti, _, _ = ev.target_rows(cache, source, target, "family", ev.CONFORM_TARGET)
            T = X[ti][:, cols]
            keep = np.isin(cols, shared)
            nnz = np.asarray((T != 0).sum(axis=1)).ravel()
            nnz_keep = np.asarray((T[:, keep] != 0).sum(axis=1)).ravel()
            mass = np.asarray(T.sum(axis=1)).ravel()
            mass_keep = np.asarray(T[:, keep].sum(axis=1)).ravel()
            has = nnz > 0
            frac = 1.0 - nnz_keep[has] / nnz[has]
            rows.append({"basis": basis, "direction": _direction(source, target), "training_protocol": ev.REFIT,
                         "evaluation_scope": ev.CONFORM_TARGET, "basis_coordinates": len(cols),
                         "shared_coordinates": len(shared), "targets_with_basis_evidence": int(np.count_nonzero(has)),
                         "mean_fraction_coordinates_dropped": float(frac.mean()),
                         "median_fraction_coordinates_dropped": float(np.median(frac)),
                         "share_targets_losing_most_coordinates": float(np.mean(frac > 0.5)),
                         "share_targets_losing_all_coordinates": float(np.mean(nnz_keep[has] == 0)),
                         "fraction_basis_events_dropped": float(1.0 - mass_keep[has].sum() / mass[has].sum())})
    return rows


def volume_null(cache: dict[str, Any], views: dict[str, dict[str, Any]], draws: int = NULL_DRAWS) -> list[dict[str, Any]]:
    """The volume control of the utility step (probes.volume_table) with seeds 0, ..., draws - 1."""
    source, target = probes.VOLUME_DIRECTION
    task, view_id = "family", probes.PREDICTION_VIEW
    cols = ev.view_columns(cache, views, view_id)
    all_mask, _, labels = ev.fitting_rows(cache, source, task, ev.ALL_SOURCE)
    _, conf_fit, _ = ev.fitting_rows(cache, source, task, ev.REFIT)
    y = cache["gold_" + task]
    ti, _, metric = ev.target_rows(cache, source, target, task, ev.CONFORM_TARGET)
    per_label: dict[str, int] = {}
    for lab in y[conf_fit].tolist():
        per_label[lab] = per_label.get(lab, 0) + 1
    pools = {lab: all_mask[y[all_mask] == lab] for lab in labels}
    rows = []
    for seed in range(draws):
        rng = np.random.default_rng(seed)
        sampled: list[int] = []
        for lab in labels:
            n = int(per_label.get(lab, 0))
            if n > len(pools[lab]):
                raise SystemExit("volume null: sample size exceeds the label pool")
            if n:
                sampled.extend(rng.choice(pools[lab], size=n, replace=False).tolist())
        fit = np.asarray(sorted(sampled), dtype=np.int64)
        shared, _ = ev.shared_projection(cache, cols, fit)
        pred, _ = probes.cosine_predict(cache, fit, ti, labels, shared, task)
        f1, acc = ev.evaluate_prediction(y[ti], pred, metric)
        rows.append({"seed": seed, "view_id": view_id, "direction": _direction(source, target),
                     "source_rows_fit": len(fit), "shared_feature_count": len(shared), "target_eval_rows": len(ti),
                     "macro_f1": f1, "accuracy": acc})
    return rows


def _per_label(y: np.ndarray, pred: np.ndarray, metric: list[str]) -> list[dict[str, Any]]:
    out = []
    for lab in metric:
        tgt = int(np.count_nonzero(y == lab))
        prd = int(np.count_nonzero(pred == lab))
        cor = int(np.count_nonzero((y == lab) & (pred == lab)))
        out.append({"metric_label": lab, "targets": tgt, "predicted": prd, "correct": cor,
                    "f1": (2.0 * cor / (prd + tgt)) if (prd + tgt) else 0.0})
    return out


def metric_labels(cache: dict[str, Any], views: dict[str, dict[str, Any]], Pi: dict[str, str], W: dict[str, str],
                  coords: dict[str, list[str]], certificates: list[dict[str, str]]
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Registry rule and cosine probe (conforming refit, registry-conforming targets) label by label."""
    X, features = cache["X"], cache["features"]
    per_label, summary = [], []
    for cert_row in certificates:
        view_id, task = cert_row["view_id"], cert_row["task"]
        basis, edges = cert_row["basis"], cert_row["carrier_edges"]
        labels = coords[task]
        gold = cache["gold_" + task]
        basis_cols = ev.selected_columns(features, {basis: "ALL"})
        view_cols = ev.view_columns(cache, views, view_id)
        for source, target in ev.DIRECTIONS:
            _, fit, src_labels = ev.fitting_rows(cache, source, task, ev.REFIT)
            ti, _, metric = ev.target_rows(cache, source, target, task, ev.CONFORM_TARGET)
            y = gold[ti]
            shared, _ = ev.shared_projection(cache, basis_cols, fit)
            M, _, unmapped = probes.mapping_matrix(features, shared, basis, task, edges, labels, Pi, W)
            if unmapped:
                raise SystemExit(f"shared basis has unmapped coordinates: {view_id} {source}->{target}")
            rule_pred, _ = probes.strict_argmax(probes.rule_scores(X, ti, shared, M), labels)
            shc, _ = ev.shared_projection(cache, view_cols, fit)
            cos_pred, _ = probes.cosine_predict(cache, fit, ti, src_labels, shc, task)
            for name, pred in (("registry_rule", rule_pred), ("cosine", cos_pred)):
                rows = _per_label(y, pred, list(metric))
                for r in rows:
                    per_label.append({"view_id": view_id, "task": task, "direction": _direction(source, target),
                                      "predictor": name, **r})
                macro, _ = ev.evaluate_prediction(y, pred, metric)
                keep = [r for r in rows if r["targets"] >= MIN_TARGETS]
                small = [r for r in rows if r["targets"] < SMALL_TARGETS]
                summary.append({"view_id": view_id, "task": task, "direction": _direction(source, target),
                                "predictor": name, "metric_labels": len(rows), "macro_f1": macro,
                                "labels_with_min_targets": len(keep),
                                "macro_f1_on_labels_with_min_targets": float(np.mean([r["f1"] for r in keep])),
                                "labels_below_small_targets": len(small),
                                "small_label_targets": "|".join(f"{r['metric_label']}={r['targets']}" for r in small)})
    return per_label, summary


def drift_type_mechanism(cache: dict[str, Any], Pi: dict[str, str], T: dict[str, str], W: dict[str, str]
                         ) -> list[dict[str, Any]]:
    """Per dataset: the drifted anchors (facet block above core.Pi) and how the copied type and family blocks treat
    the excess facet events."""
    X = cache["X"].tocsr()
    feats = [str(f) for f in cache["features"]]
    facets = sorted(T)
    fi = {f: i for i, f in enumerate(facets)}
    types = sorted(set(T.values()))
    ti_ = {t: i for i, t in enumerate(types)}
    fams = sorted(set(W.values()))
    wi = {w: i for i, w in enumerate(fams)}

    def block(prefix: str, index: dict[str, int]) -> sparse.csr_matrix:
        cols, tgt = [], []
        for j, name in enumerate(feats):
            if name.startswith(prefix):
                key = name[len(prefix):]
                if key not in index:
                    raise SystemExit(f"unknown {prefix} coordinate {key}")
                cols.append(j)
                tgt.append(index[key])
        P = sparse.csr_matrix((np.ones(len(cols)), (np.arange(len(cols)), tgt)), shape=(len(cols), len(index)))
        return (X[:, cols] @ P).tocsr()

    core_cols = [j for j, name in enumerate(feats) if name.startswith("core::")]
    rows_, tgt = [], []
    for k, j in enumerate(core_cols):
        term = feats[j][len("core::"):]
        if term not in Pi:
            raise SystemExit(f"core term without a facet: {term}")
        rows_.append(k)
        tgt.append(fi[Pi[term]])
    Pc = sparse.csr_matrix((np.ones(len(core_cols)), (rows_, tgt)), shape=(len(core_cols), len(facets)))
    core_pi = (X[:, core_cols] @ Pc).toarray()
    facet = block("facet::", fi).toarray()
    typ = block("type::", ti_).toarray()
    fam = block("family::", wi).toarray()
    Tm = np.zeros((len(facets), len(types)))
    Wm = np.zeros((len(facets), len(fams)))
    for f, i in fi.items():
        Tm[i, ti_[T[f]]] = 1
        Wm[i, wi[W[f]]] = 1
    excess = facet - core_pi
    drifted = np.any(excess != 0, axis=1)
    if np.any(excess < 0):
        raise SystemExit("an anchor's facet block falls below core.Pi")
    # compare on the materialized type and family coordinates only (type:: materializes attribute, category, part)
    tcols = [ti_[t] for t in types if "type::" + t in feats]
    wcols = [wi[w] for w in fams if "family::" + w in feats]
    expected = core_pi @ Tm
    expected[:, ti_["attribute"]] += excess.sum(axis=1)
    identity = np.all(expected[:, tcols] == typ[:, tcols], axis=1)
    t_fail = np.any((facet @ Tm)[:, tcols] != typ[:, tcols], axis=1)
    w_fail = np.any((facet @ Wm)[:, wcols] != fam[:, wcols], axis=1)
    cat = np.asarray([f.startswith("category/") for f in facets])
    color = fi["color"]
    color_exc = excess[:, color] > 0
    out = []
    for ds in ("DF", "FP"):
        m = cache["dataset"] == ds
        out.append({"dataset": ds, "rows": int(m.sum()), "drifted_rows": int(np.count_nonzero(m & drifted)),
                    "type_equals_core_types_plus_excess_as_attribute": int(np.count_nonzero(m & drifted & identity)),
                    "excess_events": int(excess[m].sum()),
                    "excess_events_on_category_facets": int(excess[m][:, cat].sum()),
                    "excess_events_on_color": int(excess[m][:, color].sum()),
                    "type_attribute_surplus_events": int((typ[m][:, ti_["attribute"]]
                                                          - (core_pi[m] @ Tm)[:, ti_["attribute"]]).sum()),
                    "T_fail_rows": int(np.count_nonzero(m & t_fail)),
                    "T_fail_drifted_rows": int(np.count_nonzero(m & t_fail & drifted)),
                    "W_fail_rows": int(np.count_nonzero(m & w_fail)),
                    "W_fail_rows_with_color_excess": int(np.count_nonzero(m & w_fail & color_exc)),
                    "W_fail_rows_without_color_excess": int(np.count_nonzero(m & w_fail & ~color_exc))})
    return out
