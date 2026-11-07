# SPDX-License-Identifier: MIT
"""Five controlled auditor mutations (M1-M5) and the label-domain cross-check of M5."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from . import certificates as cert
from . import conformance as cf
from . import verdicts as vd
from .io import array_sha
from .names import DATASETS

# One fixed seed per mutation check; each mutation writes its seed into its output.
SEEDS = {"M1": 16210331290358548553, "M2": 13368404821911376597, "M3": 7748174076895547349,
         "M4": 10992426848931531092, "M5": 17545134684022528852}
# The core identifiers that M2 remaps: one percent of the released core identifiers, drawn once with the M2 seed.
M2_REMAPPED_CORE_UIDS = (
    "hub:core:term/attribute/90e706fb816933b9", "hub:core:term/attribute/f78db8da7b522ba4",
    "hub:core:term/attribute/dd7552f89a9cdc37", "hub:core:term/attribute/a5271adfec53f301",
    "hub:core:term/attribute/7502f3098f8226de", "hub:core:term/attribute/5253cc49d6fa4a82",
    "hub:core:term/attribute/d8d1f2d680bdb813", "hub:core:term/attribute/4bde3ec350d69beb",
    "hub:core:term/attribute/bd862ee7e7ed4be1", "hub:core:term/attribute/5027d6998f68d84c",
    "hub:core:term/attribute/e4a7a15e22a83fab", "hub:core:term/attribute/0eba5983faabcca1",
)
CORE_UID_PREFIX = "hub:"
SEED_RULE = "one fixed unsigned 64-bit seed per mutation check (SEEDS)"
DESIGN_NOTE = "each mutation uses a fixed seed; M2 remaps the declared identifiers M2_REMAPPED_CORE_UIDS"
GLOBAL_MUTATIONS = "M1/M3/M4 apply to all rows in the conformance row order"
M2_FRACTION = 0.01
M4_BLOCK_NAMES = {"family::": "family3", "type::": "type3"}
DIRECT_VIEWS = {"facet": "l0_facet__facet", "family": "l0_family__family"}


def mutation_seed(mutation_id: str) -> tuple[int, str]:
    seed = SEEDS[mutation_id]
    return seed, seed.to_bytes(8, "big").hex()


def canonical_mapping_sha256(mapping: dict[str, str]) -> str:
    payload = json.dumps({str(k): str(v) for k, v in mapping.items()}, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def partition_codes(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sub = np.ascontiguousarray(a[np.flatnonzero(mask)])
    if sub.ndim == 1:
        sub = sub[:, None]
    if len(sub) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort(tuple(sub[:, j] for j in range(sub.shape[1] - 1, -1, -1)))
    ss = sub[order]
    boundary = np.empty(len(ss), dtype=bool)
    boundary[0] = True
    if len(ss) > 1:
        boundary[1:] = np.any(ss[1:] != ss[:-1], axis=1)
    inv = np.empty(len(ss), dtype=np.int64)
    inv[order] = np.cumsum(boundary, dtype=np.int64) - 1
    return inv


def partition_equivalent(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> bool:
    """Equal partitions of the masked rows iff each code determines the other."""
    ca, cb = partition_codes(a, mask), partition_codes(b, mask)
    if len(ca) == 0:
        return True
    pair = ca.astype(np.uint64) * np.uint64(int(cb.max()) + 1) + cb.astype(np.uint64)
    return len(np.unique(pair)) == len(np.unique(ca)) == len(np.unique(cb))


def select_registered_certificate(edge_specs: list[dict[str, Any]], exposed_blocks: list[str], carrier_block: str,
                                  basis_priority: dict[str, int]) -> dict[str, Any]:
    """Shortest registered path from an exposed block to the carrier (breadth-first); ties by declared basis
    priority, then basis name, then edge ids."""
    adj: dict[str, list[tuple[str, str]]] = {}
    seen_edges = set()
    for e in edge_specs:
        s, t = str(e["source_block"]), str(e["target_block"])
        if (s, t) in seen_edges:
            raise SystemExit(f"duplicate registered transform edge {s}->{t}")
        seen_edges.add((s, t))
        adj.setdefault(s, []).append((t, str(e["edge_id"])))
    candidates = []
    for basis in exposed_blocks:
        prev: dict[str, tuple[str, str] | None] = {basis: None}
        dq = deque([basis])
        while dq:
            u = dq.popleft()
            if u == carrier_block:
                break
            for v, eid in adj.get(u, []):
                if v not in prev:
                    prev[v] = (u, eid)
                    dq.append(v)
        if carrier_block not in prev:
            continue
        nodes, edges, cur = [carrier_block], [], carrier_block
        while prev[cur] is not None:
            u, eid = prev[cur]
            nodes.append(u); edges.append(eid); cur = u
        nodes.reverse(); edges.reverse()
        candidates.append((len(edges), int(basis_priority.get(basis, 10_000)), basis, edges, nodes))
    if not candidates:
        raise SystemExit("no registered path from an exposed block to the carrier")
    candidates.sort(key=lambda x: (x[0], x[1], x[2], tuple(x[3])))
    length, _p, basis, edges, nodes = candidates[0]
    return {"basis": basis, "path_length": int(length), "carrier_edges": edges, "path_nodes": nodes,
            "selection_rule": "shortest registered path from exposed block to carrier; "
                              "declared basis priority then lexical tie-break"}


def _audit_with(crow: dict[str, str], ds: str, mask: np.ndarray, labels: np.ndarray, label_names: list[str],
                reg: dict[str, Any], ev: dict[str, Any], conf: dict[str, Any], keys, values):
    return vd.audit_view(crow, ds, mask, labels, label_names, reg, ev, conf, keys, values)


# ====================================================================================================
# M1: registered invertible map y = facet * B, Path B -> Path A
# ====================================================================================================

def mutation_m1(reg, conf, ev, l0_verdict: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    facet_coords = reg["facet_coords"]
    nf = len(facet_coords)
    fidx = {x: i for i, x in enumerate(facet_coords)}
    seed, seed_hex = mutation_seed("M1")
    sup = {ds: set(reg["support"]["datasets"][ds]["S_members"]) for ds in DATASETS}
    inter = [f for f in facet_coords if all(f in sup[ds] for ds in DATASETS)]
    inter_idx = [fidx[x] for x in inter]
    key = seed.to_bytes(8, "big")
    pairs = [(i, j) for i in inter_idx for j in inter_idx if i < j]
    ranked = sorted(pairs, key=lambda ij: hashlib.sha256(key + f"{ij[0]}:{ij[1]}".encode()).digest())
    chosen = ranked[: max(1, len(inter_idx) - 1)]
    b_mat = np.eye(nf, dtype=np.int64)
    for i, j in chosen:
        b_mat[i, j] += 1
    nil = b_mat - np.eye(nf, dtype=np.int64)
    b_inv = np.eye(nf, dtype=np.int64)
    power = np.eye(nf, dtype=np.int64)
    sign = -1
    for _ in range(1, nf + 1):  # finite Neumann series: N is strictly upper triangular, hence nilpotent
        power = power @ nil
        if not np.any(power):
            break
        b_inv += sign * power
        sign *= -1
    eye = np.eye(nf, dtype=np.int64)
    if not np.array_equal(b_mat @ b_inv, eye) or not np.array_equal(b_inv @ b_mat, eye):
        raise SystemExit("M1: exact inverse of B failed")
    facet = ev["facet"].astype(np.int64)
    y = facet @ b_mat
    y_nonnegative = bool(np.all(y >= 0))
    recovered = y @ b_inv
    if not np.array_equal(recovered, facet):
        raise SystemExit("M1: y * B^-1 does not recover the facet block")
    authority = {"Pi": canonical_mapping_sha256(reg["Pi"]), "T": canonical_mapping_sha256(reg["T"]),
                 "W": canonical_mapping_sha256(reg["W"])}
    inverse_sha = array_sha(b_inv)
    edge_specs = [
        {"edge_id": "Pi", "source_block": "core::", "target_block": "facet::", "matrix_sha256": authority["Pi"]},
        {"edge_id": "T", "source_block": "facet::", "target_block": "type::", "matrix_sha256": authority["T"]},
        {"edge_id": "W", "source_block": "facet::", "target_block": "family::", "matrix_sha256": authority["W"]},
        {"edge_id": "B_inverse", "source_block": "y", "target_block": "facet::", "matrix_sha256": inverse_sha},
    ]
    sel = select_registered_certificate(edge_specs, ["family::", "y"], "facet::",
                                        {"facet::": 0, "core::": 1, "family::": 2, "type::": 3, "y": 4})
    edges = sel["carrier_edges"]
    m1_cert = {"view_id": "family3_y__facet", "task": "facet", "basis": sel["basis"],
               "basis_support": f"Supp_i({sel['basis']})", "certificate_path": "A",
               "required_maps": "|".join(edges), "carrier_edges": "|".join(edges), "rule": sel["selection_rule"],
               "path_nodes": sel["path_nodes"], "mutation_registry_edges": edge_specs,
               "selected_registered_matrix_sha256": inverse_sha if edges == ["B_inverse"] else ""}
    # Mutated support (from y itself) must equal the L0 facet support mask.
    m1_query = np.zeros(conf["n"], dtype=bool)
    for ds in DATASETS:
        dm = vd.dataset_mask(conf, ds)
        outside = [j for j in range(nf) if facet_coords[j] not in sup[ds]]
        ok = np.any(y != 0, axis=1) & (np.all(y[:, outside] == 0, axis=1) if outside else True)
        m1_query[dm] = ok[dm]
    l0_i = [r["view_id"] for r in reg["cert"]].index("l0_facet__facet")
    if not np.array_equal(m1_query, conf["qmask"][l0_i]):
        raise SystemExit("M1: mutated support mask differs from the L0 facet support mask")
    fam3 = ev["family"][:, [reg["family_coords"].index(x) for x in vd.COORDINATE_PROJECTIONS["family3__facet"]["family::"]]]
    m1_keys = np.ascontiguousarray(np.concatenate([y, fam3.astype(np.int64)], axis=1))
    labels, names = vd.labels_for("facet", reg, ev)
    strict = cert.strict_unique_max(recovered)
    m1_verdict = np.full(conf["n"], vd.UNVERIFIED, dtype=np.uint8)
    for ds in DATASETS:
        dm = vd.dataset_mask(conf, ds)
        v, *_ = vd.audit_path_a("family3_y__facet", "y", "B_inverse", m1_query, m1_keys, labels, names, dm, ev, conf,
                                strict_label=strict)
        m1_verdict[dm] = v[dm]
    equal_l0 = bool(np.array_equal(m1_verdict, l0_verdict))
    by_id = {e["edge_id"]: e["matrix_sha256"] for e in edge_specs}
    sha_ok = all(by_id[k] == authority[k] for k in ("Pi", "T", "W")) and by_id["B_inverse"] == inverse_sha
    selector_ok = (m1_cert["basis"] == "y" and m1_cert["certificate_path"] == "A"
                   and m1_cert["carrier_edges"] == "B_inverse" and m1_cert["selected_registered_matrix_sha256"] == inverse_sha
                   and sha_ok)
    passed = y_nonnegative and equal_l0 and selector_ok
    manifest = {
        "seed": seed, "support_intersection": inter,
        "N_nonzero_pairs": [[facet_coords[i], facet_coords[j]] for i, j in chosen],
        "B": b_mat.tolist(), "B_inverse": b_inv.tolist(),
        "mutated_block": "y = facet:: * B", "registered_edge": "y -> facet:: via B_inverse",
        "certificate_spec": m1_cert,
        "certificate_reselection": {"path_changed_from_original_family3_facet": "B->A", "selected_basis": m1_cert["basis"],
                                    "carrier_edges": m1_cert["carrier_edges"]},
        "all_y_nonnegative": y_nonnegative, "rowwise_verdict_equals_l0_facet": equal_l0,
        "generic_registry_selector_ok": selector_ok,
        "authority_mapping_hash_rule": "sha256(utf8(json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(',', ':'))))",
        "authority_mapping_sha256": authority, "authority_mapping_sha_verified": sha_ok, "pass": passed}
    result = {"mutation_id": "M1", "pass": int(passed),
              "primary_check": "registered B^-1; Path B->A; rowwise verdict equals L0 facet",
              "detail": f"nonnegative={y_nonnegative}; selector_ok={selector_ok}; rowwise_equal={equal_l0}"}
    return manifest, result, b_mat, b_inv


# ====================================================================================================
# M2: 1% registry remap of released core identifiers
# ====================================================================================================

def m2_selected_uids(reg: dict[str, Any], seed: int) -> list[str]:
    hub = list(reg["pops"]["hub_core_members"])
    if any(not u.startswith(CORE_UID_PREFIX) for u in hub):
        raise SystemExit("hub core identifier without the expected namespace")
    m2_n = max(1, int(round(M2_FRACTION * len(hub))))
    selected = list(M2_REMAPPED_CORE_UIDS)
    if len(selected) != m2_n or not set(selected) <= set(hub):
        raise SystemExit("the declared M2 identifiers are not one percent of the released core identifiers")
    return selected


def mutation_m2(reg, conf, ev, changed: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    seed, seed_hex = mutation_seed("M2")
    selected = m2_selected_uids(reg, seed)
    hub_n = len(reg["pops"]["hub_core_members"])
    released = set(reg["facet_coords"])
    authority_only = [f for f in reg["T"] if f not in released]
    if len(authority_only) != 1:
        raise SystemExit("M2 needs exactly one authority-only facet as remap target")
    target = authority_only[0]
    n = conf["n"]
    member_idx = {u: i for i, u in enumerate(reg["members"]["core::"])}
    sel = np.zeros(len(member_idx), dtype=bool)
    sel[[member_idx[u] for u in selected]] = True
    hit_entry = sel[ev["core_uid"]]
    rows, uids, cnts = ev["core_row"][hit_entry], ev["core_uid"][hit_entry], ev["core_count"][hit_entry]
    # Generator side: rows that contain a selected identifier (membership only).
    expected_hit = np.zeros(n, dtype=bool)
    expected_hit[rows] = True
    # Auditor side: move the selected identifiers' mass to the authority-only facet and re-test Pi conformance.
    selected_facet = np.zeros_like(ev["facet"], dtype=np.int64)
    np.add.at(selected_facet, (rows, ev["pi_facet"][uids]), cnts)
    target_mass = np.zeros(n, dtype=np.int64)
    np.add.at(target_mass, rows, cnts)
    if not np.array_equal(expected_hit, target_mass > 0):
        raise SystemExit("M2: selected-identifier witness and selected mass disagree")
    mutated26 = ev["core_facet"].astype(np.int64) - selected_facet
    if np.any(mutated26 < 0):
        raise SystemExit("M2: mutated Pi produced negative mass")
    mutated_pi_conform = np.all(mutated26 == ev["facet"].astype(np.int64), axis=1) & (target_mass == 0)
    full = conf["full"]
    mutated_full = full & mutated_pi_conform
    audit_new = full & ~mutated_full
    expected_new = expected_hit & full
    tp = int(np.count_nonzero(audit_new & expected_new))
    precision = 1.0 if audit_new.sum() == 0 else float(tp / np.count_nonzero(audit_new))
    recall = 1.0 if expected_new.sum() == 0 else float(tp / np.count_nonzero(expected_new))
    for rid in conf["row_id"][expected_hit]:
        changed.append({"mutation_id": "M2", "task": "registry_remap", "row_id": str(rid)})
    passed = precision == 1.0 and recall == 1.0 and len(selected) == max(1, round(M2_FRACTION * hub_n))
    manifest = {
        "seed": seed,
        "population": f"registry hub_core_members (released {hub_n:,} core coordinates)", "population_size": hub_n,
        "selection_fraction": M2_FRACTION, "selected_count": len(selected), "selected_core_uids": selected,
        "remap_target_facet": target, "rows_containing_selected_uid": int(expected_hit.sum()),
        "newly_failed_previously_full_conform_rows_expected": int(expected_new.sum()),
        "newly_failed_previously_full_conform_rows_audited": int(audit_new.sum()),
        "mutated_pi_conform_rows": int(mutated_pi_conform.sum()), "mutated_target_mass_total": int(target_mass.sum()),
        "baseline_defect_rows_already_containing_selected_uid": int(np.count_nonzero(expected_hit & ~full)),
        "precision": precision, "recall": recall, "pass": passed}
    result = {"mutation_id": "M2", "pass": int(passed),
              "primary_check": "mutation-induced FULL-CONFORM failures equal independently generated selected-UID "
                               "row set on baseline-conforming rows",
              "detail": f"selected={len(selected)}; hit_rows={int(expected_hit.sum())}; precision={precision}; recall={recall}"}
    return manifest, result


# ====================================================================================================
# M3: dataset-constant Path C coordinate; M4: injective post-transform and Binary
# ====================================================================================================

def mutation_m3(reg, conf, ev) -> tuple[dict[str, Any], dict[str, Any]]:
    seed, seed_hex = mutation_seed("M3")
    values = {ds: i + 1 for i, ds in enumerate(DATASETS)}
    constant = np.zeros(conf["n"], dtype=np.uint8)
    for ds in DATASETS:
        constant[vd.dataset_mask(conf, ds)] = values[ds]
    facet_nmi = float(normalized_mutual_info_score(constant, ev["gold_facet"]))
    family_nmi = float(normalized_mutual_info_score(constant, ev["gold_family"]))
    within = all(len(np.unique(constant[vd.dataset_mask(conf, ds)])) == 1 for ds in DATASETS)
    partition_ok = True
    for vid in ("inst_sem__facet", "inst_sem__family"):
        old = vd.view_keys(vid, reg, ev)
        new = np.concatenate([old.astype(np.int64), constant[:, None].astype(np.int64)], axis=1)
        for ds in DATASETS:
            partition_ok &= partition_equivalent(old, new, vd.dataset_mask(conf, ds))
    alarm = facet_nmi > 0 and family_nmi > 0
    passed = bool(within and partition_ok and alarm)
    manifest = {"seed": seed,
                "new_coordinate": "inst_sem::mutation_dataset_constant", "values": values,
                "within_dataset_constant": within, "path_c_partition_unchanged": partition_ok,
                "pooled_NMI": {"facet": facet_nmi, "family": family_nmi}, "pooled_screen_alarm": alarm, "pass": passed}
    result = {"mutation_id": "M3", "pass": int(passed),
              "primary_check": "dataset-constant Path-C coordinate preserves fibers/verdict while pooled NMI alarms",
              "detail": f"NMI_facet={facet_nmi:.6f}; NMI_family={family_nmi:.6f}"}
    return manifest, result


def mutation_m4(reg, conf, ev) -> tuple[dict[str, Any], dict[str, Any]]:
    seed, seed_hex = mutation_seed("M4")
    vid = "family3_plus_type__facet"
    base = vd.dense_projection(vid, reg, ev)
    sb = seed.to_bytes(8, "big")
    transforms, parts, offset = [], [], 0
    for block, names in reg["projection"][vid].items():
        bname = M4_BLOCK_NAMES[block]
        k = len(names)
        ranked = sorted(range(k), key=lambda j: hashlib.sha256(sb + bname.encode() + names[j].encode()).digest())
        scales = [1 + (hashlib.sha256(sb + b"scale" + names[j].encode()).digest()[0] % 4) for j in ranked]
        parts.append(base[:, offset:offset + k][:, ranked] * np.asarray(scales, dtype=np.int64))
        transforms.append({"block": bname, "permutation": [names[j] for j in ranked], "positive_scales": scales})
        offset += k
    inj = np.concatenate(parts, axis=1)
    binary = (base > 0).astype(np.uint8)
    i = [r["view_id"] for r in reg["cert"]].index(vid)
    inj_ok, binary_changed = True, False
    for ds in DATASETS:
        m = vd.dataset_mask(conf, ds) & conf["qmask"][i]
        inj_ok &= partition_equivalent(base, inj, m)
        binary_changed |= not partition_equivalent(base, binary, m)
    manifest = {"seed": seed, "view_id": vid, "injective_transforms": transforms,
                "injective_partition_unchanged": bool(inj_ok), "injective_verdict_unchanged_by_partition_invariance": bool(inj_ok),
                "binary_partition_changed": bool(binary_changed), "binary_rule": "coordinatewise indicator(count>0)",
                "pass": bool(inj_ok)}
    result = {"mutation_id": "M4", "pass": int(bool(inj_ok)),
              "primary_check": "positive rescale/permutation preserves exact partition; Binary sensitivity reported",
              "detail": f"injective_equal={bool(inj_ok)}; binary_changed={bool(binary_changed)}"}
    return manifest, result


# ====================================================================================================
# M5: within-dataset label permutation, stale-mask negative control, full re-audit
# ====================================================================================================

def m5_labels(conf: dict[str, Any], ev: dict[str, Any]) -> tuple[int, str, dict[str, np.ndarray]]:
    seed, seed_hex = mutation_seed("M5")
    children = np.random.SeedSequence(seed).spawn(4)
    out: dict[str, np.ndarray] = {}
    k = 0
    for task, raw in (("facet", ev["gold_facet"]), ("family", ev["gold_family"])):
        lab = raw.copy()
        for ds in DATASETS:
            pos = np.flatnonzero(vd.dataset_mask(conf, ds))
            perm = np.random.default_rng(children[k]).permutation(len(pos))
            k += 1
            lab[pos] = raw[pos[perm]]
        out[task] = lab
    return seed, seed_hex, out


def label_condition(carrier: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Gold label attains the positive carrier maximum."""
    c = carrier.astype(np.int64)
    mx = c.max(axis=1)
    return (mx > 0) & (c[np.arange(len(c)), labels.astype(np.int64)] == mx)


def mutation_m5(reg, conf, ev, baseline_defects: dict[tuple[str, str], int], changed: list[dict[str, str]]):
    seed, seed_hex, mutated = m5_labels(conf, ev)
    changed_counts = {}
    for task, raw in (("facet", ev["gold_facet"]), ("family", ev["gold_family"])):
        diff = mutated[task] != raw
        changed_counts[task] = int(diff.sum())
        for rid in conf["row_id"][diff]:
            changed.append({"mutation_id": "M5", "task": task, "row_id": str(rid)})
    valid = {"facet": label_condition(ev["facet"], mutated["facet"]),
             "family": label_condition(ev["family"], mutated["family"])}
    invalid_counts = {t: {ds: int(np.count_nonzero(vd.dataset_mask(conf, ds) & ~valid[t])) for ds in DATASETS}
                      for t in ("facet", "family")}
    names_for = {"facet": reg["facet_coords"], "family": reg["family_coords"]}
    cert_ids = [r["view_id"] for r in reg["cert"]]

    # Negative control: keep QUERY-CONFORM stale for the direct L0 carrier views; conflicts must appear.
    stale_checks, stale_ok = [], True
    for task, vid in DIRECT_VIEWS.items():
        i = cert_ids.index(vid)
        crow = reg["cert"][i]
        keys = vd.view_keys(vid, reg, ev)
        for ds in DATASETS:
            dm = vd.dataset_mask(conf, ds)
            invalid = conf["qmask"][i] & ~valid[task] & dm
            v, ul, m, _ = vd.audit_path_a(vid, crow["basis"], crow["carrier_edges"], conf["qmask"][i], keys,
                                          mutated[task], names_for[task], dm, ev, conf)[:4]
            unsafe_invalid = int(np.count_nonzero(invalid & (v == vd.UNSAFE)))
            conflict_invalid = int(np.count_nonzero(invalid & (v == vd.UNSAFE) & (ul != mutated[task])))
            observed = (int(invalid.sum()) > 0 and m["certificate_label_conflicts"] > 0 and unsafe_invalid > 0
                        and conflict_invalid > 0)
            stale_ok &= observed
            stale_checks.append({"view_id": vid, "task": task, "dataset": ds, "invalid_stale_query_rows": int(invalid.sum()),
                                 "certificate_label_conflicts": int(m["certificate_label_conflicts"]),
                                 "unsafe_on_invalid_stale_rows": unsafe_invalid,
                                 "conflicts_on_invalid_stale_rows": conflict_invalid,
                                 "expected_failure_observed": bool(observed)})

    # Recomputed label-domain mask, full re-audit of every view.
    failures: list[dict[str, Any]] = []
    checks = invalid_total = unsafe_on_invalid = conflicts_total = delta_mismatch = 0
    for i, crow in enumerate(reg["cert"]):
        vid, task = crow["view_id"], crow["task"]
        q_base = conf["qmask"][i]
        q_mut = q_base & valid[task]
        invalid_q = q_base & ~valid[task]
        keys, values = vd.view_inputs(crow, reg, ev)
        for ds in DATASETS:
            dm = vd.dataset_mask(conf, ds)
            inv_ds = invalid_q & dm
            expected_delta = int(inv_ds.sum())
            invalid_total += expected_delta
            v, _ul, m, _w, _c = _audit_with(crow, ds, q_mut, mutated[task], names_for[task], reg, ev, conf, keys, values)
            u_inv = int(np.count_nonzero(inv_ds & (v == vd.UNSAFE)))
            unsafe_on_invalid += u_inv
            conflicts_total += int(m["certificate_label_conflicts"])
            actual_delta = int(m["unverified_defect_rows"]) - int(baseline_defects[(vid, ds)])
            if actual_delta != expected_delta:
                delta_mismatch += 1
            checks += 1
            if not (u_inv == 0 and actual_delta == expected_delta and not np.any(v[inv_ds] != vd.UNVERIFIED)):
                failures.append({"view_id": vid, "dataset": ds, "path": crow["certificate_path"],
                                 "expected_new_query_defects": expected_delta, "actual_new_query_defects": actual_delta,
                                 "unsafe_on_invalid_rows": u_inv})
    nonvacuous = all(invalid_counts[t][ds] > 0 for t in invalid_counts for ds in DATASETS)
    passed = bool(nonvacuous and stale_ok and len(stale_checks) == 2 * len(DATASETS)
                  and checks == len(reg["cert"]) * len(DATASETS) and not failures and invalid_total > 0
                  and unsafe_on_invalid == 0 and conflicts_total == 0 and delta_mismatch == 0)
    manifest = {
        "seed": seed,
        "operation": "within-dataset label permutation; run a stale-query-mask negative control on the two direct L0 "
                     "carrier views; then recompute the label-domain mask and re-audit all 32 views",
        "scope_note": "the recomputed-mask branch shows fail-closed behaviour; failure capability is shown by the "
                      "stale-mask negative control; the label-domain recomputation is cross-checked bit for bit in "
                      "m5_view_checks.csv",
        "changed_label_rows": changed_counts, "gold_is_carrier_max_false_rows": invalid_counts,
        "full_registry_conform_unchanged_sha256": array_sha(conf["full"]),
        "failure_capable_negative_control": {
            "design": "labels permuted while QUERY-CONFORM is deliberately kept stale for direct L0 carrier views; "
                      "certificate-label conflicts must appear",
            "checks": stale_checks, "pass": bool(stale_ok)},
        "rerun_view_dataset_checks": checks,
        "invalid_baseline_query_rows_across_view_datasets": invalid_total,
        "unsafe_certificates_on_invalid_rows": unsafe_on_invalid,
        "recomputed_certificate_label_conflicts": conflicts_total,
        "query_defect_delta_mismatches": delta_mismatch, "failures": failures, "nonvacuous": nonvacuous,
        "no_unsafe_certificate_on_invalid_label_rows": unsafe_on_invalid == 0, "pass": passed}
    detail = json.dumps({"invalid_counts": invalid_counts, "view_dataset_checks": checks,
                         "invalid_baseline_query_rows": invalid_total, "stale_negative_control_pass": bool(stale_ok),
                         "stale_negative_control_checks": len(stale_checks), "unsafe_on_invalid": unsafe_on_invalid,
                         "recomputed_certificate_label_conflicts": conflicts_total,
                         "defect_delta_mismatches": delta_mismatch, "nonvacuous": nonvacuous},
                        sort_keys=True, separators=(",", ":"))
    result = {"mutation_id": "M5", "pass": int(passed),
              "primary_check": "permuted labels -> recomputed label condition -> full verdicts re-audit; "
                               "invalid rows become UNVERIFIED query defects",
              "detail": detail}
    return manifest, result, mutated, valid


# ====================================================================================================
# M5 cross-check: the conformance step's own label rule and query-mask composition under the permuted labels
# ====================================================================================================

def query_components(reg: dict[str, Any], conf: dict[str, Any], ev: dict[str, Any]) -> dict[str, np.ndarray]:
    """Non-label QUERY-CONFORM components of every row, recomputed from the evidence and the registry:
    Pi: core evidence through Pi equals the facet block; T / W: facet evidence through T / W equals the type /
    family block (a facet whose type is not a released type coordinate breaks T); supports: the block is
    nonempty and inside the dataset's declared C_i (core), S_i (facet) or w(C_i) (family) members.
    Keys follow lca.conformance.compute_query_mask."""
    n = conf["n"]
    facet = ev["facet"].astype(np.int64)
    released_type = ev["Tmat"].sum(axis=1) > 0
    t_ok = np.all(facet @ ev["Tmat"].astype(np.int64) == ev["type"].astype(np.int64), axis=1)
    t_ok &= np.all(facet[:, ~released_type] == 0, axis=1)
    out = {"pi_ok": np.all(ev["core_facet"].astype(np.int64) == facet, axis=1), "t_ok": t_ok,
           "w_ok": np.all(facet @ ev["Wmat"].astype(np.int64) == ev["family"].astype(np.int64), axis=1),
           "supp_c": np.zeros(n, dtype=bool), "supp_s": np.zeros(n, dtype=bool), "supp_f": np.zeros(n, dtype=bool)}
    has_core = np.bincount(ev["core_row"], minlength=n) > 0
    fam = ev["family"].astype(np.int64)
    for ds in DATASETS:
        dm = vd.dataset_mask(conf, ds)
        s = reg["support"]["datasets"][ds]
        c_set = set(s["C_members"])
        c_allowed = np.asarray([u in c_set for u in reg["members"]["core::"]], dtype=bool)
        bad = np.zeros(n, dtype=bool)
        bad[ev["core_row"][~c_allowed[ev["core_uid"]]]] = True
        out["supp_c"][dm] = (has_core & ~bad)[dm]
        out_s = [j for j, f in enumerate(reg["facet_coords"]) if f not in set(s["S_members"])]
        out["supp_s"][dm] = (np.any(facet > 0, axis=1) & np.all(facet[:, out_s] == 0, axis=1))[dm]
        out_f = [j for j, f in enumerate(reg["family_coords"]) if f not in set(s["W_img_members"])]
        out["supp_f"][dm] = (np.any(fam > 0, axis=1) & np.all(fam[:, out_f] == 0, axis=1))[dm]
    return out


def gold_rule_rowwise(carrier: np.ndarray, label_sets: list[np.ndarray], names: list[str]) -> list[np.ndarray]:
    """lca.conformance.gold_is_max on every row (gold label text, the row's nonzero carrier counts), once per
    label assignment."""
    n = carrier.shape[0]
    rows, cols = np.nonzero(carrier)
    vals = carrier[rows, cols].tolist()
    starts = np.searchsorted(rows, np.arange(n + 1)).tolist()
    cols = cols.tolist()
    outs = [np.zeros(n, dtype=bool) for _ in label_sets]
    labs = [ls.tolist() for ls in label_sets]
    for i in range(n):
        d = {names[c]: int(v) for c, v in zip(cols[starts[i]:starts[i + 1]], vals[starts[i]:starts[i + 1]])}
        for k, lab in enumerate(labs):
            outs[k][i] = cf.gold_is_max(names[lab[i]], d)
    return outs


def m5_label_domain_check(reg, conf, ev, mutated: dict[str, np.ndarray], valid_audit: dict[str, np.ndarray]):
    """Recompose all QUERY-CONFORM masks with the conformance step's gold_is_max and compute_query_mask, under the
    original labels (must reproduce the conformance masks bit for bit) and under the M5 labels (must equal the
    mutation audit's mask: the baseline mask restricted to rows whose permuted label still attains the carrier
    maximum)."""
    comp = query_components(reg, conf, ev)
    base_a, mut_a = dict(comp), dict(comp)
    for task, carrier, names in (("facet", ev["facet"], reg["facet_coords"]), ("family", ev["family"], reg["family_coords"])):
        base, mut = gold_rule_rowwise(carrier, [ev["gold_" + task], mutated[task]], names)
        if not np.all(base):
            raise SystemExit(f"original {task} labels do not all attain the carrier maximum")
        if not np.array_equal(mut, valid_audit[task]):
            raise SystemExit(f"conformance label rule differs from the mutation audit under M5 ({task})")
        base_a[f"gold_is_carrier_max_{task}"], mut_a[f"gold_is_carrier_max_{task}"] = base, mut
    detail, settings = [], []
    total_base = total_mut = 0
    q_base_all = np.zeros_like(conf["qmask"])
    q_mut_all = np.zeros_like(conf["qmask"])
    for i, crow in enumerate(reg["cert"]):
        q_base = cf.compute_query_mask(crow, conf["full"], base_a)
        q_mut = cf.compute_query_mask(crow, conf["full"], mut_a)
        audit_mut = conf["qmask"][i] & valid_audit[crow["task"]]
        bmis = int(np.count_nonzero(q_base != conf["qmask"][i]))
        mmis = int(np.count_nonzero(q_mut != audit_mut))
        total_base += bmis
        total_mut += mmis
        q_base_all[i], q_mut_all[i] = q_base, q_mut
        for ds in DATASETS:
            dm = vd.dataset_mask(conf, ds)
            detail.append({"array_index": i, "view_id": crow["view_id"], "task": crow["task"], "dataset": ds,
                           "baseline_query_rows": int(np.count_nonzero(q_base & dm)),
                           "mutated_query_rows": int(np.count_nonzero(q_mut & dm)),
                           "rows_leaving_label_domain": int(np.count_nonzero(q_base & dm & ~q_mut)),
                           "baseline_component_mismatches": bmis, "label_rule_vs_audit_mismatches": mmis,
                           "pass": int(bmis == 0 and mmis == 0)})
    if total_base or total_mut:
        raise SystemExit(f"M5 label-domain cross-check failed: baseline={total_base}, mutated={total_mut}")
    ids = [r["view_id"] for r in reg["cert"]]
    for task, vid in DIRECT_VIEWS.items():
        i = ids.index(vid)
        for ds in DATASETS:
            dm = vd.dataset_mask(conf, ds)
            lost = int(np.count_nonzero(q_base_all[i] & dm & ~q_mut_all[i]))
            expected = int(np.count_nonzero(conf["qmask"][i] & dm & ~valid_audit[task]))
            settings.append({"task": task, "dataset": ds, "view_id": vid, "rows_leaving_label_domain": lost,
                             "expected_rows": expected, "nonvacuous": int(lost > 0),
                             "pass": int(lost == expected and lost > 0)})
    if not all(r["pass"] == 1 for r in settings):
        raise SystemExit("M5 direct-view label-domain settings failed")
    summary = {"bits_checked": int(q_mut_all.size), "baseline_component_mismatches": total_base,
               "label_rule_vs_audit_mismatches": total_mut,
               "label_rule": "lca.conformance.gold_is_max", "query_mask_rule": "lca.conformance.compute_query_mask",
               "registered_map_failure_counts": {e: {ds: int(np.count_nonzero(vd.dataset_mask(conf, ds) & ~comp[k]))
                                                     for ds in DATASETS}
                                                 for e, k in (("Pi", "pi_ok"), ("T", "t_ok"), ("W", "w_ok"))}}
    return detail, settings, summary


M5_VIEW_FIELDS = ["array_index", "view_id", "task", "dataset", "baseline_query_rows", "mutated_query_rows",
                  "rows_leaving_label_domain", "baseline_component_mismatches", "label_rule_vs_audit_mismatches", "pass"]
M5_SETTING_FIELDS = ["task", "dataset", "view_id", "rows_leaving_label_domain", "expected_rows", "nonvacuous", "pass"]
M1_CERT_FIELDS = ["view_id", "task", "basis", "basis_support", "certificate_path", "required_maps", "carrier_edges",
                  "rule", "path_nodes", "mutation_registry_edges", "selected_registered_matrix_sha256"]
