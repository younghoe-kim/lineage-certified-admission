# SPDX-License-Identifier: MIT
"""Controls: partition identity across block combinations, simpler checks and the family-3 permutation null."""
from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from .names import DATASETS, FAMILY_ORDER

EXPECTED_ROWS = {"DF": 289202, "FP": 342182}
EXPECTED_TOTAL = 631384
SCOPES = ("ALL", "FULL_CONFORM", "DRIFT")
CHUNK_ROWS = 25000

# Build report of the released benchmark, as the source-release record (src/lca/paper/source_release.json) states
# it: the builder re-keyed core identifiers, dropped the core entries it could not map, and reported the build as
# conforming. The values are read from that record, which is their only source.
_BUILD_RECORD = json.loads((Path(__file__).resolve().parent / "paper" / "source_release.json")
                           .read_text(encoding="utf-8"))["build_record"]
BUILD_REPORT_DROPPED_CORE_ENTRIES = int(_BUILD_RECORD["dropped_core_term_entries"])
BUILD_REPORT_CONFORMS = _BUILD_RECORD["conforms"] is True

# Family-3 view (structural, surface, taxonomic family counts) used by the permutation null.
F3_KEYS = (
    "family::structural_expression",
    "family::surface_expression",
    "family::taxonomic_category",
)
SANITIZED_PREFIXES = ("facet::", "family::", "inst_family::")
PRESENCE_PREFIXES = ("core::", "type::", "inst_sem::", "rep_core::")

# Permutation null of the family-3 cosine score.
PERMUTATIONS = 999
MIN_LABEL_SUPPORT = 3
SCORE_EPS = 1e-12
GE_TOL = 1e-15
NULL_DEFINITION = "within-dataset permutation of view rows, labels fixed"
NULL_SHUFFLE_UNIT = "row"
NULL_STRATIFICATION = "dataset (source,target separately)"
NULL_RNG = "numpy.default_rng(PCG64)"
NULL_ORDER = "source_then_target"
NULL_IMPLEMENTATION = "literal_row_permutation_then_exact_signature_contingency_evaluation"
DIRECTIONS = (("DF", "FP"), ("FP", "DF"))

PARTITION_FIELDS = ["dataset", "scope", "scope_rows", "cell_id", "F", "M", "T",
                    "core_fibers", "cell_fibers", "partition_equal"]
TABLE_FIELDS = ["baseline", "what_it_tests", "result", "failure_mode", "evidence_status"]
DETAIL_FIELDS = ["baseline", "setting", "metric", "value", "status"]
VALUE_FIELDS = ["cell_id", "value", "definition", "selector"]
NULL_FIELDS = ["direction", "seed", "B", "null_definition", "shuffle_unit", "stratification", "rng",
               "permutation_order", "implementation", "observed_macro_f1", "null_macro_f1", "null_ge_observed"]
SUMMARY_FIELDS = ["direction", "status", "B", "observed_macro_f1", "null_mean", "null_sd", "null_min",
                  "null_max", "k_ge_observed", "p_value", "first20_mean"]


def num(x: Any) -> str:
    """Integers as integers, floats with 17 significant digits."""
    if isinstance(x, (bool, np.bool_)):
        return str(int(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return format(float(x), ".17g")


class Codebook:
    """Assigns consecutive integer codes in order of first appearance."""

    def __init__(self) -> None:
        self.map: dict[Any, int] = {}

    def code(self, key: Any) -> int:
        out = self.map.get(key)
        if out is None:
            out = len(self.map)
            self.map[key] = out
        return out


def _norm_lexical(x: Any) -> str:
    s = unicodedata.normalize("NFKC", str(x)).strip()
    return " ".join(s.split()).casefold()


def build_rows(anchors_path: Path, full_conform_path: Path) -> dict[str, Any]:
    """One pass over the anchors: block partition codes, family-3 counts, labels and check counters."""
    fc = pd.read_csv(full_conform_path, usecols=["dataset", "row_id", "full_registry_conform"],
                     dtype={"dataset": str, "row_id": str, "full_registry_conform": int}, keep_default_na=False)
    if len(fc) != EXPECTED_TOTAL:
        raise SystemExit("conformance row count differs from the anchor table")
    fc_dataset = fc["dataset"].to_numpy(dtype=str)
    fc_row = fc["row_id"].to_numpy(dtype=str)
    full = fc["full_registry_conform"].to_numpy(dtype=bool)

    books = {b: Codebook() for b in ("core", "facet", "family", "type", "inst_sem")}
    dataset = np.empty(EXPECTED_TOTAL, dtype="U2")
    codes = {b: np.empty(EXPECTED_TOTAL, dtype=np.int32) for b in books}
    f3 = np.empty((EXPECTED_TOTAL, 3), dtype=np.float32)
    yfacet = np.empty(EXPECTED_TOTAL, dtype=object)
    yfamily = np.empty(EXPECTED_TOTAL, dtype=object)

    direct_hits: dict[tuple[str, str], int] = defaultdict(int)
    lexical_hits: dict[tuple[str, str, str], int] = defaultdict(int)
    block_present: dict[tuple[str, str], int] = defaultdict(int)
    instfam_num = 0
    instfam_den = 0
    norm_cache: dict[str, str] = {}
    pos = 0

    reader = pd.read_csv(anchors_path, usecols=["dataset", "anchor_id", "full_feats_json", "gold_facet", "gold_family"],
                         dtype=str, keep_default_na=False, chunksize=CHUNK_ROWS)
    for chunk in reader:
        for ds, aid, raw, gfacet, gfamily in zip(chunk["dataset"], chunk["anchor_id"], chunk["full_feats_json"],
                                                 chunk["gold_facet"], chunk["gold_family"]):
            if pos >= EXPECTED_TOTAL:
                raise SystemExit("anchor table has more rows than expected")
            if ds != fc_dataset[pos] or aid != fc_row[pos]:
                raise SystemExit(f"anchor and conformance row order differ at row {pos}")
            d = json.loads(raw)
            parts: dict[str, list[tuple[str, float]]] = defaultdict(list)
            active_local: list[tuple[str, str]] = []
            instfam_vals: set[str] = set()
            for key, value in d.items():
                v = float(value)
                ns, sep, local = key.partition("::")
                if not sep:
                    ns, local = "", key
                if v != 0.0:
                    parts[ns].append((local, v))
                    active_local.append((ns, local))
                    if ns == "inst_family" and v > 0:
                        instfam_vals.add(local)
            dataset[pos] = ds
            yfacet[pos] = gfacet
            yfamily[pos] = gfamily
            for b in books:
                codes[b][pos] = books[b].code(tuple(sorted(parts.get(b, ()))))
            f3[pos] = [float(d.get(k, 0.0)) for k in F3_KEYS]

            direct_hits[(ds, "facet")] += int(float(d.get("facet::" + gfacet, 0.0)) > 0)
            direct_hits[(ds, "family")] += int(float(d.get("family::" + gfamily, 0.0)) > 0)

            normed = []
            for ns, local in active_local:
                n = norm_cache.get(local)
                if n is None:
                    n = _norm_lexical(local)
                    norm_cache[local] = n
                normed.append((ns + "::" in SANITIZED_PREFIXES, n))
            for task, gold in (("facet", gfacet), ("family", gfamily)):
                target = _norm_lexical(gold)
                lexical_hits[(ds, task, "RAW")] += int(any(n == target for _s, n in normed))
                lexical_hits[(ds, task, "SANITIZED")] += int(any(n == target for s, n in normed if not s))

            for prefix in PRESENCE_PREFIXES:
                block_present[(ds, prefix)] += int(prefix[:-2] in parts)
            if ds == "DF":
                instfam_den += 1
                instfam_num += int(gfamily in instfam_vals)
            pos += 1

    if pos != EXPECTED_TOTAL:
        raise SystemExit(f"anchor rows {pos} != {EXPECTED_TOTAL}")
    for ds, n in EXPECTED_ROWS.items():
        if int(np.count_nonzero(dataset == ds)) != n:
            raise SystemExit(f"{ds} row count differs from the released benchmark")

    return {
        "dataset": dataset, "full": full, "f3": f3,
        "core": codes["core"], "facet": codes["facet"], "family": codes["family"],
        "type": codes["type"], "sem": codes["inst_sem"].astype(np.int16),
        "yfacet": yfacet.astype(str), "yfamily": yfamily.astype(str),
        "direct_hits": dict(direct_hits), "lexical_hits": dict(lexical_hits),
        "block_present": dict(block_present),
        "instfam_num": instfam_num, "instfam_den": instfam_den,
    }


def scope_masks(rows: dict[str, Any], ds: str) -> list[tuple[str, np.ndarray]]:
    base = rows["dataset"] == ds
    return [("ALL", base), ("FULL_CONFORM", base & rows["full"]), ("DRIFT", base & ~rows["full"])]


def unique_rows(arrays: Sequence[np.ndarray], mask: np.ndarray) -> int:
    if len(arrays) == 1:
        return int(np.unique(arrays[0][mask]).size)
    return int(np.unique(np.column_stack([a[mask] for a in arrays]), axis=0).shape[0])


def partition_identity(rows: dict[str, Any]) -> list[dict[str, Any]]:
    """Fibers of core alone versus core plus each combination of facet (F), family (M), type (T)."""
    out = []
    for ds in DATASETS:
        for scope, mask in scope_masks(rows, ds):
            core_n = unique_rows([rows["core"]], mask)
            for F in (0, 1):
                for M in (0, 1):
                    for T in (0, 1):
                        arrays = [rows["core"]]
                        if F:
                            arrays.append(rows["facet"])
                        if M:
                            arrays.append(rows["family"])
                        if T:
                            arrays.append(rows["type"])
                        cell_n = unique_rows(arrays, mask)
                        out.append({"dataset": ds, "scope": scope, "scope_rows": int(np.count_nonzero(mask)),
                                    "cell_id": f"F{F}M{M}T{T}", "F": F, "M": M, "T": T,
                                    "core_fibers": core_n, "cell_fibers": cell_n,
                                    "partition_equal": int(cell_n == core_n)})
    return out


def fd_g3_stats(determinant: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[bool, float, float, int]:
    """Exact FD test, row-deletion g3 error, singleton-row mass and fiber count of determinant -> y."""
    groups: dict[int, Counter] = defaultdict(Counter)
    for a, b in zip(determinant[mask].tolist(), y[mask].tolist()):
        groups[int(a)][str(b)] += 1
    n = int(np.count_nonzero(mask))
    exact = all(len(c) <= 1 for c in groups.values())
    keep = sum(max(c.values()) for c in groups.values())
    g3 = 0.0 if n == 0 else 1.0 - keep / n
    singleton = 0.0 if n == 0 else sum(sum(c.values()) for c in groups.values() if sum(c.values()) == 1) / n
    return exact, float(g3), float(singleton), len(groups)


def label_codes(y: np.ndarray, order: Sequence[str]) -> np.ndarray:
    """Integer label codes in a declared label order."""
    pos = {x: i for i, x in enumerate(order)}
    missing = sorted(set(np.unique(y).tolist()) - set(pos))
    if missing:
        raise SystemExit(f"labels outside the declared order: {missing[:5]}")
    return np.fromiter((pos[v] for v in y.tolist()), dtype=np.int64, count=len(y))


def observed_family3_scores(probes: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Family-3 cosine macro-F1 of the utility step (all-source training, all supported target rows)."""
    out = {}
    for s, t in DIRECTIONS:
        hit = probes[(probes["view_id"] == "family3__facet") & (probes["source"] == s) & (probes["target"] == t)
                     & (probes["probe"] == "cosine") & (probes["training_protocol"] == "all_source")
                     & (probes["evaluation_scope"] == "all_supported_target") & (probes["status"] == "OK")]
        if len(hit) != 1:
            raise SystemExit(f"family-3 cosine score missing for {s}->{t}")
        out[(s, t)] = float(hit.iloc[0]["macro_f1"])
    return out


def simple_checks(rows: dict[str, Any], verdicts: pd.DataFrame, probes: pd.DataFrame,
                  facet_order: Sequence[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """Simpler checks a curator can run: summary table, per-setting details and paper values."""
    n_rows = {ds: int(np.count_nonzero(rows["dataset"] == ds)) for ds in DATASETS}
    details: list[dict[str, Any]] = []

    for ds in DATASETS:
        for task in ("facet", "family"):
            details.append({"baseline": "identity_carrier_audit", "setting": f"{ds}|{task}",
                            "metric": "gold_carrier_present_rate",
                            "value": num(rows["direct_hits"][(ds, task)] / n_rows[ds]), "status": "RECOMPUTED"})
        for prefix in PRESENCE_PREFIXES:
            details.append({"baseline": "vocabulary_sanitization", "setting": ds,
                            "metric": f"residual_block_presence_rate:{prefix}",
                            "value": num(rows["block_present"][(ds, prefix)] / n_rows[ds]), "status": "RECOMPUTED"})

    for ds in DATASETS:
        for task, y in (("facet", rows["yfacet"]), ("family", rows["yfamily"])):
            for scope, mask in scope_masks(rows, ds):
                exact, g3, singleton, fibers = fd_g3_stats(rows["core"], y, mask)
                for metric, value in (("exact_fd_holds", int(exact)), ("g3_row_deletion_error", g3),
                                      ("singleton_row_mass", singleton), ("determinant_fibers", fibers)):
                    details.append({"baseline": "targeted_fd_g3_core_signature",
                                    "setting": f"{ds}|{task}|{scope}", "metric": metric, "value": num(value),
                                    "status": "RECOMPUTED_EXPLICIT_SPEC"})

    family3 = observed_family3_scores(probes)
    for (s, t), score in family3.items():
        details.append({"baseline": "predictive_screen", "setting": f"family3__facet|{s}->{t}",
                        "metric": "all_source_cosine_macro_f1", "value": num(score), "status": "BOUND_FROM_UTILITY"})
    inst_cov = rows["instfam_num"] / rows["instfam_den"]
    details.append({"baseline": "predictive_screen", "setting": "inst_family|DF|family",
                    "metric": "gold_in_inst_family_coverage", "value": num(inst_cov), "status": "RECOMPUTED"})

    yf = label_codes(rows["yfacet"], facet_order)
    ym = label_codes(rows["yfamily"], FAMILY_ORDER)
    pooled_f = float(normalized_mutual_info_score(yf, rows["sem"]))
    pooled_m = float(normalized_mutual_info_score(ym, rows["sem"]))
    details.append({"baseline": "pooled_correlation_screen", "setting": "ALL|facet", "metric": "NMI",
                    "value": num(pooled_f), "status": "RECOMPUTED"})
    details.append({"baseline": "pooled_correlation_screen", "setting": "ALL|family", "metric": "NMI",
                    "value": num(pooled_m), "status": "RECOMPUTED"})
    within = []
    for ds in DATASETS:
        mask = rows["dataset"] == ds
        for task, y in (("facet", yf), ("family", ym)):
            nmi = float(normalized_mutual_info_score(y[mask], rows["sem"][mask]))
            within.append(nmi)
            details.append({"baseline": "pooled_correlation_screen", "setting": f"{ds}|{task}",
                            "metric": "within_dataset_NMI", "value": num(nmi), "status": "RECOMPUTED"})

    scope_views = ["inst_sem__facet", "inst_sem__family", "inst_family__family", "family3__facet"]
    scope_mass = pd.to_numeric(verdicts[verdicts["view"].isin(scope_views)]["m^U"], errors="raise")
    scope_text = "=0" if bool((scope_mass == 0).all()) else f" max={float(scope_mass.max()):.6f}"
    core_unsafe = {(str(r["view"]), str(r["dataset"])): float(r["m^U"]) for _, r in
                   verdicts[verdicts["view"] == "factorial_F0M0T0__family"].iterrows()}
    cu = (f"core-only verdicts UNSAFE family DF={core_unsafe[('factorial_F0M0T0__family', 'DF')]:.6f}, "
          f"FP={core_unsafe[('factorial_F0M0T0__family', 'FP')]:.6f}")
    rates_one = sum(rows["direct_hits"][(ds, task)] == n_rows[ds] for ds in DATASETS for task in ("facet", "family"))
    n_drift = int(np.count_nonzero(~rows["full"]))
    within_text = ("within-dataset NMI=0 for all four dataset-task cells" if all(x == 0.0 for x in within)
                   else f"within-dataset NMI max={max(within):.6f}")
    table = [
        {"baseline": "identity_carrier_audit",
         "what_it_tests": "direct protected-carrier presence / field identity",
         "result": f"{rates_one}/4 dataset-task settings carrier-present rate=1.0; "
                           f"does not reject {n_drift:,} FULL-CONFORM defects",
         "failure_mode": "misses lineage determination and materialization drift",
         "evidence_status": "RECOMPUTED"},
        {"baseline": "build_integrity_QC",
         "what_it_tests": "builder integrity / keys / counts",
         "result": f"build report of the released benchmark: conforms={str(BUILD_REPORT_CONFORMS).lower()} "
                           f"while recording {BUILD_REPORT_DROPPED_CORE_ENTRIES:,} dropped core entries",
         "failure_mode": "not a lineage audit",
         "evidence_status": "BUILD_REPORT_CONSTANT"},
        {"baseline": "vocabulary_sanitization",
         "what_it_tests": "protected-label strings in exposed vocabulary",
         "result": f"direct carrier blocks can be removed while retained core/type remain; {cu}",
         "failure_mode": "misses registered derivability after string removal",
         "evidence_status": "RECOMPUTED_PLUS_VERDICTS_BOUND"},
        {"baseline": "prefix_sanitization",
         "what_it_tests": "remove facet::/family::/inst_family:: role prefixes",
         "result": f"core/type survive; {cu}",
         "failure_mode": "misses registered derivability through retained core/type",
         "evidence_status": "RECOMPUTED_PLUS_VERDICTS_BOUND"},
        {"baseline": "targeted_FD_g3",
         "what_it_tests": "explicit core-signature exact FD / row-deletion g3",
         "result": "diagnostics per dataset, task and scope are in simple_checks_details.csv",
         "failure_mode": "exact-signature test does not cover registered partial/fiber determination",
         "evidence_status": "RECOMPUTED_SPEC"},
        {"baseline": "predictive_screen",
         "what_it_tests": "utility above trivial baseline",
         "result": f"family-3 all-source cosine={family3[('DF', 'FP')]:.6f}/{family3[('FP', 'DF')]:.6f}; "
                           f"inst_family released-label-in-block coverage={inst_cov:.6f}; "
                           f"verdicts scope-control UNSAFE mass{scope_text}",
         "failure_mode": "false positive: utility/coverage does not imply protected-query determination",
         "evidence_status": "RECOMPUTED_PLUS_UTILITY_VERDICTS_BOUND"},
        {"baseline": "pooled_correlation_screen",
         "what_it_tests": "pooled association",
         "result": f"inst_sem pooled NMI facet={pooled_f:.6f}, family={pooled_m:.6f}; {within_text}",
         "failure_mode": "false positive caused by dataset identity rather than within-dataset refinement",
         "evidence_status": "RECOMPUTED"},
        {"baseline": "registered_lineage_scope_controls",
         "what_it_tests": "protected-query fiber determination under registered paths",
         "result": f"inst_sem, inst_family, family-3 registered scope-control UNSAFE mass{scope_text} in verdicts",
         "failure_mode": "none on the registered scope-control specificity set",
         "evidence_status": "BOUND_FROM_VERDICTS"},
    ]

    raw_rates = [rows["lexical_hits"][(ds, task, "RAW")] / n_rows[ds] for ds in DATASETS for task in ("facet", "family")]
    san_rates = [rows["lexical_hits"][(ds, task, "SANITIZED")] / n_rows[ds]
                 for ds in DATASETS for task in ("facet", "family")]
    fd = {ds: fd_g3_stats(rows["core"], rows["yfamily"], (rows["dataset"] == ds) & rows["full"]) for ds in DATASETS}
    values = [
        {"cell_id": "l0_rate",
         "value": num(min(rows["direct_hits"][(ds, task)] / n_rows[ds] for ds in DATASETS for task in ("facet", "family"))),
         "definition": "minimum direct protected-carrier presence rate over four dataset-task settings",
         "selector": "all rows; exact direct carrier key"},
        {"cell_id": "l0_hit", "value": num(min(raw_rates)),
         "definition": "minimum exact lexical protected-label hit rate over four dataset-task settings before sanitization",
         "selector": "active feature local-name exact match after NFKC/strip/collapse/casefold"},
        {"cell_id": "l0_hit_san", "value": num(max(san_rates)),
         "definition": "maximum exact lexical protected-label hit rate over four dataset-task settings after removing "
                       "facet::, family::, inst_family::",
         "selector": "same lexical matcher on sanitized feature set"},
        {"cell_id": "instfam_n", "value": str(int(rows["instfam_num"])),
         "definition": "DF rows whose released family label is contained in active inst_family:: values",
         "selector": "DF all rows"},
        {"cell_id": "instfam_den", "value": str(int(rows["instfam_den"])),
         "definition": "denominator for instfam_n", "selector": "DF all rows"},
        {"cell_id": "instfam_cov", "value": num(inst_cov),
         "definition": "instfam_n / instfam_den", "selector": "DF all rows"},
        {"cell_id": "fd_single_df", "value": num(fd["DF"][2]),
         "definition": "singleton-row mass of core:: signature on FULL-CONFORM DF family task",
         "selector": "DF family FULL_CONFORM; determinant=core:: exact signature"},
        {"cell_id": "fd_single_fp", "value": num(fd["FP"][2]),
         "definition": "singleton-row mass of core:: signature on FULL-CONFORM FP family task",
         "selector": "FP family FULL_CONFORM; determinant=core:: exact signature"},
        {"cell_id": "fd_g3_df", "value": num(fd["DF"][1]),
         "definition": "row-deletion g3 of core:: signature -> family on FULL-CONFORM DF",
         "selector": "DF family FULL_CONFORM"},
        {"cell_id": "fd_g3_fp", "value": num(fd["FP"][1]),
         "definition": "row-deletion g3 of core:: signature -> family on FULL-CONFORM FP",
         "selector": "FP family FULL_CONFORM"},
    ]
    return table, details, values


class DatasetF3:
    """Family-3 rows of one dataset collapsed to exact unique count signatures."""

    def __init__(self, ds: str, rows: dict[str, Any]):
        idx = np.flatnonzero(rows["dataset"] == ds)
        self.y = rows["yfacet"][idx]
        U, inv = np.unique(rows["f3"][idx], axis=0, return_inverse=True)
        self.U = U.astype(np.float64)
        self.inv = np.asarray(inv).reshape(-1).astype(np.int32)
        norms = np.linalg.norm(self.U, axis=1)
        self.V = np.zeros_like(self.U)
        nz = norms > 0
        self.V[nz] = self.U[nz] / norms[nz, None]


class PermutationDirection:
    """Cosine prototype score source -> target after permuting view rows within each dataset."""

    def __init__(self, source: DatasetF3, target: DatasetF3):
        self.source, self.target = source, target
        counts = Counter(source.y.tolist())
        # Label order: source frequency, ties by first appearance in row order.
        self.labels = [k for k, v in counts.most_common() if int(v) >= MIN_LABEL_SUPPORT]
        pos = {x: i for i, x in enumerate(self.labels)}
        self.L = len(self.labels)
        scode = np.fromiter((pos.get(v, -1) for v in source.y), np.int32, count=len(source.y))
        tcode = np.fromiter((pos.get(v, -1) for v in target.y), np.int32, count=len(target.y))
        self.sfit_idx = np.flatnonzero(scode >= 0)
        self.sfit_code = scode[self.sfit_idx]
        self.teval_idx = np.flatnonzero(tcode >= 0)
        self.teval_code = tcode[self.teval_idx]
        self.metric_codes = sorted(set(self.teval_code.tolist()))
        self.ns = len(source.U)
        self.nt = len(target.U)

    def score(self, seed: int) -> float:
        rng = np.random.default_rng(seed)
        source_perm = rng.permutation(len(self.source.y))
        target_perm = rng.permutation(len(self.target.y))

        assigned = self.source.inv[source_perm][self.sfit_idx]
        pair = self.sfit_code.astype(np.int64) * self.ns + assigned
        counts = np.bincount(pair, minlength=self.L * self.ns).reshape(self.L, self.ns)
        prototypes = counts @ self.source.U
        norms = np.linalg.norm(prototypes, axis=1)
        nz = norms > 0
        prototypes[nz] /= norms[nz, None]
        prototypes[~nz] = 0

        scores = self.target.V @ prototypes.T
        imax = np.argmax(scores, axis=1).astype(np.int32)
        mx = scores[np.arange(self.nt), imax]
        ties = np.isclose(scores, mx[:, None], rtol=0.0, atol=SCORE_EPS).sum(axis=1) > 1
        imax[(mx <= SCORE_EPS) | ties] = -1

        assigned_t = self.target.inv[target_perm][self.teval_idx]
        pair2 = self.teval_code.astype(np.int64) * self.nt + assigned_t
        tc = np.bincount(pair2, minlength=self.L * self.nt).reshape(self.L, self.nt)
        coltot = tc.sum(axis=0)
        f1s = []
        for c in self.metric_codes:
            sel = imax == c
            tp = int(tc[c, sel].sum())
            fn = int(tc[c, :].sum() - tp)
            fp = int(coltot[sel].sum() - tp)
            den = 2 * tp + fp + fn
            f1s.append(0.0 if den == 0 else (2.0 * tp / den))
        return float(np.mean(f1s))


def permutation_engines(rows: dict[str, Any]) -> dict[tuple[str, str], PermutationDirection]:
    per = {ds: DatasetF3(ds, rows) for ds in DATASETS}
    return {(s, t): PermutationDirection(per[s], per[t]) for s, t in DIRECTIONS}


def permutation_null(engines: dict[tuple[str, str], PermutationDirection], observed: dict[tuple[str, str], float],
                     B: int = PERMUTATIONS, seeds: Sequence[int] | None = None) -> tuple[list[dict], list[dict]]:
    """Null scores for seeds 0..B-1 and the one-sided p-value (k+1)/(B+1) per direction."""
    seeds = list(range(B)) if seeds is None else list(seeds)
    null_rows, summary = [], []
    for s, t in DIRECTIONS:
        obs = observed[(s, t)]
        vals = np.array([engines[(s, t)].score(seed) for seed in seeds], dtype=np.float64)
        for seed, v in zip(seeds, vals):
            null_rows.append({"direction": f"{s}->{t}", "seed": seed, "B": B, "null_definition": NULL_DEFINITION,
                              "shuffle_unit": NULL_SHUFFLE_UNIT, "stratification": NULL_STRATIFICATION,
                              "rng": NULL_RNG, "permutation_order": NULL_ORDER,
                              "implementation": NULL_IMPLEMENTATION, "observed_macro_f1": num(obs),
                              "null_macro_f1": num(v), "null_ge_observed": int(v >= obs - GE_TOL)})
        n = len(vals)
        k = int(np.sum(vals >= obs - GE_TOL))
        summary.append({"direction": f"{s}->{t}",
                        "status": "PASS_CANONICAL" if n == PERMUTATIONS else "PASS_SIMULATION_NONCANONICAL_B",
                        "B": n, "observed_macro_f1": num(obs), "null_mean": num(float(vals.mean())),
                        "null_sd": num(float(vals.std(ddof=1)) if n > 1 else 0.0),
                        "null_min": num(float(vals.min())), "null_max": num(float(vals.max())),
                        "k_ge_observed": k, "p_value": num((k + 1) / (n + 1)),
                        "first20_mean": num(float(vals[: min(20, n)].mean()))})
    return null_rows, summary
