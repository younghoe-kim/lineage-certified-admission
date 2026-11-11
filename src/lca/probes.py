# SPDX-License-Identifier: MIT
"""Utility probes (cosine prototype, source-fiber Lookup, SrcMaj, Uniform) and the parameter-free registry rule."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from . import bounds
from . import evaluation as ev
from .evaluation import ABSTAIN, SCORE_EPS, TIE_ATOL, UNIFORM_DRAWS, score_predictions
from .io import array_sha
from .names import DATASETS, TASKS, VERDICT_CODES


# ----------------------------------------------------------------------------
# Cosine prototype probe (float32)
# ----------------------------------------------------------------------------

def fused_multiply_add32(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """float32 a*b + c with a single rounding (fused multiply-add), computed exactly in float64.

    The product of two float32 values is exact in float64; the float64 sum is rounded to odd (exact error by
    TwoSum, then the last bit set when inexact), and the rounding of that sum to float32 is then the correctly
    rounded fused result."""
    p = a.astype(np.float64) * b.astype(np.float64)
    c64 = c.astype(np.float64)
    s = c64 + p
    bb = s - c64
    e = (c64 - (s - bb)) + (p - bb)
    adj = (e != 0) & ((s.view(np.int64) & 1) == 0)
    if adj.any():
        s[adj] = np.nextafter(s[adj], np.where(e[adj] > 0, np.inf, -np.inf))
    return s.astype(np.float32)


def sparse_dense_product32(Z: sparse.csr_matrix, D: np.ndarray) -> np.ndarray:
    """float32 product Z @ D for CSR Z (rows x k) and dense D (k x L).

    Each output entry accumulates the row's nonzeros in ascending column order, starting from zero, with one
    fused multiply-add per nonzero: y <- fl32(y + z_j * d_j)."""
    Z = Z.tocsr()
    Z.sort_indices()
    D = np.ascontiguousarray(D, dtype=np.float32)
    n, L = Z.shape[0], D.shape[1]
    lens = np.diff(Z.indptr)
    order = np.argsort(-lens, kind="stable")           # rows with more nonzeros first
    starts = Z.indptr[:-1][order]
    lens_sorted = lens[order]
    out = np.zeros((n, L), dtype=np.float32)
    maxlen = int(lens_sorted[0]) if n else 0
    for k in range(maxlen):
        nk = int(np.searchsorted(-lens_sorted, -k, side="left"))   # rows with more than k nonzeros
        pos = starts[:nk] + k
        out[:nk] = fused_multiply_add32(Z.data[pos].astype(np.float32)[:, None], D[Z.indices[pos]], out[:nk])
    res = np.empty_like(out)
    res[order] = out
    return res


def cosine_score_matrix(cache: dict[str, Any], fit_rows: np.ndarray, target_idx: np.ndarray, labels: list[str],
                        cols: np.ndarray, task: str, type_weight: float | None = None) -> tuple[np.ndarray, int]:
    """Cosine scores of L2-normalised target rows against L2-normalised per-label sums of the fitting rows.

    All arithmetic is float32: prototypes are dense (labels x columns), target rows stay sparse.
    type_weight, when given, multiplies every type:: column of both sides. Returns (scores, empty prototypes)."""
    y = cache["gold_" + task]
    Zs = cache["X"][fit_rows][:, cols].tocsr().astype(np.float32)
    Zt = cache["X"][target_idx][:, cols].tocsr().astype(np.float32)
    if type_weight is not None:
        names = cache["features"][cols]
        w = np.fromiter((float(type_weight) if str(n).startswith("type::") else 1.0 for n in names),
                        dtype=np.float32, count=len(names))
        Zs = Zs.multiply(w).tocsr()
        Zt = Zt.multiply(w).tocsr()
    y_fit = y[fit_rows]
    prototypes = np.zeros((len(labels), len(cols)), dtype=np.float32)
    for j, lab in enumerate(labels):
        mask = y_fit == lab
        if np.any(mask):
            prototypes[j] = np.asarray(Zs[mask].sum(axis=0), dtype=np.float32).ravel()
    norms = np.linalg.norm(prototypes, axis=1)
    nonzero = norms > 0
    prototypes[nonzero] /= norms[nonzero, None]
    target_norm = normalize(Zt, axis=1, copy=False)
    scores = sparse_dense_product32(target_norm, prototypes.T)
    return scores, int(np.sum(~nonzero))


def cosine_predict(cache: dict[str, Any], fit_rows: np.ndarray, target_idx: np.ndarray, labels: list[str],
                   cols: np.ndarray, task: str, type_weight: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    scores, zero = cosine_score_matrix(cache, fit_rows, target_idx, labels, cols, task, type_weight=type_weight)
    pred, diag = score_predictions(scores, labels, SCORE_EPS, TIE_ATOL)
    diag["zero_prototype_labels"] = zero
    return pred, diag


# ----------------------------------------------------------------------------
# Lookup, SrcMaj and Uniform
# ----------------------------------------------------------------------------

def lookup_predict(cache: dict[str, Any], fit_rows: np.ndarray, target_idx: np.ndarray, cols: np.ndarray,
                   task: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Source-fiber lookup: the majority fitting label of the target row's fiber (ties: first seen in row
    order); a fiber unseen among the fitting rows abstains."""
    y = cache["gold_" + task]
    ss = bounds.sparse_signatures(cache["X"][fit_rows][:, cols])
    ts = bounds.sparse_signatures(cache["X"][target_idx][:, cols])
    by: dict[bytes, Counter] = defaultdict(Counter)
    for sig, lab in zip(ss, y[fit_rows].tolist()):
        by[sig][lab] += 1
    mapping = {sig: c.most_common(1)[0][0] for sig, c in by.items()}
    pred = np.asarray([mapping.get(sig, ABSTAIN) for sig in ts], dtype=object)
    ab = pred == ABSTAIN
    return pred, {"n_abstain": int(ab.sum()), "abstain_rate": float(ab.mean()) if len(ab) else float("nan"),
                  "zero_prototype_labels": 0}


def srcmaj_predict(cache: dict[str, Any], fit_rows: np.ndarray, target_idx: np.ndarray, task: str
                   ) -> tuple[np.ndarray, dict[str, Any]]:
    """SrcMaj: every target row receives the most frequent fitting label (ties: first seen in row order)."""
    counts = Counter(cache["gold_" + task][fit_rows].tolist())
    if not counts:
        raise SystemExit("empty fitting set for SrcMaj")
    majority = counts.most_common(1)[0][0]
    return np.full(len(target_idx), majority, dtype=object), {
        "n_abstain": 0, "abstain_rate": 0.0, "zero_prototype_labels": 0, "majority_label": majority}


# Uniform draws.
# The reference draws are those of numpy.random.default_rng(seed).multinomial(n, [1/k]*k, size) evaluated with
# every a*b+c of the binomial algorithms rounded once (fused multiply-add). The sampler below implements that
# algorithm (PCG64 stream; binomial by inversion when n*p <= 30, BTPE otherwise; multinomial by chained
# conditional binomials, with numpy's parameter cache) with explicit fused rounding, so the draws do not depend
# on how a local numpy build rounds these expressions.

_SPLIT = 134217729.0  # 2**27 + 1 (Veltkamp split of a double into two 26-bit halves)


def fma64(a: float, b: float, c: float) -> float:
    """Correctly rounded a*b + c for doubles: exact two-product (Dekker), then an exactly rounded sum."""
    a, b = float(a), float(b)
    p = a * b
    t = _SPLIT * a; ah = t - (t - a); al = a - ah
    t = _SPLIT * b; bh = t - (t - b); bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return math.fsum((p, e, c))


class UniformStream:
    """next_double() of numpy's PCG64 generator seeded by default_rng(seed): (next64 >> 11) * 2**-53."""

    def __init__(self, seed: int, block: int = 1 << 16):
        self._bg = np.random.default_rng(seed).bit_generator
        self._block = block
        self._buf: list[float] = []
        self._i = 0

    def next_double(self) -> float:
        if self._i >= len(self._buf):
            raw = self._bg.random_raw(self._block)
            self._buf = ((raw >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)).tolist()
            self._i = 0
        x = self._buf[self._i]
        self._i += 1
        return x


class BinomialSampler:
    """Binomial(n, p) variates (numpy's inversion and BTPE algorithms) with a per-(n, p) parameter cache."""

    def __init__(self) -> None:
        self._key: tuple[int, float] | None = None
        self._par: tuple = ()

    def __call__(self, st: UniformStream, p: float, n: int) -> int:
        if n == 0 or p == 0.0:
            return 0
        if p <= 0.5:
            return self._inversion(st, n, p) if p * n <= 30.0 else self._btpe(st, n, p)
        q = 1.0 - p
        return n - (self._inversion(st, n, q) if q * n <= 30.0 else self._btpe(st, n, q))

    def _inversion(self, st: UniformStream, n: int, p: float) -> int:
        if self._key != (n, p):
            self._key = (n, p)
            q = 1.0 - p
            qn = math.exp(n * math.log1p(-p))
            np_ = n * p
            b = fma64(10.0, math.sqrt(fma64(np_, q, 1.0)), np_)
            self._par = (q, qn, int(n if n < b else b))
        q, qn, bound = self._par
        x, px, u = 0, qn, st.next_double()
        while u > px:
            x += 1
            if x > bound:
                x, px, u = 0, qn, st.next_double()
            else:
                u -= px
                px = ((n - x + 1) * p * px) / (x * q)
        return x

    def _btpe(self, st: UniformStream, n: int, p: float) -> int:
        if self._key != (n, p):
            self._key = (n, p)
            r = min(p, 1.0 - p)
            q = 1.0 - r
            fm = fma64(n, r, r)
            m = int(math.floor(fm))
            p1 = math.floor(fma64(2.195, math.sqrt(n * r * q), -(4.6 * q))) + 0.5
            xm = m + 0.5
            xl, xr = xm - p1, xm + p1
            c = 0.134 + 20.5 / (15.3 + m)
            a = (fm - xl) / fma64(-xl, r, fm)
            laml = a * (1.0 + a / 2.0)
            a = (xr - fm) / (xr * q)
            lamr = a * (1.0 + a / 2.0)
            p2 = p1 * fma64(2.0, c, 1.0)
            p3 = p2 + c / laml
            p4 = p3 + c / lamr
            self._par = (r, q, fm, m, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4)
        r, q, fm, m, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4 = self._par
        while True:
            nrq = n * r * q
            u = st.next_double() * p4
            v = st.next_double()
            if not u > p1:                                    # triangular region: accept
                return int(math.floor(fma64(-p1, v, xm) + u))
            if not u > p2:                                    # parallelogram region
                x = xl + (u - p1) / c
                v = fma64(v, c, 1.0) - abs(m - x + 0.5) / p1
                if v > 1.0:
                    continue
                y = int(math.floor(x))
            elif not u > p3:                                  # left exponential tail
                if v == 0.0:
                    continue
                y = int(math.floor(xl + math.log(v) / laml))
                if y < 0:
                    continue
                v = v * (u - p2) * laml
            else:                                             # right exponential tail
                if v == 0.0:
                    continue
                y = int(math.floor(xr - math.log(v) / lamr))
                if y > n:
                    continue
                v = v * (u - p3) * lamr
            k = abs(y - m)
            if not (k > 20 and k < nrq / 2.0 - 1):            # explicit evaluation of f(y) / f(m)
                s = r / q
                a = s * (n + 1)
                F = 1.0
                if m < y:
                    for i in range(m + 1, y + 1):
                        F *= (a / i - s)
                elif m > y:
                    for i in range(y + 1, m + 1):
                        F /= (a / i - s)
                if v > F:
                    continue
                return y
            rho = (k / nrq) * (fma64(k, k / 3.0 + 0.625, 0.16666666666666666) / nrq + 0.5)
            t = (-k * k) / (2 * nrq)
            A = math.log(v) if v > 0.0 else -math.inf
            if A < t - rho:
                return y
            if A > t + rho:
                continue
            x1, f1 = float(y) + 1, float(m) + 1
            z, w = float(n) + 1 - float(m), float(n) - float(y) + 1
            x2, f2, z2, w2 = x1 * x1, f1 * f1, z * z, w * w
            acc = fma64(xm, math.log(f1 / x1), (n - m + 0.5) * math.log(z / w))
            acc = fma64(float(y - m), math.log(w * r / (x1 * q)), acc)
            for sq, base in ((f2, f1), (z2, z), (x2, x1), (w2, w)):
                acc = acc + (13680. - (462. - (132. - (99. - 140. / sq) / sq) / sq) / sq) / base / 166320.
            if A > acc:
                continue
            return y


def multinomial_draws(st: UniformStream, binom: BinomialSampler, n: int, k: int, draws: int) -> np.ndarray:
    """`draws` multinomial(n, [1/k]*k) vectors by chained conditional binomials."""
    pix = [1.0 / k] * k
    out = np.zeros((draws, k), dtype=np.int64)
    for i in range(draws):
        rem, dn = 1.0, n
        for j in range(k - 1):
            x = binom(st, pix[j] / rem, dn)
            out[i, j] = x
            dn -= x
            if dn <= 0:
                break
            rem -= pix[j]
        if dn > 0:
            out[i, k - 1] = dn
    return out


def uniform_macro_f1_draws(y_target: np.ndarray, candidates: Sequence[str], metric_labels: Sequence[str],
                           seed: int, draws: int = UNIFORM_DRAWS) -> np.ndarray:
    """Macro-F1 of `draws` uniform random predictors over `candidates` (each true class's rows are spread
    multinomially over the candidates). `candidates` and `metric_labels` arrive in the declared label order;
    one stream serves all labels in that order."""
    cidx = {c: i for i, c in enumerate(candidates)}
    k = len(candidates)
    ntrue = {lab: int(np.sum(y_target == lab)) for lab in metric_labels}
    st, binom = UniformStream(seed), BinomialSampler()
    mats = {lab: multinomial_draws(st, binom, ntrue[lab], k, draws) for lab in metric_labels}
    per = []
    for lab in metric_labels:
        j = cidx[lab]
        tp = mats[lab][:, j]
        fn = ntrue[lab] - tp
        fp = np.zeros(draws, dtype=np.int64)
        for other in metric_labels:
            if other != lab:
                fp += mats[other][:, j]
        den = 2 * tp + fp + fn
        per.append(np.divide(2 * tp, den, out=np.zeros(draws, dtype=float), where=den > 0))
    return np.stack(per, axis=1).mean(axis=1)


# ----------------------------------------------------------------------------
# Parameter-free registry rule
# ----------------------------------------------------------------------------

def strict_argmax(scores: np.ndarray, label_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Predict only a strictly positive unique arg-max of integer scores; otherwise abstain."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != len(label_names):
        raise SystemExit("registry rule: score matrix and label list disagree")
    if len(scores) == 0:
        return np.empty(0, dtype=object), np.empty(0, dtype=bool)
    max_score = scores.max(axis=1)
    argmax = scores.argmax(axis=1)
    ties = (scores == max_score[:, None]).sum(axis=1)
    decided = (max_score > 0) & (ties == 1)
    pred = np.full(len(scores), ABSTAIN, dtype=object)
    pred[decided] = np.asarray(label_names, dtype=object)[argmax[decided]]
    return pred, decided


def coordinate_output(basis: str, feature: str, task: str, carrier_edges: str,
                      Pi: dict[str, str], W: dict[str, str]) -> str | None:
    """Registered image of one basis coordinate: W (facet -> family), Pi (core term -> facet) or W after Pi."""
    if not feature.startswith(basis):
        raise SystemExit(f"basis prefix mismatch: {basis} / {feature}")
    coord = feature[len(basis):]
    if basis == "facet::" and task == "family" and carrier_edges == "W":
        return W.get(coord)
    if basis == "core::" and task == "facet" and carrier_edges == "Pi":
        return Pi.get(coord)
    if basis == "core::" and task == "family" and carrier_edges == "Pi|W":
        fac = Pi.get(coord)
        return None if fac is None else W.get(fac)
    raise SystemExit(f"unsupported registry map: basis={basis}, task={task}, edges={carrier_edges}")


def mapping_matrix(features: np.ndarray, cols: np.ndarray, basis: str, task: str, carrier_edges: str,
                   labels: Sequence[str], Pi: dict[str, str], W: dict[str, str]
                   ) -> tuple[np.ndarray, list[str], list[str]]:
    """0/1 matrix (basis columns x output labels) of the registered map; unmapped coordinates get a zero row."""
    li = {x: i for i, x in enumerate(labels)}
    M = np.zeros((len(cols), len(labels)), dtype=np.int64)
    names: list[str] = []
    unmapped: list[str] = []
    for local, col in enumerate(cols.tolist()):
        name = str(features[int(col)])
        names.append(name)
        out = coordinate_output(basis, name, task, carrier_edges, Pi, W)
        if out is None or out not in li:
            unmapped.append(name)
            continue
        M[local, li[out]] = 1
    return M, names, unmapped


def _integer_block(X: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray) -> sparse.csr_matrix:
    Z = X[rows][:, cols].tocsr()
    if Z.nnz:
        d = Z.data.astype(np.float64)
        if np.any(~np.isfinite(d)) or np.any(d < 0) or np.any(d != np.floor(d)):
            raise SystemExit("registry rule: basis evidence is not a nonnegative integer count")
    return Z


def rule_scores(X: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Registry-rule scores by sparse matrix product: counts of each output label's registered coordinates."""
    Z = _integer_block(X, rows, cols).astype(np.int64)
    return np.asarray(Z @ M, dtype=np.int64)


def rule_scores_by_accumulation(X: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray, M: np.ndarray
                                ) -> np.ndarray:
    """Independent computation of rule_scores: add every nonzero count into its registered output bin."""
    Z = _integer_block(X, rows, cols).tocoo()
    out = np.zeros((len(rows), M.shape[1]), dtype=np.int64)
    target = np.full(M.shape[0], -1, dtype=np.int64)
    for j in range(M.shape[0]):
        hits = np.flatnonzero(M[j])
        if len(hits) > 1:
            raise SystemExit("registry map is not a function")
        if len(hits) == 1:
            target[j] = int(hits[0])
    if Z.nnz:
        vals = Z.data.astype(np.float64)
        valid = target[Z.col] >= 0
        if np.any(~valid & (vals != 0)):
            raise SystemExit("nonzero evidence on an unmapped basis coordinate")
        np.add.at(out, (Z.row[valid], target[Z.col[valid]]), vals[valid].astype(np.int64))
    return out


def unmapped_evidence(X: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray, M: np.ndarray) -> int:
    """Number of nonzero counts on basis coordinates that the registered map leaves unmapped."""
    un = np.flatnonzero(M.sum(axis=1) == 0)
    return 0 if len(un) == 0 else int(X[rows][:, cols[un]].getnnz())


# ============================================================================
# Utility tables
# ============================================================================

PROBES = ("cosine", "lookup", "SrcMaj")
CELL_IDS = tuple(f"F{F}M{M}T{T}" for F in (0, 1) for M in (0, 1) for T in (0, 1))
PREDICTION_VIEW = "identity_admitted__family"
CODE = {name: np.uint8(code) for code, name in VERDICT_CODES.items()}

PROBE_FIELDS = [
    "task", "direction", "source", "target", "view_id", "probe", "training_protocol", "evaluation_scope",
    "fitting_mask_id", "fitting_mask_rows", "source_rows_fit", "source_label_candidates", "zero_prototype_labels",
    "target_eval_rows", "metric_labels", "feature_count", "feature_hash", "shared_feature_count",
    "shared_support_hash", "source_row_hash", "target_row_hash", "prediction_hash", "macro_f1", "accuracy",
    "abstain_rate", "status",
]
TOLERANCE_FIELDS = ["view_id", "task", "direction", "score_eps", "tie_atol", "view_feature_hash", "macro_f1",
                    "accuracy", "abstain_rate", "prediction_hash"]
CALIBRATION_FIELDS = ["source", "target", "task", "n_draws", "seed", "n_eval", "source_candidates", "metric_labels",
                      "srcmaj_label", "srcmaj_macro_f1", "uniform_macro_f1", "uniform_draw_sd", "uniform_draw_min",
                      "uniform_draw_max"]
SUPPORT_FIELDS = ["training_protocol", "evaluation_scope", "view_id", "task", "direction", "probe",
                  "fitting_mask_rows", "source_rows_fit", "raw_feature_count", "shared_feature_count",
                  "raw_feature_hash", "shared_support_hash", "independent_shared_support_hash",
                  "support_recompute_equal", "raw_prediction_hash", "shared_prediction_hash",
                  "raw_to_shared_prediction_changes", "canonical_protocol"]
FACTORIAL_FIELDS = ["task", "direction", "cell_id", "view_id", "training_protocol", "evaluation_scope",
                    "source_rows_fit", "target_eval_rows", "shared_support_hash", "prediction_hash", "macro_f1",
                    "accuracy", "abstain_rate"]
VOLUME_FIELDS = ["training_protocol", "seed", "view_id", "direction", "source_rows_fit", "target_eval_rows",
                 "shared_support_hash", "prediction_hash", "macro_f1", "accuracy"]
TYPE_WEIGHT_FIELDS = ["view_id", "base_view_id", "direction", "type_weight", "training_protocol",
                      "evaluation_scope", "shared_support_hash", "macro_f1", "accuracy", "prediction_hash"]
DETERMINATION_FIELDS = ["metric_id", "prediction_view_id", "determination_view_id",
                        "determination_certificate_path", "task", "direction", "training_protocol",
                        "evaluation_scope", "source_rows_fit", "target_eval_rows", "shared_support_hash",
                        "prediction_hash", "correct_predictions", "correct_on_unsafe", "correct_on_nondetermined",
                        "correct_on_unverified", "fraction_correct_on_unsafe"]
VERDICT_CLASS_FIELDS = DETERMINATION_FIELDS + ["declared_determination_basis"]


def _direction(source: str, target: str) -> str:
    return f"{source}->{target}"


def applicable(view_id: str, probe: str) -> bool:
    return view_id in ev.LOOKUP_VIEWS if probe == "lookup" else True


def predict(probe: str, cache: dict[str, Any], fit_rows: np.ndarray, target_idx: np.ndarray, labels: list[str],
            cols: np.ndarray, task: str) -> tuple[np.ndarray, dict[str, Any]]:
    if probe == "cosine":
        return cosine_predict(cache, fit_rows, target_idx, labels, cols, task)
    if probe == "lookup":
        return lookup_predict(cache, fit_rows, target_idx, cols, task)
    if probe == "SrcMaj":
        return srcmaj_predict(cache, fit_rows, target_idx, task)
    raise SystemExit(f"unknown probe {probe}")


def _pred_hash(cache: dict[str, Any], rows: np.ndarray, pred: np.ndarray) -> str:
    return ev.prediction_hash(cache["row_id"][rows].tolist(), pred.tolist())


def tolerance_table(cache: dict[str, Any], views: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Cosine on the shared projection V^sh of the fitting rows (as in probe_table), all-source fit, all supported
    targets, for each tolerance (score_eps = tie_atol); every main view and both directions."""
    rows = []
    for view_id in ev.MAIN_VIEW_IDS:
        task = views[view_id]["task"]
        cols = ev.view_columns(cache, views, view_id)
        for source, target in ev.DIRECTIONS:
            _, fit, labels = ev.fitting_rows(cache, source, task, ev.ALL_SOURCE)
            shared, shared_hash = ev.shared_projection(cache, cols, fit)
            ti, _, metric = ev.target_rows(cache, source, target, task, ev.ALL_TARGET)
            scores, _ = cosine_score_matrix(cache, fit, ti, labels, shared, task)
            for eps in ev.TOLERANCE_GRID:
                pred, diag = score_predictions(scores, labels, score_eps=eps, tie_atol=eps)
                f1, acc = ev.evaluate_prediction(cache["gold_" + task][ti], pred, metric)
                rows.append({"view_id": view_id, "task": task, "direction": _direction(source, target),
                             "score_eps": eps, "tie_atol": eps, "view_feature_hash": shared_hash,
                             "macro_f1": f1, "accuracy": acc, "abstain_rate": float(diag["abstain_rate"]),
                             "prediction_hash": _pred_hash(cache, ti, pred)})
    return rows


def calibration_table(cache: dict[str, Any], facets: tuple[str, ...]) -> list[dict[str, Any]]:
    """SrcMaj (majority of all source rows) and Uniform (mean Macro-F1 of seeded uniform draws over the source
    inventory) on all supported targets."""
    rows = []
    for source, target, task, seed in ev.CALIBRATION_SCHEDULE:
        key = ev.label_sort_key(task, facets)
        labels, si_all = ev.source_inventory(cache, source, task)
        y = cache["gold_" + task]
        ti, _, metric = ev.target_rows(cache, source, target, task, ev.ALL_TARGET)
        counts = Counter(y[si_all].tolist())
        majority = sorted(labels, key=lambda x: (-int(counts[x]), key(x)))[0]
        maj_f1, _ = ev.evaluate_prediction(y[ti], np.full(len(ti), majority, dtype=object), metric)
        draws = uniform_macro_f1_draws(y[ti], sorted(labels, key=key), sorted(metric, key=key), seed)
        rows.append({"source": source, "target": target, "task": task, "n_draws": UNIFORM_DRAWS, "seed": seed,
                     "n_eval": len(ti), "source_candidates": len(labels), "metric_labels": len(metric),
                     "srcmaj_label": majority, "srcmaj_macro_f1": maj_f1, "uniform_macro_f1": float(draws.mean()),
                     "uniform_draw_sd": float(draws.std(ddof=1)), "uniform_draw_min": float(draws.min()),
                     "uniform_draw_max": float(draws.max())})
    return rows


def support_check_table(cache: dict[str, Any], views: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Shared projection V^sh from the fitting rows, recomputed independently from column sums, and the number
    of predictions that change when a probe uses V^sh instead of the declared view columns."""
    rows = []
    for view_id in ev.MAIN_VIEW_IDS:
        task = views[view_id]["task"]
        raw = ev.view_columns(cache, views, view_id)
        for source, target in ev.DIRECTIONS:
            for protocol, scope in ev.PROTOCOL_SCOPES:
                mask_rows, fit, labels = ev.fitting_rows(cache, source, task, protocol)
                ti, _, _ = ev.target_rows(cache, source, target, task, scope)
                shared, shared_hash = ev.shared_projection(cache, raw, fit)
                independent = raw[np.asarray(cache["X"][fit][:, raw].sum(axis=0)).ravel() != 0]
                independent_hash = ev.feature_hash(cache, independent)
                equal = bool(np.array_equal(shared, independent) and shared_hash == independent_hash)
                if not equal:
                    raise SystemExit(f"shared-projection support disagrees: {view_id} {source}->{target} {protocol}")
                for probe in PROBES:
                    if not applicable(view_id, probe):
                        continue
                    pred_raw, _ = predict(probe, cache, fit, ti, labels, raw, task)
                    if np.array_equal(raw, shared) or probe == "SrcMaj":
                        pred_shared = pred_raw.copy()
                    else:
                        pred_shared, _ = predict(probe, cache, fit, ti, labels, shared, task)
                    rows.append({
                        "training_protocol": protocol, "evaluation_scope": scope, "view_id": view_id, "task": task,
                        "direction": _direction(source, target), "probe": probe, "fitting_mask_rows": len(mask_rows),
                        "source_rows_fit": len(fit), "raw_feature_count": len(raw),
                        "shared_feature_count": len(shared), "raw_feature_hash": ev.feature_hash(cache, raw),
                        "shared_support_hash": shared_hash, "independent_shared_support_hash": independent_hash,
                        "support_recompute_equal": int(equal),
                        "raw_prediction_hash": _pred_hash(cache, ti, pred_raw),
                        "shared_prediction_hash": _pred_hash(cache, ti, pred_shared),
                        "raw_to_shared_prediction_changes": int(np.sum(pred_raw != pred_shared)),
                        "canonical_protocol": ev.SHARED_PROTOCOL})
    return rows


def _probe_row(cache, *, task, source, target, view_id, probe, protocol, scope, mask_rows, fit, labels, ti,
               metric, cols, shared, shared_hash, pred, f1, acc, abst, zero, status) -> dict[str, Any]:
    return {
        "task": task, "direction": _direction(source, target), "source": source, "target": target,
        "view_id": view_id, "probe": probe, "training_protocol": protocol, "evaluation_scope": scope,
        "fitting_mask_id": "ALL_SOURCE" if protocol == ev.ALL_SOURCE else "FULL_CONFORM_SOURCE",
        "fitting_mask_rows": len(mask_rows), "source_rows_fit": len(fit), "source_label_candidates": len(labels),
        "zero_prototype_labels": zero, "target_eval_rows": len(ti), "metric_labels": len(metric),
        "feature_count": len(cols), "feature_hash": ev.feature_hash(cache, cols),
        "shared_feature_count": len(shared), "shared_support_hash": shared_hash,
        "source_row_hash": ev.hash_rows(cache["row_id"][fit].tolist()),
        "target_row_hash": ev.hash_rows(cache["row_id"][ti].tolist()),
        "prediction_hash": None if pred is None else _pred_hash(cache, ti, pred),
        "macro_f1": f1, "accuracy": acc, "abstain_rate": abst, "status": status,
    }


def probe_table(cache: dict[str, Any], views: dict[str, dict[str, Any]]
                ) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]]]:
    """Probes on the shared projection V^sh of the fitting rows. All-source fits are scored on all supported
    targets and on registry-conforming targets; fixed-dataset refits on registry-conforming targets.
    A probe that the design does not define for a view yields a row without values.
    Returns the rows and the cosine predictions {(view, direction, protocol, scope): (target rows, pred)}."""
    rows: list[dict[str, Any]] = []
    cosine_preds = {}
    for view_id in ev.MAIN_VIEW_IDS:
        task = views[view_id]["task"]
        cols = ev.view_columns(cache, views, view_id)
        for source, target in ev.DIRECTIONS:
            for protocol in (ev.ALL_SOURCE, ev.REFIT):
                mask_rows, fit, labels = ev.fitting_rows(cache, source, task, protocol)
                shared, shared_hash = ev.shared_projection(cache, cols, fit)
                common = dict(task=task, source=source, target=target, view_id=view_id, protocol=protocol,
                              mask_rows=mask_rows, fit=fit, labels=labels, cols=cols, shared=shared,
                              shared_hash=shared_hash)
                for probe in PROBES:
                    if not applicable(view_id, probe):
                        ti, _, metric = ev.target_rows(cache, source, target, task, ev.CONFORM_TARGET)
                        rows.append(_probe_row(cache, probe=probe, scope=ev.CONFORM_TARGET, ti=ti, metric=metric,
                                               pred=None, f1=None, acc=None, abst=None, zero=None,
                                               status="NOT_DEFINED_FOR_VIEW", **common))
                        continue
                    scope0 = ev.ALL_TARGET if protocol == ev.ALL_SOURCE else ev.CONFORM_TARGET
                    ti0, _, metric0 = ev.target_rows(cache, source, target, task, scope0)
                    pred0, diag = predict(probe, cache, fit, ti0, labels, shared, task)
                    scopes = [(scope0, ti0, pred0, metric0)]
                    if protocol == ev.ALL_SOURCE:
                        keep = cache["full_mask"][ti0]
                        ti1 = ti0[keep]
                        scopes.append((ev.CONFORM_TARGET, ti1, pred0[keep],
                                       ev.metric_labels_for(cache, task, labels, ti1)))
                    for scope, ti, pred, metric in scopes:
                        f1, acc = ev.evaluate_prediction(cache["gold_" + task][ti], pred, metric)
                        rows.append(_probe_row(cache, probe=probe, scope=scope, ti=ti, metric=metric, pred=pred,
                                               f1=f1, acc=acc, abst=ev.abstain_rate(pred),
                                               zero=int(diag.get("zero_prototype_labels", 0)), status="OK",
                                               **common))
                        if probe == "cosine":
                            cosine_preds[(view_id, _direction(source, target), protocol, scope)] = (ti, pred)
    return rows, cosine_preds


def factorial_table(cache: dict[str, Any], views: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Cosine refit on the 8-cell block factorial (facet block F, family block M, type block T present or not,
    the core block always present), registry-conforming targets."""
    rows = []
    for task in TASKS:
        for source, target in ev.DIRECTIONS:
            for cell in CELL_IDS:
                view_id = f"factorial_{cell}__{task}"
                cols = ev.view_columns(cache, views, view_id)
                _, fit, labels = ev.fitting_rows(cache, source, task, ev.REFIT)
                shared, shared_hash = ev.shared_projection(cache, cols, fit)
                ti, _, metric = ev.target_rows(cache, source, target, task, ev.CONFORM_TARGET)
                pred, diag = cosine_predict(cache, fit, ti, labels, shared, task)
                f1, acc = ev.evaluate_prediction(cache["gold_" + task][ti], pred, metric)
                rows.append({"task": task, "direction": _direction(source, target), "cell_id": cell,
                             "view_id": view_id, "training_protocol": ev.REFIT, "evaluation_scope": ev.CONFORM_TARGET,
                             "source_rows_fit": len(fit), "target_eval_rows": len(ti),
                             "shared_support_hash": shared_hash, "prediction_hash": _pred_hash(cache, ti, pred),
                             "macro_f1": f1, "accuracy": acc, "abstain_rate": float(diag["abstain_rate"])})
    return rows


VOLUME_DIRECTION = ("FP", "DF")


def volume_table(cache: dict[str, Any], views: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Volume control: cosine fitted on random source subsets drawn from all source rows with the per-label
    sizes of the registry-conforming fitting set (FP->DF, family), registry-conforming targets."""
    source, target = VOLUME_DIRECTION
    task, view_id = "family", PREDICTION_VIEW
    cols = ev.view_columns(cache, views, view_id)
    all_mask, _, labels = ev.fitting_rows(cache, source, task, ev.ALL_SOURCE)
    _, conf_fit, _ = ev.fitting_rows(cache, source, task, ev.REFIT)
    y = cache["gold_" + task]
    ti, _, metric = ev.target_rows(cache, source, target, task, ev.CONFORM_TARGET)
    per_label = Counter(y[conf_fit].tolist())
    rows = []
    for seed in ev.VOLUME_SEEDS:
        rng = np.random.default_rng(seed)
        sampled: list[int] = []
        for lab in labels:
            pool = all_mask[y[all_mask] == lab]
            n = int(per_label.get(lab, 0))
            if n > len(pool):
                raise SystemExit("volume control: sample size exceeds the label pool")
            if n:
                sampled.extend(rng.choice(pool, size=n, replace=False).tolist())
        fit = np.asarray(sorted(sampled), dtype=np.int64)
        shared, shared_hash = ev.shared_projection(cache, cols, fit)
        pred, _ = cosine_predict(cache, fit, ti, labels, shared, task)
        f1, acc = ev.evaluate_prediction(y[ti], pred, metric)
        rows.append({"training_protocol": f"volume_matched_seed_{seed}", "seed": seed, "view_id": view_id,
                     "direction": _direction(source, target), "source_rows_fit": len(fit),
                     "target_eval_rows": len(ti), "shared_support_hash": shared_hash,
                     "prediction_hash": _pred_hash(cache, ti, pred), "macro_f1": f1, "accuracy": acc})
    return rows


def type_weight_table(cache: dict[str, Any], views: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Positive rescaling of the type:: columns in the cosine refit (FP->DF, identity-admitted family view)."""
    source, target = VOLUME_DIRECTION
    task, view_id = "family", PREDICTION_VIEW
    cols = ev.view_columns(cache, views, view_id)
    _, fit, labels = ev.fitting_rows(cache, source, task, ev.REFIT)
    ti, _, metric = ev.target_rows(cache, source, target, task, ev.CONFORM_TARGET)
    shared, shared_hash = ev.shared_projection(cache, cols, fit)
    rows = []
    for idx, weight in enumerate(ev.TYPE_WEIGHTS, start=1):
        pred, _ = cosine_predict(cache, fit, ti, labels, shared, task, type_weight=weight)
        f1, acc = ev.evaluate_prediction(cache["gold_" + task][ti], pred, metric)
        rows.append({"view_id": f"type_weight_{idx}__family", "base_view_id": view_id,
                     "direction": _direction(source, target), "type_weight": float(weight),
                     "training_protocol": ev.REFIT, "evaluation_scope": ev.CONFORM_TARGET,
                     "shared_support_hash": shared_hash, "macro_f1": f1, "accuracy": acc,
                     "prediction_hash": _pred_hash(cache, ti, pred)})
    return rows


def verdict_class_tables(cache: dict[str, Any], views: dict[str, dict[str, Any]], verdict_code: np.ndarray,
                         verdict_index: list[dict[str, str]],
                         cosine_preds: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]]
                         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Where the correct cosine predictions of the identity-admitted family view fall among the verdict
    classes of every family-task view (determination matrix), and the row for the view itself."""
    task = "family"
    cols = ev.view_columns(cache, views, PREDICTION_VIEW)
    family_views = [r for r in verdict_index if r["task"] == task]
    if PREDICTION_VIEW not in {r["view_id"] for r in family_views}:
        raise SystemExit("verdicts lack the identity-admitted family view")
    matrix: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for protocol, scope in ev.PROTOCOL_SCOPES:
        for source, target in ev.DIRECTIONS:
            direction = _direction(source, target)
            _, fit, _ = ev.fitting_rows(cache, source, task, protocol)
            _, shared_hash = ev.shared_projection(cache, cols, fit)
            ti, pred = cosine_preds[(PREDICTION_VIEW, direction, protocol, scope)]
            correct = pred == cache["gold_family"][ti]
            correct_n = int(np.count_nonzero(correct))
            if correct_n <= 0:
                raise SystemExit(f"no correct cosine prediction: {protocol} {direction}")
            pred_hash = _pred_hash(cache, ti, pred)
            for rec in sorted(family_views, key=lambda r: r["view_id"]):
                v = verdict_code[int(rec["array_index"])][ti]
                n_unsafe = int(np.count_nonzero(correct & (v == CODE["UNSAFE"])))
                n_nondet = int(np.count_nonzero(correct & (v == CODE["NONDETERMINED"])))
                n_unver = int(np.count_nonzero(correct & (v == CODE["UNVERIFIED"])))
                if n_unsafe + n_nondet + n_unver != correct_n:
                    raise SystemExit("verdict classes do not partition the correct predictions")
                row = {"metric_id": "cosine_correct_on_unsafe", "prediction_view_id": PREDICTION_VIEW,
                       "determination_view_id": rec["view_id"],
                       "determination_certificate_path": rec["certificate_path"], "task": task,
                       "direction": direction, "training_protocol": protocol, "evaluation_scope": scope,
                       "source_rows_fit": len(fit), "target_eval_rows": len(ti), "shared_support_hash": shared_hash,
                       "prediction_hash": pred_hash, "correct_predictions": correct_n,
                       "correct_on_unsafe": n_unsafe, "correct_on_nondetermined": n_nondet,
                       "correct_on_unverified": n_unver, "fraction_correct_on_unsafe": n_unsafe / correct_n}
                matrix.append(row)
                if rec["view_id"] == PREDICTION_VIEW:
                    summary.append({**row, "declared_determination_basis": f"{PREDICTION_VIEW}/UNSAFE"})
    if len(summary) != 4 or len(matrix) != 4 * len(family_views):
        raise SystemExit("verdict-class tables have an unexpected number of rows")
    return matrix, summary


# ----------------------------------------------------------------------------
# Registry-rule tables
# ----------------------------------------------------------------------------

# Main Path-A certificates: view -> (task, basis, registered carrier edges).
RULE_VIEWS = {
    "identity_admitted__family": ("family", "facet::", "W"),
    "facet_only__family": ("family", "facet::", "W"),
    "core_plus_type__family": ("family", "core::", "Pi|W"),
    "identity_admitted__facet": ("facet", "core::", "Pi"),
}
RULE_SCOPE = "shared_source_fit_support_on_pathA_basis"
FULLVIEW_FIELDS = ["view_id", "task", "dataset", "certificate_path", "basis", "carrier_edges", "query_conform_rows",
                   "decided_rows", "unsafe_rows", "decided_equals_unsafe_rows", "decided_equals_unsafe_rate",
                   "decided_label_correct_rows", "decided_precision", "primary_score_sha256",
                   "verifier_score_sha256", "pass"]
RULE_FIELDS = [
    "view_id", "task", "direction", "source", "target", "certificate_path", "basis", "carrier_edges",
    "training_protocol", "evaluation_scope", "rule_scope", "learned_parameters", "random_seeds",
    "basis_feature_count", "shared_basis_feature_count", "shared_basis_support_sha256", "source_rows_mask",
    "source_rows_fit", "source_label_candidates", "target_eval_rows", "metric_labels", "decided_rows", "coverage",
    "correct_decided_rows", "precision_on_decisions", "accuracy", "macro_f1", "predicted_outside_source_inventory",
    "target_unsafe_rows", "decided_and_unsafe_rows", "fraction_decided_that_is_unsafe", "fraction_unsafe_decided",
    "cosine_refit_f1", "cosine_all_f1", "beta_upper_bound", "rule_le_beta", "target_row_sha256",
    "prediction_sha256", "score_matrix_sha256", "verifier_score_matrix_sha256", "pass",
]
LOOKUP_COVERAGE_FIELDS = ["view_id", "task", "direction", "macro_f1", "target_eval_rows", "coverage", "source"]


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def rule_certificates(certificate_spec: list[dict[str, str]]) -> list[dict[str, str]]:
    """The main Path-A certificates, checked against the declared basis and carrier edges."""
    rows = []
    for r in certificate_spec:
        if r["view_id"] not in RULE_VIEWS:
            continue
        task, basis, edges = RULE_VIEWS[r["view_id"]]
        if (r["certificate_path"], r["task"], r["basis"], r["carrier_edges"]) != ("A", task, basis, edges):
            raise SystemExit(f"certificate of {r['view_id']} differs from the declared Path-A basis")
        rows.append(r)
    if {r["view_id"] for r in rows} != set(RULE_VIEWS):
        raise SystemExit("the main Path-A certificate set differs from the declared views")
    return sorted(rows, key=lambda r: r["view_id"])


def _one(rows: list[dict[str, str]], **eq: str) -> dict[str, str]:
    hit = [r for r in rows if all(r[k] == v for k, v in eq.items())]
    if len(hit) != 1:
        raise SystemExit(f"expected one row for {eq}, found {len(hit)}")
    return hit[0]


def registry_rule_tables(cache: dict[str, Any], certificates: list[dict[str, str]], Pi: dict[str, str],
                         W: dict[str, str], coords: dict[str, list[str]], query_conform: dict[str, np.ndarray],
                         verdicts: dict[str, np.ndarray], probe_rows: list[dict[str, str]],
                         bound_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parameter-free registry rule on the Path-A basis.

    Full view: on every query-conforming row the rule decides exactly the UNSAFE rows, always correctly.
    Shared projection: the basis is cut to the coordinates supported by the fixed-dataset fitting rows and the
    rule is scored like a probe on the registry-conforming targets; it is compared with the cosine probe and
    with beta read from probes.csv and bounds.csv of this step."""
    X, features = cache["X"], cache["features"]
    all_rows = np.arange(X.shape[0], dtype=np.int64)
    full_rows: list[dict[str, Any]] = []
    shared_rows: list[dict[str, Any]] = []
    for cert in certificates:
        view_id, task, basis, edges = cert["view_id"], cert["task"], cert["basis"], cert["carrier_edges"]
        labels = coords[task]
        gold = cache["gold_" + task]
        basis_cols = ev.selected_columns(features, {basis: "ALL"})
        M, _, _ = mapping_matrix(features, basis_cols, basis, task, edges, labels, Pi, W)
        if unmapped_evidence(X, all_rows, basis_cols, M):
            raise SystemExit(f"unmapped basis coordinates carry evidence: {view_id}")
        primary = rule_scores(X, all_rows, basis_cols, M)
        verifier = rule_scores_by_accumulation(X, all_rows, basis_cols, M)
        if not np.array_equal(primary, verifier):
            raise SystemExit(f"full-view rule scores differ between the two computations: {view_id}")
        pred, decided = strict_argmax(primary, labels)
        for ds in DATASETS:
            idx = np.flatnonzero((cache["dataset"] == ds) & query_conform[view_id])
            unsafe = verdicts[view_id][idx] == CODE["UNSAFE"]
            dec = decided[idx]
            match = int(np.count_nonzero(dec == unsafe))
            decided_n = int(np.count_nonzero(dec))
            correct = int(np.count_nonzero(dec & (pred[idx] == gold[idx])))
            if match != len(idx):
                raise SystemExit(f"full-view rule decisions differ from UNSAFE: {view_id} {ds}")
            if correct != decided_n:
                raise SystemExit(f"full-view rule decision with a wrong label: {view_id} {ds}")
            full_rows.append({
                "view_id": view_id, "task": task, "dataset": ds, "certificate_path": "A", "basis": basis,
                "carrier_edges": edges, "query_conform_rows": len(idx), "decided_rows": decided_n,
                "unsafe_rows": int(np.count_nonzero(unsafe)), "decided_equals_unsafe_rows": match,
                "decided_equals_unsafe_rate": _ratio(match, len(idx)), "decided_label_correct_rows": correct,
                "decided_precision": _ratio(correct, decided_n), "primary_score_sha256": array_sha(primary[idx]),
                "verifier_score_sha256": array_sha(verifier[idx]), "pass": 1})
        del primary, verifier, pred, decided

        for source, target in ev.DIRECTIONS:
            direction = _direction(source, target)
            mask_rows, fit, source_labels = ev.fitting_rows(cache, source, task, ev.REFIT)
            ti, _, metric = ev.target_rows(cache, source, target, task, ev.CONFORM_TARGET)
            shared, shared_hash = ev.shared_projection(cache, basis_cols, fit)
            M, _, unmapped = mapping_matrix(features, shared, basis, task, edges, labels, Pi, W)
            if unmapped:
                raise SystemExit(f"shared basis has unmapped coordinates: {view_id} {direction}")
            scores = rule_scores(X, ti, shared, M)
            check = rule_scores_by_accumulation(X, ti, shared, M)
            if not np.array_equal(scores, check):
                raise SystemExit(f"shared rule scores differ between the two computations: {view_id} {direction}")
            pred, decided = strict_argmax(scores, labels)
            y_true = gold[ti]
            f1, acc = ev.evaluate_prediction(y_true, pred, metric)
            decided_n = int(np.count_nonzero(decided))
            correct = int(np.count_nonzero(decided & (pred == y_true)))
            outside = int(np.count_nonzero(decided & ~np.isin(pred, np.asarray(source_labels, dtype=object))))
            unsafe = verdicts[view_id][ti] == CODE["UNSAFE"]
            dec_unsafe = int(np.count_nonzero(decided & unsafe))
            unsafe_n = int(np.count_nonzero(unsafe))
            refit = _one(probe_rows, view_id=view_id, direction=direction, probe="cosine",
                         training_protocol=ev.REFIT, evaluation_scope=ev.CONFORM_TARGET)
            allsrc = _one(probe_rows, view_id=view_id, direction=direction, probe="cosine",
                        training_protocol=ev.ALL_SOURCE, evaluation_scope=ev.ALL_TARGET)
            bound = _one(bound_rows, view_id=view_id, direction=direction, training_protocol=ev.REFIT,
                         evaluation_scope=ev.CONFORM_TARGET)
            beta = float(bound["macro_f1_upper_bound_C2"])
            target_hash = ev.hash_rows(cache["row_id"][ti].tolist())
            if len(ti) != int(refit["target_eval_rows"]) or target_hash != refit["target_row_hash"]:
                raise SystemExit(f"rule target rows differ from the probe target rows: {view_id} {direction}")
            if len(metric) != int(refit["metric_labels"]):
                raise SystemExit(f"rule metric labels differ from the probe metric labels: {view_id} {direction}")
            if f1 > beta + ev.BOUND_SLACK:
                raise SystemExit(f"registry rule exceeds beta: {view_id} {direction}")
            shared_rows.append({
                "view_id": view_id, "task": task, "direction": direction, "source": source, "target": target,
                "certificate_path": "A", "basis": basis, "carrier_edges": edges, "training_protocol": ev.REFIT,
                "evaluation_scope": ev.CONFORM_TARGET, "rule_scope": RULE_SCOPE, "learned_parameters": 0,
                "random_seeds": 0, "basis_feature_count": len(basis_cols), "shared_basis_feature_count": len(shared),
                "shared_basis_support_sha256": shared_hash, "source_rows_mask": len(mask_rows),
                "source_rows_fit": len(fit), "source_label_candidates": len(source_labels),
                "target_eval_rows": len(ti), "metric_labels": len(metric), "decided_rows": decided_n,
                "coverage": _ratio(decided_n, len(ti)), "correct_decided_rows": correct,
                "precision_on_decisions": _ratio(correct, decided_n), "accuracy": acc, "macro_f1": f1,
                "predicted_outside_source_inventory": outside, "target_unsafe_rows": unsafe_n,
                "decided_and_unsafe_rows": dec_unsafe,
                "fraction_decided_that_is_unsafe": _ratio(dec_unsafe, decided_n),
                "fraction_unsafe_decided": _ratio(dec_unsafe, unsafe_n),
                "cosine_refit_f1": float(refit["macro_f1"]), "cosine_all_f1": float(allsrc["macro_f1"]),
                "beta_upper_bound": beta, "rule_le_beta": int(f1 <= beta + ev.BOUND_SLACK),
                "target_row_sha256": target_hash, "prediction_sha256": _pred_hash(cache, ti, pred),
                "score_matrix_sha256": array_sha(scores), "verifier_score_matrix_sha256": array_sha(check),
                "pass": 1})
    if len(full_rows) != 8 or len(shared_rows) != 8:
        raise SystemExit("registry-rule tables have an unexpected number of rows")
    return full_rows, shared_rows


def lookup_coverage_table(probe_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Coverage (non-abstained share) and Macro-F1 of the Lookup refit, read from probes.csv."""
    rows = []
    for r in probe_rows:
        if (r["probe"] == "lookup" and r["training_protocol"] == ev.REFIT and r["evaluation_scope"] == ev.CONFORM_TARGET
                and r["status"] == "OK" and r["view_id"] in ev.LOOKUP_VIEWS):
            rows.append({"view_id": r["view_id"], "task": r["task"], "direction": r["direction"],
                         "macro_f1": float(r["macro_f1"]), "target_eval_rows": int(r["target_eval_rows"]),
                         "coverage": 1.0 - float(r["abstain_rate"]), "source": "utility/probes.csv"})
    if len(rows) != 2 * len(ev.LOOKUP_VIEWS):
        raise SystemExit("lookup coverage table has an unexpected number of rows")
    return rows
