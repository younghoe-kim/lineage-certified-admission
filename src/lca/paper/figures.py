# SPDX-License-Identifier: MIT
"""Figures 1-3 of the paper. For each figure, data_<fig>(outputs) collects the plotted values (also written as a
CSV with columns panel, series, x, y) and draw_<fig>(data, path) draws it. Only drawing imports matplotlib."""
from __future__ import annotations

import csv
import io
import json
from math import comb
from decimal import Decimal
from pathlib import Path
from typing import Any

from .common import Outputs, PaperError, dec, frac, ival, one

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, PANEL, EDGE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#f6f5f1", "#c9c8c2"
FAMILY_CODE = {"taxonomic_category": "TAX", "structural_expression": "STR", "surface_expression": "SURF",
               "attribute_expression": "ATTR", "tag_terms": "TAG"}
CSV_FIELDS = ["panel", "series", "x", "y"]
# Fixed PDF metadata: no creation date and no producer or creator string, so the bytes depend only on the data.
PDF_METADATA = {"CreationDate": None, "Producer": None, "Creator": None}
RC = {
    "font.family": "DejaVu Sans", "font.size": 6.4, "axes.titlesize": 6.8, "axes.labelsize": 6.4,
    "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "axes.edgecolor": INK2, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "svg.hashsalt": "lca", "path.simplify": True,
}
FIG1_MAX_FACETS = 4


def unit_label(v: float, _pos: Any = None) -> str:
    """Tick label of a [0, 1] axis, as in the text: 0, 0.2, 0.25, ..., 1."""
    if abs(v) < 1e-12:
        return "0"
    if abs(v - 1) < 1e-12:
        return "1"
    return f"{v:.2f}".rstrip("0")


def unit_ticks(axis: Any) -> None:
    from matplotlib.ticker import FuncFormatter
    axis.set_major_formatter(FuncFormatter(unit_label))


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fam(code: str) -> str:
    if code not in FAMILY_CODE:
        raise PaperError(f"no display code for family {code!r}")
    return FAMILY_CODE[code]


def num(x: Any) -> str:
    """CSV text of a plotted number."""
    if isinstance(x, bool):
        raise PaperError("boolean in figure data")
    if isinstance(x, int):
        return str(x)
    return repr(float(x))


def rows_csv(rows: list[tuple[str, str, Any, Any]]) -> str:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_FIELDS)
    for panel, series, x, y in rows:
        w.writerow([panel, series, x if isinstance(x, str) else num(x), y if isinstance(y, str) else num(y)])
    return buf.getvalue()


def save(fig, path: Path, **kw) -> None:
    fig.savefig(path, metadata=PDF_METADATA, **kw)


# ---------------------------------------------------------------------- Fig. 1
FIG1_PATHB = ("type3__family", "FP", "[1,2,0]")   # the type value of the Path B example of Sec. I


def data_fig1(out: Outputs) -> dict[str, Any]:
    f1 = out.json("verdicts/fig1_example.json")
    inst = f1["selected"]
    fmap = out.csv("registry/facet_map.csv")
    wmap = {r["facet"]: r["family"] for r in fmap}
    items = sorted(((k, ival(v)) for k, v in inst["facet_nonzero"].items()), key=lambda kv: (-kv[1], kv[0]))
    fams = sorted(((k, ival(v)) for k, v in inst["declared_W_family"].items()), key=lambda kv: (-kv[1], fam(kv[0])))
    if not items or len(items) > FIG1_MAX_FACETS or len(fams) > FIG1_MAX_FACETS:
        raise PaperError("the Fig. 1 anchor must have one to four facet and family coordinates")
    if len(fams) > 1 and fams[1][1] == fams[0][1]:
        raise PaperError("the Fig. 1 anchor must have a strict unique maximum")
    if any(wmap[name] not in dict(fams) for name, _ in items):
        raise PaperError("a facet of the Fig. 1 anchor maps to a family outside its family block")
    # panel (b): the certified Path B type value of the introduction, read from its certificate record
    view, ds, value_json = FIG1_PATHB
    cert = [r for r in out.csv("verdicts/path_b_certificates.csv")
            if (r["view_id"], r["dataset"], r["fiber_value"]) == FIG1_PATHB]
    if len(cert) != 1 or cert[0]["pass"] != "1":
        raise PaperError("the Path B example of Fig. 1 must have one passing certificate")
    proj = {r["view_id"]: json.loads(r["projection_json"]) for r in out.csv("registry/view_projection.csv")}
    types = proj[view]["type::"]
    value = json.loads(value_json)
    support = out.json("registry/dataset_support.json")["datasets"][ds]["S_members"]
    ttype = {r["facet"]: r["term_type"] for r in fmap}
    per_type = {t: sorted(f for f in support if ttype[f] == t) for t in types}
    states = 1
    for t, k in zip(types, value):
        states *= comb(len(per_type[t]) + k - 1, k)
    if states != ival(cert[0]["integer_states_checked"]):
        raise PaperError("the Path B example of Fig. 1 does not have the state count of its certificate")
    label = cert[0]["certified_label"]
    rows_b = [(t, k, len(per_type[t]), {c: sum(wmap[f] == c for f in per_type[t]) for c in sorted({wmap[f] for f in per_type[t]})})
              for t, k in zip(types, value) if k]
    rows = [("a", "facet", name, c) for name, c in items]
    rows += [("a", "family", fam(code), c) for code, c in fams]
    rows += [("a", "W", name, fam(wmap[name])) for name, _ in items]
    rows += [("a", "candidates", "", ival(f1["candidate_rows"]))]
    rows += [("b", "type", t, k) for t, k, _, _ in rows_b]
    rows += [("b", "facets of type", t, n) for t, _, n, _ in rows_b]
    rows += [("b", f"W {t}", fam(c), n) for t, _, _, fc in rows_b for c, n in fc.items()]
    rows += [("b", "states", "", states), ("b", "certified label", "", fam(label))]
    return {"items": items, "fams": fams, "wmap": wmap, "rows_b": rows_b, "states": states, "label": label,
            "rows": rows}


def draw_fig1(d: dict[str, Any], path: Path) -> None:
    """(a) Path A on Example 1 and (b) Path B on one FP type value, side by side; (c) the admission gate."""
    plt = pyplot()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    grey, peach = "#eeeeec", "#fdeee7"

    def box(ax, x, y, w, h, text, fc="white", ec=EDGE, lw=0.6, fs=5.0, color=INK, ls="-", weight="normal", ha="center"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015", fc=fc, ec=ec, lw=lw,
                                    ls=ls, transform=ax.transAxes))
        tx = x + w / 2 if ha == "center" else x + 0.02
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=color, transform=ax.transAxes,
                weight=weight, linespacing=1.1)

    def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.6, ms=5):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                     transform=ax.transAxes, shrinkA=0, shrinkB=0))

    def panel(ax, title, fs=6.4):
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03", fc=PANEL, ec="none",
                                    transform=ax.transAxes))
        ax.text(0.04, 0.955, title, ha="left", va="top", fontsize=fs, weight="bold", color=INK, transform=ax.transAxes)

    items, fams, wmap = d["items"], d["fams"], d["wmap"]
    top = fams[0][0]
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(3.45, 2.62))
        # (a) Path A on Example 1
        ax = fig.add_axes([0, 0.405, 0.492, 0.595])
        panel(ax, "(a) Path A: Example 1 (DF)")
        ax.text(0.04, 0.80, "retained facet::", fontsize=4.9, color=INK2, transform=ax.transAxes)
        ax.text(0.64, 0.80, "carrier family::", fontsize=4.9, color=INK2, transform=ax.transAxes)
        ys, h = [0.605, 0.455, 0.305], 0.115
        fy, famy = {}, {}
        for (name, c), y in zip(items, ys):
            lab = name.replace("attribute/", "attr: ").replace("category/", "cat: ")
            if len(lab) > 22:
                cut = lab.rfind(" ", 0, len(lab) // 2 + 4)
                lab = lab[:cut] + "\n" + lab[cut + 1:]
            box(ax, 0.04, y, 0.47, h, lab, fs=4.6, ha="left")
            ax.text(0.49, y + h / 2, f"{c}", ha="right", va="center", fontsize=5.6, weight="bold", color=BLUE,
                    transform=ax.transAxes)
            fy[name] = y
        for (code, c), y in zip(fams, ys):
            t = code == top
            box(ax, 0.66, y, 0.30, h, f"{fam(code)} {c}", fc=peach if t else "white", ec=ORANGE if t else EDGE, ls="--",
                fs=5.2, weight="bold" if t else "normal")
            famy[code] = y
        for name, y in fy.items():
            arrow(ax, 0.52, y + h / 2, 0.655, famy[wmap[name]] + h / 2)
        ax.text(0.585, 0.735, r"$W$", ha="center", fontsize=6.0, color=INK, transform=ax.transAxes)
        box(ax, 0.04, 0.05, 0.40, 0.19, "identity\ncheck: PASS", fc=grey, fs=4.8)
        box(ax, 0.48, 0.05, 0.48, 0.19, f"carrier fixed,\n$q$ = {fam(top)}: UNSAFE", fc=peach, ec=ORANGE, fs=4.8,
            weight="bold")
        # (b) Path B on one FP type value
        bx = fig.add_axes([0.508, 0.405, 0.492, 0.595])
        panel(bx, "(b) Path B: one FP type value")
        bx.text(0.04, 0.80, "retained type::", fontsize=4.9, color=INK2, transform=bx.transAxes)
        bx.text(0.64, 0.80, "carrier family::", fontsize=4.9, color=INK2, transform=bx.transAxes)
        hb = 0.15
        for (t, k, n, fc), y in zip(d["rows_b"], [0.585, 0.395]):
            box(bx, 0.04, y, 0.47, hb, f"{t}\n({k} of {n} facets)", fs=4.5, ha="left")
            bx.text(0.49, y + hb / 2, f"{k}", ha="right", va="center", fontsize=5.6, weight="bold", color=BLUE,
                    transform=bx.transAxes)
            if len(fc) == 1:
                box(bx, 0.66, y, 0.30, hb, f"{fam(next(iter(fc)))} {k}", fc=peach, ec=ORANGE, ls="--", fs=5.2,
                    weight="bold")
            else:
                order = sorted(fc, key=lambda c: (-fc[c], fam(c)))
                box(bx, 0.66, y, 0.30, hb, ", ".join(fam(c) for c in order[:-1]) + f"\nor {fam(order[-1])} {k}", ls="--",
                    fs=4.5)
            arrow(bx, 0.52, y + hb / 2, 0.655, y + hb / 2)
        bx.text(0.585, 0.735, r"$W$", ha="center", fontsize=6.0, color=INK, transform=bx.transAxes)
        bx.text(0.50, 0.30, f"{d['states']} facet-count assignments", ha="center", va="center", fontsize=4.8, color=INK2,
                transform=bx.transAxes)
        box(bx, 0.04, 0.05, 0.40, 0.19, "no forward\nmap to carrier", fc=grey, fs=4.8)
        box(bx, 0.48, 0.05, 0.48, 0.19, f"carrier varies,\n$q$ = {fam(d['label'])}: UNSAFE", fc=peach, ec=ORANGE,
            fs=4.8, weight="bold")
        # (c) the admission gate
        cx = fig.add_axes([0, 0, 1, 0.385])
        panel(cx, "(c) The admission gate (before any model is trained)", fs=6.8)
        steps = [("declare\n$q$, $L^{rel}$, spec", "#eef4fc", "schema and\nregistry only"),
                 ("FULL-\nCONFORM", "white", "per row,\nlabel-free"),
                 ("QUERY-\nCONFORM", "white", "per view\nand query")]
        x, sw, sg = 0.025, 0.14, 0.03
        for i, (t, col, note) in enumerate(steps):
            box(cx, x, 0.34, sw, 0.34, t, fc=col, fs=5.4)
            cx.text(x + sw / 2, 0.18, note, ha="center", va="center", fontsize=4.9, color=INK2, transform=cx.transAxes,
                    linespacing=1.1)
            if i < 2:
                arrow(cx, x + sw, 0.51, x + sw + sg, 0.51, color=INK2, ms=6)
            x += sw + sg
        fx, ax0, aw, vx = x - sg, 0.555, 0.198, 0.810
        for t, y in [("A: linear certificate", 0.66), ("B: fiber audit", 0.45), ("C: witness pairs", 0.24)]:
            box(cx, ax0, y - 0.065, aw, 0.13, t, fs=4.8)
            arrow(cx, fx + 0.006, 0.51, ax0 - 0.004, y, color=INK2, ms=6)
            arrow(cx, ax0 + aw + 0.004, y, vx - 0.004, 0.51, color=INK2, lw=0.7, ms=6)
        box(cx, vx, 0.20, 0.175, 0.62, "UNSAFE\nNONDETERMINED\nUNVERIFIED\nmasses $\\to$ admit", fc=peach, ec=ORANGE,
            fs=4.6)
        save(fig, path)
        plt.close(fig)


# ---------------------------------------------------------------------- Fig. 2
FIG2_SETTINGS = [("DF->FP", "facet", "DF\u2192FP facet"), ("DF->FP", "family", "DF\u2192FP family"),
                 ("FP->DF", "facet", "FP\u2192DF facet"), ("FP->DF", "family", "FP\u2192DF family")]
EIGHT = [f"F{f}M{m}T{t}" for f in (0, 1) for m in (0, 1) for t in (0, 1)]


def data_fig2(out: Outputs) -> dict[str, Any]:
    fx = out.csv("utility/factorial_8cell.csv")
    tw = out.csv("utility/type_weight.csv")
    bd = out.csv("utility/bounds.csv")
    settings = []
    for d, task, lab in FIG2_SETTINGS:
        vals = {c: float(dec(one(fx, direction=d, task=task, cell_id=c)["macro_f1"])) for c in EIGHT}
        settings.append((d, task, lab, vals))
    weights = sorted(((r["type_weight"], float(dec(r["macro_f1"]))) for r in tw), key=lambda kv: float(kv[0]))
    c2 = float(dec(one(bd, view_id="identity_admitted__family", direction="FP->DF",
                       training_protocol="fixed_dataset_refit")["macro_f1_upper_bound_C2"]))
    rows = [("a", f"{d} {task}", c, v) for d, task, _, vals in settings for c, v in vals.items()]
    rows += [("b", "type weight", w, v) for w, v in weights]
    rows += [("b", "F1 bound", "", c2)]
    return {"settings": settings, "weights": weights, "c2": c2, "rows": rows}


def draw_fig2(d: dict[str, Any], path: Path) -> None:
    plt = pyplot()
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.78),
                                 gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.30})
        ax = axes[0]
        settings = d["settings"]
        for i, (_, task, _, vals) in enumerate(settings):
            y = len(settings) - 1 - i
            carrier = (lambda c: c[1] == "1") if task == "facet" else (lambda c: c[3] == "1")
            lo, hi = min(vals.values()), max(vals.values())
            ax.plot([lo, hi], [y, y], color=GRID, lw=3.2, solid_capstyle="round", zorder=1)
            for c, v in vals.items():
                if carrier(c):
                    ax.scatter(v, y, s=16, marker="o", facecolor=ORANGE, edgecolor="white", lw=0.6, zorder=3)
                else:
                    ax.scatter(v, y, s=16, marker="D", facecolor=BLUE, edgecolor="white", lw=0.6, zorder=3)
            ax.text(hi + 0.02, y, f"width {frac(Decimal(str(hi)) - Decimal(str(lo)), 3)}", va="center", fontsize=5.6,
                    color=INK2)
        ax.set_yticks(range(len(settings)))
        ax.set_yticklabels([s[2] for s in reversed(settings)])
        ax.set_xlim(0, 0.92)
        unit_ticks(ax.xaxis)
        ax.set_ylim(-0.6, len(settings) - 0.4)
        ax.set_xlabel("Macro-F1 (conforming fit, registry-conforming targets)")
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.scatter([], [], s=16, marker="D", facecolor=BLUE, edgecolor="white", label="carrier excluded")
        ax.scatter([], [], s=16, marker="o", facecolor=ORANGE, edgecolor="white", label="carrier exposed (L0)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2, fontsize=5.4, frameon=False,
                  handletextpad=0.2, columnspacing=1.2)
        ax.set_title("(a) Eight block combinations", loc="left", weight="bold")

        ax = axes[1]
        weights = d["weights"]
        xs = range(len(weights))
        vals = [v for _, v in weights]
        ax.bar(xs, vals, width=0.62, color=BLUE, edgecolor="white", lw=0.6)
        for x_, v in zip(xs, vals):
            ax.text(x_, v + 0.02, frac(Decimal(str(v)), 3), ha="center", fontsize=5.4, color=INK)
        c2 = d["c2"]
        ax.axhline(c2, color=INK2, lw=0.7, ls="--")
        ax.text(len(weights) - 0.6, c2 - 0.05, f"F1 bound {frac(Decimal(str(c2)), 3)} (unchanged)", ha="right",
                va="top", fontsize=5.4, color=INK2)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([w.rstrip("0").rstrip(".") if "." in w else w for w, _ in weights])
        ax.set_xlabel("type:: weight (FP\u2192DF family)")
        ax.set_ylim(0, 1.05)
        unit_ticks(ax.yaxis)
        ax.set_ylabel("Macro-F1")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title("(b) Rescaling type::", loc="left", weight="bold")

        save(fig, path)
        plt.close(fig)


# ---------------------------------------------------------------------- Fig. 3
def data_fig3(out: Outputs) -> dict[str, Any]:
    eq = sorted(out.csv("external/query_results.csv"), key=lambda r: r["query_id"])
    agg = sorted(out.csv("scaling/scaling.csv"), key=lambda r: float(r["fraction"]))
    queries = []
    rows = []
    for r in eq:
        ne, l0, l1 = ival(r["nonempty_rows"]), ival(r["L0_direct_rows"]), ival(r["L1_exact_derived_rows"])
        name = f"{r['domain']} / {r['target_attribute']}"
        queries.append((name, ne, l0, l1))
        rows += [("a", name, "L0", l0 / ne), ("a", name, "L1", l1 / ne), ("a", name, "UNVERIFIED", (ne - l0 - l1) / ne),
                 ("a", name, "L1 rows", l1), ("a", name, "non-empty rows", ne)]
    n = [ival(r["sampled_rows_total"]) / 1000 for r in agg]
    g = [float(dec(r["sum_grouping_seconds_median"])) for r in agg]
    s = [float(dec(r["sum_milp_seconds_median"])) for r in agg]
    fibers = [ival(r["sum_solver_fibers"]) for r in agg]
    rows += [("b", "group-by seconds", x, y) for x, y in zip(n, g)]
    rows += [("b", "integer-program seconds", x, y) for x, y in zip(n, s)]
    rows += [("b", "single-label fibers", x, y) for x, y in zip(n, fibers)]
    return {"queries": queries, "n": n, "g": g, "s": s, "fibers": fibers, "rows": rows}


def draw_fig3(d: dict[str, Any], path: Path) -> None:
    plt = pyplot()
    from matplotlib.patches import Patch
    with plt.rc_context(RC):
        fw, fh = 7.16, 1.95
        aw, ah, bottom = 2.30, 1.20, 0.55
        left_a, left_b = 0.98, 0.98 + 2.30 + 0.62 + 0.52
        fig = plt.figure(figsize=(fw, fh))
        axes = [fig.add_axes([left_a / fw, bottom / fh, aw / fw, ah / fh]),
                fig.add_axes([left_b / fw, bottom / fh, aw / fw, ah / fh])]
        ax = axes[0]
        qs = d["queries"]
        for i, (name, ne, l0, l1) in enumerate(qs):
            y = len(qs) - 1 - i
            un = ne - l0 - l1
            ax.barh(y, l0 / ne, color=BLUE, edgecolor="white", lw=0.6, height=0.62)
            ax.barh(y, l1 / ne, left=l0 / ne, color=ORANGE, edgecolor="white", lw=0.6, height=0.62)
            ax.barh(y, un / ne, left=(l0 + l1) / ne, color="#dcdbd5", edgecolor="white", lw=0.6, height=0.62)
            ax.text(1.015, y, f"{l1:,} / {ne:,}", va="center", fontsize=5.4, color=INK if l1 else INK2)
        ax.set_yticks(range(len(qs)))
        ax.set_yticklabels([q[0] for q in reversed(qs)])
        ax.set_xlim(0, 1)
        unit_ticks(ax.xaxis)
        ax.set_xlabel("share of nonempty source cells")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        handles = [Patch(color=BLUE, label="L0: protected value or alias"),
                   Patch(color=ORANGE, label="L1: determined through the taxonomy"),
                   Patch(color="#dcdbd5", label="UNVERIFIED")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=5.4,
                  frameon=False, handlelength=1.0, columnspacing=1.0)
        ax.text(1.015, len(qs) - 0.35, "L1 / nonempty", fontsize=5.2, color=INK2, va="bottom")
        ax.set_title("(a) External benchmark: six queries", loc="left", weight="bold")

        ax = axes[1]
        n, g, s, fibers = d["n"], d["g"], d["s"], d["fibers"]
        ax.plot(n, g, color=BLUE, lw=1.4, marker="o", ms=2.6)
        ax.plot(n, s, color=ORANGE, lw=1.4, marker="s", ms=2.6)
        ax.text(n[-1] + 8, g[-1], "group-by", va="center", fontsize=5.6, color=INK)
        ax.text(n[-1] + 8, s[-1] + 0.03, "integer\nprograms", va="center", fontsize=5.6, color=INK)
        ax.text(n[0], s[0] - 0.09, f"{fibers[0]} single-label fibers", fontsize=5.2, color=INK2)
        ax.text(n[-1], s[-1] + 0.10, f"{fibers[-1]} single-label fibers", fontsize=5.2, color=INK2, ha="right")
        ax.set_xlim(0, 760)
        ax.set_ylim(0, 0.85)
        ax.set_xlabel("anchors sampled (thousands)")
        ax.set_ylabel("seconds (median of three runs)")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title("(b) Path B time versus rows", loc="left", weight="bold")
        save(fig, path)
        plt.close(fig)


FIGURES = (("fig1_example", data_fig1, draw_fig1), ("fig2_partition", data_fig2, draw_fig2),
           ("fig3_breadth_cost", data_fig3, draw_fig3))
