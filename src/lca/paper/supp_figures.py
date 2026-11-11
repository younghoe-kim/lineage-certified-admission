# SPDX-License-Identifier: MIT
"""Figures S1-S3 of the extended audit details, with the same data/draw split as figures.py."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import Outputs, PaperError, dec, frac, ival, one, select, view_parts
from .figures import pyplot, save, unit_ticks

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, LIGHT = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#d4d3cd"
RC = {
    "font.family": "DejaVu Sans", "font.size": 6.2, "axes.titlesize": 6.6, "axes.labelsize": 6.2,
    "xtick.labelsize": 5.6, "ytick.labelsize": 5.6, "axes.edgecolor": INK2, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.fontsize": 5.6, "legend.frameon": False,
    "svg.hashsalt": "lca", "path.simplify": True,
}


def clean(ax, grid_axis="x") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def title(ax, text: str) -> None:
    ax.set_title(text, loc="left", weight="bold")


# ---------------------------------------------------------------------- Fig. S1
PATH_ORDER = {"A": 0, "B": 1, "C": 2}
SEGMENTS = [("m^U", ORANGE, "UNSAFE"), ("m^N", AQUA, "NONDETERMINED"),
            ("m^V_obs", LIGHT, "UNVERIFIED: tie or single observed label"), ("m^V_defect", MUTED, "UNVERIFIED: query defect")]


def data_figS1(out: Outputs) -> dict[str, Any]:
    vr = out.csv("verdicts/verdicts.csv")
    spec = {r["view_id"]: r for r in out.csv("registry/certificate_spec.csv")}
    views = sorted({r["view"] for r in vr})
    if set(views) != set(spec):
        raise PaperError("verdict views and specification views differ")

    def key(v):
        name, task = view_parts(v)
        s = spec[v]
        carrier = "facet::" if task == "facet" else "family::"
        b = s["basis"]
        border = 0 if b == carrier else (1 if b == "facet::" else (2 if b == "core::" else 3))
        return (0 if task == "family" else 1, PATH_ORDER[s["certificate_path"]], border, name)

    views.sort(key=key)
    lines = []
    last_task = None
    for v in views:
        name, task = view_parts(v)
        if task != last_task:
            lines.append(("header", f"{task} task", None, None))
            last_task = task
        lines.append(("view", name, spec[v]["certificate_path"], v))
    masses = {}
    rows = []
    for panel, ds in (("a", "DF"), ("b", "FP")):
        for kind, _, _, v in lines:
            if kind != "view":
                continue
            r = one(vr, view=v, dataset=ds)
            ws = [float(dec(r[col])) for col, _, _ in SEGMENTS]
            if abs(sum(ws) - 1.0) > 1e-9:
                raise PaperError(f"masses of {v} {ds} sum to {sum(ws)}")
            masses[(v, ds)] = (ws, dec(r["m^U"]))
            rows += [(panel, f"{ds} {v}", col, w) for (col, _, _), w in zip(SEGMENTS, ws)]
    return {"lines": lines, "masses": masses, "rows": rows}


def draw_figS1(d: dict[str, Any], path: Path) -> None:
    plt = pyplot()
    from matplotlib.patches import Patch
    lines, masses = d["lines"], d["masses"]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45), sharey=True, gridspec_kw={"wspace": 0.13})
        n = len(lines)
        for ax, ds in zip(axes, ("DF", "FP")):
            for i, (kind, _, _, v) in enumerate(lines):
                y = n - 1 - i
                if kind == "header":
                    continue
                ws, mu = masses[(v, ds)]
                left = 0.0
                for (_, color, _), w in zip(SEGMENTS, ws):
                    if w > 0:
                        ax.barh(y, w, left=left, height=0.72, color=color, edgecolor="white", lw=0.4)
                    left += w
                txt = frac(mu, 3) if mu >= dec("0.0005") or mu == 0 else "<0.001"
                ax.text(1.015, y, txt, va="center", ha="left", fontsize=5.2, color=INK)
            ax.set_xlim(0, 1.0)
            ax.set_ylim(-0.6, n - 0.4)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
            ax.set_xlabel(f"share of {ds} anchors")
            clean(ax, grid_axis=None)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            title(ax, f"({'a' if ds == 'DF' else 'b'}) {ds}")
            ax.text(1.015, n - 0.2, "$m^{\\mathrm{U}}$", fontsize=5.4, color=INK2, ha="left", va="center")
        labels, ticks = [], []
        for i, (kind, label, path_, _) in enumerate(lines):
            y = n - 1 - i
            ticks.append(y)
            labels.append(label if kind == "view" else "")
            if kind == "header":
                axes[0].text(-0.02, y, label, transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                             fontsize=6.0, weight="bold", color=INK)
            else:
                axes[0].text(-0.01, y, path_, transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                             fontsize=5.4, color=INK2)
        axes[0].set_yticks(ticks)
        axes[0].set_yticklabels(labels)
        axes[0].tick_params(axis="y", pad=11)
        handles = [Patch(fc=c, ec="none", label=lab) for _, c, lab in SEGMENTS]
        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.035))
        save(fig, path)
        plt.close(fig)


# ---------------------------------------------------------------------- Fig. S2
PROBE_STYLE = {"SrcMaj": (MUTED, "o"), "cosine": (ORANGE, "^")}  # the two probes the paper reports


def data_figS2(out: Outputs) -> dict[str, Any]:
    pb = out.csv("utility/probe_bound_check.csv")
    if any(r["violation"] != "0" for r in pb):
        raise PaperError("a probe score exceeds its bound")
    points = {p: [(float(r["C2_upper_bound"]), float(r["macro_f1"])) for r in pb if r["probe"] == p]
              for p in PROBE_STYLE}
    rows = [("a", p, x, y) for p, pts in points.items() for x, y in pts]
    return {"points": points, "n_points": sum(len(v) for v in points.values()), "rows": rows}


def draw_figS2(d: dict[str, Any], path: Path) -> None:
    plt = pyplot()
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.3, 2.1))
        for p, (col, mk) in PROBE_STYLE.items():
            pts = d["points"][p]
            ax.scatter([x for x, _ in pts], [y for _, y in pts], s=6, color=col, marker=mk, lw=0,
                       label=f"{p} ({len(pts)})", zorder=3)
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls="--", zorder=2)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        unit_ticks(ax.xaxis)
        unit_ticks(ax.yaxis)
        ax.set_xlabel("F1 bound $\\beta$")
        ax.set_ylabel("Macro-F1")
        clean(ax, grid_axis="both")
        ax.legend(loc="upper left", handletextpad=0.1, borderaxespad=0.2)
        save(fig, path)
        plt.close(fig)


# ---------------------------------------------------------------------- Fig. S3
def data_figS3(out: Outputs) -> dict[str, Any]:
    sm = out.csv("scaling/scaling_summary.csv")
    views = sorted({r["view_id"] for r in sm})
    lines = {}
    rows = []
    for v in views:
        for ds in ("DF", "FP"):
            rs = sorted(select(sm, view_id=v, dataset=ds), key=lambda r: float(r["fraction"]))
            xs = [ival(r["sampled_dataset_rows"]) / 1000 for r in rs]
            fib = [ival(r["single_label_fibers"]) for r in rs]
            ms = [float(r["milp_seconds_median"]) * 1000 for r in rs]
            lines[(v, ds)] = (xs, fib, ms)
            rows += [("a", f"{v} {ds}", x, y) for x, y in zip(xs, fib)]
            rows += [("b", f"{v} {ds}", x, y) for x, y in zip(xs, ms)]
    return {"views": views, "lines": lines, "rows": rows}


def draw_figS3(d: dict[str, Any], path: Path) -> None:
    plt = pyplot()
    from matplotlib.lines import Line2D
    views, lines = d["views"], d["lines"]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.55), gridspec_kw={"wspace": 0.45})
        colors = [BLUE, ORANGE, AQUA, "#8e5bb5"]
        for v, col in zip(views, colors):
            for ds, ls in (("DF", "-"), ("FP", "--")):
                xs, fib, ms = lines[(v, ds)]
                axes[0].plot(xs, fib, color=col, ls=ls, lw=0.8, marker="o", ms=1.6)
                axes[1].plot(xs, ms, color=col, ls=ls, lw=0.8, marker="o", ms=1.6)
        for ax, lab in zip(axes, ("single-label fibers", "integer programs (ms)")):
            ax.set_xlabel("rows sampled (thousands)")
            ax.set_ylabel(lab)
            clean(ax, grid_axis="y")
        title(axes[0], "(a) Single-label fibers")
        title(axes[1], "(b) Program time")
        handles = [Line2D([], [], color=c, lw=0.9, label=f"{view_parts(v)[0]} ({view_parts(v)[1]})")
                   for v, c in zip(views, colors)]
        handles += [Line2D([], [], color=INK2, lw=0.9, ls="-", label="DF"),
                    Line2D([], [], color=INK2, lw=0.9, ls="--", label="FP")]
        fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
        save(fig, path)
        plt.close(fig)


FIGURES = (("figS1_verdicts", data_figS1, draw_figS1), ("figS2_bounds", data_figS2, draw_figS2),
           ("figS3_pathb", data_figS3, draw_figS3))
