# SPDX-License-Identifier: MIT
"""Tables of the extended audit details: one LaTeX macro per table (\\SuppTable<Name>), entries read from the
aggregate outputs. Per-view and per-probe grids stay in the outputs as CSV files."""
from __future__ import annotations

from .common import DATASETS, Outputs, PaperError, count, dec, family_code, frac, sci, view_parts


def tt(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&")


def vname(v: str) -> str:
    name, _ = view_parts(v)
    if name.endswith("::"):
        return "\\texttt{" + tt(name) + "}"
    return name.replace("{", "$\\{$").replace("}", "$\\}$")


def yes(v) -> str:
    if str(v) not in ("1", "True", "true", "PASS"):
        raise PaperError(f"unexpected check value {v}")
    return "yes"


def table(name, caption, label, colspec, header, rows, size="\\scriptsize", sep="3pt") -> str:
    out = ["\\newcommand{\\SuppTable" + name + "}{%", "\\begin{table}[htbp]", "\\centering",
           "\\caption{" + caption + "}", "\\label{" + label + "}", size, "\\setlength{\\tabcolsep}{" + sep + "}",
           "\\begin{tabular}{" + colspec + "}", "\\toprule", header + " \\\\", "\\midrule"]
    out += [r + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}}", ""]
    return "\n".join(out)


def build(out: Outputs) -> str:
    parts = []

    # ---------------------------------------------------------------- Path B certificates
    pbx = out.csv("verdicts/path_b_certificates.csv")
    rows = []
    for r in sorted(pbx, key=lambda r: (r["view_id"], r["dataset"], r["fiber_value"])):
        _, task = view_parts(r["view_id"])
        lab = family_code(r["certified_label"]) if task == "family" else tt(r["certified_label"].split("/", 1)[-1])
        meth = "enumeration" if r["verification_method"].startswith("exhaustive") else "argument"
        states = count(r["integer_states_checked"]) if r["integer_states_checked"].isdigit() else "unbounded"
        fv = r["fiber_value"].replace("[", "(").replace("]", ")")
        yes(r["pass"])
        rows.append(f"{vname(r['view_id'])} & {r['dataset']} & {fv} & {lab} & {meth} & {states} & {r['minimum_strict_margin']}")
    parts.append(table(
        "PathB", f"Exact verification of the {len(pbx)} Path B \\textsc{{unsafe}} certificates; every certificate "
        "passes. Value: type counts (attribute, category, part) for the type view of the family task, (attribute, part) for "
        "type$\\{$attr., part$\\}$, and (STR, SURF, TAX, attribute, category, part) for family-3+type of the facet task. "
        "Values: basis-block values of the fiber, each checked; an unbounded fiber is checked by the argument of "
        "Section~A.2. Margin: smallest gap between the certified label and the largest rival over the states of the fiber.",
        "tab:s-pathb", "@{}lllllrr@{}", "View & Set & Value & Label & Method & Values & Margin", rows, sep="2.5pt"))

    # ---------------------------------------------------------------- registry maps
    reg = out.csv("registry/facet_map.csv")
    sup = out.json("registry/dataset_support.json")["datasets"]
    diff = out.json("registry/populations.json")["facet_set_differences"]
    materialized = {r["facet"]: r["materialized_family"] for r in out.csv("registry/declared_vs_materialized.csv")
                    if r["record_type"] == "MAP" and r["mapping_equal"] == "0"}
    rows = []
    for r in sorted(reg, key=lambda r: (r["term_type"], r["facet"])):
        f = r["facet"]
        note = ("authority only" if f in diff["authority_only"] else
                "no active core term" if f in diff["materialized_not_core_reachable"] else "")
        ind = ["$\\bullet$" if f in sup[d]["S_members"] else "" for d in DATASETS]
        name = f.split("/", 1)[1] if "/" in f else f
        wcode = family_code(r["family"]) + ("$^\\ast$" if f in materialized else "")
        rows.append(f"{tt(name)} & {tt(r['term_type'])} & {wcode} & {ind[0]} & {ind[1]} & {note}")
    star = "; ".join(f"{tt(f.split('/', 1)[-1])} as {family_code(c)}" for f, c in sorted(materialized.items()))
    star = f" $^\\ast$The released family block materializes {star}." if materialized else ""
    parts.append(table(
        "Registry", "Registry maps. $T$ (facet to type) and $W$ (facet to family) on the "
        f"{len(reg)} authority facets, with membership in the dataset supports $S_i=\\Pi(K_i)$. $T(f)$ is the term "
        "type; \\texttt{type::} materializes the attribute, category and part types. The family \\texttt{other} lies "
        "outside the label set." + star,
        "tab:s-registry", "@{}>{\\raggedright\\arraybackslash}p{0.43\\columnwidth}lllc>{\\raggedright\\arraybackslash}p{0.13\\columnwidth}@{}",
        "Facet $f$ & $T(f)$ & $W(f)$ & $S_{\\mathrm{DF}}$ & $S_{\\mathrm{FP}}$ & Note", rows))

    parts.append(support_table(out))
    return "\n".join(parts)


SUPPORT_REGIMES = (("released", "Released"), ("union", "Union"), ("intersection", "Intersection"),
                   ("released_plus_color", "Released${}+{}$color"))
PATH_B_VIEWS = ("type3__family", "type_attr_part__family", "family3__facet", "family3_plus_type__facet")


def mass(x) -> str:
    """Mass display of Table IV: four decimals, or scientific notation for a positive mass below .00005."""
    v = dec(x)
    return sci(v) if 0 < v < dec("0.00005") else frac(v, 4)


def support_table(out: Outputs) -> str:
    """Table S3: Path B on the released supports and on the three counterfactual coordinate sets."""
    sv = out.csv("support/support_verdicts.csv")
    ce = out.csv("support/support_certificates.csv")
    counts = out.json("support/support.json")["counts"]
    spec = {r["view_id"]: r for r in out.csv("registry/certificate_spec.csv")}
    if sorted(v for v, s in spec.items() if s["certificate_path"] == "B") != sorted(PATH_B_VIEWS):
        raise PaperError("Table S3 expects the four Path B views")
    rows = []
    for v in PATH_B_VIEWS:
        _, task = view_parts(v)
        for ds in DATASETS:
            cells = []
            for reg, _ in SUPPORT_REGIMES:
                r = [x for x in sv if x["regime"] == reg and x["view"] == v and x["dataset"] == ds]
                if len(r) != 1:
                    raise PaperError(f"support verdicts: {reg} {v} {ds}")
                r = r[0]
                n_cert = sum(1 for x in ce if x["regime"] == reg and x["view_id"] == v and x["dataset"] == ds)
                if (n_cert > 0) != (dec(r["m^U"]) > 0):
                    raise PaperError(f"certificates and UNSAFE mass disagree: {reg} {v} {ds}")
                mv = dec(r["m^V_obs"]) + dec(r["m^V_defect"])
                cells += [str(n_cert), mass(r["m^U"]), mass(mv)]
            rows.append(f"{vname(v)} ({task}) & {ds} & " + " & ".join(cells))
    tot = []
    for reg, _ in SUPPORT_REGIMES:
        c = counts[reg]
        if (c["certificates_verified_exactly"] != c["path_b_certificates"]
                or c["open_fibers_verified_exactly"] != c["path_b_open_fibers"]
                or c["path_b_certificates"] != sum(1 for x in ce if x["regime"] == reg)):
            raise PaperError(f"support counts are inconsistent for {reg}")
        tot.append(f"\\multicolumn{{3}}{{c}}{{{c['path_b_certificates']} / {count(c['path_b_open_fibers'])}}}")
    head = ("View & Set & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in SUPPORT_REGIMES)
            + " \\\\\n" + "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(4)) + "\n"
            + " & & " + " & ".join(["Cert. & $m^{\\mathrm{U}}$ & $m^{\\mathrm{V}}$"] * 4))
    caption = ("Path~B on the released supports and on the three counterfactual coordinate sets of Section~E: "
               "certificates (Cert.), \\textsc{unsafe} mass $m^{\\mathrm{U}}$ and \\textsc{unverified} mass "
               "$m^{\\mathrm{V}}$ per view and dataset. Last row: all Path~B certificates / single-label fibers that "
               "stay open, each checked exactly. The registry maps, \\textsc{full-conform}, the evidence, the labels, "
               "the view projections, the paths and the solver are fixed; \\textsc{query-conform} is recomputed on each set.")
    out_ = ["\\newcommand{\\SuppTableSupport}{%", "\\begin{table*}[!t]", "\\centering", "\\caption{" + caption + "}",
            "\\label{tab:s-support}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{@{}ll" + "rrr" * 4 + "@{}}", "\\toprule", head + " \\\\", "\\midrule"]
    out_ += [r + " \\\\" for r in rows]
    out_ += ["\\midrule", "All Path~B views & & " + " & ".join(tot) + " \\\\", "\\bottomrule", "\\end{tabular}",
             "\\end{table*}}", ""]
    return "\n".join(out_)
