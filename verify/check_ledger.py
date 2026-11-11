# SPDX-License-Identifier: MIT
"""Recompute every number cell, claim, extended table and figure data file from the aggregate outputs and the
source-release record (file hashes, source and build counts), and compare them with the paper folder, with the
SHA-256 of every file the ledger names and with the result line that README.md states; exit 1 on a mismatch."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca.paper import claims, figures, numbers, supp_figures, supp_tables  # noqa: E402
from lca.paper.common import RESTRICTED, Outputs, resolve, sha256_file  # noqa: E402

MAX_REPORTED = 20


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--paper", type=Path, default=REPO / "paper")
    a = ap.parse_args()
    problems: list[str] = []

    def bad(msg: str) -> None:
        problems.append(msg)

    # number cells: recompute, then compare with the ledger and with numbers.tex
    out = Outputs(a.outputs)
    cells = numbers.build(out)
    ledger = numbers.read_ledger(a.paper / "number_ledger.csv")
    tex = numbers.parse_numbers_tex((a.paper / "numbers.tex").read_text(encoding="utf-8"))
    if [c.cell_id for c in cells] != [r["cell_id"] for r in ledger]:
        bad("the cell ids of number_ledger.csv differ from the recomputed cells")
    if list(tex) != [r["cell_id"] for r in ledger]:
        bad("numbers.tex and number_ledger.csv list different cells")
    by_id = {r["cell_id"]: r for r in ledger}
    for c in cells:
        r = by_id.get(c.cell_id)
        if r is None:
            continue
        for field, value in (("display", c.display), ("raw_value", c.raw), ("output", ";".join(c.outputs)),
                             ("selector", c.selector)):
            if r[field] != value:
                bad(f"cell {c.cell_id}: {field} {r[field]!r} in the ledger, recomputed {value!r}")
        if tex.get(c.cell_id) != c.display:
            bad(f"cell {c.cell_id}: numbers.tex shows {tex.get(c.cell_id)!r}, recomputed {c.display!r}")

    # every output named by the ledger: aggregate only, and the recorded SHA-256
    recorded: dict[str, str] = {}
    for r in ledger:
        for name, digest in zip(r["output"].split(";"), r["output_sha256"].split(";")):
            if recorded.setdefault(name, digest) != digest:
                bad(f"the ledger records two SHA-256 values for {name}")
    for name, digest in sorted(recorded.items()):
        if name.startswith("outputs/") and name[len("outputs/"):] in RESTRICTED:
            bad(f"the ledger names the per-anchor output {name}")
            continue
        path = resolve(name, a.outputs)
        if not path.is_file():
            bad(f"missing {name}")
        elif sha256_file(path) != digest:
            bad(f"SHA-256 of {name} differs from the ledger")

    # claims
    rows = claims.evaluate(out, cells)
    shipped = claims.read_claims(a.paper / "claims.csv")
    if rows != shipped:
        bad("claims.csv differs from the claims recomputed from the outputs")
    bad_status = [r["claim_id"] for r in rows if r["status"] != "PASS"]
    if bad_status:
        bad(f"claims that do not hold: {', '.join(bad_status)}")

    # extended tables and the plotted values of every figure
    if supp_tables.build(out) != (a.paper / "tables" / "supp_tables.tex").read_text(encoding="utf-8"):
        bad("tables/supp_tables.tex differs from the tables recomputed from the outputs")
    n_fig = 0
    for name, data, _ in figures.FIGURES + supp_figures.FIGURES:
        n_fig += 1
        csv_path = a.paper / "figures" / f"{name}.csv"
        if not csv_path.is_file() or csv_path.read_text(encoding="utf-8") != figures.rows_csv(data(out)["rows"]):
            bad(f"figures/{name}.csv differs from the values recomputed from the outputs")
        if not (a.paper / "figures" / f"{name}.pdf").is_file():
            bad(f"figures/{name}.pdf is missing")

    n_files = len(recorded)
    line = (f"check_ledger: PASS: {len(cells)} cells, {n_files} ledger files, {len(rows)} claims, "
            f"1 table file, {n_fig} figure data files checked against {a.outputs.name}/ and the source-release record")
    if line not in (REPO / "README.md").read_text(encoding="utf-8"):
        bad("README.md states a different result line than this check prints")

    for p in problems[:MAX_REPORTED]:
        print("MISMATCH", p)
    if len(problems) > MAX_REPORTED:
        print(f"MISMATCH ... and {len(problems) - MAX_REPORTED} more")
    status = "PASS" if not problems else f"FAIL, {len(problems)} mismatch(es)"
    print(f"check_ledger: {status}: {len(cells)} cells, {n_files} ledger files, {len(rows)} claims, "
          f"1 table file, {n_fig} figure data files checked against {a.outputs.name}/ and the source-release record")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
