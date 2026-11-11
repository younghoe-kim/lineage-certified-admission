# SPDX-License-Identifier: MIT
"""Paper step: number cells and ledger, claim list, extended tables and figures, from the aggregate outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca import io as lio  # noqa: E402
from lca.paper import claims, figures, numbers, supp_figures, supp_tables  # noqa: E402
from lca.paper.common import Outputs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--paper", type=Path, default=REPO / "paper")
    ap.add_argument("--claims-md", type=Path, default=None, help="also write the claim map (CLAIMS.md) here")
    a = ap.parse_args()
    out = Outputs(a.outputs)

    cells = numbers.build(out)
    claim_rows = claims.evaluate(out, cells)
    failed = [r["claim_id"] for r in claim_rows if r["status"] != "PASS"]
    if failed:
        raise SystemExit(f"claims do not hold on these outputs: {', '.join(failed)}")
    tables = supp_tables.build(out)
    figure_data = [(name, data(out), draw) for name, data, draw in figures.FIGURES + supp_figures.FIGURES]

    paper = a.paper
    lio.write_text(paper / "numbers.tex", numbers.numbers_tex(cells))
    lio.write_text(paper / "number_ledger.csv", numbers.ledger_csv(numbers.ledger_rows(cells, out.used)))
    lio.write_text(paper / "claims.csv", claims.claims_csv(claim_rows))
    lio.write_text(paper / "tables" / "supp_tables.tex", tables)
    fig_dir = paper / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, d, draw in figure_data:
        lio.write_text(fig_dir / f"{name}.csv", figures.rows_csv(d["rows"]))
        draw(d, fig_dir / f"{name}.pdf")
    if a.claims_md is not None:
        lio.write_text(a.claims_md, claims.claims_md(claim_rows, cells))
    print(f"paper: {len(cells)} cells from {len(out.used)} files, {len(claim_rows)} claims, "
          f"{len(figure_data)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
