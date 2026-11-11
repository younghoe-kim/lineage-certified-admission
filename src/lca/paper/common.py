# SPDX-License-Identifier: MIT
"""Readers of the aggregate outputs and the display formats shared by the paper tools.

CSV files are read as strings so that every number keeps the exact text the step wrote; rounding for display is
done with decimal arithmetic on that text. Every file read is recorded with its SHA-256 under its repo-relative
name (outputs/<step>/<file>), which is what the number ledger names."""
from __future__ import annotations

import csv
import hashlib
import json
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import yaml

csv.field_size_limit(10 ** 9)

PACKAGE_DIR = Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parents[2]

# Per-anchor outputs and copies of external-benchmark content: the paper tools read aggregate outputs only.
RESTRICTED = frozenset({
    "conformance/full_conform.csv.gz", "conformance/query_masks.npz", "verdicts/anchor_verdicts.npz",
    "verdicts/witnesses.json", "verdicts/fig1_instance.json", "mutations/changed_rows.csv.gz",
    "external/taxonomy_rows.jsonl", "external/gold_schema_mappings.csv", "external/target_schema_attributes.csv",
    "external/source_metadata.csv", "external/labeled_set_headers.csv", "external/source_fields.csv",
    "external/source_inventory.csv", "external/unmapped_source_columns.csv", "verdicts/path_b_open_witnesses.csv",
})
# Facts about the source release that no step can recompute from the input view (see DATASHEET.md).
SOURCE_RELEASE = "src/lca/paper/source_release.json"


class PaperError(SystemExit):
    """A selector, a consistency check or a format rule failed."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Outputs:
    """Read-only access to an outputs folder; records the repo-relative name and SHA-256 of every file read."""

    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise PaperError(f"outputs folder not found: {self.root}")
        self.used: dict[str, str] = {}
        self._cache: dict[str, Any] = {}

    def path(self, rel: str) -> Path:
        if rel in RESTRICTED:
            raise PaperError(f"{rel} is a per-anchor output; the paper tools read aggregate outputs only")
        p = self.root / rel
        if not p.is_file():
            raise PaperError(f"missing output {rel}")
        self.used.setdefault("outputs/" + rel, sha256_file(p))
        return p

    def csv(self, rel: str) -> list[dict[str, str]]:
        key = "csv:" + rel
        if key not in self._cache:
            with self.path(rel).open(encoding="utf-8", newline="") as fh:
                self._cache[key] = list(csv.DictReader(fh))
        else:
            self.path(rel)
        return [dict(r) for r in self._cache[key]]

    def json(self, rel: str) -> Any:
        return json.loads(self.path(rel).read_text(encoding="utf-8"))

    def yaml(self, rel: str) -> Any:
        return yaml.safe_load(self.path(rel).read_text(encoding="utf-8"))

    def source_release(self) -> dict[str, Any]:
        p = REPO / SOURCE_RELEASE
        self.used.setdefault(SOURCE_RELEASE, sha256_file(p))
        return json.loads(p.read_text(encoding="utf-8"))


def resolve(name: str, outputs_dir: Path) -> Path:
    """File behind a ledger name: outputs/<rel> lives in the outputs folder, anything else in the repository."""
    if name.startswith("outputs/"):
        return Path(outputs_dir) / name[len("outputs/"):]
    return REPO / name


def select(rows: list[dict[str, str]], **conds: str) -> list[dict[str, str]]:
    """Rows whose string fields equal the given values."""
    return [r for r in rows if all(r.get(k) == v for k, v in conds.items())]


def one(rows: list[dict[str, str]], **conds: str) -> dict[str, str]:
    hit = select(rows, **conds)
    if len(hit) != 1:
        raise PaperError(f"selector {conds} matched {len(hit)} rows")
    return hit[0]


# ------------------------------------------------------------------ display formats
def dec(x: Any) -> Decimal:
    return Decimal(str(x).strip())


def ival(x: Any) -> int:
    """Integer value of a count written as '12' or '12.0'."""
    d = dec(x)
    if d != d.to_integral_value():
        raise PaperError(f"not an integer count: {x!r}")
    return int(d)


def frac(x: Any, digits: int) -> str:
    """Value in [0, 1] shown with `digits` decimals and a leading zero (0.6263)."""
    q = Decimal(1).scaleb(-digits)
    v = dec(x).quantize(q, rounding=ROUND_HALF_UP)
    if v == 1:
        return "1." + "0" * digits if digits else "1"
    return f"{v:.{digits}f}"


def floor_frac(x: Any, digits: int) -> str:
    """Value in [0, 1) rounded down to `digits` decimals, for bounds stated as "at least" or "below" (0.148)."""
    q = Decimal(1).scaleb(-digits)
    v = dec(x).quantize(q, rounding=ROUND_FLOOR)
    if not Decimal(0) <= v < 1:
        raise PaperError(f"floor display expects a value in [0, 1): {x!r}")
    return f"{v:.{digits}f}"


def signed(x: Any, digits: int) -> str:
    q = Decimal(1).scaleb(-digits)
    v = dec(x).quantize(q, rounding=ROUND_HALF_UP)
    s = f"{abs(v):.{digits}f}"
    return ("$-$" if v < 0 else "$+$") + s


def pct(x: Any, digits: int = 1) -> str:
    q = Decimal(1).scaleb(-digits)
    v = (dec(x) * 100).quantize(q, rounding=ROUND_HALF_UP)
    return f"{v:.{digits}f}\\%"


def count(x: Any) -> str:
    return f"{ival(x):,}"


def secs(x: Any, digits: int = 1) -> str:
    q = Decimal(1).scaleb(-digits)
    v = dec(x).quantize(q, rounding=ROUND_HALF_UP)
    return f"{v:.{digits}f}"


def ratio(a: Any, b: Any, digits: int = 2) -> str:
    q = Decimal(1).scaleb(-digits)
    v = (dec(a) / dec(b)).quantize(q, rounding=ROUND_HALF_UP)
    return f"{v:.{digits}f}"


def sci(x: Any) -> str:
    """Small positive value in scientific notation with two significant digits ($2.9{\\cdot}10^{-6}$)."""
    mant, exp = f"{dec(x):.1e}".split("e")
    return f"${mant}{{\\cdot}}10^{{{int(exp)}}}$"


def plain(display: str) -> str:
    """A display string as plain text (for claim statements and figure labels)."""
    return (display.replace("\\%", "%").replace("$-$", "-").replace("$+$", "+").replace("{\\cdot}", "x")
            .replace("$", "").replace("10^{", "10^").replace("}", ""))


# ------------------------------------------------------------------ names of the paper
FAMILY_CODE = {"taxonomic_category": "TAX", "structural_expression": "STR", "surface_expression": "SURF",
               "attribute_expression": "ATTR", "tag_terms": "TAG", "other_expression": "other"}
VIEW_BASE = {"identity_admitted": "Identity-admitted", "core_plus_type": "Core+type", "facet_only": "Facet only",
             "type3": "Type", "type_attr_part": "Type{attr., part}", "family3": "Family-3",
             "family3_plus_type": "Family-3+type", "l0_facet": "Direct carrier", "l0_family": "Direct carrier",
             "inst_family": "inst_family::", "inst_sem": "inst_sem::", "rep_core": "rep_core::"}
DATASETS = ("DF", "FP")
DIRECTIONS = {"dffp": "DF->FP", "fpdf": "FP->DF"}


def family_code(f: str) -> str:
    """Family code of the paper; the family outside the label set prints as 'other'."""
    if f not in FAMILY_CODE:
        raise PaperError(f"unknown family identifier {f!r}")
    return FAMILY_CODE[f]


def combo_blocks(cell: str) -> str:
    """'F1M0T1' -> 'core+facet+type' (every block combination contains the core)."""
    names = ["core"] + [n for flag, n in zip(cell[1::2], ("facet", "family", "type")) if flag == "1"]
    return "+".join(names)


def view_parts(v: str) -> tuple[str, str]:
    """(display name, task) of a registered view id."""
    base, task = v.rsplit("__", 1)
    if task not in ("facet", "family"):
        raise PaperError(f"unknown task in view id {v}")
    name = combo_blocks(base.split("_", 1)[1]) if base.startswith("factorial_") else VIEW_BASE[base]
    return name, task
