# SPDX-License-Identifier: MIT
"""External study on MaDI-Bench: source freeze, protocol authority, frozen protocol, audit and independent check."""
from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import math
import os
import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from lca import io as lio

STEP = "external"
SCHEMA_VERSION = "1"

# ---------------------------------------------------------------------------
# Pinned benchmark source
# ---------------------------------------------------------------------------
BENCHMARK = "MaDI-Bench"
OFFICIAL_REPOSITORY = "https://github.com/wbsg-uni-mannheim/MaDI-Bench"
OFFICIAL_REMOTE = OFFICIAL_REPOSITORY + ".git"
BRANCH = "main"
PINNED_COMMIT = "062058d09c8abd97026e6d4e9908b433753f58d3"
PINNED_TREE = "05f6a54f48eafb9d1f08d249eea82e66a77d161d"
PINNED_COMMIT_TIME = "2026-10-09T11:31:32+02:00"
TIER = "base"

# The audited domains in their declared order. This order numbers the queries, orders the
# domain tables and orders the list of qualifying domains.
DOMAINS = ("games", "companies", "music", "products", "papers")
TAXONOMY_DOMAINS = ("games", "companies", "music")
NO_TRANSFORM_DOMAINS = ("products", "papers")

# SHA-256 of the canonical JSON list [[relative_path, size_bytes, sha256], ...] (paths in code-point
# order) of every file under use cases/<domain>/base/input at the pinned commit.
PINNED_DOMAIN_INPUT_HASHES = {
    "games": "d82201f478a3f2f43224ef949934ade390740725b8f25b78c90a8b049161f5c9",
    "companies": "efd374b29669c06d075e4e1ddcea38d629b3db3678cddf24716393b0c9636d06",
    "music": "ee9a7b9f578949435ac20d63244bd17f206acdb72f3d0a883a3e19bdc7161955",
    "products": "ddefb8e307f091ed4853e630fc60974f773414cface8f1586f7b0ae9cad61aaf",
    "papers": "4b694015fb15b5c49b0c0e36914f9ea88a21c64d6a2f16d3ede2f7772e4d9f09",
}
PINNED_INVENTORY_FILES = 133
PINNED_INVENTORY_BYTES = 130653392
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"   # first bytes of a Git-LFS pointer file
LICENSE_PATTERNS = ("LICENSE*", "LICENCE*", "COPYING*", "NOTICE*")

# Structural cardinalities of the pinned authorities. They guard the definitions; they are not
# result targets.
AUTHORITY_COUNTS = {
    "target_schema_attributes": 66,
    "positive_gold_schema_mappings": 202,
    "unmapped_source_columns": 4,
    "declared_taxonomy_files": 6,
    "transform_authority_candidates": 6,
    "structurally_complete_transform_authorities": 6,
    "source_files_field_names_only": 16,
}
PROTOCOL_COUNTS = {"queries": 6, "source_mappings": 14, "exact_carriers": 17, "noncarriers": 14}

# Qualification rule of the external study (protocol constants).
MIN_L1_ROWS = 100
MIN_L1_RATE = 0.01
MIN_DISTINCT_DOMAINS = 2
WITNESS_LIMIT_PER_SOURCE = 20
CSV_FIELD_LIMIT = 16 * 1024 * 1024

CLASSES = ("L0_DIRECT", "L1_EXACT_DERIVED", "UNVERIFIED_AMBIGUOUS", "UNVERIFIED_NO_MATCH")

# Output file names (under <outputs>/external/).
OUT = {
    "inventory": "source_inventory.csv",
    "domain_summary": "domain_summary.csv",
    "freeze": "source_freeze.json",
    "target_attributes": "target_schema_attributes.csv",
    "gold_mappings": "gold_schema_mappings.csv",
    "unmapped": "unmapped_source_columns.csv",
    "taxonomy_summary": "taxonomy_summary.csv",
    "taxonomy_rows": "taxonomy_rows.jsonl",
    "source_fields": "source_fields.csv",
    "source_metadata": "source_metadata.csv",
    "labeled_headers": "labeled_set_headers.csv",
    "transform_authorities": "transform_authorities.csv",
    "protocol": "protocol.json",
    "query_registry": "query_registry.csv",
    "carrier_registry": "carrier_registry.csv",
    "noncarrier_registry": "noncarrier_registry.csv",
    "source_scope": "source_scope.csv",
    "domain_scope": "domain_scope.csv",
    "source_results": "source_results.csv",
    "query_results": "query_results.csv",
    "domain_results": "domain_results.csv",
    "witnesses": "hashed_witnesses.csv",
    "checks": "independent_checks.csv",
    "verification": "independent_verification.json",
    "gold_check": "gold_check.csv",
    "gold_protocol": "gold_check.json",
}

FIELDS = {
    "inventory": ["domain", "tier", "relative_path", "size_bytes", "sha256", "lfs_pointer"],
    "domain_summary": [
        "domain", "tier", "task_path", "input_file_count", "input_bytes", "source_table_count",
        "source_metadata_count", "taxonomy_file_count", "normalization_csv_count",
        "entitymatching_test_count", "fusion_file_count", "gold_version_present", "target_schema_present",
        "sm_mapping_gold_present", "target_attribute_count", "schema_mapping_gold_count",
        "domain_input_hash", "lfs_pointer_count", "missing_required",
    ],
    "target_attributes": [
        "domain", "target_attribute", "required", "type_json", "format", "pattern", "minimum", "maximum",
        "minLength", "maxLength", "minItems", "maxItems", "taxonomy_reference", "taxonomy_column",
        "taxonomy_exhaustive", "taxonomy_alias_count", "taxonomy_aliases_json", "property_spec_sha256",
    ],
    "gold_mappings": [
        "domain", "mapping_index", "source_dataset", "source_column", "target_dataset", "target_column",
        "label", "score",
    ],
    "unmapped": ["domain", "unmapped_index", "source_dataset", "source_column", "target_column", "label", "reason"],
    "taxonomy_summary": [
        "domain", "taxonomy_path", "taxonomy_sha256", "row_count", "column_count", "columns_json",
        "unique_counts_json", "duplicate_full_rows", "referenced_by_json",
    ],
    "source_fields": [
        "domain", "source_path", "format", "field_discovery_method", "field_count", "fields_json",
        "source_sha256",
    ],
    "source_metadata": ["domain", "metadata_path", "json_pointer", "value_json"],
    "labeled_headers": ["domain", "task", "relative_path", "format", "header_only_json", "content_rows_read", "sha256"],
    "transform_authorities": [
        "domain", "target_attribute", "authority_type", "taxonomy_reference", "taxonomy_column",
        "taxonomy_column_present", "taxonomy_exhaustive", "taxonomy_alias_count", "taxonomy_row_count",
        "positive_gold_source_mapping_count", "positive_gold_source_mappings_json",
        "structural_authority_complete", "NOTE",
    ],
    "query_registry": [
        "query_id", "domain", "target_attribute", "target_type_json", "required", "taxonomy_path",
        "taxonomy_sha256", "protected_taxonomy_column", "protected_unique_values",
        "protected_value_set_sha256", "taxonomy_exhaustive", "target_schema_alias_count",
        "target_schema_aliases_json", "normalized_alias_map_sha256", "positive_gold_source_mapping_count",
        "exact_carrier_count", "rejected_noncarrier_count", "protected_query",
    ],
    "carrier_registry": [
        "query_id", "carrier_id", "domain", "target_attribute", "taxonomy_path", "protected_taxonomy_column",
        "carrier_taxonomy_column", "taxonomy_rows", "carrier_nonblank_rows", "carrier_unique_values",
        "protected_unique_values", "fd_conflict_keys", "exact_determinacy", "carrier_map_sha256",
        "selection_rule",
    ],
    "noncarrier_registry": [
        "query_id", "domain", "target_attribute", "taxonomy_path", "protected_taxonomy_column",
        "candidate_taxonomy_column", "taxonomy_rows", "carrier_nonblank_rows", "carrier_unique_values",
        "fd_conflict_keys", "exact_determinacy", "reason",
    ],
    "source_scope": [
        "query_id", "domain", "target_attribute", "source_dataset", "source_column", "gold_label",
        "gold_score", "scope_role",
    ],
    "domain_scope": [
        "domain", "declared", "base_tier_only", "taxonomy_backed_query_count", "handling",
        "interpretation_if_no_query",
    ],
    "source_results": [
        "query_id", "domain", "source_dataset", "source_relative_path", "source_sha256", "source_column",
        "source_rows", "empty_rows", "nonempty_rows", "L0_direct_rows", "L1_exact_derived_rows",
        "unverified_ambiguous_rows", "unverified_no_match_rows", "L0_direct_rate_nonempty",
        "L1_exact_derived_rate_nonempty", "nonempty_accounting_ok",
    ],
    "query_results": [
        "query_id", "domain", "target_attribute", "source_columns", "source_rows_sum", "empty_rows",
        "nonempty_rows", "L0_direct_rows", "L1_exact_derived_rows", "unverified_ambiguous_rows",
        "unverified_no_match_rows", "L0_direct_rate_nonempty", "L1_exact_derived_rate_nonempty",
        "nontrivial_L1_min_rows", "nontrivial_L1_min_rate", "nontrivial_L1_query",
    ],
    "domain_results": [
        "domain", "registered_queries", "executed_queries", "qualifying_nontrivial_L1_queries",
        "domain_qualifies", "L0_direct_rows", "L1_exact_derived_rows", "nonempty_rows",
        "handling", "interpretation",
    ],
    "witnesses": [
        "query_id", "domain", "source_dataset", "source_column", "row_locator_sha256", "token_sha256",
        "derived_target_sha256", "carrier_ids", "class",
    ],
    "checks": ["scope", "key", "primary_L1", "independent_L1", "exact_match", "note"],
    "gold_check": [
        "query_id", "domain", "target_attribute", "gold_cells", "L0_direct", "L1_exact_derived",
        "unverified_ambiguous", "unverified_no_match", "empty", "classified", "consistent", "contradictions",
        "complete_set_equal", "component_only", "unverified_gold_normalization", "independent_match",
    ],
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def content_hash(obj: Any) -> str:
    """SHA-256 of canonical UTF-8 JSON (sorted keys, compact separators, non-ASCII kept, trailing newline)."""
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_remote(url: str) -> str:
    """Comparable form of a Git remote URL (https form, no trailing slash or .git, case-folded)."""
    s = url.strip().rstrip("/")
    if s.endswith(".git"):
        s = s[:-4]
    if s.startswith("git@github.com:"):
        s = "https://github.com/" + s[len("git@github.com:"):]
    return s.casefold()


def _intish(x: Any) -> int:
    return int(float(str(x)))


def _boolish(x: Any) -> bool:
    s = str(x).strip().casefold()
    if s in {"1", "true", "yes"}:
        return True
    if s in {"0", "false", "no", ""}:
        return False
    raise SystemExit(f"not a boolean value: {x!r}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """One canonical JSON object per line, in the given order (non-ASCII kept)."""
    lio.write_text(path, "\n".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for r in rows))


def _scope_prefix(domain: str) -> str:
    return f"use cases/{domain}/{TIER}/input/"


def _in_scope(rel: str) -> bool:
    return any(rel.startswith(_scope_prefix(d)) for d in DOMAINS)


def _direct_children(files: list[str], folder: str, pattern: str = "*") -> list[str]:
    """Files directly inside folder (a relative path ending in '/') whose name matches pattern."""
    out = []
    for rel in files:
        if rel.startswith(folder):
            name = rel[len(folder):]
            if "/" not in name and fnmatch.fnmatchcase(name, pattern):
                out.append(rel)
    return sorted(out)


# ---------------------------------------------------------------------------
# Git access (never contacts a remote)
# ---------------------------------------------------------------------------

def _git(git_dir: Path, *args: str) -> str:
    env = dict(os.environ)
    env.update(GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
    proc = subprocess.run(["git", "-C", str(git_dir), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=600, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return proc.stdout.decode("utf-8")


def read_git_metadata(git_dir: Path) -> dict[str, str]:
    """Origin, HEAD commit, tree and commit time; stops unless they are the pinned official ones."""
    meta = {
        "remote": _git(git_dir, "remote", "get-url", "origin").strip(),
        "commit": _git(git_dir, "rev-parse", "HEAD").strip(),
        "tree": _git(git_dir, "rev-parse", "HEAD^{tree}").strip(),
        "commit_time": _git(git_dir, "show", "-s", "--format=%cI", "HEAD").strip(),
    }
    if normalize_remote(meta["remote"]) != normalize_remote(OFFICIAL_REMOTE):
        raise SystemExit(f"origin is not the official {BENCHMARK} repository: {meta['remote']}")
    if meta["commit"] != PINNED_COMMIT:
        raise SystemExit(f"checkout is at commit {meta['commit']}, expected {PINNED_COMMIT}")
    if meta["tree"] != PINNED_TREE:
        raise SystemExit(f"checkout tree is {meta['tree']}, expected {PINNED_TREE}")
    if meta["commit_time"] != PINNED_COMMIT_TIME:
        raise SystemExit(f"commit time {meta['commit_time']} differs from {PINNED_COMMIT_TIME}")
    return meta


def pinned_tree_blobs(git_dir: Path) -> dict[str, str]:
    """Path -> git blob id for every file of the pinned commit (read from tree objects only)."""
    blobs = {}
    for entry in _git(git_dir, "ls-tree", "-r", "-z", "--full-tree", PINNED_COMMIT).split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        _mode, kind, oid = meta.split()
        if kind == "blob":
            blobs[path] = oid
    return blobs


def hash_file(path: Path) -> dict[str, Any]:
    """Size, SHA-256, git blob id and Git-LFS pointer flag of one file (streamed)."""
    size = path.stat().st_size
    sha = hashlib.sha256()
    blob = hashlib.sha1(b"blob %d\0" % size)
    head = None
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            if head is None:
                head = chunk[:len(LFS_MAGIC)]
            sha.update(chunk)
            blob.update(chunk)
    return {"size_bytes": size, "sha256": sha.hexdigest(), "blob_id": blob.hexdigest(),
            "lfs_pointer": int(head == LFS_MAGIC)}


def domain_input_hash(rows: list[tuple[str, int, str]]) -> str:
    return content_hash([[rel, int(size), sha] for rel, size, sha in sorted(rows)])


class BenchmarkSource:
    """Verified view of the pinned benchmark files in a local checkout.

    With a Git directory, the file listing is the pinned commit's tree and every present file must
    have the blob id recorded there. Without one, the checkout must be complete and reproduce the
    pinned per-domain input hashes. Files are opened only through path(), which re-checks SHA-256."""

    def __init__(self, root: Path, git_dir: Path | None, allow_partial: bool = False):
        self.root = Path(root).resolve()
        self.git_dir = None if git_dir is None else Path(git_dir).resolve()
        for d in DOMAINS:
            if not (self.root / _scope_prefix(d)).is_dir():
                raise SystemExit(f"missing task input folder of a declared domain: {_scope_prefix(d)}")
        walked = sorted(p.relative_to(self.root).as_posix()
                        for d in DOMAINS for p in (self.root / _scope_prefix(d)).rglob("*") if p.is_file())
        if self.git_dir is not None:
            self.git = read_git_metadata(self.git_dir)
            blobs = pinned_tree_blobs(self.git_dir)
            self.listing = sorted(p for p in blobs if _in_scope(p))
            self.root_entries = sorted(p for p in blobs if "/" not in p)
            extra = sorted(set(walked) - set(self.listing))
            if extra:
                raise SystemExit(f"files not in the pinned commit inside the audited scope: {extra[:10]}")
            self.verification = "pinned_tree_blob_ids"
        else:
            if allow_partial:
                raise SystemExit("an incomplete checkout can only be verified against a Git directory (--git-dir)")
            self.git = {}
            blobs = None
            self.listing = walked
            self.root_entries = sorted(p.name for p in self.root.iterdir() if p.is_file())
            self.verification = "pinned_domain_input_hashes"
        self.records: dict[str, dict[str, Any]] = {}
        self.missing: list[str] = []
        mismatched = []
        for rel in self.listing:
            p = self.root / rel
            if not p.is_file():
                self.missing.append(rel)
                continue
            rec = hash_file(p)
            if blobs is not None and rec["blob_id"] != blobs[rel]:
                mismatched.append(rel)
                continue
            self.records[rel] = rec
        if mismatched:
            raise SystemExit(f"files differ from the pinned commit: {mismatched[:10]}")
        if self.missing and not allow_partial:
            raise SystemExit(f"{len(self.missing)} pinned files are missing from the checkout, e.g. {self.missing[:5]}")
        for d in DOMAINS:
            if self.domain_complete(d) and self.domain_hash(d) != PINNED_DOMAIN_INPUT_HASHES[d]:
                raise SystemExit(f"input files of domain {d} differ from the pinned inventory")
        if not self.missing:
            total = sum(r["size_bytes"] for r in self.records.values())
            if len(self.listing) != PINNED_INVENTORY_FILES or total != PINNED_INVENTORY_BYTES:
                raise SystemExit(f"inventory has {len(self.listing)} files / {total} bytes, expected "
                                 f"{PINNED_INVENTORY_FILES} / {PINNED_INVENTORY_BYTES}")

    @property
    def complete(self) -> bool:
        return not self.missing

    def scope_files(self, domain: str) -> list[str]:
        return [r for r in self.listing if r.startswith(_scope_prefix(domain))]

    def domain_complete(self, domain: str) -> bool:
        return all(r in self.records for r in self.scope_files(domain))

    def domain_hash(self, domain: str) -> str:
        return domain_input_hash([(r, self.records[r]["size_bytes"], self.records[r]["sha256"])
                                  for r in self.scope_files(domain)])

    def has(self, rel: str) -> bool:
        return rel in self.records

    def path(self, rel: str) -> Path:
        if rel not in self.records:
            raise SystemExit(f"benchmark file not available in the checkout: {rel}")
        return lio.require_sha(self.root / rel, self.records[rel]["sha256"])


# ---------------------------------------------------------------------------
# Source freeze: inventory, domain summary, licence files
# ---------------------------------------------------------------------------

def inventory_rows(src: BenchmarkSource) -> list[dict[str, Any]]:
    rows = []
    for d in DOMAINS:
        for rel in src.scope_files(d):
            if rel in src.records:
                r = src.records[rel]
                rows.append({"domain": d, "tier": TIER, "relative_path": rel, "size_bytes": r["size_bytes"],
                             "sha256": r["sha256"], "lfs_pointer": r["lfs_pointer"]})
    return rows


def _target_attribute_count(path: Path) -> int | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    props = obj.get("properties")
    if isinstance(props, dict):
        return len(props)
    for key in ("items", "schema"):
        node = obj.get(key)
        if isinstance(node, dict) and isinstance(node.get("properties"), dict):
            return len(node["properties"])
    return None


def _mapping_count(path: Path) -> int | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(obj, dict):
        for key in ("mappings", "mapping", "correspondences"):
            if key in obj and isinstance(obj[key], list):
                return len(obj[key])
    return None


def domain_summaries(src: BenchmarkSource) -> list[dict[str, Any]]:
    """Per-domain structure counts; byte, hash and LFS totals only when the domain is complete."""
    rows = []
    for d in DOMAINS:
        inp = _scope_prefix(d)
        files = src.scope_files(d)
        ts = inp + "schemamatching/target_schema.json"
        gm = inp + "schemamatching/sm_mapping_gold.json"
        complete = src.domain_complete(d)
        recs = [src.records[r] for r in files if r in src.records]
        rows.append({
            "domain": d, "tier": TIER, "task_path": f"use cases/{d}/{TIER}",
            "input_file_count": len(files),
            "input_bytes": sum(r["size_bytes"] for r in recs) if complete else "",
            "source_table_count": len(_direct_children(files, inp + "data/", "*.csv")),
            "source_metadata_count": len(_direct_children(files, inp + "data/", "*.json")),
            "taxonomy_file_count": len(_direct_children(files, inp + "schemamatching/", "*Taxonomy.csv")),
            "normalization_csv_count": len(_direct_children(files, inp + "normalization/", "*.csv")),
            "entitymatching_test_count": len(_direct_children(files, inp + "entitymatching/", "*_test.csv")),
            "fusion_file_count": sum(1 for r in files if r.startswith(inp + "fusion/")),
            "gold_version_present": int(inp + "fusion/GOLD_VERSION" in files),
            "target_schema_present": int(ts in files),
            "sm_mapping_gold_present": int(gm in files),
            "target_attribute_count": _target_attribute_count(src.path(ts)) if src.has(ts) else "",
            "schema_mapping_gold_count": _mapping_count(src.path(gm)) if src.has(gm) else "",
            "domain_input_hash": src.domain_hash(d) if complete else "",
            "lfs_pointer_count": sum(r["lfs_pointer"] for r in recs) if complete else "",
            "missing_required": [r for r in (ts, gm) if r not in files],
        })
    return rows


def license_files(src: BenchmarkSource) -> list[dict[str, Any]]:
    """Top-level licence / notice files of the pinned commit."""
    hits = []
    for name in src.root_entries:
        if any(fnmatch.fnmatchcase(name, pat) for pat in LICENSE_PATTERNS):
            p = src.root / name
            if not p.is_file():
                raise SystemExit(f"licence file {name} of the pinned commit is not in the checkout")
            rec = hash_file(p)
            hits.append({"name": name, "size_bytes": rec["size_bytes"], "sha256": rec["sha256"]})
    return sorted(hits, key=lambda x: x["name"])


PROTOCOL_CONSTRAINTS = [
    "The protocol consumes only files of the frozen inventory.",
    "The protected queries and the registered transformation interface are defined before any external audit verdict is computed.",
    "All five base domains remain in scope; no result-driven domain dropping.",
    "Easy/medium/hard variants are excluded from the external audit and may not be added after seeing base-domain results without a new protocol version.",
    "Zero-hit domains are reported as valid external specificity evidence rather than silently excluded.",
    "MaDI-Bench reference pipeline scores are not used to choose an audit interface.",
]


def source_freeze_object(src: BenchmarkSource, inv: list[dict[str, Any]], licenses: list[dict[str, Any]]) -> dict[str, Any]:
    complete_hashes = {d: src.domain_hash(d) for d in DOMAINS if src.domain_complete(d)}
    return {
        "step": STEP,
        "schema_version": SCHEMA_VERSION,
        "inputs_sha256": complete_hashes,
        "status": "FROZEN_BEFORE_EXTERNAL_RESULTS",
        "benchmark": BENCHMARK,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_git_remote": OFFICIAL_REMOTE,
        "git_commit": PINNED_COMMIT,
        "git_tree": PINNED_TREE,
        "git_branch": BRANCH,
        "git_commit_time": PINNED_COMMIT_TIME,
        "declared_domains": list(DOMAINS),
        "declared_tier": TIER,
        "authority_scope": ("all files under use cases/<domain>/base/input for each of the five "
                            "declared domains"),
        "domain_selection_after_results": False,
        "variants_in_scope": False,
        "benchmark_performance_results_read": False,
        "external_audit_run_before_freeze": False,
        "protocol_constraints": PROTOCOL_CONSTRAINTS,
        "license_files_detected": licenses,
        "redistribution_policy": "metadata and hashes only; benchmark files are not redistributed",
        "verification_method": src.verification,
        "inventory": {
            "domains": len(DOMAINS),
            "inventory_files": len(inv),
            "inventory_bytes": sum(int(r["size_bytes"]) for r in inv),
            "lfs_pointer_files": sum(int(r["lfs_pointer"]) for r in inv),
            "license_files": len(licenses),
        },
        "domain_input_hashes": complete_hashes,
        "inventory_verified": {
            "verified_files": len(src.records),
            "verified_bytes": sum(r["size_bytes"] for r in src.records.values()),
            "mismatches": 0,
        },
        "checkout_complete": src.complete,
        "pinned_files": len(src.listing),
        "missing_files": list(src.missing),
    }


# ---------------------------------------------------------------------------
# Protocol authority: structural reads only (schemas, gold mappings, taxonomies, field names)
# ---------------------------------------------------------------------------

def _flatten_json_scalars(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """JSON-pointer / scalar pairs; lists contribute their length and, when short and flat, their members."""
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k in sorted(obj):
            out.extend(_flatten_json_scalars(obj[k], f"{prefix}/{k}" if prefix else f"/{k}"))
    elif isinstance(obj, list):
        out.append((f"{prefix}/@length", len(obj)))
        if len(obj) <= 20 and all(not isinstance(x, (dict, list)) for x in obj):
            for i, v in enumerate(obj):
                out.append((f"{prefix}/{i}", v))
    else:
        out.append((prefix or "/", obj))
    return out


def _source_field_names(path: Path) -> tuple[str, list[str]]:
    """Field names of one source table without exporting any value."""
    ext = path.suffix.lower()
    if ext == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            header = next(csv.reader(fh), [])
        return "csv_header", [str(x) for x in header]
    if ext == ".jsonl":
        keys: set[str] = set()
        with path.open("r", encoding="utf-8") as fh:
            for _ in range(10):
                line = fh.readline()
                if not line:
                    break
                obj = json.loads(line)
                if isinstance(obj, dict):
                    keys.update(map(str, obj.keys()))
        return "jsonl_top_level_keys_first10", sorted(keys)
    if ext == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            keys = set()
            for x in obj[:10]:
                if isinstance(x, dict):
                    keys.update(map(str, x.keys()))
            return "json_list_top_level_keys_first10", sorted(keys)
        if isinstance(obj, dict):
            nested = [v for v in list(obj.values())[:10] if isinstance(v, dict)]
            if nested:
                keys = set()
                for x in nested:
                    keys.update(map(str, x.keys()))
                return "json_nested_record_keys_first10", sorted(keys)
            return "json_top_level_keys", sorted(map(str, obj.keys()))
    return "unsupported_for_field_names", []


def _header_only(path: Path) -> list[str]:
    if path.suffix.lower() != ".csv":
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [str(x) for x in next(csv.reader(fh), [])]


def _target_schema_rows(domain: str, path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    props = obj.get("properties")
    if not isinstance(props, dict):
        raise SystemExit(f"{domain}: target_schema has no properties")
    required = set(map(str, obj.get("required", [])))
    rows = []
    for attr in sorted(props):
        spec = props[attr]
        if not isinstance(spec, dict):
            spec = {"value": spec}
        aliases = spec.get("x-pydi-taxonomy-aliases", {})
        if not isinstance(aliases, dict):
            aliases = {}
        rows.append({
            "domain": domain,
            "target_attribute": attr,
            "required": int(attr in required),
            "type_json": json.dumps(spec.get("type", ""), ensure_ascii=False, sort_keys=True),
            "format": spec.get("format", ""),
            "pattern": spec.get("pattern", ""),
            "minimum": spec.get("minimum", ""),
            "maximum": spec.get("maximum", ""),
            "minLength": spec.get("minLength", ""),
            "maxLength": spec.get("maxLength", ""),
            "minItems": spec.get("minItems", ""),
            "maxItems": spec.get("maxItems", ""),
            "taxonomy_reference": spec.get("x-pydi-taxonomy", ""),
            "taxonomy_column": spec.get("x-pydi-taxonomy-column", ""),
            "taxonomy_exhaustive": spec.get("x-pydi-taxonomy-exhaustive", ""),
            "taxonomy_alias_count": len(aliases),
            "taxonomy_aliases_json": json.dumps(aliases, ensure_ascii=False, sort_keys=True),
            "property_spec_sha256": content_hash(spec),
        })
    return rows


def _gold_mapping_rows(domain: str, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    mappings = obj.get("mappings")
    if not isinstance(mappings, list):
        raise SystemExit(f"{domain}: sm_mapping_gold has no mappings list")
    positive = []
    for i, r in enumerate(mappings):
        if not isinstance(r, dict):
            raise SystemExit(f"{domain}: mapping {i} is not an object")
        positive.append({
            "domain": domain, "mapping_index": i,
            "source_dataset": str(r.get("source_dataset", "")), "source_column": str(r.get("source_column", "")),
            "target_dataset": str(r.get("target_dataset", "")), "target_column": str(r.get("target_column", "")),
            "label": int(bool(r.get("label", False))), "score": r.get("score", ""),
        })
    unmapped = []
    for i, r in enumerate(obj.get("unmapped_source_columns", [])):
        if not isinstance(r, dict):
            continue
        unmapped.append({
            "domain": domain, "unmapped_index": i,
            "source_dataset": str(r.get("source_dataset", "")), "source_column": str(r.get("source_column", "")),
            "target_column": r.get("target_column", ""), "label": int(bool(r.get("label", False))),
            "reason": str(r.get("reason", "")),
        })
    return positive, unmapped


def _taxonomy(domain: str, path: Path, rel: str, refs: list[tuple[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = [str(x) for x in (reader.fieldnames or [])]
        rows = [{str(k): "" if v is None else str(v) for k, v in r.items()} for r in reader]
    duplicates = len(rows) - len({tuple(r.get(c, "") for c in fields) for r in rows})
    unique = {c: len({r.get(c, "") for r in rows}) for c in fields}
    norm_rows = [{
        "domain": domain, "taxonomy_path": rel, "row_index": i,
        "row_json": json.dumps({c: r.get(c, "") for c in fields}, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")),
    } for i, r in enumerate(rows)]
    summary = {
        "domain": domain, "taxonomy_path": rel, "taxonomy_sha256": lio.sha256_file(path),
        "row_count": len(rows), "column_count": len(fields),
        "columns_json": json.dumps(fields, ensure_ascii=False),
        "unique_counts_json": json.dumps(unique, ensure_ascii=False, sort_keys=True),
        "duplicate_full_rows": duplicates,
        "referenced_by_json": json.dumps([{"target_attribute": a, "taxonomy_column": c} for a, c in refs],
                                         ensure_ascii=False, sort_keys=True),
    }
    return summary, norm_rows


AUTHORITY_NOTE = ("Candidate authority only. No protected query, derivation rule, or external audit verdict "
                  "is defined by the authority extraction.")


def extract_authorities(src: BenchmarkSource) -> dict[str, Any]:
    """Structural authorities of the five base tasks; labeled sets contribute headers only."""
    listed = set(src.listing)
    ext: dict[str, list] = {k: [] for k in (
        "target_attributes", "gold_mappings", "unmapped", "taxonomy_summary", "taxonomy_rows",
        "source_fields", "source_metadata", "labeled_headers", "transform_authorities", "skipped")}
    for domain in DOMAINS:
        inp = _scope_prefix(domain)
        sr = _target_schema_rows(domain, src.path(inp + "schemamatching/target_schema.json"))
        mr, ur = _gold_mapping_rows(domain, src.path(inp + "schemamatching/sm_mapping_gold.json"))
        ext["target_attributes"].extend(sr)
        ext["gold_mappings"].extend(mr)
        ext["unmapped"].extend(ur)

        refs_by_ref: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for r in sr:
            if str(r["taxonomy_reference"]):
                refs_by_ref[str(r["taxonomy_reference"])].append((str(r["target_attribute"]), str(r["taxonomy_column"])))
        tax_lookup = {}
        for tax_ref, refs in sorted(refs_by_ref.items()):
            rel = f"use cases/{domain}/{TIER}/{tax_ref}"
            if rel not in listed:
                raise SystemExit(f"{domain}: taxonomy reference is not a pinned file: {tax_ref}")
            summary, rows = _taxonomy(domain, src.path(rel), rel, refs)
            ext["taxonomy_summary"].append(summary)
            ext["taxonomy_rows"].extend(rows)
            tax_lookup[tax_ref] = summary

        files = src.scope_files(domain)
        for rel in _direct_children(files, inp + "data/"):
            if not src.has(rel):
                ext["skipped"].append(rel)
                continue
            p = src.path(rel)
            if p.name.endswith("_metadata.json"):
                for ptr, val in _flatten_json_scalars(json.loads(p.read_text(encoding="utf-8"))):
                    ext["source_metadata"].append({
                        "domain": domain, "metadata_path": rel, "json_pointer": ptr,
                        "value_json": json.dumps(val, ensure_ascii=False, sort_keys=True)})
            else:
                method, names = _source_field_names(p)
                ext["source_fields"].append({
                    "domain": domain, "source_path": rel, "format": p.suffix.lower().lstrip("."),
                    "field_discovery_method": method, "field_count": len(names),
                    "fields_json": json.dumps(names, ensure_ascii=False), "source_sha256": src.records[rel]["sha256"]})

        for task in ("normalization", "entitymatching", "fusion"):
            for rel in _direct_children(files, inp + task + "/"):
                if not src.has(rel):
                    ext["skipped"].append(rel)
                    continue
                p = src.path(rel)
                ext["labeled_headers"].append({
                    "domain": domain, "task": task, "relative_path": rel, "format": p.suffix.lower().lstrip("."),
                    "header_only_json": json.dumps(_header_only(p), ensure_ascii=False), "content_rows_read": 0,
                    "sha256": src.records[rel]["sha256"]})

        positive_by_target: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for r in mr:
            if int(r["label"]) == 1:
                positive_by_target[str(r["target_column"])].append((str(r["source_dataset"]), str(r["source_column"])))
        for r in sr:
            tax_ref = str(r["taxonomy_reference"])
            if not tax_ref:
                continue
            summ = tax_lookup[tax_ref]
            tax_col = str(r["taxonomy_column"])
            present = tax_col in json.loads(summ["columns_json"])
            srcs = positive_by_target.get(str(r["target_attribute"]), [])
            ext["transform_authorities"].append({
                "domain": domain, "target_attribute": r["target_attribute"],
                "authority_type": "TARGET_SCHEMA_DECLARED_TAXONOMY", "taxonomy_reference": tax_ref,
                "taxonomy_column": tax_col, "taxonomy_column_present": int(present),
                "taxonomy_exhaustive": r["taxonomy_exhaustive"], "taxonomy_alias_count": r["taxonomy_alias_count"],
                "taxonomy_row_count": summ["row_count"], "positive_gold_source_mapping_count": len(srcs),
                "positive_gold_source_mappings_json": json.dumps(
                    [{"source_dataset": a, "source_column": b} for a, b in srcs], ensure_ascii=False, sort_keys=True),
                "structural_authority_complete": int(present and len(srcs) > 0),
                "NOTE": AUTHORITY_NOTE,
            })

    ext["taxonomy_authority_by_domain"] = []
    for d in DOMAINS:
        trs = [r for r in ext["transform_authorities"] if r["domain"] == d]
        complete = [r for r in trs if int(r["structural_authority_complete"]) == 1]
        ext["taxonomy_authority_by_domain"].append({
            "domain": d, "declared_taxonomy_attributes": len(trs),
            "structurally_complete_taxonomy_authorities": len(complete),
            "status": "HAS_DECLARED_TAXONOMY_AUTHORITY" if complete else "NO_DECLARED_TAXONOMY_AUTHORITY"})

    counts = {
        "target_schema_attributes": len(ext["target_attributes"]),
        "positive_gold_schema_mappings": len(ext["gold_mappings"]),
        "unmapped_source_columns": len(ext["unmapped"]),
        "declared_taxonomy_files": len(ext["taxonomy_summary"]),
        "transform_authority_candidates": len(ext["transform_authorities"]),
        "structurally_complete_transform_authorities": sum(int(r["structural_authority_complete"])
                                                           for r in ext["transform_authorities"]),
        "source_files_field_names_only": len(ext["source_fields"]),
    }
    for k, v in AUTHORITY_COUNTS.items():
        if k == "source_files_field_names_only" and ext["skipped"]:
            continue
        if counts[k] != v:
            raise SystemExit(f"structural authority count {k}={counts[k]}, expected {v}")
    ext["counts"] = counts
    return ext


# ---------------------------------------------------------------------------
# Frozen protocol: queries, exact carriers, source scope, qualification rule (no source values read)
# ---------------------------------------------------------------------------

def normalize_token(x: Any) -> str:
    """Unicode NFKC, strip, collapse whitespace runs to one space, casefold."""
    if x is None:
        return ""
    s = unicodedata.normalize("NFKC", str(x))
    s = re.sub(r"\s+", " ", s.strip())
    return s.casefold()


def slug(s: str) -> str:
    x = re.sub(r"[^a-z0-9]+", "-", s.casefold()).strip("-")
    return x or "x"


def taxonomy_tables(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    """(domain, taxonomy_path) -> taxonomy rows in file order."""
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for rec in rows:
        row = json.loads(str(rec["row_json"]))
        if not isinstance(row, dict):
            raise SystemExit("taxonomy row_json is not an object")
        grouped[(str(rec["domain"]), str(rec["taxonomy_path"]))].append(
            (int(rec["row_index"]), {str(k): "" if v is None else str(v) for k, v in row.items()}))
    return {k: [r for _, r in sorted(v, key=lambda t: t[0])] for k, v in grouped.items()}


def _parse_aliases(s: str) -> dict[str, str]:
    obj = json.loads(s or "{}")
    if not isinstance(obj, dict):
        raise SystemExit("taxonomy_aliases_json is not an object")
    return {str(k): str(v) for k, v in obj.items()}


SELECTION_RULE = ("ALL non-protected taxonomy columns with non-empty support and zero normalized "
                  "carrier->protected conflicts")


def build_protocol(target_attributes: list[dict[str, str]], gold_mappings: list[dict[str, str]],
                   transform_authorities: list[dict[str, str]], taxonomy_rows: list[dict[str, Any]]) -> dict[str, list]:
    """Registered queries, exact carriers (functional determinacy inside the taxonomy) and source scope.

    Queries are numbered in DOMAINS order and then by target attribute in code-point order; carriers of a
    query are numbered by taxonomy column name in code-point order."""
    attrs = {(str(r["domain"]), str(r["target_attribute"])): r for r in target_attributes}
    tables = taxonomy_tables(taxonomy_rows)
    pos_maps = [r for r in gold_mappings if _intish(r["label"]) == 1]
    queries, carriers, noncarriers, scope = [], [], [], []
    ordered = sorted(transform_authorities, key=lambda r: (DOMAINS.index(str(r["domain"])), str(r["target_attribute"])))
    for q_index, auth in enumerate(ordered):
        domain, target = str(auth["domain"]), str(auth["target_attribute"])
        if domain not in TAXONOMY_DOMAINS:
            raise SystemExit(f"unexpected transform-authority domain: {domain}")
        if _intish(auth["structural_authority_complete"]) != 1:
            raise SystemExit(f"incomplete authority: {domain}/{target}")
        ar = attrs.get((domain, target))
        if ar is None:
            raise SystemExit(f"target attribute record missing: {domain}/{target}")
        tax_path = f"use cases/{domain}/{TIER}/{auth['taxonomy_reference']}"
        target_col = str(auth["taxonomy_column"])
        table = tables.get((domain, tax_path))
        if not table:
            raise SystemExit(f"taxonomy rows missing: {domain}/{tax_path}")
        if target_col not in table[0]:
            raise SystemExit(f"protected taxonomy column missing: {domain}/{target}/{target_col}")

        aliases = _parse_aliases(str(ar.get("taxonomy_aliases_json", "{}")))
        normalized_aliases = {normalize_token(k): normalize_token(v) for k, v in aliases.items()
                              if normalize_token(k) and normalize_token(v)}
        protected = sorted({normalize_token(r.get(target_col, "")) for r in table
                            if normalize_token(r.get(target_col, ""))})
        if not protected:
            raise SystemExit(f"protected taxonomy domain is empty: {domain}/{target}")
        qid = f"Q{q_index + 1:02d}-{slug(domain)}-{slug(target)}"

        maps = [r for r in pos_maps if str(r["domain"]) == domain and str(r["target_column"]) == target]
        if len(maps) != _intish(auth["positive_gold_source_mapping_count"]):
            raise SystemExit(f"positive source-mapping count mismatch for {qid}")
        for mr in sorted(maps, key=lambda r: (str(r["source_dataset"]), str(r["source_column"]))):
            scope.append({
                "query_id": qid, "domain": domain, "target_attribute": target,
                "source_dataset": str(mr["source_dataset"]), "source_column": str(mr["source_column"]),
                "gold_label": _intish(mr["label"]), "gold_score": str(mr["score"]), "scope_role": "AUDIT_SOURCE_COLUMN"})

        n_exact = n_rejected = 0
        for carrier_col in sorted(c for c in table[0].keys() if c != target_col):
            mapping: dict[str, set[str]] = defaultdict(set)
            nonblank = 0
            for tr in table:
                cv, tv = normalize_token(tr.get(carrier_col, "")), normalize_token(tr.get(target_col, ""))
                if not cv or not tv:
                    continue
                nonblank += 1
                mapping[cv].add(tv)
            conflicts = sorted(k for k, vals in mapping.items() if len(vals) != 1)
            if nonblank > 0 and not conflicts:
                n_exact += 1
                carriers.append({
                    "query_id": qid, "carrier_id": f"{qid}-C{n_exact:02d}-{slug(carrier_col)}",
                    "domain": domain, "target_attribute": target, "taxonomy_path": tax_path,
                    "protected_taxonomy_column": target_col, "carrier_taxonomy_column": carrier_col,
                    "taxonomy_rows": len(table), "carrier_nonblank_rows": nonblank,
                    "carrier_unique_values": len(mapping), "protected_unique_values": len(protected),
                    "fd_conflict_keys": 0, "exact_determinacy": 1,
                    "carrier_map_sha256": content_hash({k: next(iter(v)) for k, v in sorted(mapping.items())}),
                    "selection_rule": SELECTION_RULE})
            else:
                n_rejected += 1
                noncarriers.append({
                    "query_id": qid, "domain": domain, "target_attribute": target, "taxonomy_path": tax_path,
                    "protected_taxonomy_column": target_col, "candidate_taxonomy_column": carrier_col,
                    "taxonomy_rows": len(table), "carrier_nonblank_rows": nonblank,
                    "carrier_unique_values": len(mapping), "fd_conflict_keys": len(conflicts), "exact_determinacy": 0,
                    "reason": ("NO_NONBLANK_SUPPORT" if nonblank == 0
                               else "ONE_CARRIER_VALUE_MAPS_TO_MULTIPLE_PROTECTED_VALUES")})

        queries.append({
            "query_id": qid, "domain": domain, "target_attribute": target,
            "target_type_json": str(ar.get("type_json", "")), "required": _intish(ar.get("required", 0)),
            "taxonomy_path": tax_path,
            "taxonomy_sha256": content_hash([{k: normalize_token(v) for k, v in sorted(tr.items())} for tr in table]),
            "protected_taxonomy_column": target_col, "protected_unique_values": len(protected),
            "protected_value_set_sha256": content_hash(protected),
            "taxonomy_exhaustive": int(_boolish(ar.get("taxonomy_exhaustive", False))),
            "target_schema_alias_count": len(aliases),
            "target_schema_aliases_json": json.dumps(aliases, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "normalized_alias_map_sha256": content_hash(normalized_aliases),
            "positive_gold_source_mapping_count": len(maps),
            "exact_carrier_count": n_exact, "rejected_noncarrier_count": n_rejected,
            "protected_query": (f"canonical value of target attribute {target!r} under the frozen taxonomy "
                                f"column {target_col!r}"),
        })

    per_domain: dict[str, int] = defaultdict(int)
    for q in queries:
        per_domain[q["domain"]] += 1
    domain_scope = [{
        "domain": d, "declared": 1, "base_tier_only": 1, "taxonomy_backed_query_count": per_domain.get(d, 0),
        "handling": "RUN_ALL_REGISTERED_QUERIES" if per_domain.get(d, 0) else "NO_REGISTERED_TAXONOMY_TRANSFORM",
        "interpretation_if_no_query": ("" if per_domain.get(d, 0)
                                       else "Structural out-of-scope control; do not infer NONDETERMINED."),
    } for d in DOMAINS]

    got = {"queries": len(queries), "source_mappings": len(scope), "exact_carriers": len(carriers),
           "noncarriers": len(noncarriers)}
    if got != PROTOCOL_COUNTS:
        raise SystemExit(f"protocol cardinalities {got} differ from {PROTOCOL_COUNTS}")
    if {q["domain"] for q in queries} != set(TAXONOMY_DOMAINS):
        raise SystemExit("taxonomy-domain scope differs from TAXONOMY_DOMAINS")
    if {r["domain"] for r in domain_scope if r["handling"] == "NO_REGISTERED_TAXONOMY_TRANSFORM"} != set(NO_TRANSFORM_DOMAINS):
        raise SystemExit("no-transform domain scope differs from NO_TRANSFORM_DOMAINS")
    return {"queries": queries, "carriers": carriers, "noncarriers": noncarriers, "source_scope": scope,
            "domain_scope": domain_scope}


def qualification_rule() -> dict[str, Any]:
    return {
        "scientific_execution_requirements": {
            "run_all_registered_queries": True,
            "run_all_gold_mapped_source_columns": True,
            "selective_domain_dropping_after_results": False,
            "selective_query_dropping_after_results": False,
            "manual_rule_addition_after_results": False,
            "products_and_papers": "retain as structural no-registered-transform domains; do not infer non-determination",
        },
        "nontrivial_L1_query_definition": {
            "minimum_L1_exact_derived_rows": MIN_L1_ROWS,
            "minimum_L1_rate_among_nonempty_mapped_values": MIN_L1_RATE,
            "both_conditions_required": True,
            "note": ("A domain qualifies when one of its queries meets both thresholds; the thresholds are "
                     "frozen before any source value is read and change no measurement."),
        },
        "breadth_qualification_rule": {
            "minimum_distinct_domains_with_at_least_one_nontrivial_L1_query": MIN_DISTINCT_DOMAINS,
            "protocol_integrity_failures_required": 0,
            "reporting": ("All registered queries and all gold-mapped source columns are reported whatever the "
                          "outcome."),
        },
        "never_permitted": [
            "Keep only domains with favorable L1 results.",
            "Add an alias or delimiter parser after inspecting source values.",
            "Change the protected target or carrier set after audit results.",
            "Treat no match as NONDETERMINED.",
            "Describe a structurally unregistered domain as evidence against derivability.",
        ],
    }


def protocol_object(reg: dict[str, list], authority_by_domain: list[dict[str, Any]],
                    inputs_sha256: dict[str, str]) -> dict[str, Any]:
    return {
        "step": STEP,
        "schema_version": SCHEMA_VERSION,
        "inputs_sha256": inputs_sha256,
        "status": "FROZEN_BEFORE_EXTERNAL_AUDIT",
        "madi_commit": PINNED_COMMIT,
        "madi_tree": PINNED_TREE,
        "domains": list(DOMAINS),
        "tier": TIER,
        "result_values_read": False,
        "external_verdicts_computed": False,
        "query_count": len(reg["queries"]),
        "source_mapping_count": len(reg["source_scope"]),
        "exact_carrier_count": len(reg["carriers"]),
        "noncarrier_count": len(reg["noncarriers"]),
        "protected_query_semantics": ("For each taxonomy-backed target attribute, q asks for the canonical "
                                      "target taxonomy value represented by the frozen target taxonomy column."),
        "source_scope": ("Only source columns with positive gold schema mappings to the protected target "
                         "attribute are audited. All such mappings are included."),
        "normalization": {
            "steps": ["Unicode NFKC", "strip leading/trailing whitespace",
                      "collapse one-or-more Unicode whitespace characters to one ASCII space", "Unicode casefold"],
            "forbidden": ["punctuation stripping", "stemming", "fuzzy matching", "edit-distance matching",
                          "embedding similarity", "learned normalization", "manual source-specific aliases"],
        },
        "tokenization": {
            "native_JSON_array": "audit each scalar array element independently",
            "native_scalar": "audit as one scalar token",
            "CSV_string": "audit the whole cell as one token",
            "JSON_encoded_list_inside_string": "do not parse or split; audit as one scalar string",
            "delimiter_splitting": False,
            "reason": "Conservative pre-result rule prevents result-driven parsing of source-specific delimiters.",
        },
        "classification_order": [
            {"class": "L0_DIRECT",
             "rule": ("Normalized token equals a protected-taxonomy value, or equals a target-schema-declared "
                      "alias whose mapped value is a protected taxonomy value.")},
            {"class": "L1_EXACT_DERIVED",
             "rule": ("Token is not L0_DIRECT; it matches one or more registered exact carrier columns, and "
                      "every matched carrier yields the same protected target value.")},
            {"class": "UNVERIFIED_AMBIGUOUS",
             "rule": ("Token is not L0_DIRECT and matches registered carriers that imply different protected "
                      "target values.")},
            {"class": "UNVERIFIED_NO_MATCH",
             "rule": "Token is not L0_DIRECT and matches no registered exact carrier."},
        ],
        "important_semantics": [
            "UNVERIFIED is not evidence of non-determination.",
            "A domain with no declared taxonomy transform is not NONDETERMINED.",
            "No cross-row, multi-column, fuzzy, learned, or manually curated derivation is permitted.",
            "Taxonomy-exhaustive=false means unmatched values remain UNVERIFIED.",
            "The audit reports all six registered queries and all fourteen gold-mapped source columns.",
        ],
        "all_query_registry_sha256": content_hash(reg["queries"]),
        "all_carrier_registry_sha256": content_hash(reg["carriers"]),
        "all_noncarrier_registry_sha256": content_hash(reg["noncarriers"]),
        "all_source_scope_sha256": content_hash(reg["source_scope"]),
        "all_domain_scope_sha256": content_hash(reg["domain_scope"]),
        "qualification_rule": qualification_rule(),
        "taxonomy_authority_by_domain": authority_by_domain,
        "taxonomy_authority_note": ("HAS_DECLARED_TAXONOMY_AUTHORITY is not evidence of leakage, derivability, "
                                    "or audit success; the authority extraction computes no external verdict."),
    }


# ---------------------------------------------------------------------------
# Empirical audit: the only part that reads benchmark source values
# ---------------------------------------------------------------------------

def query_models(queries: list[dict[str, str]], carriers: list[dict[str, str]],
                 taxonomy_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Protected sets, aliases and reverse carrier maps rebuilt from the taxonomy and checked against the registry."""
    tables = taxonomy_tables(taxonomy_rows)
    by_query: dict[str, list[dict[str, str]]] = defaultdict(list)
    for c in carriers:
        by_query[str(c["query_id"])].append(c)
    models = {}
    for q in queries:
        qid = str(q["query_id"])
        table = tables.get((str(q["domain"]), str(q["taxonomy_path"])))
        if not table:
            raise SystemExit(f"taxonomy table missing for {qid}")
        target_col = str(q["protected_taxonomy_column"])
        protected = sorted({normalize_token(r.get(target_col, "")) for r in table
                            if normalize_token(r.get(target_col, ""))})
        if len(protected) != _intish(q["protected_unique_values"]) or content_hash(protected) != str(q["protected_value_set_sha256"]):
            raise SystemExit(f"protected value set differs from the registry for {qid}")
        aliases = {normalize_token(k): normalize_token(v)
                   for k, v in json.loads(str(q["target_schema_aliases_json"]) or "{}").items()}
        aliases = {k: v for k, v in aliases.items() if k and v}
        if content_hash(aliases) != str(q["normalized_alias_map_sha256"]):
            raise SystemExit(f"alias map differs from the registry for {qid}")
        reverse: dict[str, list[tuple[str, str]]] = defaultdict(list)
        n_carriers = 0
        for c in sorted(by_query.get(qid, []), key=lambda x: str(x["carrier_id"])):
            ccol = str(c["carrier_taxonomy_column"])
            mapping: dict[str, set[str]] = defaultdict(set)
            for tr in table:
                cv, tv = normalize_token(tr.get(ccol, "")), normalize_token(tr.get(target_col, ""))
                if cv and tv:
                    mapping[cv].add(tv)
            if any(len(v) != 1 for v in mapping.values()):
                raise SystemExit(f"registered carrier is not exact: {c['carrier_id']}")
            carrier_map = {k: next(iter(v)) for k, v in sorted(mapping.items())}
            if (content_hash(carrier_map) != str(c["carrier_map_sha256"])
                    or len(carrier_map) != _intish(c["carrier_unique_values"])):
                raise SystemExit(f"carrier map differs from the registry: {c['carrier_id']}")
            n_carriers += 1
            for token, value in carrier_map.items():
                reverse[token].append((str(c["carrier_id"]), value))
        if n_carriers != _intish(q["exact_carrier_count"]):
            raise SystemExit(f"carrier count mismatch for {qid}")
        models[qid] = {"protected": set(protected), "aliases": aliases, "reverse": dict(reverse),
                       "carriers": n_carriers}
    if len(models) != PROTOCOL_COUNTS["queries"]:
        raise SystemExit(f"query-model count {len(models)}")
    return models


def classify_token(raw: Any, model: dict[str, Any]) -> tuple[str, str | None, list[str], str]:
    """Class, derived/protected value, carrier ids and normalized token, in the frozen class order."""
    token = normalize_token(raw)
    if not token:
        return "EMPTY", None, [], token
    if token in model["protected"]:
        return "L0_DIRECT", token, [], token
    if token in model["aliases"] and model["aliases"][token] in model["protected"]:
        return "L0_DIRECT", model["aliases"][token], [], token
    hits = model["reverse"].get(token, [])
    if not hits:
        return "UNVERIFIED_NO_MATCH", None, [], token
    targets = sorted({t for _, t in hits})
    carrier_ids = sorted({cid for cid, _ in hits})
    if len(targets) == 1:
        return "L1_EXACT_DERIVED", targets[0], carrier_ids, token
    return "UNVERIFIED_AMBIGUOUS", None, carrier_ids, token


def resolve_sources(src: BenchmarkSource, source_fields: list[dict[str, str]],
                    source_scope: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    """(domain, source_dataset) -> verified CSV source table with its declared field names."""
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in source_fields:
        by_key[(str(r["domain"]), Path(str(r["source_path"])).stem)].append(r)
    out = {}
    for key in sorted({(str(r["domain"]), str(r["source_dataset"])) for r in source_scope}):
        rows = by_key.get(key, [])
        if len(rows) != 1:
            raise SystemExit(f"source dataset {key} does not resolve to exactly one table")
        r = rows[0]
        if str(r["format"]) != "csv":
            raise SystemExit(f"registered source {key} is not a CSV table")
        if src.records.get(str(r["source_path"]), {}).get("sha256") != str(r["source_sha256"]):
            raise SystemExit(f"source table {key} differs from the frozen inventory")
        out[key] = {"path": src.path(str(r["source_path"])), "relative_path": str(r["source_path"]),
                    "sha256": str(r["source_sha256"]), "fields": [str(x) for x in json.loads(str(r["fields_json"]))]}
    for r in source_scope:
        if str(r["source_column"]) not in out[(str(r["domain"]), str(r["source_dataset"]))]["fields"]:
            raise SystemExit(f"registered source column {r['source_column']} is not a field of {r['source_dataset']}")
    return out


def audit_source_column(source: dict[str, Any], column: str, query_id: str, model: dict[str, Any],
                        domain: str, dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Per-cell classification of one registered source column; first L1 cells kept as hashed witnesses."""
    csv.field_size_limit(max(csv.field_size_limit(), CSV_FIELD_LIMIT))
    counts: dict[str, int] = defaultdict(int)
    witnesses = []
    total = 0
    with source["path"].open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise SystemExit(f"column {column} missing from {source['relative_path']}")
        for row_index, row in enumerate(reader, start=1):
            total += 1
            cls, target, carrier_ids, token = classify_token(row.get(column, ""), model)
            counts[cls] += 1
            if cls == "L1_EXACT_DERIVED" and len(witnesses) < WITNESS_LIMIT_PER_SOURCE:
                witnesses.append({
                    "query_id": query_id, "domain": domain, "source_dataset": dataset, "source_column": column,
                    "row_locator_sha256": lio.sha256_bytes(f"{source['sha256']}|{row_index}|{column}".encode("utf-8")),
                    "token_sha256": lio.sha256_bytes(token.encode("utf-8")),
                    "derived_target_sha256": lio.sha256_bytes(str(target).encode("utf-8")),
                    "carrier_ids": carrier_ids, "class": cls})
    nonempty = total - counts["EMPTY"]
    classified = sum(counts[c] for c in CLASSES)
    if classified != nonempty:
        raise SystemExit("classification accounting mismatch")
    return {
        "query_id": query_id, "domain": domain, "source_dataset": dataset,
        "source_relative_path": source["relative_path"], "source_sha256": source["sha256"], "source_column": column,
        "source_rows": total, "empty_rows": counts["EMPTY"], "nonempty_rows": nonempty,
        "L0_direct_rows": counts["L0_DIRECT"], "L1_exact_derived_rows": counts["L1_EXACT_DERIVED"],
        "unverified_ambiguous_rows": counts["UNVERIFIED_AMBIGUOUS"],
        "unverified_no_match_rows": counts["UNVERIFIED_NO_MATCH"],
        "L0_direct_rate_nonempty": counts["L0_DIRECT"] / nonempty if nonempty else 0.0,
        "L1_exact_derived_rate_nonempty": counts["L1_EXACT_DERIVED"] / nonempty if nonempty else 0.0,
        "nonempty_accounting_ok": int(classified == nonempty),
    }, witnesses


def aggregate_results(source_results: list[dict[str, Any]], queries: list[dict[str, str]],
                      domain_scope: list[dict[str, str]]) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Per-query and per-domain totals and the qualification decision."""
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in source_results:
        by_q[str(r["query_id"])].append(r)
    q_lookup = {str(q["query_id"]): q for q in queries}
    query_rows = []
    for qid in sorted(q_lookup):
        rr, q = by_q.get(qid, []), q_lookup[qid]
        if len(rr) != _intish(q["positive_gold_source_mapping_count"]):
            raise SystemExit(f"query {qid} has {len(rr)} audited source columns")
        tot = {k: sum(int(r[k]) for r in rr) for k in (
            "source_rows", "empty_rows", "nonempty_rows", "L0_direct_rows", "L1_exact_derived_rows",
            "unverified_ambiguous_rows", "unverified_no_match_rows")}
        if (tot["L0_direct_rows"] + tot["L1_exact_derived_rows"] + tot["unverified_ambiguous_rows"]
                + tot["unverified_no_match_rows"]) != tot["nonempty_rows"]:
            raise SystemExit(f"query accounting mismatch for {qid}")
        nonempty = tot["nonempty_rows"]
        rate = tot["L1_exact_derived_rows"] / nonempty if nonempty else 0.0
        query_rows.append({
            "query_id": qid, "domain": str(q["domain"]), "target_attribute": str(q["target_attribute"]),
            "source_columns": len(rr), "source_rows_sum": tot["source_rows"], "empty_rows": tot["empty_rows"],
            "nonempty_rows": nonempty, "L0_direct_rows": tot["L0_direct_rows"],
            "L1_exact_derived_rows": tot["L1_exact_derived_rows"],
            "unverified_ambiguous_rows": tot["unverified_ambiguous_rows"],
            "unverified_no_match_rows": tot["unverified_no_match_rows"],
            "L0_direct_rate_nonempty": tot["L0_direct_rows"] / nonempty if nonempty else 0.0,
            "L1_exact_derived_rate_nonempty": rate,
            "nontrivial_L1_min_rows": MIN_L1_ROWS, "nontrivial_L1_min_rate": MIN_L1_RATE,
            "nontrivial_L1_query": int(tot["L1_exact_derived_rows"] >= MIN_L1_ROWS and rate >= MIN_L1_RATE),
        })
    scope = {str(r["domain"]): r for r in domain_scope}
    by_d: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in query_rows:
        by_d[q["domain"]].append(q)
    domain_rows, qualifying = [], []
    for d in DOMAINS:
        qs = by_d.get(d, [])
        registered = _intish(scope[d]["taxonomy_backed_query_count"])
        if registered != len(qs):
            raise SystemExit(f"domain {d}: {registered} registered queries, {len(qs)} executed")
        n_qual = sum(int(q["nontrivial_L1_query"]) for q in qs)
        if n_qual >= 1:
            qualifying.append(d)
        domain_rows.append({
            "domain": d, "registered_queries": registered, "executed_queries": len(qs),
            "qualifying_nontrivial_L1_queries": n_qual, "domain_qualifies": int(n_qual >= 1),
            "L0_direct_rows": sum(int(q["L0_direct_rows"]) for q in qs),
            "L1_exact_derived_rows": sum(int(q["L1_exact_derived_rows"]) for q in qs),
            "nonempty_rows": sum(int(q["nonempty_rows"]) for q in qs),
            "handling": str(scope[d]["handling"]),
            "interpretation": ("AUDITED_REGISTERED_TRANSFORMS" if registered > 0
                               else "NO_REGISTERED_TRANSFORM_DO_NOT_INFER_NONDETERMINATION"),
        })
    decision = {
        "breadth_qualification": "PASS" if len(qualifying) >= MIN_DISTINCT_DOMAINS else "FAIL",
        "qualifying_domains": qualifying,
        "qualifying_domain_count": len(qualifying),
        "required_distinct_domains": MIN_DISTINCT_DOMAINS,
        "per_query_threshold": {"minimum_L1_exact_derived_rows": MIN_L1_ROWS,
                                "minimum_L1_rate_among_nonempty_mapped_values": MIN_L1_RATE},
        "note": "Qualification summarizes the measurements and does not change them.",
    }
    return query_rows, domain_rows, decision


def run_audit(src: BenchmarkSource, out_dir: Path) -> dict[str, Any]:
    """Audit all registered source columns under the frozen protocol read back from out_dir."""
    queries = read_csv_rows(out_dir / OUT["query_registry"])
    carriers = read_csv_rows(out_dir / OUT["carrier_registry"])
    scope = read_csv_rows(out_dir / OUT["source_scope"])
    domain_scope = read_csv_rows(out_dir / OUT["domain_scope"])
    source_fields = read_csv_rows(out_dir / OUT["source_fields"])
    taxonomy = read_jsonl(out_dir / OUT["taxonomy_rows"])
    sources = resolve_sources(src, source_fields, scope)
    models = query_models(queries, carriers, taxonomy)
    results, witnesses = [], []
    for s in sorted(scope, key=lambda r: (str(r["query_id"]), str(r["source_dataset"]), str(r["source_column"]))):
        key = (str(s["domain"]), str(s["source_dataset"]))
        r, w = audit_source_column(sources[key], str(s["source_column"]), str(s["query_id"]),
                                   models[str(s["query_id"])], key[0], key[1])
        results.append(r)
        witnesses.extend(w)
    if len(results) != PROTOCOL_COUNTS["source_mappings"]:
        raise SystemExit("not every registered source mapping was audited")
    query_rows, domain_rows, decision = aggregate_results(results, queries, domain_scope)
    if len(query_rows) != PROTOCOL_COUNTS["queries"] or {r["domain"] for r in domain_rows} != set(DOMAINS):
        raise SystemExit("not every registered query and domain is reported")
    lio.write_csv(out_dir / OUT["source_results"], results, FIELDS["source_results"])
    lio.write_csv(out_dir / OUT["query_results"], query_rows, FIELDS["query_results"])
    lio.write_csv(out_dir / OUT["domain_results"], domain_rows, FIELDS["domain_results"])
    lio.write_csv(out_dir / OUT["witnesses"], witnesses, FIELDS["witnesses"])
    integrity = {
        "registered_queries": len(queries), "executed_queries": len(query_rows),
        "registered_source_mappings": len(scope), "executed_source_mappings": len(results),
        "registered_exact_carriers": len(carriers),
        "reconstructed_exact_carriers": sum(m["carriers"] for m in models.values()),
        "source_files_verified": len(sources),
        "hashed_witness_limit_per_source_mapping": WITNESS_LIMIT_PER_SOURCE,
        "madi_commit": PINNED_COMMIT, "madi_tree": PINNED_TREE, "madi_repo_clean": True,
        "raw_external_values_written_to_outputs": False, "post_result_rules_added": False,
        "source_specific_delimiter_parsing": False, "fuzzy_or_embedding_matching": False,
    }
    return {"decision": decision, "integrity": integrity}


# ---------------------------------------------------------------------------
# Independent second implementation. It shares no normalization, carrier, classification or
# aggregation code with the audit above: it re-reads the frozen registries and the audit outputs from
# disk, recomputes every count from the source tables, and compares.
# ---------------------------------------------------------------------------

def _ind_norm(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _ind_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ind_table(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _ind_models(queries, carriers, taxonomy_records) -> dict[str, dict[str, Any]]:
    tables: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for rec in taxonomy_records:
        obj = json.loads(rec["row_json"])
        tables[(rec["domain"], rec["taxonomy_path"])].append(
            (int(rec["row_index"]), {str(k): "" if v is None else str(v) for k, v in obj.items()}))
    carriers_of: dict[str, list[dict[str, str]]] = defaultdict(list)
    for c in carriers:
        carriers_of[c["query_id"]].append(c)
    models, total_carriers = {}, 0
    for q in queries:
        qid = q["query_id"]
        rows = [r for _, r in sorted(tables.get((q["domain"], q["taxonomy_path"]), []), key=lambda t: t[0])]
        if not rows:
            raise SystemExit(f"independent check: taxonomy missing for {qid}")
        pcol = q["protected_taxonomy_column"]
        protected = sorted({_ind_norm(r.get(pcol, "")) for r in rows} - {""})
        if len(protected) != int(float(q["protected_unique_values"])) or _ind_hash(protected) != q["protected_value_set_sha256"]:
            raise SystemExit(f"independent check: protected set mismatch for {qid}")
        aliases = {}
        for k, v in json.loads(q["target_schema_aliases_json"] or "{}").items():
            nk, nv = _ind_norm(k), _ind_norm(v)
            if nk and nv:
                aliases[nk] = nv
        if _ind_hash(aliases) != q["normalized_alias_map_sha256"]:
            raise SystemExit(f"independent check: alias map mismatch for {qid}")
        reverse: dict[str, list[tuple[str, str]]] = defaultdict(list)
        n = 0
        for c in sorted(carriers_of[qid], key=lambda r: r["carrier_id"]):
            pairs: dict[str, set[str]] = defaultdict(set)
            for r in rows:
                x, y = _ind_norm(r.get(c["carrier_taxonomy_column"], "")), _ind_norm(r.get(pcol, ""))
                if x and y:
                    pairs[x].add(y)
            if any(len(s) != 1 for s in pairs.values()):
                raise SystemExit(f"independent check: carrier {c['carrier_id']} is ambiguous")
            cmap = {k: min(s) for k, s in sorted(pairs.items())}
            if _ind_hash(cmap) != c["carrier_map_sha256"] or len(cmap) != int(float(c["carrier_unique_values"])):
                raise SystemExit(f"independent check: carrier map mismatch for {c['carrier_id']}")
            for token, derived in cmap.items():
                reverse[token].append((c["carrier_id"], derived))
            n += 1
        if n != int(float(q["exact_carrier_count"])):
            raise SystemExit(f"independent check: carrier count mismatch for {qid}")
        total_carriers += n
        models[qid] = {"protected": set(protected), "aliases": aliases, "reverse": dict(reverse), "carriers": n}
    if len(models) != PROTOCOL_COUNTS["queries"] or total_carriers != PROTOCOL_COUNTS["exact_carriers"]:
        raise SystemExit("independent check: reconstructed query or carrier count mismatch")
    return models


def _ind_classify(raw: Any, model: dict[str, Any]) -> tuple[str, str | None, list[str], str]:
    token = _ind_norm(raw)
    if token == "":
        return "EMPTY", None, [], ""
    if token in model["protected"]:
        return "L0_DIRECT", token, [], token
    alias = model["aliases"].get(token)
    if alias is not None and alias in model["protected"]:
        return "L0_DIRECT", alias, [], token
    hits = model["reverse"].get(token, [])
    if not hits:
        return "UNVERIFIED_NO_MATCH", None, [], token
    derived = sorted({d for _, d in hits})
    ids = sorted({cid for cid, _ in hits})
    return ("L1_EXACT_DERIVED", derived[0], ids, token) if len(derived) == 1 else ("UNVERIFIED_AMBIGUOUS", None, ids, token)


def _ind_recount(path: Path, rel: str, sha: str, qid: str, domain: str, dataset: str, column: str,
                 model: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    csv.field_size_limit(max(csv.field_size_limit(), CSV_FIELD_LIMIT))
    tally = {k: 0 for k in ("EMPTY",) + CLASSES}
    witnesses, n = [], 0
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise SystemExit(f"independent check: column {column} missing from {rel}")
        for row_number, row in enumerate(reader, start=1):
            n += 1
            cls, derived, ids, token = _ind_classify(row.get(column, ""), model)
            tally[cls] += 1
            if cls == "L1_EXACT_DERIVED" and len(witnesses) < WITNESS_LIMIT_PER_SOURCE:
                witnesses.append({
                    "query_id": qid, "domain": domain, "source_dataset": dataset, "source_column": column,
                    "row_locator_sha256": hashlib.sha256(f"{sha}|{row_number}|{column}".encode("utf-8")).hexdigest(),
                    "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
                    "derived_target_sha256": hashlib.sha256(str(derived).encode("utf-8")).hexdigest(),
                    "carrier_ids": ids, "class": cls})
    nonempty = n - tally["EMPTY"]
    if sum(tally[c] for c in CLASSES) != nonempty:
        raise SystemExit("independent check: source accounting mismatch")
    return {
        "query_id": qid, "domain": domain, "source_dataset": dataset, "source_relative_path": rel,
        "source_sha256": sha, "source_column": column, "source_rows": n, "empty_rows": tally["EMPTY"],
        "nonempty_rows": nonempty, "L0_direct_rows": tally["L0_DIRECT"],
        "L1_exact_derived_rows": tally["L1_EXACT_DERIVED"], "unverified_ambiguous_rows": tally["UNVERIFIED_AMBIGUOUS"],
        "unverified_no_match_rows": tally["UNVERIFIED_NO_MATCH"],
        "L0_direct_rate_nonempty": tally["L0_DIRECT"] / nonempty if nonempty else 0.0,
        "L1_exact_derived_rate_nonempty": tally["L1_EXACT_DERIVED"] / nonempty if nonempty else 0.0,
        "nonempty_accounting_ok": 1,
    }, witnesses


def _ind_aggregate(sources: list[dict[str, Any]], queries, domain_scope):
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sources:
        grouped[s["query_id"]].append(s)
    qrows = []
    for q in sorted(queries, key=lambda r: r["query_id"]):
        rr = grouped[q["query_id"]]
        if len(rr) != int(float(q["positive_gold_source_mapping_count"])):
            raise SystemExit(f"independent check: source count mismatch for {q['query_id']}")
        s = lambda k: sum(int(r[k]) for r in rr)
        nonempty, l0, l1 = s("nonempty_rows"), s("L0_direct_rows"), s("L1_exact_derived_rows")
        amb, nom = s("unverified_ambiguous_rows"), s("unverified_no_match_rows")
        if l0 + l1 + amb + nom != nonempty:
            raise SystemExit(f"independent check: query accounting mismatch for {q['query_id']}")
        l1_rate = l1 / nonempty if nonempty else 0.0
        qrows.append({
            "query_id": q["query_id"], "domain": q["domain"], "target_attribute": q["target_attribute"],
            "source_columns": len(rr), "source_rows_sum": s("source_rows"), "empty_rows": s("empty_rows"),
            "nonempty_rows": nonempty, "L0_direct_rows": l0, "L1_exact_derived_rows": l1,
            "unverified_ambiguous_rows": amb, "unverified_no_match_rows": nom,
            "L0_direct_rate_nonempty": l0 / nonempty if nonempty else 0.0, "L1_exact_derived_rate_nonempty": l1_rate,
            "nontrivial_L1_min_rows": MIN_L1_ROWS, "nontrivial_L1_min_rate": MIN_L1_RATE,
            "nontrivial_L1_query": int(l1 >= MIN_L1_ROWS and l1_rate >= MIN_L1_RATE)})
    ds = {r["domain"]: r for r in domain_scope}
    drows, qualifying = [], []
    for d in DOMAINS:
        qs = [q for q in qrows if q["domain"] == d]
        registered = int(float(ds[d]["taxonomy_backed_query_count"]))
        if registered != len(qs):
            raise SystemExit(f"independent check: domain query count mismatch for {d}")
        nq = sum(q["nontrivial_L1_query"] for q in qs)
        if nq >= 1:
            qualifying.append(d)
        drows.append({
            "domain": d, "registered_queries": registered, "executed_queries": len(qs),
            "qualifying_nontrivial_L1_queries": nq, "domain_qualifies": int(nq >= 1),
            "L0_direct_rows": sum(q["L0_direct_rows"] for q in qs),
            "L1_exact_derived_rows": sum(q["L1_exact_derived_rows"] for q in qs),
            "nonempty_rows": sum(q["nonempty_rows"] for q in qs), "handling": ds[d]["handling"],
            "interpretation": ("AUDITED_REGISTERED_TRANSFORMS" if registered > 0
                               else "NO_REGISTERED_TRANSFORM_DO_NOT_INFER_NONDETERMINATION")})
    decision = {"breadth_qualification": "PASS" if len(qualifying) >= MIN_DISTINCT_DOMAINS else "FAIL",
                "qualifying_domains": qualifying, "qualifying_domain_count": len(qualifying)}
    return qrows, drows, decision


_IND_SOURCE_COUNTS = ["source_rows", "empty_rows", "nonempty_rows", "L0_direct_rows", "L1_exact_derived_rows",
                      "unverified_ambiguous_rows", "unverified_no_match_rows", "nonempty_accounting_ok"]
_IND_RATES = ["L0_direct_rate_nonempty", "L1_exact_derived_rate_nonempty"]
_IND_QUERY_COUNTS = ["source_columns", "source_rows_sum", "empty_rows", "nonempty_rows", "L0_direct_rows",
                     "L1_exact_derived_rows", "unverified_ambiguous_rows", "unverified_no_match_rows",
                     "nontrivial_L1_min_rows", "nontrivial_L1_query"]
_IND_DOMAIN_COUNTS = ["registered_queries", "executed_queries", "qualifying_nontrivial_L1_queries",
                      "domain_qualifies", "L0_direct_rows", "L1_exact_derived_rows", "nonempty_rows"]


def _ind_same_float(a: Any, b: Any) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-15)


def _ind_compare(scope: str, primary: dict[str, dict[str, str]], mine: dict[str, dict[str, Any]],
                 strings: list[str], counts: list[str], rates: list[str]) -> list[dict[str, Any]]:
    if set(primary) != set(mine):
        raise SystemExit(f"independent check: {scope} key sets differ")
    checks = []
    for key in sorted(primary):
        p, x = primary[key], mine[key]
        bad = [c for c in strings if str(p[c]) != str(x[c])]
        bad += [c for c in counts if int(float(p[c])) != int(x[c])]
        bad += [c for c in rates if not _ind_same_float(p[c], x[c])]
        checks.append({"scope": scope, "key": key, "primary_L1": int(float(p["L1_exact_derived_rows"])),
                       "independent_L1": int(x["L1_exact_derived_rows"]), "exact_match": int(not bad),
                       "note": ",".join(bad)})
    return checks


def _ind_witness_key(r: dict[str, Any]) -> tuple:
    ids = r["carrier_ids"]
    if isinstance(ids, str):
        ids = json.loads(ids)
    return (r["query_id"], r["domain"], r["source_dataset"], r["source_column"], r["row_locator_sha256"],
            r["token_sha256"], r["derived_target_sha256"], tuple(str(i) for i in ids), r["class"])


def _ind_repo_status(root: Path, git_dir: Path | None, freeze: dict[str, Any]) -> dict[str, bool]:
    if git_dir is not None:
        def git(*args: str) -> str:
            env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
            p = subprocess.run(["git", "-C", str(git_dir), *args], capture_output=True, timeout=600, env=env)
            return p.stdout.decode("utf-8").strip() if p.returncode == 0 else ""
        origin = normalize_remote(git("remote", "get-url", "origin")) == normalize_remote(OFFICIAL_REMOTE)
        commit = git("rev-parse", "HEAD") == PINNED_COMMIT
        tree = git("rev-parse", "HEAD^{tree}") == PINNED_TREE
    else:
        origin = commit = tree = freeze.get("domain_input_hashes") == PINNED_DOMAIN_INPUT_HASHES
    clean = int(freeze.get("inventory_verified", {}).get("mismatches", -1)) == 0
    return {"official_origin": origin, "commit_match": commit, "tree_match": tree, "clean": clean}


def verify_independently(root: Path, git_dir: Path | None, out_dir: Path,
                         primary_decision: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute the audit from the frozen registries and source tables and compare with the audit outputs."""
    t = {k: _ind_table(out_dir / OUT[k]) for k in (
        "query_registry", "carrier_registry", "source_scope", "domain_scope", "source_fields",
        "source_results", "query_results", "domain_results", "witnesses")}
    taxonomy = []
    with open(out_dir / OUT["taxonomy_rows"], "r", encoding="utf-8") as fh:
        taxonomy = [json.loads(line) for line in fh if line.strip()]
    with open(out_dir / OUT["freeze"], "r", encoding="utf-8") as fh:
        freeze = json.load(fh)
    with open(out_dir / OUT["protocol"], "r", encoding="utf-8") as fh:
        rule = json.load(fh)["qualification_rule"]
    if len(t["source_results"]) != PROTOCOL_COUNTS["source_mappings"] or len(t["query_results"]) != PROTOCOL_COUNTS["queries"] \
            or len(t["domain_results"]) != len(DOMAINS):
        raise SystemExit("independent check: audit output cardinality mismatch")

    models = _ind_models(t["query_registry"], t["carrier_registry"], taxonomy)
    tables: dict[tuple[str, str], dict[str, Any]] = {}
    for r in t["source_fields"]:
        key = (r["domain"], r["source_path"].rsplit("/", 1)[-1].rsplit(".", 1)[0])
        if key in tables:
            raise SystemExit(f"independent check: source {key} resolves twice")
        tables[key] = r
    sources, witnesses = [], []
    for s in sorted(t["source_scope"], key=lambda r: (r["query_id"], r["source_dataset"], r["source_column"])):
        f = tables.get((s["domain"], s["source_dataset"]))
        if f is None or f["format"] != "csv":
            raise SystemExit(f"independent check: source {s['source_dataset']} is not a registered CSV table")
        path = Path(root) / f["source_path"]
        if lio.sha256_file(path) != f["source_sha256"]:
            raise SystemExit(f"independent check: {f['source_path']} differs from the frozen inventory")
        if s["source_column"] not in json.loads(f["fields_json"]):
            raise SystemExit(f"independent check: column {s['source_column']} not declared for {f['source_path']}")
        res, wit = _ind_recount(path, f["source_path"], f["source_sha256"], s["query_id"], s["domain"],
                                s["source_dataset"], s["source_column"], models[s["query_id"]])
        sources.append(res)
        witnesses.extend(wit)
    qrows, drows, decision = _ind_aggregate(sources, t["query_registry"], t["domain_scope"])

    def keyed(rows, cols):
        return {"|".join(str(r[c]) for c in cols): r for r in rows}
    src_key = ["query_id", "domain", "source_dataset", "source_column"]
    checks = _ind_compare("source_mapping", keyed(t["source_results"], src_key), keyed(sources, src_key),
                          ["source_relative_path", "source_sha256"], _IND_SOURCE_COUNTS, _IND_RATES)
    checks += _ind_compare("query", keyed(t["query_results"], ["query_id"]), keyed(qrows, ["query_id"]),
                           ["domain", "target_attribute"], _IND_QUERY_COUNTS, _IND_RATES + ["nontrivial_L1_min_rate"])
    checks += _ind_compare("domain", keyed(t["domain_results"], ["domain"]), keyed(drows, ["domain"]),
                           ["handling", "interpretation"], _IND_DOMAIN_COUNTS, [])

    pw = sorted(_ind_witness_key(r) for r in t["witnesses"])
    iw = sorted(_ind_witness_key(r) for r in witnesses)
    checks.append({"scope": "witnesses", "key": "all_hashed_L1_witnesses", "primary_L1": len(pw),
                   "independent_L1": len(iw), "exact_match": int(pw == iw), "note": ""})

    thr = rule["nontrivial_L1_query_definition"]
    qual_rule = rule["breadth_qualification_rule"]
    qual_match = (
        primary_decision["breadth_qualification"] == decision["breadth_qualification"]
        and list(primary_decision["qualifying_domains"]) == decision["qualifying_domains"]
        and int(primary_decision["qualifying_domain_count"]) == decision["qualifying_domain_count"]
        and int(primary_decision["required_distinct_domains"]) == MIN_DISTINCT_DOMAINS
        and int(qual_rule["minimum_distinct_domains_with_at_least_one_nontrivial_L1_query"]) == MIN_DISTINCT_DOMAINS
        and int(primary_decision["per_query_threshold"]["minimum_L1_exact_derived_rows"]) == MIN_L1_ROWS
        and int(thr["minimum_L1_exact_derived_rows"]) == MIN_L1_ROWS
        and _ind_same_float(primary_decision["per_query_threshold"]["minimum_L1_rate_among_nonempty_mapped_values"], MIN_L1_RATE)
        and _ind_same_float(thr["minimum_L1_rate_among_nonempty_mapped_values"], MIN_L1_RATE))
    checks.append({"scope": "qualification", "key": "breadth_qualification",
                   "primary_L1": primary_decision["breadth_qualification"],
                   "independent_L1": decision["breadth_qualification"], "exact_match": int(qual_match),
                   "note": ""})

    total_keys = {"nonempty_values": "nonempty_rows", "L0_direct_rows": "L0_direct_rows",
                  "L1_exact_derived_rows": "L1_exact_derived_rows",
                  "unverified_ambiguous_rows": "unverified_ambiguous_rows",
                  "unverified_no_match_rows": "unverified_no_match_rows"}
    primary_totals = {k: sum(int(r[c]) for r in t["source_results"]) for k, c in total_keys.items()}
    totals = {k: sum(int(r[c]) for r in sources) for k, c in total_keys.items()}
    checks.append({"scope": "global", "key": "global_classification_totals",
                   "primary_L1": primary_totals["L1_exact_derived_rows"],
                   "independent_L1": totals["L1_exact_derived_rows"], "exact_match": int(totals == primary_totals),
                   "note": ""})

    failed = [c for c in checks if int(c["exact_match"]) != 1]
    inputs = {OUT[k]: lio.sha256_file(out_dir / OUT[k]) for k in (
        "query_registry", "carrier_registry", "source_scope", "domain_scope", "source_fields", "taxonomy_rows",
        "protocol", "source_results", "query_results", "domain_results", "witnesses")}
    verification = {
        "step": STEP,
        "schema_version": SCHEMA_VERSION,
        "inputs_sha256": inputs,
        "status": "PASS" if not failed else "FAIL",
        "independent_implementation": True,
        "madi_commit": PINNED_COMMIT,
        "madi_tree": PINNED_TREE,
        "repo": _ind_repo_status(root, git_dir, freeze),
        "recomputed": {
            "queries": len(qrows), "source_mappings": len(sources),
            "exact_carriers": sum(m["carriers"] for m in models.values()),
            "hashed_witnesses": len(witnesses), "global_totals": totals,
            "qualifying_domains": decision["qualifying_domains"],
            "breadth_qualification": decision["breadth_qualification"],
        },
        "checks_total": len(checks),
        "checks_failed": len(failed),
        "failed": failed,
        "raw_external_values_written_to_outputs": False,
    }
    return checks, verification


# ---------------------------------------------------------------------------
# Comparison with the benchmark's normalization gold. The audit is applied unchanged (classify_token with the
# models of query_models, and the second implementation with its own models); no rule, alias, carrier or
# threshold is added, and only counts are written.
# ---------------------------------------------------------------------------

GOLD_SPLITS = ("test", "validation")
_GOLD_KEY = {"L0_DIRECT": "L0_direct", "L1_EXACT_DERIVED": "L1_exact_derived",
             "UNVERIFIED_AMBIGUOUS": "unverified_ambiguous", "UNVERIFIED_NO_MATCH": "unverified_no_match",
             "EMPTY": "empty"}


def gold_files() -> list[str]:
    """Normalization gold files of the taxonomy-bound domains (part of the pinned inventory)."""
    return [f"{_scope_prefix(d)}normalization/{s}.csv" for d in TAXONOMY_DOMAINS for s in GOLD_SPLITS]


def gold_value_set(expected: Any, array: bool, norm) -> set[str]:
    """Normalized gold values of one cell: the members of a JSON array for array targets, else the value."""
    if array:
        v = json.loads(str(expected))
        v = v if isinstance(v, list) else [v]
    else:
        v = [expected]
    return {norm(x) for x in v if norm(x)}


def run_gold_check(src: BenchmarkSource, out_dir: Path) -> dict[str, Any] | None:
    """Classify every gold cell of the audited source columns and compare the derived value with the gold value."""
    rels = gold_files()
    if any(rel in src.missing for rel in rels):
        return None
    queries = read_csv_rows(out_dir / OUT["query_registry"])
    carriers = read_csv_rows(out_dir / OUT["carrier_registry"])
    scope = {(str(s["query_id"]), str(s["source_dataset"]), str(s["source_column"]))
             for s in read_csv_rows(out_dir / OUT["source_scope"])}
    taxonomy = read_jsonl(out_dir / OUT["taxonomy_rows"])
    primary = query_models(queries, carriers, taxonomy)
    second = _ind_models(queries, carriers, taxonomy)
    by_attr = {(str(q["domain"]), str(q["target_attribute"])): q for q in queries}
    counts = {impl: {str(q["query_id"]): defaultdict(int) for q in queries} for impl in ("primary", "second")}
    files = []
    for rel in rels:
        path = src.root / rel
        domain = rel.split("/")[1]
        files.append({"path": rel, "sha256": lio.sha256_file(path)})
        for r in read_csv_rows(path):
            q = by_attr.get((domain, str(r["attribute"])))
            if q is None or (str(q["query_id"]), str(r["source"]), str(r["source_column"])) not in scope:
                continue
            qid = str(q["query_id"])
            array = json.loads(str(q["target_type_json"])) == "array"
            for impl, classify, models, norm in (("primary", classify_token, primary, normalize_token),
                                                 ("second", _ind_classify, second, _ind_norm)):
                c = counts[impl][qid]
                c["gold_cells"] += 1
                cls, value, _, _ = classify(r["raw_value"], models[qid])
                c[_GOLD_KEY[cls]] += 1
                if cls.startswith("UNVERIFIED") and str(r["category"]) == "normalization":
                    c["unverified_gold_normalization"] += 1
                if value is None:
                    continue
                gold = gold_value_set(r["expected_value"], array, norm)
                c["classified"] += 1
                if value in gold:
                    c["consistent"] += 1
                    c["complete_set_equal" if gold == {value} else "component_only"] += 1
                else:
                    c["contradictions"] += 1
    rows = []
    for q in sorted(queries, key=lambda x: str(x["query_id"])):
        qid = str(q["query_id"])
        a, b = counts["primary"][qid], counts["second"][qid]
        row = {"query_id": qid, "domain": q["domain"], "target_attribute": q["target_attribute"]}
        row.update({k: a.get(k, 0) for k in FIELDS["gold_check"][3:-1]})
        row["independent_match"] = int(all(a.get(k, 0) == b.get(k, 0) for k in FIELDS["gold_check"][3:-1]))
        rows.append(row)
    protocol = {
        "step": STEP, "schema_version": SCHEMA_VERSION,
        "role": "comparison of the external audit with the benchmark's normalization gold",
        "classifier": "classify_token with query_models; second implementation _ind_classify with _ind_models",
        "scope": "gold rows whose attribute is a registered query and whose source column is in source_scope.csv",
        "agreement": {"array": "derived value is a member of the normalized gold value set",
                      "string": "derived value equals the normalized gold value"},
        "complete_set_equal": "the normalized gold value set equals the derived value alone",
        "gold_files": files, "rules_added": False, "aliases_added": False, "carriers_added": False,
        "thresholds_added": False, "raw_external_values_written_to_outputs": False,
        "madi_commit": PINNED_COMMIT, "madi_tree": PINNED_TREE,
    }
    return {"rows": rows, "protocol": protocol}



# ---------------------------------------------------------------------------
# Step driver
# ---------------------------------------------------------------------------

def run(checkout: Path, git_dir: Path | None, out_dir: Path, allow_partial: bool = False) -> dict[str, Any]:
    """Freeze the pinned benchmark files, extract authorities, freeze the protocol, audit, verify, and compare the
    audit with the normalization gold."""
    out_dir = Path(out_dir)
    src = BenchmarkSource(checkout, git_dir, allow_partial=allow_partial)

    inv = inventory_rows(src)
    summaries = domain_summaries(src)
    bad = {s["domain"]: s["missing_required"] for s in summaries if s["missing_required"]}
    lfs = sum(int(s["lfs_pointer_count"] or 0) for s in summaries)
    if bad or lfs:
        raise SystemExit(f"benchmark authority closure failed: missing={bad}, lfs_pointers={lfs}")
    lio.write_csv(out_dir / OUT["inventory"], inv, FIELDS["inventory"])
    lio.write_csv(out_dir / OUT["domain_summary"], summaries, FIELDS["domain_summary"])
    lio.write_json(out_dir / OUT["freeze"], source_freeze_object(src, inv, license_files(src)))

    ext = extract_authorities(src)
    for key in ("target_attributes", "gold_mappings", "unmapped", "taxonomy_summary", "source_fields",
                "source_metadata", "labeled_headers", "transform_authorities"):
        lio.write_csv(out_dir / OUT[key], ext[key], FIELDS[key])
    write_jsonl(out_dir / OUT["taxonomy_rows"], ext["taxonomy_rows"])

    upstream = ("target_attributes", "gold_mappings", "transform_authorities", "taxonomy_rows")
    reg = build_protocol(read_csv_rows(out_dir / OUT["target_attributes"]), read_csv_rows(out_dir / OUT["gold_mappings"]),
                         read_csv_rows(out_dir / OUT["transform_authorities"]), read_jsonl(out_dir / OUT["taxonomy_rows"]))
    for key, rows in (("query_registry", reg["queries"]), ("carrier_registry", reg["carriers"]),
                      ("noncarrier_registry", reg["noncarriers"]), ("source_scope", reg["source_scope"]),
                      ("domain_scope", reg["domain_scope"])):
        lio.write_csv(out_dir / OUT[key], rows, FIELDS[key])
    lio.write_json(out_dir / OUT["protocol"], protocol_object(
        reg, ext["taxonomy_authority_by_domain"], {OUT[k]: lio.sha256_file(out_dir / OUT[k]) for k in upstream}))

    audit = run_audit(src, out_dir)
    checks, verification = verify_independently(src.root, src.git_dir, out_dir, audit["decision"])
    verification["qualification_decision"] = audit["decision"]
    verification["audit_integrity"] = audit["integrity"]
    lio.write_csv(out_dir / OUT["checks"], checks, FIELDS["checks"])
    lio.write_json(out_dir / OUT["verification"], verification)
    if verification["status"] != "PASS":
        raise SystemExit(f"independent verification failed: {verification['checks_failed']} checks differ")
    gold = run_gold_check(src, out_dir)
    if gold is not None:
        if any(r["independent_match"] != 1 for r in gold["rows"]):
            raise SystemExit("the second implementation of the gold comparison disagrees")
        lio.write_csv(out_dir / OUT["gold_check"], gold["rows"], FIELDS["gold_check"])
        lio.write_json(out_dir / OUT["gold_protocol"], gold["protocol"])
    return {"complete": src.complete, "missing": len(src.missing), "skipped_rows": len(ext["skipped"]),
            "checks": len(checks), "decision": audit["decision"]["breadth_qualification"],
            "qualifying_domains": audit["decision"]["qualifying_domains"],
            "gold_cells": None if gold is None else sum(r["gold_cells"] for r in gold["rows"]),
            "gold_contradictions": None if gold is None else sum(r["contradictions"] for r in gold["rows"])}
