# SPDX-License-Identifier: MIT
"""Fetch the pinned MaDI-Bench commit, or check an existing copy, and verify the audited files against FETCH.md."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca import external as ext  # noqa: E402

LIST_BEGIN = "<!-- sha256-list:begin -->"
LIST_END = "<!-- sha256-list:end -->"
LIST_LINE = re.compile(r"^([0-9a-f]{64})  (use cases/.+)$")


def pinned_list(fetch_md: Path) -> dict[str, str]:
    """Relative path -> SHA-256 from the list block of FETCH.md."""
    text = fetch_md.read_text(encoding="utf-8")
    if LIST_BEGIN not in text or LIST_END not in text:
        raise SystemExit(f"{fetch_md.name} has no SHA-256 list block")
    block = text.split(LIST_BEGIN, 1)[1].split(LIST_END, 1)[0]
    out = {}
    for line in block.splitlines():
        m = LIST_LINE.match(line)
        if m:
            out[m.group(2)] = m.group(1)
    if len(out) != ext.PINNED_INVENTORY_FILES:
        raise SystemExit(f"{fetch_md.name} lists {len(out)} files, expected {ext.PINNED_INVENTORY_FILES}")
    return out


def git(*args: str) -> None:
    proc = subprocess.run(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600)
    if proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.decode('utf-8', 'replace').strip()}")


def clone(dest: Path, remote: str, sparse: bool) -> None:
    """Clone without file contents, then check out the pinned commit (only the audited folders when sparse)."""
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"destination is not empty: {dest} (use --checkout to verify an existing copy)")
    official = ext.normalize_remote(remote) == ext.normalize_remote(ext.OFFICIAL_REMOTE)
    git("clone", "--quiet", "--no-checkout", *(["--filter=blob:none"] if official else []), remote, str(dest))
    if not official:
        git("-C", str(dest), "remote", "set-url", "origin", ext.OFFICIAL_REMOTE)
    probe = subprocess.run(["git", "-C", str(dest), "cat-file", "-e", ext.PINNED_COMMIT + "^{commit}"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if probe.returncode != 0:
        git("-C", str(dest), "fetch", "--quiet", remote, ext.PINNED_COMMIT)
    if sparse:
        git("-C", str(dest), "config", "core.sparseCheckout", "true")
        patterns = "".join(f"/use cases/{d}/{ext.TIER}/input/\n" for d in ext.DOMAINS)
        (dest / ".git" / "info").mkdir(parents=True, exist_ok=True)
        (dest / ".git" / "info" / "sparse-checkout").write_text(patterns, encoding="utf-8")
    git("-C", str(dest), "checkout", "--quiet", "--detach", ext.PINNED_COMMIT)


def verify(checkout: Path, git_dir: Path | None, pinned: dict[str, str]) -> int:
    """Check origin/commit/tree (with a Git directory), every listed file and the per-domain hashes."""
    checkout = checkout.resolve()
    print(f"benchmark      : {ext.BENCHMARK} {ext.OFFICIAL_REPOSITORY}")
    print(f"pinned commit  : {ext.PINNED_COMMIT}  tree {ext.PINNED_TREE}")
    blobs = None
    if git_dir is not None:
        meta = ext.read_git_metadata(git_dir)
        blobs = ext.pinned_tree_blobs(git_dir)
        scope = {p for p in blobs if any(p.startswith(f"use cases/{d}/{ext.TIER}/input/") for d in ext.DOMAINS)}
        if scope != set(pinned):
            raise SystemExit("the FETCH.md list does not match the audited scope of the pinned tree")
        print(f"git directory  : origin, commit and tree match ({meta['commit_time']})")
    else:
        print("git directory  : none (files are checked against FETCH.md and the per-domain hashes only)")
    records, missing, differ = {}, [], []
    for rel, sha in sorted(pinned.items()):
        p = checkout / rel
        if not p.is_file():
            missing.append(rel)
            continue
        rec = ext.hash_file(p)
        if rec["sha256"] != sha or (blobs is not None and rec["blob_id"] != blobs[rel]):
            differ.append(rel)
            continue
        records[rel] = rec
    print(f"files          : {len(records)} match, {len(differ)} differ, {len(missing)} missing (of {len(pinned)})")
    for d in ext.DOMAINS:
        files = [r for r in pinned if r.startswith(f"use cases/{d}/{ext.TIER}/input/")]
        if all(r in records for r in files):
            got = ext.domain_input_hash([(r, records[r]["size_bytes"], records[r]["sha256"]) for r in files])
            status = "match" if got == ext.PINNED_DOMAIN_INPUT_HASHES[d] else "DIFFERS"
        else:
            status = f"incomplete ({sum(r in records for r in files)}/{len(files)} files)"
        print(f"domain {d:10s}: input hash {status}")
        if status == "DIFFERS":
            differ.append(f"use cases/{d}/{ext.TIER}/input/ (domain hash)")
    if missing:
        groups = Counter(r.rsplit("/", 1)[0] for r in missing)
        print("missing by folder:")
        for folder, n in sorted(groups.items()):
            print(f"  {n:3d}  {folder}/")
        for rel in missing:
            print(f"  missing  {rel}")
    for rel in differ:
        print(f"  differs  {rel}")
    ok = not missing and not differ
    print("result         : " + ("all pinned files verified" if ok else "NOT VERIFIED"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=HERE / "MaDI-Bench", help="where to clone the pinned commit")
    ap.add_argument("--checkout", type=Path, default=None,
                    help="verify this existing copy instead of cloning (folder that contains 'use cases/')")
    ap.add_argument("--git-dir", type=Path, default=None,
                    help="Git repository holding the pinned commit (default: the checkout when it has .git)")
    ap.add_argument("--remote", default=ext.OFFICIAL_REMOTE,
                    help="clone from this URL instead (a mirror); commit, tree and file ids are still verified")
    ap.add_argument("--full", action="store_true", help="check out the whole commit, not only the audited folders")
    ap.add_argument("--list", type=Path, default=HERE / "FETCH.md", help="file holding the pinned SHA-256 list")
    a = ap.parse_args()
    pinned = pinned_list(a.list)
    if a.checkout is None:
        clone(a.dest, a.remote, sparse=not a.full)
        checkout = a.dest
    else:
        checkout = a.checkout
    git_dir = a.git_dir if a.git_dir is not None else (checkout if (checkout / ".git").exists() else None)
    return verify(checkout, git_dir, pinned)


if __name__ == "__main__":
    raise SystemExit(main())
