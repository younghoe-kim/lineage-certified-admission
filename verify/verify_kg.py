# SPDX-License-Identifier: MIT
"""Independent check of the link-prediction study (extended audit details Section C).

This file imports nothing from the pipeline library (src/lca). It re-reads the pinned files of
data/external/kg/sources.json, rebuilds the reflection of the WordNet table from data/external/kg/wordnet_reflexive.yaml
with its own code, recounts every quantity of outputs/kg/ and writes outputs/checks/kg_verification.json. A WordNet
synset is an offset with a part of speech; the WordNet check gives each entity of a certified pair one part of
speech under which the triple and its training witness are pointers between the same two synsets."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
KG = REPO / "data" / "external" / "kg"
EXPECTED = {("WN18", "test"): (5000, 4504, 4658), ("WN18", "valid"): (5000, 4512, 4671),
            ("WN18RR", "test"): (3134, 1052, 1052), ("WN18RR", "valid"): (3034, 1046, 1046)}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pinned(root: Path, e: dict) -> bytes:
    b = (root / e["path"]).read_bytes()
    if sha(b) != e["sha256"]:
        sys.exit(f"SHA-256 mismatch for {e['path']}")
    return b


def triples(text: str) -> list[tuple[str, str, str]]:
    return [tuple(x.strip() for x in line.split("\t")) for line in text.splitlines() if line.strip()]


def reflections(doc: dict, symmetrize_domain: bool) -> dict[str, set[str]]:
    """relation symbol r -> symbols P such that a pointer of type P from t to h declares (h, r, t)."""
    sy = doc["pointer_symbols"]
    out: dict[str, set[str]] = {}
    for a, b in doc["rows"]:
        for x, y in zip(sy[a], sy[b]):
            out.setdefault(y, set()).add(x)
            if symmetrize_domain and a == "Domain of synset":
                out.setdefault(x, set()).add(y)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", type=Path, default=REPO / "outputs")
    ap.add_argument("--kg", type=Path, default=KG / "files")
    a = ap.parse_args()
    src = json.loads((KG / "sources.json").read_text(encoding="utf-8"))
    ent = {e["key"]: e for e in src["files"]}
    doc = yaml.safe_load((KG / "wordnet_reflexive.yaml").read_text(encoding="utf-8"))
    rel_sym = doc["wn18_relations"]
    data = {"WN18": {s: triples(pinned(a.kg, ent[f"wn18_{s}"]).decode("utf-8")) for s in ("train", "valid", "test")}}
    with tarfile.open(fileobj=io.BytesIO(pinned(a.kg, ent["wn18rr_archive"])), mode="r:gz") as tf:
        data["WN18RR"] = {n[:-4]: triples(tf.extractfile(tf.getmember(n)).read().decode("utf-8"))
                          for n in ("train.txt", "valid.txt", "test.txt")}
    tree = ast.parse(pinned(a.kg, ent["wn18rr_script"]).decode("utf-8"))
    removed = [ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "predicates_to_remove" for t in n.targets)][0]

    checks: list[dict] = []

    def check(name: str, expected, got) -> None:
        checks.append({"check": name, "expected": expected, "independent_value": got, "pass": expected == got})

    def reader(rel: str) -> list[dict[str, str]]:
        with (a.outputs / "kg" / rel).open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    summary = {(r["release"], r["split"]): r for r in reader("release_summary.csv")}
    breakdown = {(r["release"], r["split"], r["relation"]): r for r in reader("relation_breakdown.csv")}
    directed, symm = reflections(doc, False), reflections(doc, True)
    certified = []
    for (release, split), (n_exp, l1_exp, l1s_exp) in sorted(EXPECTED.items()):
        train = set(data[release]["train"])
        held = data[release][split]
        sym_rel = {s: r for r, s in rel_sym.items()}

        def l1_of(refl: dict[str, set[str]]) -> list[tuple[str, str, str] | None]:
            out = []
            for h, r, t in held:
                hits = [(t, sym_rel[p], h) for p in sorted(refl.get(rel_sym[r], ())) if p in sym_rel
                        and (t, sym_rel[p], h) in train]
                out.append(hits[0] if hits else None)
            return out

        d, s = l1_of(directed), l1_of(symm)
        got = (len(held), sum(x is not None for x in d), sum(x is not None for x in s))
        row = summary[(release, split)]
        shipped = (int(row["held_out"]), int(row["l1"]), int(row["l1_symmetrized_domain"]))
        check(f"kg.counts.{release}.{split}", list((n_exp, l1_exp, l1s_exp)), list(got))
        check(f"kg.shipped.{release}.{split}", list(got), list(shipped))
        check(f"kg.identity_overlap.{release}.{split}", 0, sum(x in train for x in held))
        check(f"kg.identity_overlap_shipped.{release}.{split}", 0, int(row["identity_overlap_with_train"]))
        per = {}
        for x, w in zip(held, d):
            per.setdefault(x[1], [0, 0])
            per[x[1]][0] += 1
            per[x[1]][1] += int(w is not None)
        shipped_per = {k[2]: [int(v["held_out"]), int(v["l1"])] for k, v in breakdown.items()
                       if k[0] == release and k[1] == split}
        check(f"kg.relations.{release}.{split}", per, shipped_per)
        check(f"kg.also_see_uncertified.{release}.{split}", 0, per.get("_also_see", [0, 0])[1])
        certified += [(x, w) for x, w in zip(held, d) if w is not None]

    # construction of WN18RR
    filtered = {s: [x for x in data["WN18"][s] if x[1] not in removed] for s in ("train", "valid", "test")}
    check("kg.wn18rr_equals_filtered_wn18", True, all(filtered[s] == data["WN18RR"][s] for s in filtered))
    check("kg.removed_relations", 7, len(set(removed)))
    fc = json.loads((a.outputs / "kg" / "wn18rr_filter_check.json").read_text(encoding="utf-8"))
    check("kg.removed_relations_shipped", sorted(set(removed)), fc["removed_relations"])

    # WordNet 3.0 pointers with the part of speech of both ends (file of the source, field of the target)
    pointers, full, parts = set(), set(), {}
    with zipfile.ZipFile(io.BytesIO(pinned(a.kg, ent["wordnet"]))) as z:
        for pos, letter in (("noun", "n"), ("verb", "v"), ("adj", "a"), ("adv", "r")):
            for line in z.read(f"wordnet/data.{pos}").decode("latin-1").splitlines():
                if not line or line[0] == " ":
                    continue
                f = line.split()
                parts.setdefault(f[0], set()).add(letter)
                k = 4 + 2 * int(f[3], 16)
                for i in range(int(f[k])):
                    sym, off, tpos = f[k + 1 + 4 * i], f[k + 2 + 4 * i], f[k + 3 + 4 * i]
                    pointers.add((f[0], sym, off))
                    full.add((f[0], letter, sym, off, {"s": "a"}.get(tpos, tpos)))
    tags = {"NN": "n", "VB": "v", "JJ": "a", "RB": "r"}
    defpos = {}
    for line in pinned(a.kg, ent["wn18_definitions"]).decode("utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) >= 2:
            w = cols[1].strip().split("_")
            defpos[cols[0].strip()] = tags.get(w[-2], "?") if len(w) >= 3 else "?"

    def wn(x: tuple[str, str, str]) -> bool:
        return (x[0], rel_sym[x[1]], x[2]) in pointers

    raw = json.loads((a.outputs / "kg" / "raw_authority.json").read_text(encoding="utf-8"))
    check("kg.raw.certified", [11114, 11114, 11114],
          [len(certified), sum(wn(x) for x, _ in certified), sum(wn(w) for _, w in certified)])
    check("kg.raw.certified_shipped", [len(certified)] * 3,
          [raw["certified_triples"], raw["certified_targets_that_are_wordnet_pointers"],
           raw["certified_witnesses_that_are_wordnet_pointers"]])
    allwn = [x for s in ("train", "valid", "test") for x in data["WN18"][s]]
    check("kg.raw.wn18", [151442, 151436], [len(allwn), sum(wn(x) for x in allwn)])

    def pair_ok(x, w) -> bool:
        return any((x[0], p, rel_sym[x[1]], x[2], q) in full and (x[2], q, rel_sym[w[1]], x[0], p) in full
                   for p in parts.get(x[0], ()) for q in parts.get(x[2], ()))

    def by_definitions(x) -> bool:
        return (x[0], defpos.get(x[0]), rel_sym[x[1]], x[2], defpos.get(x[2])) in full

    shared = {o for o, ps in parts.items() if len(ps) >= 2}
    ents = {e for x in allwn for e in (x[0], x[2])}
    seen = {}
    for x in allwn:
        for e, side in ((x[0], 0), (x[2], 1)):
            if e not in shared:
                continue
            fits = {(p, q)[side] for p in parts.get(x[0], ()) for q in parts.get(x[2], ())
                    if (x[0], p, rel_sym[x[1]], x[2], q) in full}
            if len(fits) == 1:
                seen.setdefault(e, set()).update(fits)
    misses = [(x, w) for x, w in certified if not (by_definitions(x) and by_definitions(w))]
    pos_got = [sum(pair_ok(x, w) for x, w in certified),
               sum(not ({x[0], x[2]} & shared) for x, _ in certified),
               len(shared), len(ents), len(ents & shared), sum(len(v) >= 2 for v in seen.values()),
               len(certified) - len(misses), len(misses), all({x[0], x[2]} & shared for x, _ in misses)]
    check("kg.raw.one_synset_pair", 11114, pos_got[0])
    check("kg.raw.unshared_offsets", 10946, pos_got[1])
    check("kg.raw.shared_offsets_and_entities", [282, 40943, 232, 161], pos_got[2:6])
    check("kg.raw.definitions_pos", [11033, 81, True], pos_got[6:9])
    dp = raw["definitions_file_pos"]
    check("kg.raw.pos_shipped", pos_got,
          [raw["certified_pairs_on_one_synset_pair"], raw["certified_pairs_without_shared_offsets"],
           raw["wordnet_offsets_in_two_or_more_pos_files"], raw["wn18_entities"],
           raw["wn18_entities_on_shared_offsets"], raw["wn18_entities_with_triples_from_two_synsets"],
           dp["certified_pairs_matching"], dp["others"], dp["others_all_on_shared_offsets"]])
    check("kg.raw.wn18_shipped", [len(allwn), sum(wn(x) for x in allwn)],
          [raw["wn18_triples"], raw["wn18_triples_that_are_wordnet_pointers"]])
    audit = json.loads((a.outputs / "kg" / "kg_audit.json").read_text(encoding="utf-8"))
    check("kg.no_nondetermined", False, audit["nondetermined_issued"])

    failed = [c["check"] for c in checks if not c["pass"]]
    out = a.outputs / "checks" / "kg_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"step": "verify_kg", "schema_version": "1", "pipeline_library_imported": False,
                               "checks_total": len(checks), "checks_failed": len(failed), "all_pass": not failed,
                               "failed_checks": failed, "checks": checks},
                              sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"verify_kg: checks={len(checks)} failed={len(failed)}")
    if failed:
        sys.exit("kg verification failed: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
