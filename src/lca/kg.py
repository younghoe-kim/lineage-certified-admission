# SPDX-License-Identifier: MIT
"""Link-prediction study (paper Sec. VI-C, extended audit details Section C).

A link-prediction release splits a set of (head, relation, tail) triples into training, validation and test triples.
For a held-out triple (h, r, t), the audit asks whether the training triples and the registry entail it; the split
withholds each held-out triple by its identity and admits the rest of the graph to training, an analogue of
identity-based admission for splits (not a feature-column view). The registry is the table of reflexive pointers of the
WordNet documentation (data/external/kg/wordnet_reflexive.yaml), read row by row: a pointer of type P from x to y
declares a pointer of type reflect(P) from y to x. A held-out triple is L1 (UNSAFE) when training contains a triple
(t, P, h) with reflect(P) = r, and UNVERIFIED otherwise. A missing partner does not show that a triple is
undetermined, so the study issues no NONDETERMINED verdict.

The step reads the WN18 splits, the WN18RR archive, the construction script released with WN18RR (as text: its
list literal of removed relations is parsed, the script is never run), the WordNet 3.0 database files and the WN18
definitions file (part of speech of each entity, for one diagnostic). Every file is checked against
data/external/kg/sources.json before it is read. Only aggregates are written.

A WordNet synset is an offset together with a part of speech. WN18 identifiers are offsets only, and WN18 merges
the synsets that share an offset across parts of speech, so the WordNet check gives each entity of a certified
triple and its training witness one part of speech under which both are pointers between the same two synsets."""
from __future__ import annotations

import ast
import hashlib
import io
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import yaml

from .io import require_sha, write_csv, write_json

KG_DIR = Path(__file__).resolve().parents[2] / "data" / "external" / "kg"
SOURCES, REGISTRY = KG_DIR / "sources.json", KG_DIR / "wordnet_reflexive.yaml"
SPLITS = ("train", "valid", "test")
HELD_OUT = ("valid", "test")
RELEASES = ("WN18", "WN18RR")
WORDNET_DATA = ("wordnet/data.noun", "wordnet/data.verb", "wordnet/data.adj", "wordnet/data.adv")
FILE_POS = {"wordnet/data.noun": "n", "wordnet/data.verb": "v", "wordnet/data.adj": "a", "wordnet/data.adv": "r"}
DEFINITION_POS = {"NN": "n", "VB": "v", "JJ": "a", "RB": "r"}
SCRIPT_LIST = "predicates_to_remove"
SUMMARY_FIELDS = ["release", "split", "held_out", "identity_overlap_with_train", "l1", "unverified",
                  "l1_symmetrized_domain", "symmetrized_increment"]
RELATION_FIELDS = ["release", "split", "relation", "pointer_symbol", "declared", "held_out", "l1",
                   "l1_symmetrized_domain"]


class KGError(RuntimeError):
    pass


def load_sources(path: Path = SOURCES) -> dict:
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_registry(path: Path = REGISTRY) -> dict:
    """Symbol-level reflection of the table, read as written (directed) and with the Domain row read both ways."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if doc.get("directed") is not True:
        raise KGError("the registry must be read row by row")
    syms = doc["pointer_symbols"]
    reflect: dict[str, str] = {}
    for src, dst in doc["rows"]:
        a, b = syms[src], syms[dst]
        if len(a) != len(b):
            raise KGError(f"row {src} -> {dst} pairs pointer types with different subtypes")
        for x, y in zip(a, b):
            if reflect.get(x, y) != y:
                raise KGError(f"pointer symbol {x} has two reflections")
            reflect[x] = y
    symmetrized = dict(reflect)
    for x, y in zip(syms["Domain of synset"], syms["Member of Domain"]):
        symmetrized.setdefault(y, x)
    rel_sym = dict(doc["wn18_relations"])
    undeclared = {s for v in doc["not_declared"].values() for s in v}
    if undeclared & set(reflect):
        raise KGError("a pointer type is both declared and undeclared")
    return {"doc": doc, "reflect": reflect, "symmetrized": symmetrized, "rel_sym": rel_sym,
            "sym_rel": {s: r for r, s in rel_sym.items()}, "undeclared": undeclared}


def _pinned(root: Path, entry: dict) -> bytes:
    return require_sha(root / entry["path"], entry["sha256"]).read_bytes()


def parse_triples(text: str) -> list[tuple[str, str, str]]:
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise KGError("a triple line does not have three tab-separated fields")
        out.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return out


def read_archive(raw: bytes, members: list[str]) -> dict[str, list[tuple[str, str, str]]]:
    """Reads the split files of a tar.gz archive in memory; nothing is extracted to disk."""
    out = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        names = {m.name: m for m in tf.getmembers()}
        for name in members:
            m = names.get(name)
            if m is None or not m.isfile():
                raise KGError(f"archive member {name} is missing or not a regular file")
            out[name.rsplit(".", 1)[0]] = parse_triples(tf.extractfile(m).read().decode("utf-8"))
    return out


def script_removed_relations(text: str) -> list[str]:
    """The list literal assigned to predicates_to_remove in the construction script (parsed, never executed)."""
    tree = ast.parse(text)
    found = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == SCRIPT_LIST for t in n.targets)]
    if len(found) != 1:
        raise KGError(f"expected one assignment to {SCRIPT_LIST}")
    value = ast.literal_eval(found[0])
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise KGError(f"{SCRIPT_LIST} is not a list of relation names")
    return value


def wordnet_index(raw_zip: bytes) -> tuple[set[tuple[str, str, str, str, str]], dict[str, set[str]]]:
    """Every pointer of the WordNet database files as (source offset, source part of speech, pointer symbol, target
    offset, target part of speech), and the parts of speech whose data file holds each offset.

    A synset is an offset together with a part of speech; adjective satellites (s) count as adjectives (a)."""
    pointers: set[tuple[str, str, str, str, str]] = set()
    pos_of: dict[str, set[str]] = {}
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        for name in WORDNET_DATA:
            fpos = FILE_POS[name]
            for line in z.read(name).decode("latin-1").split("\n"):
                if not line or line.startswith("  "):
                    continue
                tok = line.split(" ")
                pos_of.setdefault(tok[0], set()).add(fpos)
                i = 4 + 2 * int(tok[3], 16)
                for _ in range(int(tok[i])):
                    tpos = "a" if tok[i + 3] == "s" else tok[i + 3]
                    pointers.add((tok[0], fpos, tok[i + 1], tok[i + 2], tpos))
                    i += 4
    return pointers, pos_of


def wordnet_pointers(raw_zip: bytes) -> set[tuple[str, str, str]]:
    """(source offset, pointer symbol, target offset) of every pointer: the identity that WN18 uses."""
    return {(s, sym, t) for s, _, sym, t, _ in wordnet_index(raw_zip)[0]}


def definitions_pos(text: str) -> dict[str, str]:
    """Part of speech that the WN18 definitions file gives each entity (the tag of its word, as in __stool_NN_2)."""
    out = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0].strip():
            continue
        word = parts[1].strip().rsplit("_", 2)
        out[parts[0].strip()] = DEFINITION_POS.get(word[-2], "?") if len(word) == 3 else "?"
    return out


def same_synset_pair(target: tuple[str, str, str], witness: tuple[str, str, str], rel_sym: dict[str, str],
                     pointers: set[tuple[str, str, str, str, str]], pos_of: dict[str, set[str]]) -> bool:
    """True if one part of speech for each of the two entities makes the target triple and its training witness
    WordNet pointers in opposite directions between the same two synsets."""
    (h, r, t), (_, p, _) = target, witness
    return any((h, ph, rel_sym[r], t, pt) in pointers and (t, pt, rel_sym[p], h, ph) in pointers
               for ph in sorted(pos_of.get(h, ())) for pt in sorted(pos_of.get(t, ())))


def classify(held: list[tuple[str, str, str]], train: set[tuple[str, str, str]], reflect: dict[str, str],
             reg: dict) -> list[tuple[str, str, str] | None]:
    """For each held-out triple, the training witness (t, P, h) with reflect(P) = r, or None (UNVERIFIED)."""
    sources: dict[str, list[str]] = {}
    for p, q in reflect.items():
        if p in reg["sym_rel"] and q in reg["sym_rel"]:
            sources.setdefault(reg["sym_rel"][q], []).append(reg["sym_rel"][p])
    out = []
    for h, r, t in held:
        w = next(((t, p, h) for p in sorted(sources.get(r, [])) if (t, p, h) in train), None)
        out.append(w)
    return out


def _sha_lines(rows) -> str:
    return hashlib.sha256("".join(f"{x}\n" for x in rows).encode("utf-8")).hexdigest()


def run(files: Path, out: Path, sources_path: Path = SOURCES, registry_path: Path = REGISTRY) -> dict:
    src = load_sources(sources_path)
    ent = {e["key"]: e for e in src["files"]}
    reg = load_registry(registry_path)
    data = {"WN18": {s: parse_triples(_pinned(files, ent[f"wn18_{s}"]).decode("utf-8")) for s in SPLITS},
            "WN18RR": read_archive(_pinned(files, ent["wn18rr_archive"]), ent["wn18rr_archive"]["members"])}
    removed = script_removed_relations(_pinned(files, ent["wn18rr_script"]).decode("utf-8"))
    pos_pointers, pos_of = wordnet_index(_pinned(files, ent["wordnet"]))
    pointers = {(s, sym, t) for s, _, sym, t, _ in pos_pointers}
    dpos = definitions_pos(_pinned(files, ent["wn18_definitions"]).decode("utf-8"))
    for rel in {r for rows in data.values() for split in rows.values() for _, r, _ in split}:
        if rel not in reg["rel_sym"]:
            raise KGError(f"relation {rel} has no WordNet pointer symbol")

    # construction of WN18RR: the released script's relation filter applied to the WN18 splits
    filt = {s: [x for x in data["WN18"][s] if x[1] not in set(removed)] for s in SPLITS}
    kept = sorted({r for _, r, _ in data["WN18RR"]["train"]})
    pair_of = {}  # the two relations of each inverse pair of the table (a row read in either direction)
    for p, q in reg["reflect"].items():
        if p != q and p in reg["sym_rel"] and q in reg["sym_rel"]:
            pair_of[reg["sym_rel"][p]] = reg["sym_rel"][q]
            pair_of[reg["sym_rel"][q]] = reg["sym_rel"][p]
    removed_side = all(r in pair_of and pair_of[r] in kept for r in removed)
    pairs = {frozenset((a, b)) for a, b in pair_of.items()}
    filter_check = {
        "script_sha256": ent["wn18rr_script"]["sha256"], "removed_relations": sorted(removed),
        "removed_relation_count": len(removed),
        "inverse_pairs_in_wn18": len(pairs),
        "each_removed_relation_has_its_inverse_kept": removed_side,
        "wn18rr_relations": kept,
        "splits": {s: {"wn18_rows": len(data["WN18"][s]), "filtered_rows": len(filt[s]),
                       "wn18rr_rows": len(data["WN18RR"][s]), "equal_triple_for_triple": filt[s] == data["WN18RR"][s]}
                   for s in SPLITS},
    }
    filter_check["reproduces_all_splits"] = all(v["equal_triple_for_triple"] for v in filter_check["splits"].values())
    if not filter_check["reproduces_all_splits"] or not removed_side:
        raise KGError("the released construction script does not reproduce WN18RR from WN18")

    # audit of the held-out splits
    summary, breakdown, classification, certified = [], [], {}, []
    for rel_name in RELEASES:
        train = set(data[rel_name]["train"])
        for split in HELD_OUT:
            held = data[rel_name][split]
            wit = classify(held, train, reg["reflect"], reg)
            wsym = classify(held, train, reg["symmetrized"], reg)
            l1 = sum(w is not None for w in wit)
            l1s = sum(w is not None for w in wsym)
            if any(a is not None and b is None for a, b in zip(wit, wsym)):
                raise KGError("the symmetrized registry loses a certificate")
            overlap = sum(x in train for x in held)
            summary.append({"release": rel_name, "split": split, "held_out": len(held),
                            "identity_overlap_with_train": overlap, "l1": l1, "unverified": len(held) - l1,
                            "l1_symmetrized_domain": l1s, "symmetrized_increment": l1s - l1})
            per = Counter(r for _, r, _ in held)
            per_l1 = Counter(x[1] for x, w in zip(held, wit) if w is not None)
            per_l1s = Counter(x[1] for x, w in zip(held, wsym) if w is not None)
            for r in sorted(per):
                sym = reg["rel_sym"][r]
                breakdown.append({"release": rel_name, "split": split, "relation": r, "pointer_symbol": sym,
                                  "declared": int(sym in reg["reflect"]), "held_out": per[r], "l1": per_l1[r],
                                  "l1_symmetrized_domain": per_l1s[r]})
            classification[f"{rel_name}/{split}"] = _sha_lines(
                f"{h}\t{r}\t{t}\t{'L1' if w is not None else 'UNVERIFIED'}" for (h, r, t), w in zip(held, wit))
            certified += [(x, w) for x, w in zip(held, wit) if w is not None]

    # raw authority: every certified triple and its training witness are pointers of WordNet 3.0 between the same two
    # synsets (an offset with a part of speech); WN18 identifiers are offsets only
    rel_sym = reg["rel_sym"]

    def is_pointer(x: tuple[str, str, str]) -> bool:
        return (x[0], rel_sym[x[1]], x[2]) in pointers

    def strict(x: tuple[str, str, str]) -> bool:
        return (x[0], dpos.get(x[0]), rel_sym[x[1]], x[2], dpos.get(x[2])) in pos_pointers

    shared = {o for o, ps in pos_of.items() if len(ps) > 1}
    wn18_all = [x for s in SPLITS for x in data["WN18"][s]]
    entities = {e for x in wn18_all for e in (x[0], x[2])}
    forced: dict[str, set[str]] = {}  # entity on a shared offset -> parts of speech that some triple forces on it
    for h, r, t in wn18_all:
        for e, side in ((h, 0), (t, 1)):
            if e in shared:
                ps = {(ph, pt)[side] for ph in pos_of.get(h, ()) for pt in pos_of.get(t, ())
                      if (h, ph, rel_sym[r], t, pt) in pos_pointers}
                if len(ps) == 1:
                    forced.setdefault(e, set()).update(ps)
    one_pair = [same_synset_pair(x, w, rel_sym, pos_pointers, pos_of) for x, w in certified]
    other = [(x, w) for x, w in certified if not (strict(x) and strict(w))]
    raw = {"matching": "a certified triple and its training witness count as WordNet pointers when one part of speech "
                       "for each of their two entities makes both pointers of the database, in opposite directions "
                       "between the same two synsets (a synset is an offset with a part of speech; WN18 identifiers "
                       "are offsets only); wn18_triples_that_are_wordnet_pointers and the certified_*_that_are_"
                       "wordnet_pointers counts match by offset and pointer symbol",
           "certified_triples": len(certified),
           "certified_targets_that_are_wordnet_pointers": sum(is_pointer(x) for x, _ in certified),
           "certified_witnesses_that_are_wordnet_pointers": sum(is_pointer(w) for _, w in certified),
           "certified_pairs_on_one_synset_pair": sum(one_pair),
           "certified_pairs_without_shared_offsets": sum(not ({x[0], x[2]} & shared) for x, _ in certified),
           "wordnet_offsets_in_two_or_more_pos_files": len(shared),
           "wn18_entities": len(entities),
           "wn18_entities_on_shared_offsets": len(entities & shared),
           "wn18_entities_with_triples_from_two_synsets": sum(len(v) > 1 for v in forced.values()),
           "definitions_file_pos": {"sha256": ent["wn18_definitions"]["sha256"],
                                    "certified_pairs_matching": len(certified) - len(other),
                                    "others": len(other),
                                    "others_all_on_shared_offsets": all({x[0], x[2]} & shared for x, _ in other)},
           "wn18_triples": len(wn18_all),
           "wn18_triples_that_are_wordnet_pointers": sum(is_pointer(x) for x in wn18_all),
           "wordnet_version": "3.0", "wordnet_sha256": ent["wordnet"]["sha256"]}
    if not (raw["certified_targets_that_are_wordnet_pointers"] == raw["certified_witnesses_that_are_wordnet_pointers"]
            == raw["certified_pairs_on_one_synset_pair"] == raw["certified_triples"]):
        raise KGError("a certified triple and its witness are not WordNet pointers between the same two synsets")
    if not raw["definitions_file_pos"]["others_all_on_shared_offsets"]:
        raise KGError("a certified pair fails with the definitions part of speech on an unshared offset")

    audit = {"step": "kg", "schema_version": "1",
             "inputs_sha256": {e["path"]: e["sha256"] for e in src["files"]},
             "registry": {"authority": reg["doc"]["authority"], "url": reg["doc"]["url"],
                          "accessed": reg["doc"]["accessed"], "directed": True,
                          "rows": [list(x) for x in reg["doc"]["rows"]],
                          "reflection": dict(sorted(reg["reflect"].items())),
                          "not_declared": sorted(reg["undeclared"])},
             "verdicts_issued": ["UNSAFE (L1)", "UNVERIFIED"],
             "nondetermined_issued": False, "model_scores_read": False,
             "classification_sha256": classification,
             "headline": {f"{r['release']}/{r['split']}": {"held_out": r["held_out"], "l1": r["l1"]} for r in summary}}
    write_csv(out / "release_summary.csv", summary, SUMMARY_FIELDS)
    write_csv(out / "relation_breakdown.csv", breakdown, RELATION_FIELDS)
    write_json(out / "wn18rr_filter_check.json", {"step": "kg", "schema_version": "1", **filter_check})
    write_json(out / "raw_authority.json", {"step": "kg", "schema_version": "1", **raw})
    write_json(out / "kg_audit.json", audit)
    return {"summary": summary, "filter": filter_check, "raw": raw}
