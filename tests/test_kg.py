# SPDX-License-Identifier: MIT
"""Checks of the link-prediction step: a synthetic release and WordNet fixture (plain asserts), then the shipped
outputs of WN18 and WN18RR."""
import csv
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lca import kg  # noqa: E402

E = [f"{i:08d}" for i in range(10)]
TRAIN = [(E[2], "_hyponym", E[1]), (E[4], "_member_of_domain_topic", E[3]), (E[6], "_synset_domain_topic_of", E[5]),
         (E[2], "_also_see", E[1]), (E[2], "_derivationally_related_form", E[1]), (E[7], "_hypernym", E[8])]
TEST = [(E[1], "_hypernym", E[2]),                    # L1 through the Hyponym -> Hypernym row
        (E[3], "_synset_domain_topic_of", E[4]),      # no row from Member of Domain: UNVERIFIED (L1 if symmetrized)
        (E[5], "_member_of_domain_topic", E[6]),      # L1 through the Domain of synset -> Member of Domain row
        (E[1], "_also_see", E[2]),                    # also-see has no row: UNVERIFIED
        (E[1], "_derivationally_related_form", E[2])]  # self-reflexive row: L1
VALID = [(E[8], "_hyponym", E[7])]                   # L1 through the Hypernym -> Hyponym row
SCRIPT = ("import os\nopen('kg_script_ran', 'w').write('x')\n"
          "predicates_to_remove = [\n    '_hyponym',\n    '_member_of_domain_topic',\n]\n")


def _text(rows) -> bytes:
    return "".join(f"{h}\t{r}\t{t}\n" for h, r, t in rows).encode()


def _fixture(root: Path) -> tuple[Path, Path]:
    reg = kg.load_registry()
    removed = ["_hyponym", "_member_of_domain_topic"]
    wn18 = {"train": TRAIN, "valid": VALID, "test": TEST}
    rr = {s: [x for x in rows if x[1] not in removed] for s, rows in wn18.items()}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for s, rows in rr.items():
            b = _text(rows)
            info = tarfile.TarInfo(f"{s}.txt")
            info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    pointers: dict[str, list[tuple[str, str]]] = {}
    for rows in wn18.values():
        for h, r, t in rows:
            pointers.setdefault(h, []).append((reg["rel_sym"][r], t))
    noun = "  header line\n" + "".join(
        f"{h} 03 n 01 word 0 {len(ps):03d} " + " ".join(f"{s} {t} n 0000" for s, t in ps) + " | gloss\n"
        for h, ps in sorted(pointers.items()))
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("wordnet/data.noun", noun)
        for pos in ("verb", "adj", "adv"):
            z.writestr(f"wordnet/data.{pos}", "  header line\n")
    defs = "".join(f"{e}\t__word_{i}_NN_1\tgloss\n" for i, e in enumerate(E)).encode()
    files = {"WN18/train.txt": _text(TRAIN), "WN18/valid.txt": _text(VALID), "WN18/test.txt": _text(TEST),
             "conve/WN18RR.tar.gz": buf.getvalue(), "conve/create_WN18RR.py": SCRIPT.encode(),
             "wordnet/wordnet.zip": zbuf.getvalue(), "WN18/wordnet-mlj12-definitions.txt": defs}
    keys = {"WN18/train.txt": "wn18_train", "WN18/valid.txt": "wn18_valid", "WN18/test.txt": "wn18_test",
            "conve/WN18RR.tar.gz": "wn18rr_archive", "conve/create_WN18RR.py": "wn18rr_script",
            "wordnet/wordnet.zip": "wordnet", "WN18/wordnet-mlj12-definitions.txt": "wn18_definitions"}
    entries = []
    for p, b in files.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_bytes(b)
        e = {"key": keys[p], "path": p, "sha256": hashlib.sha256(b).hexdigest()}
        if keys[p] == "wn18rr_archive":
            e["members"] = ["train.txt", "valid.txt", "test.txt"]
        entries.append(e)
    sources = root / "sources.json"
    sources.write_text(json.dumps({"files": entries}))
    return root, sources


def test_toy_release() -> None:
    with tempfile.TemporaryDirectory() as d:
        root, sources = _fixture(Path(d))
        cwd = os.getcwd()
        os.chdir(d)
        try:
            r = kg.run(root, root / "out", sources_path=sources)
        finally:
            os.chdir(cwd)
        assert not (Path(d) / "kg_script_ran").exists(), "the construction script must never run"
        s = {(x["release"], x["split"]): x for x in r["summary"]}
        assert (s[("WN18", "test")]["l1"], s[("WN18", "test")]["l1_symmetrized_domain"]) == (3, 4)
        assert s[("WN18", "test")]["identity_overlap_with_train"] == 0
        assert s[("WN18", "valid")]["l1"] == 1
        assert r["filter"]["reproduces_all_splits"] and r["filter"]["removed_relation_count"] == 2
        raw = r["raw"]
        assert raw["certified_triples"] == raw["certified_targets_that_are_wordnet_pointers"] \
            == raw["certified_witnesses_that_are_wordnet_pointers"] == raw["certified_pairs_on_one_synset_pair"] > 0
        assert raw["wordnet_offsets_in_two_or_more_pos_files"] == 0
        assert raw["definitions_file_pos"]["certified_pairs_matching"] == raw["certified_triples"]
        with (root / "out" / "relation_breakdown.csv").open(newline="") as fh:
            br = {(x["release"], x["split"], x["relation"]): x for x in csv.DictReader(fh)}
        assert br[("WN18", "test", "_also_see")]["declared"] == "0" and br[("WN18", "test", "_also_see")]["l1"] == "0"
        audit = json.loads((root / "out" / "kg_audit.json").read_text())
        assert audit["nondetermined_issued"] is False and audit["model_scores_read"] is False


def test_part_of_speech() -> None:
    """A synset is an offset with a part of speech: the target and its witness must join the same two synsets."""
    rel_sym = {"_hypernym": "@", "_hyponym": "~"}
    pos_of = {"00000001": {"n", "v"}, "00000002": {"v"}}  # offset 00000001 is shared by a noun and a verb
    target, witness = ("00000001", "_hypernym", "00000002"), ("00000002", "_hyponym", "00000001")
    joined = {("00000001", "v", "@", "00000002", "v"), ("00000002", "v", "~", "00000001", "v")}
    assert kg.same_synset_pair(target, witness, rel_sym, joined, pos_of)
    split = {("00000001", "n", "@", "00000002", "v"), ("00000002", "v", "~", "00000001", "v")}
    assert not kg.same_synset_pair(target, witness, rel_sym, split, pos_of), \
        "an offset-only match must not pass when the two pointers use different synsets of a shared offset"
    assert kg.definitions_pos("00000001\t__walk_VB_2\tgloss\n00000002\t__spiritual_bouquet_NN_1\tg\n") == \
        {"00000001": "v", "00000002": "n"}


def test_registry() -> None:
    reg = kg.load_registry()
    assert reg["reflect"]["~"] == "@" and reg["reflect"]["@"] == "~" and reg["reflect"]["+"] == "+"
    assert reg["reflect"][";c"] == "-c" and "-c" not in reg["reflect"], "the Domain row is read in one direction"
    assert reg["symmetrized"]["-c"] == ";c"
    assert "^" in reg["undeclared"] and "^" not in reg["reflect"]


def test_shipped_outputs() -> None:
    out = REPO / "outputs" / "kg"
    if not out.exists():
        print("test_shipped_outputs: outputs/kg not present, skipped")
        return
    with (out / "release_summary.csv").open(newline="") as fh:
        s = {(x["release"], x["split"]): x for x in csv.DictReader(fh)}
    golden = {("WN18", "test"): (5000, 4504, 154), ("WN18", "valid"): (5000, 4512, 159),
              ("WN18RR", "test"): (3134, 1052, 0), ("WN18RR", "valid"): (3034, 1046, 0)}
    for k, (n, l1, inc) in golden.items():
        x = s[k]
        assert (int(x["held_out"]), int(x["l1"]), int(x["symmetrized_increment"])) == (n, l1, inc), k
        assert int(x["identity_overlap_with_train"]) == 0, k
    fc = json.loads((out / "wn18rr_filter_check.json").read_text())
    assert fc["reproduces_all_splits"] is True and fc["removed_relation_count"] == 7
    raw = json.loads((out / "raw_authority.json").read_text())
    assert raw["certified_triples"] == raw["certified_targets_that_are_wordnet_pointers"] \
        == raw["certified_witnesses_that_are_wordnet_pointers"] == raw["certified_pairs_on_one_synset_pair"] == 11114
    assert (raw["certified_pairs_without_shared_offsets"], raw["wordnet_offsets_in_two_or_more_pos_files"],
            raw["wn18_entities"], raw["wn18_entities_on_shared_offsets"],
            raw["wn18_entities_with_triples_from_two_synsets"]) == (10946, 282, 40943, 232, 161)
    dp = raw["definitions_file_pos"]
    assert (dp["certified_pairs_matching"], dp["others"], dp["others_all_on_shared_offsets"]) == (11033, 81, True)


if __name__ == "__main__":
    test_registry()
    test_part_of_speech()
    test_toy_release()
    test_shipped_outputs()
    print("test_kg: ok")
