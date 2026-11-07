# Input view (restricted)

The audit reads five files from this folder. They derive from DeepFashion and Fashionpedia annotations and are
not redistributed (see `../../DATA_LICENSE.md`). `SHA256SUMS` lists their hashes, and every step
checks a file against it before reading it (`make check-inputs` checks all five at once).

| File | Used by |
|---|---|
| `anchors.csv.gz` | registry, conformance, verdicts, support, mutations, utility, evidence, controls, scaling, verify |
| `core_terms.csv` | registry, verify |
| `registry_edges.csv` | registry, conformance, verify |
| `label_rule.py` | registry, verify (read as text, never executed) |
| `sample_order.csv.gz` | scaling, verify |

`../../DATASHEET.md` describes the columns and how the view was made.
