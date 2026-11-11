# Datasheet: input view

`data/hub/` holds every column the audit reads (the input files of Section V of the paper). Terms of use are in
`DATA_LICENSE.md`.

## Composition

| File | Rows | Columns |
|---|---:|---|
| `anchors.csv.gz` | 631,384 anchors (DF 289,202; FP 342,182) | `dataset` (DF, FP), `anchor_id` (DF-000001, …, in release order), `full_feats_json` (the 1,337 released feature columns as a sparse map from column to event count), `gold_facet`, `gold_family` |
| `core_terms.csv` | 2,715 terms of the semantic authority | `core_term_uid`, `term_type`, `facet` |
| `registry_edges.csv` | 15,006 declared relations | `edge_uid`, `src_uid`, `edge_type`, `dst_uid` |
| `label_rule.py` | the generator's family-derivation rule | read as text; the audit never executes it |
| `sample_order.csv.gz` | 631,384 | `dataset`, `anchor_id`, `sample_rank` (unsigned 64-bit) |

Feature columns form blocks by prefix: `core::` (1,181), `facet::` (26), `family::` (5), `type::` (3),
`rep_core::` (113), `inst_family::` (7) and `inst_sem::` (2). DF anchors are images of the Category and Attribute
Prediction Benchmark of DeepFashion; FP anchors are annotated instances of Fashionpedia (CC BY 4.0). The view holds
no image, image path, file name or person-level attribute.

## Collection

The first four files come from four hash-identified files of the released benchmark, the frozen release that Section V
of the paper describes (`src/lca/paper/source_release.json`), by column selection and one-to-one renaming of codes and
identifiers, for example one family identifier to `tag_terms` (the paper's TAG). The row order is kept, so no reported
number depends on the renaming. `sample_order.csv.gz` holds the ranks of the nested samples of the scaling step,
computed from a salted SHA-256 of the source identifiers.

The release record in the same file gives the build facts that the paper reports and the audit checks:

- Re-keying the core features to the cleaned authority dropped 117,540 core-term entries (66,440 in the core
  block, holding 117,475 facet events, and 51,100 in the representative-core block), and the role blocks were
  copied unchanged. This yields the 65,076 drifted anchors of Section VI-A. The build report's `conforms: true`
  compares only row counts.
- The released labels maximize the materialized carrier on every anchor. The generator's event order, which breaks
  ties, is not part of the release.
- Anchors were matched one to one to the cleaned anchor table on identifier keys, and labels were copied after
  matching.

| Anchors | DF | FP |
|---|---:|---:|
| before re-keying | 341,934 | 342,182 |
| of these, In-shop Clothes Retrieval images outside the cleaned authority | 52,712 | 0 |
| in the cleaned authority | 289,222 | 342,182 |
| matched one to one and released | 289,202 | 342,182 |
| set aside: 10 path pairs differing only in letter case | 20 | 0 |

## Names in the outputs

| Output name | Paper |
|---|---|
| verdict masses `m^U`, `m^N`, `m^V_obs` + `m^V_defect` (`verdicts/verdicts.csv`) | $`m^{\mathrm{U}}`$, $`m^{\mathrm{N}}`$, $`m^{\mathrm{V}}`$ (Section III) |
| `C_i`, `w(C_i)` (`registry/certificate_spec.csv`) | dataset support $`K_i`$ and its family image $`w(K_i)`$ |
| protocol `fixed_dataset_refit` | conforming fit (Section V) |
| `macro_f1_upper_bound_C2`, `accuracy_ceiling` (`utility/bounds.csv`) | F1 bound $`\beta`$ and $`1-g_3`$ (Lemma 2) |
| row classes `no_rep_core`, `ghost_color`, `unsupported_category_unit` | row classes of Table V |
| support regimes `released`, `union`, `intersection`, `released_plus_color` | supports of Remark 1 and the extended audit details |

## Uses

Intended: reproducing and checking the audit, and auditing other views of this benchmark under the same registry.
Not intended: training or evaluating deployed models, or commercial use. `make check-inputs` verifies the view
against `data/hub/SHA256SUMS`.
