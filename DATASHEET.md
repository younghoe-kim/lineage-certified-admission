# Datasheet: input view of the audited benchmark

This datasheet describes the input view that the audit reads (`data/hub/`), which the paper calls the input files (Section V). Section headings follow
*Datasheets for Datasets* (Gebru et al., CACM 2021). Terms of use are in `DATA_LICENSE.md`.

## Motivation

The view lets the audit of the paper be reproduced from the released benchmark: it holds every column the audit reads. It serves one purpose, deciding whether candidate feature views determine the benchmark's
protected labels under the registered lineage of the semantic hub; it is not a training set.

## Composition

| File | Rows | Columns |
|---|---:|---|
| `anchors.csv.gz` | 631,384 instance anchors (DF 289,202; FP 342,182) | `dataset` (DF, FP), `anchor_id` (DF-000001, …, in release order), `full_feats_json` (the 1,337 released feature columns as a sparse map from column to event count), `gold_facet`, `gold_family` |
| `core_terms.csv` | 2,715 terms of the semantic authority | `core_term_uid`, `term_type`, `facet` |
| `registry_edges.csv` | 15,006 declared relations | `edge_uid`, `src_uid`, `edge_type`, `dst_uid` |
| `label_rule.py` | the generator's family-derivation rule | read as text; the audit never executes it |
| `sample_order.csv.gz` | 631,384 | `dataset`, `anchor_id`, `sample_rank` (unsigned 64-bit) |

Feature columns are grouped into blocks by prefix: `core::` (1,181), `facet::` (26), `family::` (5),
`type::` (3), `rep_core::` (113), `inst_family::` (7) and `inst_sem::` (2). The authors built the benchmark release
from the Category and Attribute Prediction Benchmark of DeepFashion, used under the DeepFashion agreement (DF
anchors), and from the Fashionpedia annotations, licensed under CC BY 4.0 (FP anchors). The view holds no image, image path, file name or
person-level attribute.

## Collection and preprocessing

The first four files of the view were produced from four hash-identified files of the released benchmark, by column
selection and one-to-one renaming only:

| Source file | SHA-256 (first 8 hex digits) |
|---|---|
| anchor table | `0446d27d` |
| term table | `f117373f` |
| relation table | `04817133` |
| generator specification (label rule) | `45f68a22` |

The renamings give the view the paper's names and opaque anchor identifiers: dataset code of the DeepFashion-family dataset to `DF`; the namespace prefix of hub
identifiers to `hub:`; one family identifier to `tag_terms` (the paper's TAG), wherever it occurs;
anchor identifiers to opaque serial numbers in release order. Every renaming is a bijection, so no count, mass,
score, bound or verdict depends on the new names. The illustrative anchor of Fig. 1 is the candidate that the
label-free selection rule picks in the released benchmark (the smallest SHA-256 of its evidence text). Renaming
changes that text and keeps the row order, so this view binds the same anchor by its zero-based row; the verdict
step stops unless the candidate count and that row reproduce the record in `src/lca/paper/source_release.json`.

`sample_order.csv.gz` stores, for each anchor, the rank that defines the nested 10%–100% samples of the scaling
step; it was computed once from a salted SHA-256 of the source anchor identifiers.

Facts about the released benchmark that the paper reports and the audit checks:

- The release builder re-keyed core features to the cleaned semantic authority and dropped 117,540 core-term
  entries without a cleaned counterpart (66,440 in the core block, carrying 117,475 facet events, and 51,100 in the
  representative-core block); role blocks were copied unchanged. As a result, 65,076 anchors carry role blocks that
  differ from the registry applied to their core block (Section VI-A of the paper). The build report recorded
  `conforms: true`, a flag that compares only the number of written rows with the number of matched anchors. The
  release record `src/lca/paper/source_release.json` (`build_record`) gives these counts and the SHA-256 of the two build
  scripts, the bridge table and the build report. `make evidence` shows from the view alone that the copied type block counts every lost event
  as an attribute event (`evidence/drift_type_mechanism.csv`), and claim CL-DRIFTMECH checks this pattern on every drifted anchor.
- The released labels maximize the materialized carrier on every anchor. The generator's tie state (the event
  order that breaks ties) is not part of the release, so a tie fiber stays UNVERIFIED unless two of its anchors
  carry different labels (NONDETERMINED). Declaring that order would change no admission decision (`make tie_order`,
  `outputs/tie_order/`, claim CL-TIEORDER).
- The release builder matched each instance anchor one to one to the cleaned anchor table on keys from the anchor
  identifier: the dataset code plus the lower-cased identifier body (DF: image path; FP: annotation file and instance
  index), tried in a fixed order; a key that occurs more than once on the cleaned side is not matched. The match
  decision uses only these keys; labels are copied and compared after matching. The build scripts run on the source annotation
  tables upstream of the input view; `src/lca/paper/source_release.json` (`build_record`, `release_builder`) records
  their SHA-256, the rules they apply and the counts:

| Instance anchors | DF | FP |
|---|---:|---:|
| before re-keying | 341,934 | 342,182 |
| of these, images of the DeepFashion In-shop Clothes Retrieval Benchmark, outside the cleaned authority | 52,712 | 0 |
| in the cleaned authority (DF: the images of the Category and Attribute Prediction Benchmark) | 289,222 | 342,182 |
| matched one to one and released | 289,202 | 342,182 |
| set aside: paths that form 10 pairs differing only in letter case | 20 | 0 |

- The source anchor table also carries two cleaned-side auxiliary columns (the gold facet and the term type of that
  facet); the view keeps the generator's labels `gold_facet` and `gold_family`, the released facet and family labels.

Output names and the paper's terms:

| Output name | Paper |
|---|---|
| anchor, `anchor_id` | a released row: an anchor in Sections III–IV, a row in Sections I and V–VI |
| `unsupported_facet_mass` (`conformance/drift_anatomy.csv`) | facet excess over core·Π (Table V) |
| row class `no_rep_core` | row class: empty representative core |
| row class `ghost_color` | row class: excess color counts |
| row class `unsupported_category_unit` | the remaining drifted anchors (one category event) |
| `tag_terms` | TAG |
| protocol `fixed_dataset_refit` | conforming fit (Section V, Protocol) |
| protocol `all_source` | fitting on all source anchors |
| `macro_f1_upper_bound_C2` (`utility/bounds.csv`) | F1 bound β (Lemma 2) |
| `accuracy_ceiling` (`utility/bounds.csv`) | accuracy ceiling 1 − g₃ (Lemma 2) |
| `C_i`, `w(C_i)` (`registry/certificate_spec.csv`, column `basis_support`) | K_i, w(K_i) (Section III) |
| `carrier` in `external/*` (MaDI-Bench) | determinant: the taxonomy column that determines the protected column (Section V) |
| regimes `released`, `union`, `intersection`, `released_plus_color` (`support/*`) | released supports and the three counterfactual coordinate sets (Remark 1; extended audit details, Section E) |
| verdict codes `UNSAFE`, `NONDETERMINED`, `UNVERIFIED`; masses `m^U`, `m^N`, `m^V_obs`, `m^V_defect` (`verdicts/verdicts.csv`) | verdicts and masses (Section III); `m^V_obs` + `m^V_defect` is the UNVERIFIED mass |

## Uses

Intended: reproducing and checking the audit of the paper; auditing other candidate views of this benchmark under
the same registry. Not intended: training or evaluating models for deployment, or any commercial use (see
`DATA_LICENSE.md`).

## Distribution and maintenance

The view stays restricted under the DeepFashion agreement (`DATA_LICENSE.md`). The public aggregate outputs
and the source-release record suffice to recompute or check every number of the paper (`make check`), and `make check-inputs` verifies a copy of the view
against `data/hub/SHA256SUMS` before any step reads it. The link-prediction and SEN12MS studies read public
files, not this view (`data/external/kg/FETCH.md`, `data/external/sen12ms/FETCH.md`).
