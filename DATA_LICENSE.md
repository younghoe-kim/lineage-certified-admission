# Data license and availability

The code in this repository is released under the MIT license (`LICENSE`). The data that the code reads and
writes fall into six groups with different terms.

## 1. Input view of the released benchmark (restricted)

`data/hub/` lists, in `SHA256SUMS`, the five files that the audit reads:

| File | Content |
|---|---|
| `anchors.csv.gz` | one row per instance anchor: dataset, anchor id, evidence blocks, facet and family labels |
| `core_terms.csv` | core terms of the semantic hub: identifier, term type, facet |
| `registry_edges.csv` | declared relations of the semantic hub |
| `label_rule.py` | the generator's label rule, as released |
| `sample_order.csv.gz` | one fixed sample rank per anchor, used by the scaling step |

The anchors derive from annotations of two datasets:

- **DeepFashion** (DF anchors). The DeepFashion agreement permits non-commercial research use only and does not
  permit further copying, publishing or distributing any portion of the dataset, except for internal use at a
  single site within one organization.
- **Fashionpedia** (FP anchors). The Fashionpedia annotations and ontology are licensed under the Creative
  Commons Attribution 4.0 license; Fashionpedia does not own the copyright of the images.

Because the files of the input view contain rows, terms or rules derived from DeepFashion, the input view is not
redistributed. This repository identifies each of its files by SHA-256 (`data/hub/SHA256SUMS`), and
`make check-inputs` verifies a copy against these hashes before any step reads it.

## 2. Per-anchor outputs (restricted)

The following outputs have one row or array element per anchor and are therefore derived from DeepFashion
annotations at row level. They are produced by the pipeline, restricted like the input view, and listed with their
SHA-256 in `outputs/MANIFEST.json`:

`outputs/conformance/full_conform.csv.gz`, `outputs/conformance/query_masks.npz`,
`outputs/verdicts/anchor_verdicts.npz`, `outputs/verdicts/witnesses.json`,
`outputs/verdicts/fig1_instance.json`, `outputs/mutations/changed_rows.csv.gz`.

The full table of open single-label Path B fibers, `outputs/verdicts/path_b_open_witnesses.csv`, is fiber-level
rather than per-anchor, but it is restricted as well and listed with its SHA-256 in `outputs/MANIFEST.json`: it contains
192 rows that each describe a DeepFashion fiber with a single query-conforming anchor (section 3).

## 3. Aggregate outputs (public, CC BY-NC 4.0)

Every other file under `outputs/` and `paper/`, except the MaDI-Bench metadata of section 4, holds counts, masses,
scores, bounds, certificates of fibers,
timings and checks aggregated over anchors; the registry maps over opaque term identifiers (`outputs/registry/`,
`outputs/mutations/m1_registry.yaml`); and the facet and family counts of the one illustrative DF anchor that
Fig. 1 of the paper prints, without its identifier (`outputs/verdicts/fig1_example.json`,
`paper/figures/fig1_example.csv`). They derive in part from DeepFashion annotations, whose agreement
limits use to non-commercial research, so they are released under the Creative Commons
Attribution-NonCommercial 4.0 license, with attribution to DeepFashion and Fashionpedia. Together with the
source-release record `src/lca/paper/source_release.json` (file hashes and source and build counts), they are
sufficient to check every number of the paper (`make check`).

Three public Path B verification files are fiber-level rather than per-anchor: `outputs/verdicts/path_b_certificates.csv`,
`outputs/support/support_certificates.csv` and `outputs/verdicts/path_b_open_witnesses_public.csv`. They list
low-dimensional fiber values, labels and certificate state-set hashes or witness states, without anchor identifiers,
image paths or evidence vectors. Every DeepFashion row in them names a view value that at least two DeepFashion
anchors carry (at least 11,927 in `path_b_certificates.csv` and 203 in `support_certificates.csv`; in the public
open-witness subset at least two query-conforming anchors), which the independent verifier recounts from the input
view; in the two certificate files the label is the one that the registry certifies from that value. The complete
659-row open-witness table is identified by SHA-256 and restricted, because 192 DeepFashion rows each describe a fiber with a
single query-conforming anchor; counts and verifier results for all 659 fibers remain public
(`outputs/verdicts/path_b_open_summary.json`, `outputs/checks/independent_checks.csv`).

## 4. External benchmark (not redistributed)

The external study reads MaDI-Bench at a pinned commit. Its files are fetched from the official repository by
`data/external/fetch_external.py` and checked against the hashes in `data/external/FETCH.md`. The public outputs keep
the minimal source-declared metadata needed to reproduce the audit: the names of the audited source columns, target
attributes and taxonomy columns and the declared aliases of the six audited attributes, with counts and hashes
(`outputs/external/query_registry.csv`; `source_scope.csv` and `transform_authorities.csv` give the gold
correspondences of the 14 audited source columns to target attributes). Benchmark rows, free-text notes, source
metadata, taxonomies and schema-mapping files are not redistributed: outputs that copy benchmark content (taxonomy
rows, schema mappings, source metadata and headers, source inventories, the free-text reasons of unmapped source
columns) or describe the local checkout (domain summary, source freeze record) are regenerated after fetching and are
not shipped; `outputs/MANIFEST.json` lists their hashes. This metadata is attributed to the pinned MaDI-Bench release
and is not relicensed by this artifact.

## 5. Link-prediction study (public data, not redistributed)

The link-prediction study reads the WN18 splits and definitions file (TuckER repository), the WN18RR splits and the
WN18RR construction script (ConvE repository; the script is read as text and never run) and the WordNet 3.0 database
(NLTK data repository) at pinned commits. WN18 and WN18RR derive from WordNet 3.0 and are subject to its license; the
hosting repositories (TuckER, ConvE) are MIT-licensed; the WordNet 3.0 database is read from nltk_data, whose wordnet
archive carries the WordNet license.
`data/external/kg/fetch_kg.py` downloads these files and checks them against `data/external/kg/sources.json`; none
of them is part of this artifact. The outputs of the study (`outputs/kg/`, `outputs/checks/kg_verification.json`)
are counts, relation names and file hashes; they contain no triple of either release and are released with the
aggregate outputs of group 3 (CC BY-NC 4.0).

Attribution: WordNet 3.0 Copyright 2006 by Princeton University. All rights reserved.

## 6. SEN12MS study (public data, not redistributed)

The SEN12MS study reads four files from the SEN12MS repository at a pinned commit: the class fractions
(`labels/IGBP_probability_labels.pkl`), the released 17-class and 10-class labels
(`labels/single_label_IGBPfull_ClsNum.txt`, `labels/single_label_IGBPsimple_ClsNum.txt`) and the official loader
(`classification/dataset.py`, read as text and never run). SEN12MS is released under CC BY 4.0
(Schmitt et al., 2019). `data/external/sen12ms/fetch_sen12ms.py` downloads these files and checks them against
`data/external/sen12ms/sources.json` (SHA-256 and git blob id); none is part of this artifact. The outputs of the study
(`outputs/external/sen12ms_pathb.json`, `outputs/checks/sen12ms_verification.json`) hold counts, file hashes and the
released keys of the patches in four witness pairs; they contain no patch data and are released with the aggregate
outputs of group 3 (CC BY-NC 4.0).

Attribution: SEN12MS, M. Schmitt, L. H. Hughes, C. Qiu and X. X. Zhu, ISPRS Annals IV-2/W7, 2019, CC BY 4.0.

## Sources of the terms

- DeepFashion agreement: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/DeepFashionAgreement.pdf
- Fashionpedia terms of use: https://fashionpedia.github.io/home/data_license.html
- Creative Commons Attribution 4.0: https://creativecommons.org/licenses/by/4.0/
- Creative Commons Attribution-NonCommercial 4.0: https://creativecommons.org/licenses/by-nc/4.0/
- WordNet 3.0 license: `wordnet/LICENSE` in the WordNet 3.0 archive of `data/external/kg/sources.json`
- SEN12MS repository and license: https://github.com/schmitt-muc/SEN12MS (CC BY 4.0)
