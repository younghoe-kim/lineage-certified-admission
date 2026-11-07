# Lineage-certified admission: artifact

Code and outputs that regenerate every table and figure and recompute or check every number of the paper
*When Field Exclusion Fails: Lineage-Certified Admission for Integrated-Data Benchmarks*, and a map from each claim of the
paper to the command and the output cells that support it (`CLAIMS.md`).

The extended audit details ([`paper/extended_audit_details.pdf`](paper/extended_audit_details.pdf)) list further
tables and checks of this artifact.

## Quick check (no data access needed)

With Python 3.11 or later (tested with 3.11 and 3.13), from the repository root:

```
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r environment/requirements.lock.txt
make check      # unit tests, every Path B certificate and released open-fiber witness re-verified, then every ledger entry and claim
                # checked against the shipped outputs and the source-release record
```

The check needs neither network access nor the input view. It takes about 15 seconds and 150 MB of memory
(measured on a two-core Linux machine) and ends with the line

```
check_ledger: PASS: 1371 cells, 65 ledger files, 36 claims, 1 table file, 6 figure data files checked against outputs/ and the source-release record
```

The artifact supports three levels of verification:

1. **Every reported number and claim** (`make check`, above). Apart from the code and the WordNet pointer table that a
   unit test reads (`data/external/kg/wordnet_reflexive.yaml`), it reads `outputs/`, `paper/`, the source-release
   record `src/lca/paper/source_release.json` and the result line of this README only. It recomputes each of the
   paper's numbers from the shipped output named in `paper/number_ledger.csv` or, for the file hashes and the source
   and build counts, reads it from the source-release record, compares it with the ledger value and the file's
   SHA-256, and confirms that every claim in `paper/claims.csv` holds on the outputs.
2. **Every Path B certificate and the public studies.** `make check` also runs `verify/verify_support.py`, which
   re-verifies every Path B certificate of every support regime from the shipped outputs, without the pipeline
   library, and `verify/verify_tie_order.py`, which rechecks each of the 467 witness records shipped here from its
   fiber value, its witness state and `outputs/registry/` alone and recomputes the rival optima of the 5 tie-only
   fibers by enumerating their states; both write their reports to `outputs/checks/`. With the
   input view and the fetched public files, `make independent` rechecks all 659. The external, link-prediction and SEN12MS steps, with their independent
   checks, rerun from their fetched public files (below); `make verify-sen12ms` downloads the four SEN12MS files and
   reruns that study's independent check in about ten seconds.
3. **The full pipeline.** Rerunning the audit from the input view requires access to DeepFashion, from which the input
   files, identified by SHA-256 in the source-release record, derive (`make all`, below).

## What the public files reproduce

| Result | With the public files only | Needs the restricted input view |
|---|---|---|
| Every number of the paper and the extended audit details, and every claim | `make check`: recomputed from the shipped outputs or checked against the source-release record | |
| Path B certificates of the released and the three counterfactual supports (38) | `make check` (`verify/verify_support.py`) rechecks each in integer arithmetic | |
| Open single-label Path B fibers (659) | each of the 467 released rows lists its fiber value and witness state, which `make check` (`verify/verify_tie_order.py`) rechecks with `outputs/registry/` alone; `outputs/verdicts/path_b_open_summary.json` gives the counts and hashes of all 659 | all 659 rows: `make independent` |
| External (MaDI-Bench), link-prediction and SEN12MS studies | rerun from their fetched public files (`make fetch-external external`, `make fetch-kg kg`, `make fetch-sen12ms sen12ms`); `make verify-sen12ms` downloads the SEN12MS files and reruns that study's independent check alone | |
| Release construction (anchor matching, re-keying, 117,540 dropped entries, 65,076 drifted anchors) | the rule and the counts: `src/lca/paper/source_release.json` and `DATASHEET.md`; the drift pattern that the rule leaves, on every drifted anchor: `make check` (claim CL-DRIFTMECH) | the build runs on the source rows |
| Registry, FULL-CONFORM, QUERY-CONFORM, verdicts, support, declared event order, mutations, probes, evidence, controls, scaling | | `make all INPUTS=...` |
| Independent verifier (both conformance layers recomputed per anchor, verdict rules, open fibers) | | `make independent` |

## Main results and where to check them

The first table of `CLAIMS.md` lists the main results of the paper, each with its value, its claim and the output
files that hold it; the sections below it list every claim with its number cells. `make paper` writes `CLAIMS.md`
from the ledger, so the table always shows the values of the shipped outputs.

**Further outputs.** Section VI-B of the paper reports the exact source-fiber lookup coverage
(`outputs/utility/lookup_coverage.csv`) next to the registry rule. Some outputs hold further analyses that neither the paper nor the extended audit details report: a
fiber-lookup probe (the `lookup` rows of `outputs/utility/probes.csv` and `probe_bound_check.csv`), refits on random rows and their split into rows and shared coordinates
(`outputs/utility/volume_control.csv`, `outputs/evidence/volume_null.csv`, `refit_separation.csv`), probe accuracy by verdict class (`outputs/utility/verdict_class_accuracy.csv`,
`cosine_determination_matrix.csv`), Macro-F1 label by label (`outputs/evidence/metric_label_*.csv`) and the
coordinates that the shared projection drops (`outputs/evidence/shared_basis_drop.csv`). No number, claim, table or
figure of the paper uses them. The ledger records some of their cells, so `make check` recomputes these too, and its
F1-bound check covers the lookup and registry-rule scores as well.

## Full reproduction (needs the input view)

```
make all INPUTS=/path/to/hub      # about 56 minutes on one core; the external, link-prediction and SEN12MS steps need their fetched files
```

The input view (the input files of Section V of the paper) is not distributed (`DATA_LICENSE.md`); `make all` runs where a licensed copy of it exists, and
`make check-inputs` verifies that copy. It goes in `data/hub/` (the default `INPUTS`) or any folder named by `INPUTS`. The audit steps
(`registry` to `mutations`, with `support` and `tie_order`) take about 6 minutes; the SEN12MS step about 15 minutes (its integer programs for every fiber); the probes, the evidence step, the controls and scaling take the rest. (The paper's 151.5 s is the reference run of FULL-CONFORM, QUERY-CONFORM and
the verdicts with the mutation checks on one process of the reference workstation with Python 3.13.7, NumPy 2.4.2,
pandas 3.0.3 and SciPy 1.17.1, as recorded in `outputs/scaling/reference_times.json`.)
`make all` runs the steps in order and writes under `outputs/`; each step is also a target:

| Target | Step | Writes | Paper |
|---|---|---|---|
| `registry` | registry maps, protected query, certificate specification of every view | `outputs/registry/` | Sec. III, Table II |
| `conformance` | FULL-CONFORM and QUERY-CONFORM, defects, drift anatomy | `outputs/conformance/` | Table V |
| `verdicts` | fiber verdicts of 32 views, witnesses, the identity-admitted against the direct-carrier view of each task (`carrier_equivalence.csv`), exact checks of every Path B certificate and open fiber (the full open-fiber table is restricted like the per-anchor outputs; its public subset and a summary with counts and hashes are shipped) | `outputs/verdicts/` | Tables III, IV; Sec. VI-A; extended audit details Table S1, Fig. S1, Section C |
| `support` | every view audited again on three counterfactual supports (union, intersection, with color); exact checks of every Path B result | `outputs/support/` | Remark 1, Proposition 2; extended audit details Section E, Table S3 |
| `tie_order` | the UNVERIFIED mass that a declared event order could turn into UNSAFE mass: the single-label tie fibers of Path A, and every open Path B fiber with the rival programs of all rivals solved and each strictly winning rival state re-checked in integer arithmetic; the admission decisions that such an order could change; the public per-fiber rows (`tie_order_fibers_public.csv`): `verify/verify_tie_order.py` rechecks the 462 strictly winning rival states in integer arithmetic from `outputs/registry/` alone and recomputes the rival optima of the 5 tie-only fibers by enumerating all their states (`make check`, `make independent`) | `outputs/tie_order/` | Sec. III (after Proposition 1); Sec. VI-A |
| `mutations` | mutation checks M1–M5 | `outputs/mutations/` | Sec. VI-A; extended audit details Section A.3 |
| `utility` | probes and their baselines, F1 bounds and the bound check, the eight block combinations and the type-block rescaling, the tolerance check | `outputs/utility/` | Table IV, Fig. 2; Sec. VI-B |
| `evidence` | family-3 verdict classes, type and family blocks of the drifted anchors | `outputs/evidence/` | Table V; Sec. VI-A; extended audit details Section D |
| `controls` | partition identity, simpler checks, permutation null | `outputs/utility/` | Table VI, Fig. 2a |
| `external` | external benchmark protocol, its second implementation and the comparison with the benchmark's normalization gold | `outputs/external/` | Sec. VI-C, Fig. 3a; extended audit details Section C |
| `kg` | link-prediction study: the held-out triples of WN18 and WN18RR against the WordNet table of reflexive pointers, the relation filter of the WN18RR construction script, and every certificate and its witness checked against WordNet 3.0 as pointers between the same two synsets (one part of speech per synset) | `outputs/kg/` | Sec. VI-C; extended audit details Section C |
| `sen12ms` | SEN12MS study: QUERY-CONFORM of every patch, the released 10-class label checked against the view, the fibers of the merged-count view and their verdicts for the released 17-class label, every certificate, witness pair and rival state checked in integer arithmetic, a second state of every certified fiber whose 17 counts vary, and the rival integer programs of every fiber that is not NONDETERMINED, compared with the closed form | `outputs/external/sen12ms_pathb.json` | Sec. VI-C |
| `scaling` | Path B time on nested samples | `outputs/scaling/` | Fig. 3b |
| `independent` | independent verifier that imports no pipeline code: reads Π and T from the input term and relation tables and checks the registry maps of the step outputs against them; it checks W against the map stated in Table S2 of the extended audit details rather than rebuilding it from the generator specification; recomputes FULL-CONFORM and the QUERY-CONFORM masks of all 32 views for every anchor from the input view and matches them row by row with the step outputs; rebuilds the fibers and witness pairs of all 32 views, tallies the verdict codes, checks the rules that tie verdicts to certificate paths on every anchor, recounts the carrier equivalence and rechecks all 659 open Path B fibers from the restricted full table (fiber sizes recounted from the input view); independent checks of the support step, the declared-order step, the link-prediction step and the SEN12MS study; each check compares a recomputation with the pipeline's output, with a value the paper states (written in the verifier) or with an invariant such as zero conflicts | `outputs/checks/` | all |
| `paper` | numbers, tables and figures of the paper and the extended audit details | `paper/` | all |
| `ledger` | every ledger entry and claim checked against the outputs and the source-release record | | all |
| `verify` | `independent`, then `ledger` | `outputs/checks/` | all |

Apart from the fetched files of the external, link-prediction and SEN12MS steps and their pinned records under
`data/external/` (below), and the source-release record `src/lca/paper/source_release.json` (the verdicts step
reads the Fig. 1 selection from it, the controls step the build report), every step reads only `--inputs` (the input view, checked against `data/hub/SHA256SUMS` before it is read) and
the outputs of earlier steps (`--outputs`). Paths, solver settings, thread counts and seeds are arguments or
constants written into each step's output; apart from the timing columns, no output depends on the working directory or the clock.

The external step reads MaDI-Bench at a pinned commit: `make fetch-external` clones it and checks every file
against `data/external/FETCH.md`. The link-prediction step reads the WN18 splits, the WN18RR archive, the WN18RR construction script (as
text; its list of removed relations is parsed, the script is never run), the WordNet 3.0 database and the WN18
definitions file (part of speech of each entity, for one diagnostic) at pinned commits:
`make fetch-kg` downloads them and checks every file against `data/external/kg/sources.json`
(`data/external/kg/FETCH.md`); this step and its check do not need the input view. The SEN12MS step reads the
SEN12MS class fractions, its released 17-class and 10-class labels and its official loader (as text; the loader is never
run) at a pinned commit: `make fetch-sen12ms` downloads them and checks all four against `data/external/sen12ms/sources.json`
(SHA-256 and git blob id; `data/external/sen12ms/FETCH.md`); this step and its check do not need the input view either.

**Label access.** `outputs/checks/data_access_checks.json`, written by the conformance step (`make conformance`),
records the columns that the registry and conformance steps load (for the registry step, its declared loader columns)
and the order of the conformance step's reads and writes up to the QUERY-CONFORM masks. The registry step and the
FULL-CONFORM pass load no label column, and the label columns are first loaded after
`conformance/full_conform.csv.gz` is written; the conformance step stops otherwise.

## Input identification and evaluation constants

Section V of the paper states these conventions in summary; the artifact fixes them as follows.

- **Input files.** The four inputs of the hub audit (term table, relation table, anchor table and generator specification)
  are identified by their full SHA-256 in the source-release record `src/lca/paper/source_release.json`; a local copy is
  checked against `data/hub/SHA256SUMS` before any step reads it (`make check-inputs`). The audit reads copies of these
  files with identifiers renamed one to one (`DATASHEET.md`). The inputs, the per-anchor outputs and the sample order
  of the scaling step derive from DeepFashion and are not redistributed (`DATA_LICENSE.md`). Label access: above.
- **Exact check.** Each solver solution is rounded to integers (`INTEGRALITY_TOL = 1e-7`, `src/lca/certificates.py`) and
  its feasibility and objective are verified again in integer arithmetic without the solver; coefficient integrality is
  checked with `EXACT_TOL = 1e-12`.
- **Probe tolerances.** The cosine probe abstains when its top score is at most 1e-12 or tied within 1e-12
  (`SCORE_EPS`, `TIE_ATOL`, `src/lca/evaluation.py`). Its prototypes (one l2-normalized mean prototype per source
  label) are computed in single precision (float32). Raising the abstention tolerance to 1e-9 or 1e-8 leaves the
  cosine predictions fitted on all source anchors unchanged for all eight views and both directions
  (`outputs/utility/tolerance_sensitivity.csv`).
- **Uniform baseline.** Uniform is the mean of a fixed number of uniform draws over the candidate labels; the draw count
  is the ledger cell `unidraws` (10,000).
- **Views.** The `registry` step writes the certificate specification (path, basis block, required maps) of each of the
  32 audited views to `outputs/registry/`; their fiber verdicts are in `outputs/verdicts/` (Fig. S1 of the extended audit details).

## Data

The input view and the per-anchor outputs derive from DeepFashion annotations and stay restricted under the
DeepFashion agreement; the aggregate outputs and the source-release record that `make check` reads are public and
give every number of the paper. The full table of open Path B witnesses is identified by its SHA-256 and restricted like
the per-anchor outputs, because some of its DeepFashion rows each describe a fiber with a single query-conforming anchor; its public subset and
a summary of all its rows are shipped (`DATA_LICENSE.md`, section 3). WN18 and WN18RR derive
from WordNet 3.0 and are subject to its license; the hosting repositories (TuckER, ConvE) are MIT-licensed; the WordNet
3.0 database is read from nltk_data, whose wordnet archive carries the WordNet license; the WN18RR construction script
is read as text, never run. SEN12MS is released under CC BY 4.0. These files are fetched, not redistributed; the
link-prediction outputs are counts, and the SEN12MS outputs are counts with the released keys of four witness pairs.
`DATA_LICENSE.md` gives the terms and the list of restricted files; `DATASHEET.md` describes the input view and how it
was made.

## Determinism

Deterministic logical outputs are byte-identical across runs in the pinned environment:
`outputs/checks/rerun_equivalence.json` records two full runs (Python 3.11.15 and 3.11.17) in which all
110 non-timing output files are byte-identical; it also records the SEN12MS step and its independent check, run
from the fetched files in both environments, with byte-identical outputs (2 files), and the tie-order step, run from the input view in both environments, with byte-identical outputs (4 files). The scaling outputs hold wall-clock times and are left out of that
comparison; the independent verifier recomputes their summaries from the raw runs, and the timings reported in the
paper come from the reference run described in `outputs/scaling/reference_times.json`. The orders of datasets,
tasks, blocks and labels are declared once in `src/lca/names.py`, and every pipeline output is written by the
deterministic writer in `src/lca/io.py`.

## Environment

Python 3.11 or later (tested with 3.11 and 3.13) with the packages in `environment/requirements.lock.txt`, or the
container in `environment/Dockerfile` (`docker build -f environment/Dockerfile -t lca . && docker run --rm lca`).

## Notation of the paper

| Notation | Meaning (where the paper defines it) |
|---|---|
| $`L^{\mathrm{rel}} = (\Pi, T, W)`$ | release registry: $`\Pi`$ sends a core term to its facet, $`T`$ a facet to its type and $`W`$ a facet to its family (Sec. III) |
| FULL-CONFORM, $`O_i^{\mathrm{rel}}`$ | label-free check against the full registry; $`O_i^{\mathrm{rel}}`$ holds the anchors whose core block is nonzero and whose role blocks equal the registry applied to their core block (Definition 2(a)) |
| certificate specification | path (A, B or C), basis block $`b`$ and required maps $`R(V,q)`$, fixed by the view schema and the registry alone (Definition 2(b), Remark 2) |
| Path A, Path B, Path C | A: the view fully exposes the basis block; B: each block of the view is a registered image of the basis block or a coordinate projection of one; C: every other view ($`R(V,q) = \emptyset`$), which never receives UNSAFE (Definitions 2(b) and 3) |
| QUERY-CONFORM, $`O_i^{V,q}`$, $`D_i^{V,q}`$ | $`O_i^{V,q}`$ holds the anchors whose evidence meets the conditions of the query-audit domain $`D_i^{V,q}`$ and whose label maximizes the carrier (Definition 2(c)) |
| UNSAFE | the solver of the path certifies that one label is the strict unique maximum of the carrier at every state of the fiber (Definition 3) |
| NONDETERMINED | two query-conforming anchors with the same view value carry different labels; this rules out determination, not dependence (Definition 3) |
| UNVERIFIED | every other observed value, and every anchor outside $`O_i^{V,q}`$ (query defects) (Definition 3) |
| $`m^{\mathrm{U}}`$, $`m^{\mathrm{N}}`$, $`m^{\mathrm{V}}`$ | shares of a dataset's anchors that receive each verdict (Definition 3) |
| L0, L1, L1', L2, L3 | audit findings of Table III: direct exposure, linear determination (Path A), fiber determination (Path B), materialization drift, information without determination |
| $`V^{\mathrm{sh}}`$ | shared projection: the coordinates supported in the fitting rows (Sec. V, Probes) |
| $`\beta`$, $`g_3`$ | F1 bound of every deterministic predictor of a view, and the smallest fraction of anchors whose removal makes the label a function of the view (Lemma 2) |

## Layout

`src/lca/` library · `pipeline/` one command per step · `verify/` independent checks · `tests/` unit tests ·
`data/` input hashes and the fetchers of the external benchmark, the link-prediction and the SEN12MS files · `outputs/` results and `MANIFEST.json` ·
`paper/` number ledger, claims, tables, figures and the extended audit details.

Code: MIT (`LICENSE`). Data and outputs: `DATA_LICENSE.md`.
