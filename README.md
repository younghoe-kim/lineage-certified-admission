# Lineage-certified admission: artifact

Code, input files and audit outputs of the paper *When Field Exclusion Fails: Lineage-Certified Admission for
Integrated-Data Benchmarks*.

## Quick check

With Python 3.11 or later, from the repository root:

```
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r environment/requirements.lock.txt
make check
```

`make check` runs the unit tests, re-verifies every Path B certificate and every open-fiber record of
`outputs/tie_order/` in integer arithmetic, and checks every number and claim of the paper against `outputs/` and
the source-release record. It needs no network access, takes about 15 seconds, and ends with the line

```
check_ledger: PASS: 1371 cells, 65 ledger files, 35 claims, 1 table file, 6 figure data files checked against outputs/ and the source-release record
```

`docker build -f environment/Dockerfile -t lca . && docker run --rm lca` runs the same check in a container.

## Full reproduction

```
make fetch-external fetch-kg fetch-sen12ms
make all
```

The first command downloads the public MaDI-Bench, link-prediction and SEN12MS files at pinned commits and checks
them. `make all` then reruns every step from the input view in `data/hub/` (about 56 minutes on one core) and
rewrites `outputs/` byte for byte in the pinned environment, apart from timings. `make verify` runs an independent
verifier that imports no pipeline code and rechecks both conformance layers, all fibers, every verdict and every
open Path B fiber from the input view.

## Contents

`CLAIMS.md` maps each claim to its place in the paper and its command, `paper/number_ledger.csv` binds each
reported number to an output file and its SHA-256, and `paper/extended_audit_details.pdf` gives the tables of all
views, certificates and support variants. `DATASHEET.md` describes the input view. Code is in `src/lca/`, one
script per step in `pipeline/` and the independent checks in `verify/`.

## License

Code: MIT (`LICENSE`). Input view, outputs and paper files: CC BY-NC 4.0; `DATA_LICENSE.md` gives the terms and
attribution. External datasets keep their own licenses.
