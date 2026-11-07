# Commands of the artifact. The audit steps read INPUTS (the input view), the external, link-prediction and SEN12MS steps
# their fetched files; every step writes under OUTPUTS.
#   make quick     unit tests only
#   make check     tests, the output-only re-verification of every Path B certificate and every released open-fiber witness,
#                  then every ledger entry and claim checked against the shipped outputs and the source-release record
#                  (no input view needed)
#   make all       the whole audit from the input view, the independent checks, then the paper's numbers, tables and figures
#   make verify-sen12ms  download the SEN12MS files and rerun the independent check of that study
#   make verify    independent verifier (no pipeline code) of the verdict counts, the verdict rules, the carrier equivalence and the
#                  checks, independent checks of the support, link-prediction and SEN12MS steps, then the ledger and claim checks
PYTHON  ?= python3
INPUTS  ?= data/hub
OUTPUTS ?= outputs
PAPER   ?= paper
CHECKOUT ?= data/external/MaDI-Bench
KG       ?= data/external/kg/files
SEN12MS  ?= data/external/sen12ms/files
RUN = $(PYTHON) -I -B

.PHONY: all check quick test check-inputs registry conformance verdicts support tie_order mutations utility evidence \
        controls external fetch-external kg fetch-kg sen12ms fetch-sen12ms verify-sen12ms scaling paper tables figures independent \
        ledger verify

all: registry conformance verdicts support tie_order mutations utility evidence controls external kg sen12ms scaling independent \
     paper ledger

quick: test

check: test
	$(RUN) verify/verify_support.py --outputs $(OUTPUTS)
	$(RUN) verify/verify_tie_order.py --outputs $(OUTPUTS)
	$(RUN) verify/check_ledger.py --outputs $(OUTPUTS) --paper $(PAPER)

test:
	$(RUN) tests/test_registry.py
	$(RUN) tests/test_certificates.py
	$(RUN) tests/test_bounds.py
	$(RUN) tests/test_paper.py
	$(RUN) tests/test_support.py
	$(RUN) tests/test_tie_order.py
	$(RUN) tests/test_semantics.py
	$(RUN) tests/test_kg.py
	$(RUN) tests/test_sen12ms.py

check-inputs:
	$(PYTHON) -I -c "import hashlib, pathlib, sys; d = pathlib.Path('$(INPUTS)'); \
	rows = [l.split(None, 1) for l in pathlib.Path('data/hub/SHA256SUMS').read_text().splitlines() if l.strip()]; \
	sums = [(s, n.strip()) for s, n in rows]; missing = [n for s, n in sums if not (d / n).is_file()]; \
	bad = [n for s, n in sums if n not in missing and hashlib.sha256((d / n).read_bytes()).hexdigest() != s]; \
	sys.exit('input view incomplete in $(INPUTS): ' + ' '.join(missing) if missing else ('input hash mismatch: ' + ' '.join(bad) if bad else 0))"

registry: check-inputs
	$(RUN) pipeline/01_registry.py --inputs $(INPUTS) --outputs $(OUTPUTS)

conformance: registry
	$(RUN) pipeline/02_conformance.py --inputs $(INPUTS) --outputs $(OUTPUTS)

verdicts: conformance
	$(RUN) pipeline/03_verdicts.py --inputs $(INPUTS) --outputs $(OUTPUTS)

support: verdicts
	$(RUN) pipeline/03b_support.py --inputs $(INPUTS) --outputs $(OUTPUTS)

tie_order: verdicts
	$(RUN) pipeline/03c_tie_order.py --inputs $(INPUTS) --outputs $(OUTPUTS)

mutations: verdicts
	$(RUN) pipeline/04_mutations.py --inputs $(INPUTS) --outputs $(OUTPUTS)

utility: verdicts
	$(RUN) pipeline/05_utility.py --inputs $(INPUTS) --outputs $(OUTPUTS)

evidence: utility
	$(RUN) pipeline/05b_evidence.py --inputs $(INPUTS) --outputs $(OUTPUTS)

controls: utility
	$(RUN) pipeline/06_controls.py --inputs $(INPUTS) --outputs $(OUTPUTS)

fetch-external:
	$(RUN) data/external/fetch_external.py --dest $(CHECKOUT)

external:
	$(RUN) pipeline/07_external.py --inputs $(INPUTS) --outputs $(OUTPUTS) --checkout $(CHECKOUT)

fetch-kg:
	$(RUN) data/external/kg/fetch_kg.py --dest $(KG)

kg:
	$(RUN) pipeline/07c_kg.py --outputs $(OUTPUTS) --kg $(KG)

fetch-sen12ms:
	$(RUN) data/external/sen12ms/fetch_sen12ms.py --dest $(SEN12MS)

sen12ms:
	$(RUN) pipeline/07d_sen12ms.py --sen12ms $(SEN12MS) --outputs $(OUTPUTS)

verify-sen12ms: fetch-sen12ms
	$(RUN) verify/verify_sen12ms.py --outputs $(OUTPUTS) --sen12ms $(SEN12MS)

scaling: verdicts
	$(RUN) pipeline/08_scaling.py --inputs $(INPUTS) --outputs $(OUTPUTS)

paper:
	$(RUN) pipeline/09_tables_figures.py --outputs $(OUTPUTS) --paper $(PAPER) --claims-md CLAIMS.md

tables figures: paper

independent:
	$(RUN) verify/verify_outputs.py --inputs $(INPUTS) --outputs $(OUTPUTS)
	$(RUN) verify/verify_support.py --outputs $(OUTPUTS)
	$(RUN) verify/verify_tie_order.py --outputs $(OUTPUTS)
	$(RUN) verify/verify_kg.py --outputs $(OUTPUTS) --kg $(KG)
	$(RUN) verify/verify_sen12ms.py --outputs $(OUTPUTS) --sen12ms $(SEN12MS)

ledger:
	$(RUN) verify/check_ledger.py --outputs $(OUTPUTS) --paper $(PAPER)

verify: independent ledger
