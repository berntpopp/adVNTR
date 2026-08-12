# adVNTR (berntpopp hard fork) -- development targets.
#
# Everything runs in the envadvntr conda env (Python 2.7). muscle must be on PATH.

ENVBIN := /home/bernt-popp/miniforge3/envs/envadvntr/bin
export PATH := $(ENVBIN):$(PATH)

VNTYPER_DATA ?= /home/bernt-popp/development/VNtyper/tests/data

.PHONY: build test gate coverage-ratchet no-upstream-remote loc-ratchet tier2 clean

build:
	python setup_hmm.py build_ext --inplace

test:
	python -m unittest discover tests

# This is a hard fork (see FORK.md). An upstream remote implies a reconciliation that is
# not going to happen, and makes an accidental `git pull upstream` a real risk.
no-upstream-remote:
	@if git remote | grep -qx upstream; then \
		echo "ERROR: an 'upstream' remote exists. This is a hard fork; see FORK.md."; \
		exit 1; \
	fi
	@echo "no upstream remote: ok"

# Files already over 650 LOC may only shrink. New files must start under it.
loc-ratchet:
	python scripts/loc_ratchet.py

coverage-ratchet:
	coverage run --source=advntr,hmm -m unittest discover tests
	@coverage report --include='advntr/*,hmm/*' | tail -1 | awk '{print $$NF}' | tr -d '%' > .coverage-now
	@python -c "import sys; \
base = float(open('.coverage-baseline').read()); \
now = float(open('.coverage-now').read()); \
print('coverage %.1f%% (baseline %.1f%%)' % (now, base)); \
sys.exit(0 if now >= base else 'coverage fell: %.1f%% -> %.1f%%' % (base, now))"

# Full-corpus equivalence. Needs VNtyper's tests/data; hours to run. Not part of `gate`.
tier2:
	python -m advntr_harness.capture --tier 2 --out /tmp/advntr-tier2 \
		--vntyper-data $(VNTYPER_DATA) --verify tests/golden

gate: no-upstream-remote loc-ratchet build test coverage-ratchet
	@echo "gate: PASS"

clean:
	rm -f hmm/*.so hmm/*.c .coverage .coverage-now
	rm -rf build
