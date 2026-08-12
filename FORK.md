# Fork status

`berntpopp/adVNTR` is a **hard fork**. It does not track
`mehrdadbakhtiari/adVNTR` and no upstream merges are performed.

## Divergence point

- This fork's `main` is `05fd98a4db4ce833546a673be48a7e07a39dd146`.
- That commit is upstream `enhanced_hmm` (`6bb94d4`) plus one commit
  ("fix: Guard against `NoneType` in mutation-sequence logic", also filed
  upstream as PR #73).
- Against `upstream/master` (`b1d91cb`) this fork is 111 ahead / 94 behind.
  `enhanced_hmm` and `master` are long-diverged lineages and were never
  going to reconcile.

## Why detach

This fork exists to serve [VNtyper](https://github.com/hassansaei/VNtyper)'s
MUC1-VNTR genotyping. That is one locus, one code path (`genotype -fs` on
Illumina short reads), and it is where essentially all of the runtime lives:
profiled on `example_7a61_hg19_subset.bam`, `select_illumina_reads` was 196.3 s
of a 197 s run — 99.7 %. Optimising for that path means changes upstream has no
reason to want, and carrying an upstream remote implies a reconciliation that is
not going to happen.

## Supported surface

| Path | Status |
|---|---|
| `genotype -fs` (Illumina frameshift, `USE_ENHANCED_HMM=True`) | **Supported, tested, benchmarked** |
| `genotype` copy-number, `makedb`, PacBio, `plot` | Compiles and imports; not tested, not supported |
| `USE_ENHANCED_HMM=False` (pomegranate backend) | **Unsupported — raises on import** |

`pomegranate/` remains in the tree for reference but is no longer compiled: it is
dead code on every supported path (`advntr/settings.py` sets
`USE_ENHANCED_HMM = True`, which routes to `hmm/hmm.pyx`) and it does not build on
gcc >= 14, because `pomegranate/distributions.pyx` uses old scipy `cython_blas`
bindings that pass `long*` where `int*` is expected. Compiling it made
`setup.py build_ext` fail outright.

## Applying an upstream change

Cherry-pick by hand, then run the full gate:

```bash
git fetch https://github.com/mehrdadbakhtiari/adVNTR.git <branch>
git cherry-pick <sha>
make gate
```

Do **not** add a permanent `upstream` remote; `make no-upstream-remote` and CI
both fail if one exists.

Upstream changes that alter decoder output are not accepted on the strength of
being upstream. They go through the same two-tier gate as anything else: either
byte-identical decoder output, or identical genotype calls on the golden cohort
with a demonstration that the gate actually exercises the changed branch.
