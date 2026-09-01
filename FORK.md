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

## Releases

| Version | What changed |
|---|---|
| 2.1.0 | Decoder throughput, proven byte-identical at corpus scale. Four Tier A changes to the Viterbi DP -- a pop-time duplicate skip, deriving the predecessor column instead of storing `vpath_table_col`, rolling the score table to two columns, and per-thread vpath scratch with a LUT encoder and an off-GIL traceback -- take serial decoding from 3.07 to ~2.22 ms/attempt (-28%; ~27.5x against pristine 61.1). Thread scaling is the larger win: `-t 32` went from **6.9x, slower than its own `-t 8`**, to 15.23x and monotonic, because the per-call score table fell from 3.12 MB to 41 KB and the per-thread footprint from 6.24 MB to 1.60 MB. **The first pristine Tier 2 baseline exists**: `make tier2` had never been satisfiable, since no baseline manifest had ever been captured, so AGENTS.md's "no decoder change without Tier 2" rule was unenforceable. It is now committed and green -- all four Tier A changes are byte-identical to pristine `05fd98a` over **288,096 decode attempts** across the eight public `example_*` BAMs, where Tier 1 alone covers 4,000. New default-off `--prune-reverse` (Tier B): runs the reverse decode against the forward score, 1.50-1.66x, with identical selected-read digests on all eight corpus BAMs; its safety valve is load-bearing, because the `> 1e-10` relaxation guard is not covered by the threshold-monotonicity argument and only the 1e-6 margin closes the gap. `Model.log_probability` is **removed** -- uncalled on every path and mathematically wrong (it multiplied probability transitions by log-emissions). Also folds in the unreleased 2.0.4 `-aln` fixes. Five overclaims corrected rather than quietly dropped: Tasks 3+4 measured -26%, not the planned -31/-35%; `--prune-reverse` measured 1.5-1.66x, not ~1.9x; a claimed -27.5% median did not reproduce (-23% did); reusing the **small** 41 KB score buffer costs +36% while reusing the **large** 1.56 MB one costs ~0%, the opposite of the plan's assumption, with the mechanism recorded as unidentified; and a `loc ratchet` transcript quoting 67 files was requoted at 69. |
| 2.0.3 | Code-review follow-up on 2.0.2. **No executable line under `advntr/` or `hmm/` differs from 2.0.2** -- the only change to shipped runtime code is a docstring -- so the decoder and the genotype path are byte-identical. Closes a latent false PASS in `scripts/assert_gil_release.py`: it excluded the forward declaration and definition by testing the *line* prefix for `static`, so a real call sharing a line with a declaration was dropped, and one dropped call with another guarded call present made the whole check exit 0. It now keys on the enclosing statement. `baseline_manifest()` gains a caller (`make tier3-baseline`) and a drift test, having shipped uncalled. Three overclaims corrected: dropping the losing traceback does NOT restore pristine's retention profile (pristine keeps one path per *selected* read, this keeps one per *eligible* read -- 1,047 against 1,617 on example_7a61); `writeable = False` stops a stray assignment, it is not memory safety, since `np.asarray(view.base)` still aliases the same storage; and the CI check proves the call is *emitted* inside a nogil region in the generated C, not that the compiled `.so` ran concurrently. |
| 2.0.2 | Adversarial-review fixes, none of which change decoder output. `transition_matrix_view()` now returns an array with `writeable = False`, which stops the accident it exists to prevent -- a stray assignment in a test moved a score by 1.0 while the CSR copy the main DP reads stayed put. It is not a memory-safety guarantee: `np.asarray(view.base)` still aliases the same storage, and `nbr_logp` is `cdef public`. `_decode_one` keeps only the winning traceback: peak RSS on the 50,619-read BAM at `-t 16` falls 1592.9 -> 936.6 MB with an identical selection digest. The CI GIL check now asserts the DP call itself runs GIL-free rather than counting releases file-wide. `--verify` refuses a missing baseline before capturing instead of after. The Tier 3 manifest records the kernel that produced it and states that it is a post-rewrite regression baseline, not a pristine one -- Tier 1 is the pristine gate. |
| 2.0.1 | Packaging fix: `find_packages()` was shipping `advntr_harness` (the equivalence harness) and `scripts/` into the installed egg, putting development tooling on the user's path. Caught by installing in Docker and importing from outside the repo. |
| 2.0.0 | `-t` became real: the Viterbi DP moved into a `nogil` block and the read loop is threaded. 19.6x serial, ~119x end-to-end at `-t 16`. Byte-identical decoder output. `USE_ENHANCED_HMM=False` now raises; `pomegranate/` is no longer compiled. |
| 1.3.3 | Inherited from upstream `enhanced_hmm`. `-t` was a genuine no-op on the `genotype -fs` path. |

The major bump is not cosmetic: `-t N` previously set `settings.CORES`, which nothing on
this path read. Anyone relying on `-t` being inert now gets real threads.
