# AGENTS.md

Instructions for anyone — human or agent — changing this repository.

## Project

`berntpopp/adVNTR` is a **hard fork** of `mehrdadbakhtiari/adVNTR`, maintained to serve
[VNtyper](https://github.com/hassansaei/VNtyper)'s MUC1-VNTR genotyping. Read
[FORK.md](FORK.md) first: it records the divergence point, the supported surface, and why
there is no `upstream` remote.

The supported path is exactly one command:

```
advntr genotype -fs -vid 25561 --alignment_file X.bam -o out.vcf \
       -m muc1.db --working_directory D -t N -aln
```

Everything else (`makedb`, copy-number genotyping, PacBio, plotting) still compiles and
imports, but is untested and unsupported.

**Where the time goes.** Profiled on `example_7a61_hg19_subset.bam`:
`select_illumina_reads` was 196.3 s of a 197 s run — **99.7 %**. It calls `hmm.viterbi`
twice per read (forward and reverse complement) against a ~2565-state profile HMM. Model
construction is 0.33 s and frameshift analysis 0.29 s; both are noise. Optimise the
decoder or you are not optimising this tool.

## Setup

Everything runs in the `envadvntr` conda environment (Python 2.7.15, Cython 0.29.15,
NumPy 1.16.5).

```bash
export PATH=/home/bernt-popp/miniforge3/envs/envadvntr/bin:$PATH
```

`muscle` must be on `$PATH` — it lives in that env, and model construction shells out to
it. Without it you get `Bio.Application.ApplicationError: Non-zero return code 127`, which
does not mention muscle in its first line.

## Commands

| Command | Does |
|---|---|
| `make build` | Rebuild `hmm/` in place (fast dev loop) |
| `make test` | Unit suite |
| `make gate` | Everything CI runs: remote check, build, tests, coverage ratchet |
| `make coverage-ratchet` | Fail if coverage of `advntr/` + `hmm/` fell |
| `make no-upstream-remote` | Fail if an `upstream` remote exists |
| `python setup.py build_ext --inplace` | Full package build |
| `python -m advntr_harness.capture --tier 1 --out tests/golden` | Re-capture Tier 1 fixtures |
| `make tier2` | Full-corpus equivalence check against `tests/golden/tier2_manifest.json` |
| `python -m advntr_harness.capture --tier 2 --out /tmp/c --verify tests/golden` | What `make tier2` runs, with an explicit `--out` |
| `advntr genotype -fs -vid 25561 ... --frameshift-calibration-out F.jsonl` | Append one calibration capture record per VNTR. Default-off; changes no call |

## Layout

```
advntr/            The tool. genotype -fs lives in vntr_finder.py + genome_analyzer.py.
hmm/               The live HMM backend (Cython). hmm.pyx is the Viterbi decoder,
                   compiled to module hmm.hmm (production). Its DP fill lives in
                   _viterbi_fill_core.pxi, `include`-d a second time by
                   hmm_instrumented.pyx (module hmm.hmm_instrumented, test-only) with a
                   different compile-time DEF -- counters and a skip_enabled toggle
                   compile in there and nowhere else. See _viterbi_fill_core.pxi's
                   docstring for why (a runtime guard, measured, was rejected).
pomegranate/       DEAD. Not compiled, not supported. See FORK.md.
advntr_harness/    Equivalence harness. Not shipped; imported by tests.
tests/golden/      Committed fixtures + manifests. The decoder's regression net.
filtering/         Standalone C++ k-mer prefilter. Not built. See Traps.
```

## Code style

- **Python 2.7.** No f-strings, no `pathlib`, no type hints, no `concurrent.futures`.
  Use `%` formatting.
- Docstrings say *why*, not *what*. If a line encodes a non-obvious decision, cite the
  file:line it mirrors or the measurement that justifies it.
- Cite measurements, not intuitions. "61.1 ms/attempt on the pristine build" beats "slow".

## Changing existing code

### File size

New files must be under **650 LOC**. These are already over and may **only shrink** — if
you touch one, leave it smaller than you found it:

| File | LOC |
|---|---|
| `advntr/plot.py` | 1445 |
| `advntr/vntr_finder.py` | 1212 |
| `hmm/hmm.pyx` | 693 |
| `advntr/hmm_utils.py` | 900 |
| `hmm/_viterbi_fill_core.pxi` | 199 |

`pomegranate/` is excluded: not compiled, not maintained.

### Coverage

Coverage of `advntr/` + `hmm/` must never fall below `.coverage-baseline`. Target is
**> 89 %**; the recorded starting point was 8 %, so this is a ratchet you push, not a bar
you clear in one commit. `make coverage-ratchet` enforces the floor.

Note coverage does **not** trace the Cython kernel. `hmm/hmm.pyx` is covered by the
equivalence gates, not by line coverage — do not read a high percentage as decoder
coverage.

## Testing

### The decoder must be proven byte-identical

Any change to `hmm/*.pyx` or the read-selection path is guilty until proven equivalent.
The harness compares at **decode-attempt** granularity: one record per `(read,
orientation)` *before* selection, carrying the exact IEEE-754 bits of `logp`, the full
Viterbi path, the exit status, and the selection decision.

That granularity is not fussiness. Comparing only *selected* reads and only the *winning*
orientation hides a broken forward decode whenever the reverse still wins, and compares
nothing at all for rejected reads.

| Tier | Scope | Cost | When |
|---|---|---|---|
| 1 | 2,000 stratified fixtures, committed | ~15 s | Every CI run |
| 2 | Full corpus from VNtyper's `tests/data` | hours | Before merging a decoder change |
| 3 | Occurrence-level, whole BAMs through `select_illumina_reads` | minutes | Any threading change |

`tests/golden/tier2_manifest.json` holds the pristine Tier 2 baseline: 8 files (the
public `TIER2_FILES` corpus), 288,096 decode attempts, captured from this fork's
pristine kernel (05fd98a) in an isolated worktree. Its `baseline_kind` and `note` fields
say so directly -- see `advntr_harness/capture.py`'s `BASELINE_KIND`/`BASELINE_NOTE` and
`tests/test_tier2_baseline.py`.

Tier 1 strata are deliberate, and **every stratum must be non-empty** — capture fails
otherwise. The `reverse_complement_wins` stratum is the one that matters most: measured
across 29,998 reads it fires for 77 (0.26 %), and for `example_66bf` it fires **zero**
times in 12,608 reads. A fixture set drawn from that sample alone would go green without
testing the branch it covers.

### Two-tier rule for behaviour changes

- **Tier A** — must be byte-identical. Merge on a green gate.
- **Tier B** — may change paths, but must produce identical genotype calls on the golden
  cohort, **and** must first demonstrate the gate actually exercises the changed branch.
  Land behind a default-off flag.

An upstream patch gets no exemption from either.

### `--prune-reverse`: a worked Tier B example

The one Tier B flag shipped so far. `--prune-reverse` (Task 8; `advntr/settings.py:55`
`PRUNE_REVERSE_DECODE`, default `False`) runs the reverse-complement decode with
`threshold = max(dp_score_threshold, fwd_logp)` instead of `dp_score_threshold` alone
(`advntr/read_selection.py:_decode_one`, via `hmm.hmm.Model.viterbi`'s
`min_threshold`). Sound because every DP edge weight is `<= 0` (log probabilities), so a
path's running score is non-increasing column by column: if the true best reverse path
beats `fwd_logp`, every prefix of it also clears the raised threshold, so raising it this
way can only prune paths that could never have won.

- **Default-off, and the default path is untouched.** With the flag off, `_decode_one`
  is byte-for-byte the pre-Task-8 code path. Tier 2 (all 8 public `example_*` BAMs,
  288,096 decode attempts) is VERIFIED identical against the pristine 05fd98a kernel
  with the flag off.
- **The safety valve is load-bearing, not defence-in-depth.** The monotonicity argument
  above covers only the non-strict half of `_viterbi_fill_core.pxi`'s write guard
  (`log_prob >= threshold`, `:149,176`) -- not its other half, the `log_prob -
  dynamic_table[...] > 1e-10` relaxation epsilon. Pruning removes writes, so a pruned
  run's incumbent at a cell can be lower than the unpruned run's, letting it accept a
  relaxation the unpruned run rejects as sub-epsilon -- a different value and push,
  hence a different visit order downstream (this file's Traps section, "Visit order is semantic").
  `_SAFETY_VALVE_MARGIN = 1e-6` (`advntr/read_selection.py`) re-runs the reverse decode
  unpruned whenever the pruned score comes within that margin of `fwd_logp`; bit-exactness
  with the flag on rests on that margin being four orders of magnitude wider than the
  `1e-10` epsilon that creates the gap, not on the threshold argument alone.
- **Measured (own harness, public `example_*` corpus only).** Flag-on vs flag-off
  selected-read digests identical on all 8 public `TIER2_FILES` (count, query-name
  order, and digest all matched). Safety valve fired on 17/20,556 = 0.083% of
  reverse-decode attempts (a 3,000-read prefix of each of the 8 files). Speedup
  1.50-1.66x by min/median -- BELOW the plan's ~1.9x estimate: pruning touches only the
  reverse decode, never forward, and its benefit is strongly margin-dependent (median
  per-read relaxation-count drop 95.6%, matching the estimate, but the aggregate
  sum-weighted drop only 72.2-91.2%, since a minority of weak-margin reads dominates the
  sum).
- **Worth enabling** on throughput-sensitive runs where that 1.5-1.66x reverse-decode
  speedup matters. Turning it on by default is a separate decision that has not been
  taken -- it stays opt-in.

### `--exact-frameshift-caller`: a Tier B flag that cannot run unconfigured

The second Tier B flag (Task 8; `advntr/settings.py` `EXACT_FRAMESHIFT_CALLER`, default
`False`, written from `args.exact_frameshift_caller` at `advntr/advntr_commands.py:76`).
With it off, the decision is the shipped `identify_frameshift`
(`advntr/vntr_finder.py:187-197`) and the path is byte-for-byte the pre-Task-8 one. With
it on, `advntr/exact_caller.py` decides with a one-sided exact binomial
(`advntr/exact_tail.py`) over Task 7's integer `(k, N)` and a frozen background loaded
from `--frameshift-background <file>`.

- **It is not usable without an artifact, by design.** There is no built-in `p0` and
  there must not be one: SPEC Q-RATE shows the public candidate-conditioned rates
  (3.0e-4 pooled, 1.7e-4 median) are conditional on candidates already selected at
  support >= 3 and are not plug-in estimates for a production null. Falling back to the
  shipped statistic would make the flag a lie. `genotype` therefore refuses at startup
  both when `--frameshift-background` is missing and when the file it names does not
  validate. Setting `settings.EXACT_FRAMESHIFT_CALLER` directly, bypassing the CLI,
  raises instead at the top of `find_frameshift_from_selected_reads` -- which is *after*
  `select_illumina_reads` has decoded every read, so that path is a backstop, not a
  fail-fast.
- **`State` and the six-column table are untouched.** Only the p-value moves;
  `MeanCoverage` stays the legacy quantity, so a flag-on run stays diffable against a
  flag-off one.
- **Sibling support is attributed per read, and the denominator is the scored state's
  own.** Task 7's records are per `(read, occurrence)`; the emitted `State` is per read.
  `advntr/exact_caller.py` unions the identities each row attributed to that `State`
  (`state_identities`, from the fusion THAT read's own whole-read map produced), per SPEC
  line 131 ("must never sum overlapping counts"), and takes `N` from the `State`'s own
  row. Task 8a unioned both halves instead, which credited a state with siblings' whole
  support (measured: `k = 300` against its own row's 11 on `example_dfc3_hg19_subset.bam`)
  and with trials no occurrence offered it; `advntr/exact_caller.py`'s docstring records
  that correction. `k <= N` is consequently NOT structural -- the `support >
  opportunities` guard logs and refuses the call, and clamping is forbidden. That guard
  compares two integers, not two sets: `k`'s occurrences and `N`'s trials are different
  sets (measured, 0 identities outside their state's own spans on all eight public BAMs),
  so a calibration, which holds the span signature table, must assert the subset property
  itself.
- **Not yet calibrated or measured.** The background must be frozen on a partition that
  is not the holdout, and the operating point measured once afterwards. Until that is
  done the flag is a mechanism, not a recommendation. Three things a calibration has to
  condition on are written into `advntr/exact_caller.py`'s docstring rather than only
  here: candidates reach the statistic only after the legacy support floor and the flank
  boundary gates, so the null is for a truncated population; a compound `State`'s
  aggregated event is "at least one component", whose null rate is not a per-slot `p0`;
  and an aggregated `k == 0` is reported at a tail of 1.0 with a warning.

### `--frameshift-calibration-out`: a capture surface, not a caller

The third default-off flag (Task 8h; `advntr/settings.py` `FRAMESHIFT_CALIBRATION_OUT`,
default `None`, written from `args.frameshift_calibration_out` at
`advntr/advntr_commands.py:81`). It moves no decision at all: with it set,
`OpportunityCounter.finalise` appends one record and returns exactly what it returned
before. **Measured through the real CLI on `example_66bf_hg19_subset.bam`: the emitted
six-column table is byte-identical across pre-change flag-off, post-change flag-off and
post-change flag-on.** That is a one-BAM measurement, not a corpus claim.

- **Why it exists at all.** PLAN Task 8 Step 3 has to freeze a background on a calibration
  partition, and its estimator needs `N` for a `State` in the samples where that `State`
  did NOT fire. Nothing carried it: `finalise` emits rows only for
  `set(legacy_support) | set(self._support)`, the encoded diagnostics drop the identity
  and span fields (`UNENCODED_FIELDS`), and `advntr/vntr_finder.py:429` publishes only the
  records. The counter's span inventory -- the one object that generates every missing
  denominator -- had never left the process.
- **Independent of `--exact-frameshift-caller`, deliberately.** A calibration capture runs
  with the caller OFF so it cannot perturb the calls it is measuring; the two flags are
  read separately and neither implies the other.
- **The format is JSON Lines, append.** One line per `finalise`, so a multi-VNTR
  `-vid a,b` run does not overwrite itself, and every line carries `schema`, `version`,
  `vntr_id`, `read_length` and `is_haploid` so a duplicate from a resumed run is
  detectable offline. `sort_keys`, compact separators, no read name -- the anonymity
  property `tests/test_frameshift_context.py:199` pins, re-checked here against the real
  capture: 0 of 19,884 query names in the BAM appear anywhere in the file.
- **It stores primitives and derives nothing, and that is checkable.** The line carries
  the candidate rows and the span signature table with a distinct-identity count per
  signature; `opportunity_spans` is excluded because it is exactly derivable from that
  table with the shipped `parse_components` and `_signature_supports`. On the
  `example_66bf` capture: 443,002 bytes total (1,014 rows, 1,373 spans), of which the span
  table is 59,082, against 2,644,839 bytes for `opportunity_spans` alone -- 45x the input
  that regenerates it. Recomputing every row's `opportunities` from the span table
  reproduced the run exactly, 0 mismatches over all 1,014 rows.
- **One obligation the format cannot discharge.** `advntr/exact_caller.py` says a consumer
  holding the span table must assert that the identities behind `k` are among the trials
  `N` counts. The exported table carries a COUNT per signature, not the identities, so an
  offline consumer can re-check the cardinality but not the set; carrying the identities
  costs 28x on the span table (1,683,975 bytes, 26,593 pairs). The set property is pinned
  in-process instead, by `tests/test_frameshift_calibration.py`'s `TestSubsetObligation`.
- **One file per sample, and the line never names the sample.** `vntr_id` says which VNTR
  was scored; nothing says which sample, deliberately -- adVNTR is not told a cohort sample
  id and the confidentiality boundary is cleaner when it cannot learn one. Sample identity
  comes from the sink's path and the capture controller's manifest. The consequence, which
  has to be stated rather than discovered: append several samples into one file and the
  partition a line came from is unrecoverable.
- **A shared path is a mistake the writer survives, not a supported mode.** A line is
  443 KB, far above the size at which an append is atomic: eight barrier-synchronised
  writers into one file leave an arbitrary number of the eight lines unparseable without a
  lock. How many varies run to run, and
  `advntr/frameshift_calibration.py`'s module docstring states that once -- cite it rather
  than quoting a run. The stable figure is the test's: it catches a removed lock in 20 of
  20 runs. The append holds an exclusive `fcntl.flock` across the whole write including
  the flush, which makes the same eight-way test come out 8 of 8 parseable with every
  `vntr_id` recovered. `flock` is advisory and binds only writers that take it, so
  it is a guard against a mistake, not a licence to share a path. The writer also checks
  that the file ends with a newline and supplies the missing one first, so a process
  killed mid-write costs its own line and not the next good one.
- **A consumer must ABORT on an unparseable line, and must never skip one.** This is the
  contract, not a suggestion. Skipping a torn line silently drops that sample's
  denominators, including every zero-support state it was the only witness for, and biases
  the estimate in a direction nobody can bound afterwards. The writer's job is to confine
  the damage to one line; refusing to continue is what makes that confinement worth
  anything.
- **The path is preflighted at startup** (`advntr/advntr_commands.py:82-95`), opened
  `a+b` -- the writer's own mode, because it reads the last byte back -- and closed,
  exactly as `--frameshift-background` is validated ten lines below. Without it an
  unwritable path raises `IOError` only inside `finalise`, after every read has been
  decoded. The preflight CREATES the file, so a run that fails later leaves a 0-byte sink:
  an empty file therefore does not distinguish "the run failed" from "there were no
  decision sites". That distinction belongs to the capture controller's manifest, which
  records each sample's exit status, and deliberately not to this flag.
- **`ru_length` is a row field.** `ru_length`, `ru_bp_coverage`, `ru_bp_coverage_ratio` and
  `avg_bp_coverage` sit on candidate rows, so a pattern that produced no row at all
  contributes none of them and its repeat-unit length is not recoverable from the line. `N`
  is unaffected -- it comes from `spans` alone.
- **The write is inside a counter method, which is a smell that is argued rather than
  hidden.** `OpportunityCounter._spans` is the only place the inventory exists, and
  `advntr/frameshift_calibration.py`'s module docstring carries the argument. The write is
  not routed through `advntr/vntr_finder.py`, which stays at exactly 1212 lines; the
  counter is handed the finder it belongs to at its single construction site
  (`advntr/vntr_finder.py:238`) purely so a line can say which VNTR and which read length
  it scored.

## Git and PRs

- Conventional commits (`feat:`, `fix:`, `perf:`, `test:`, `build:`, `docs:`, `refactor:`).
- The commit body explains *why*, and cites the measurement or file:line that justifies it.
- `make gate` green before opening a PR.
- A decoder PR body must contain: pristine ms/attempt, new ms/attempt, which tiers ran,
  how many files and attempts were compared.

## Release workflow

1. `make gate`
2. Bump `advntr/__init__.py:__version__`
3. Tag and release from `main`
4. Bump `GIT_COMMIT` in VNtyper's `vntyper/dependencies/advntr/install_advntr.cfg`
5. Run VNtyper's golden-cohort gate — the adVNTR cases are `a5c1_hg19_advntr`,
   `b178_hg19_advntr`, `dfc3_hg19_advntr` at `--advntr-max-coverage 300`

VNtyper pins an exact commit, so nothing reaches users until step 4.

## Traps

- **Visit order is semantic.** `hmm/hmm.pyx` relaxes with `> 1e-10`, not `> 0`. The DP is
  therefore *not* an order-independent fixpoint: a chain of sub-epsilon improvements is
  truncated differently under a different order. Neighbour order (the `sorted()` in
  `bake()`) and FIFO push/pop order must be preserved exactly by any rewrite. This is the
  single easiest way to silently change a genotype call.

- **Never recompute `log()` for edge weights.** Copy the double already in
  `transition_matrix`. libm's `log` is not correctly rounded and can differ in the last
  ulp across versions, which breaks bit-equivalence for no benefit.

- **Never add `-ffast-math`.** It reassociates floating point and breaks the `-inf`
  propagation the decoder relies on for its rejection test.

- **`wraparound=False` segfaults.** The code relies on negative indexing and `boundscheck`
  is already off. Verified empirically, not theorised.

- **`select_illumina_reads` ignores its `hmm` argument.** `advntr/vntr_finder.py:883`
  unconditionally rebuilds the model from a read length derived from `samfile.head(5)`.
  On the corpus BAMs the derived length is 151, giving a **2565**-state model — a
  hand-built `read_length=150` model has **2559**. Fingerprint `finder.hmm` *after* the
  call, never before. `genotype -fs -u` does **not** silently fail to converge:
  `iteratively_update_model` (`advntr/vntr_finder.py:826-856`) rebuilds through the
  non-enhanced `get_read_matcher_model`, whose `Model.from_matrix` call at
  `advntr/hmm_utils.py:745` raises `AttributeError` on the enhanced backend.

- **`derive_read_length` can IndexError.** It is `sorted(head(5) lengths)[3]`, so a BAM
  whose head yields fewer than four records crashes. Mirrored in the harness rather than
  papered over.

- **`USE_TRAINED_HMMS = True` crashes.** `advntr/settings.py:9` disables it. The enhanced
  `Model` has no `to_json`/`from_json`, so `advntr/vntr_finder.py:114,121` would raise
  `AttributeError`. The flag was turned off in 2018 for disk usage — a year before the
  backend that lacks the API was written.

- **`adVNTR-Filtering` is not built or installed.** `advntr/genome_analyzer.py:166` calls
  it via `os.system` and never checks the return; the shell redirect creates an empty
  file, so unmapped-read recruitment silently yields nothing. Affects the copy-number
  path, not `-fs`.

- **`hmm/__init__.py` calls `pyximport.install()` at import** and mutates
  `os.environ['CFLAGS']` in-process, which leaks into every subprocess it later spawns.

- **The egg is zip-safe but ships `.so` files**, so they are extracted to
  `~/.cache/Python-Eggs` at first import. Stale after a rebuild; breaks on read-only
  `$HOME`.

- **Final-column silent states are never drained.** The main loop is
  `for col in range(sequence_length)`, so silent states activated at `col == L` are
  dropped; one hardcoded relaxation runs afterwards and writes a DP cell *without
  enqueuing it*. That last detail breaks any scratch-reuse scheme that tracks only the
  work queue.

- **A hung `Model.viterbi` traceback cannot be interrupted by `signal.alarm`.** If `logp`
  ever comes out finite for a read whose DP fill broke early (work queue empty before
  column `sequence_length`), the traceback loop (`while row != 0 or col != 0: ...`) walks
  `vpath_table_row` cells that were never legitimately written for that path and can spin
  forever. It is a tight `nogil`-compiled C loop holding the GIL, so it never returns to
  the bytecode dispatch point that delivers a pending Python signal -- `signal.alarm`, and
  therefore a bare in-process `unittest` assertion, cannot stop it. Task 5's naive
  no-reset rolled-table prototype hit exactly this (task-5-report.md): a wrong finite
  `logp`, then a hang killed only by an external `timeout`/SIGTERM (exit 143/124). The
  regression test for it (`tests/test_decoder_workload.py`,
  `tests/_early_break_worker.py`) therefore runs the production call in a
  `timeout`-wrapped subprocess, never in-process.

- **`recruit_read` needs the vpath, not indices.** It calls
  `get_number_of_matches_in_vpath`, which unpacks `(idx, state)` tuples. Passing a tuple
  of ints raises `TypeError: 'int' object is not iterable`.

- **`self.subModels[1]` segfaults, not IndexErrors, on a model with only one
  subModel.** `subseq_viterbi` still reads it (`hmm.pyx`) to find the repeat matcher;
  `viterbi()` used to as well, for two variables (`repeat_start_index`/
  `repeat_end_index`) never read again afterward -- Task 6 found and deleted that dead
  block. `subModels` is `cdef list`, and this file's `boundscheck=False` plus that
  method's own `@cython.wraparound(False)` make Cython emit the unchecked
  `PyList_GET_ITEM` for it, so indexing past the end reads adjacent memory instead of
  raising. Every real MUC1 model has >= 2 subModels (prefix/repeat/suffix,
  concatenated), so this never fires in production; a single-subModel synthetic test
  model is what surfaces it (task-6-report.md).

- **Reusing `viterbi()`'s small (41 KB) score table across calls via
  `threading.local()` is measurably SLOWER, not faster, for a reason no profiling
  tool in this sandbox could pin down.** Task 6 fix round 1 measured it directly: an
  isolated score-only-reuse build cost +36% serial (2.16-2.18 ms fresh-every-call
  baseline -> 2.94-2.99 ms/attempt); the shipped combined-reuse build cost +8%
  against the same baseline (2.32-2.36 ms). Every mechanism that could plausibly
  explain it was checked and ruled out --
  `hmm/hmm.c`'s `_viterbi_fill` body is byte-identical before/after (the DP loop's own
  codegen cannot be the cause); `/usr/bin/time -v` shows minor page faults and
  `Maximum resident set size` essentially unchanged across every variant, with the
  entire gap landing in `User time`; `strace -c` shows near-identical `mmap`/`munmap`/
  `brk` counts. `perf_event_paranoid=4` in this sandbox blocks hardware counters
  (`cache-misses`, `cycles`), so the exact micro-architectural mechanism (cache
  associativity conflict from a persistent small allocation's fixed address,
  something in glibc's per-arena free-list state, or another effect below what
  `/usr/bin/time`/`strace` can see) was never identified. The FIX is not to explain
  it but to avoid it: only the large (~1.56 MB) vpath table is amortised via
  `_thread_scratch`; the score table stays a fresh `np.empty` every call, exactly as
  it was before Task 6, and costs nothing to keep that way (task-6-report.md's fix
  round 1 section has the full ablation matrix). Do not "simplify" this by folding
  the score table back into the reused scratch without re-measuring serial first.
  `hmm/hmm.pyx`'s `_traceback` keeps its own malloc'd path buffer fresh-per-call
  (never `_thread_scratch`) for the same class of reason -- but this ablation only
  ever measured the score table; the traceback buffer's cost was never separately
  profiled, so treat that as an informed inference from the same mechanism, not an
  independent measurement of its own.

## Never

- Add an `upstream` remote, or merge from upstream. Cherry-pick and re-gate instead.
- Change production code to make a test pass. If a test contradicts the code, establish
  which is right; if the code wins, skip the test with the reason written down.
- Claim equivalence from a gate that compares only selected reads or only one orientation.
- Land a decoder change without Tier 2, or a threading change without Tier 3.
- Recompile `pomegranate/`.
