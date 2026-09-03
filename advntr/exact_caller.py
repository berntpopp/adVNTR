"""The default-off exact frameshift caller: Task 7's `(k, N)`, the exact tail, one `p0`.

PLAN Task 8 Steps 4 and 6. Activation is Tier B (AGENTS.md), mirroring the one Tier B
flag shipped so far: `settings.EXACT_FRAMESHIFT_CALLER` is `False`, written from
`args.exact_frameshift_caller` in `advntr/advntr_commands.py` exactly as
`--prune-reverse` writes `PRUNE_REVERSE_DECODE`. With the flag off nothing here runs and
the decision is the shipped `identify_frameshift` (`advntr/vntr_finder.py:187-197`),
unchanged.

**It cannot run without an artifact.** `configured_background` raises when the flag is on
and no background path is configured. Falling back to the shipped statistic would make
the flag mean nothing, and there is no plug-in default to fall back to instead -- SPEC
Q-RATE, and `advntr/frameshift_background.py`'s docstring, say why.

## Aggregating Task 7's rows onto an emitted `State`

`VNTRFinder.last_frameshift_opportunities` is keyed on *occurrence-scoped* candidate
names, and `advntr/frameshift_opportunities.py:per_occurrence_candidates` documents the
two ways those diverge from the shipped `State`: adjacent deletions in different
occurrences are fused by the whole-read map (`D11_2` + `D12_2` -> `D11_2&D12_2`), and an
insertion's `_LEN` suffix is renumbered (`I2_1_T_LEN1` twice -> `I2_1_T_LEN2`). In both
cases the legacy-named row sits at `support == 0` while its siblings carry the support,
and each sibling's `legacy_states` field names the `State` its support belongs to.

**Both halves are a union over `(read, occurrence)` identities.** SPEC line 131: "Any
future merge must union read/occurrence identities; it must never sum overlapping
counts." So:

- **`k`** is the size of the union of the siblings' `support_identities`.
- **`N`** is the size of the union of the siblings' `opportunity_spans` -- the span ids
  are unioned and their counts added, which is the identity union because spans
  partition the identities (`advntr/frameshift_opportunities.py:finalise`).

That makes `k <= N` structural rather than hoped for: each sibling's support identities
lie inside the spans that sibling matches, so the union of supports lies inside the
union of spans. The `support > opportunities` guard below is kept as a defensive check
against a hand-built or future record shape, not as a live decision path.

The earlier `sum(k)` / `max(N)` pair was wrong in both directions and is recorded here
so it is not reinvented. `max` is a *lower bound* on the union of the siblings' trials,
and `sum` double-counts an occurrence two siblings share, so p-values came out too small
in the regime the guard never saw. On the fusion example above, with two siblings, one
shared supporting occurrence and disjoint span sets: `sum/max` gives `(k=6, N=20)` and
`p = 3.29e-04`, a call at the shipped 1e-3 cutoff; the union gives `(k=5, N=40)` and
`p = 4.80e-02`, not a call. That is 146x, in the anti-conservative direction -- the one
`advntr/frameshift_opportunities.py:126-129` identifies as the wrong one to optimise
against, since a smaller `N` at the same `k` lowers the p-value.

**Residual, and the calibration sub-task must condition on it.** For a fused `State` the
aggregated event is "this occurrence supported at least one component", whose null rate
under a per-slot `p0` is nearer `1 - (1 - p0)^components` than `p0`, so scoring it with a
per-slot rate is anti-conservative for compound candidates. Nothing here can fix that:
the background is keyed on the emitted `State` string, so the fix is to calibrate `p0`
for a compound `State` on the same aggregated statistic this function computes.

**Two more divergences a frozen background has to match.**

- Only candidates that already passed `settings.MIN_SUPPORTING_READ_COUNT` and the flank
  boundary gates reach a decision site (`advntr/vntr_finder.py:475-480`, `:519-521`,
  `:529-531`), so the selection truncation SPEC Q-RATE warns about is still UPSTREAM of
  this statistic. A background estimated over all candidate slots would not be the null
  for the truncated population that actually arrives here.
- An aggregated `k == 0` has a well-defined tail of exactly 1.0 and is reported as such,
  but it means every occurrence supporting a state the legacy caller found at or above
  the support floor was ineligible. It is logged as a divergence rather than dropped
  quietly.

`p0` is looked up on the emitted `State` string, falling back to the artifact's declared
default. `State` is byte-identical by SPEC 3.5, which is what makes it a stable key.

The decision is `log_tail < log(cutoff)` against `settings.INDEL_MUTATION_MIN_PVALUE`;
the returned probability is for the `Pvalue` column and the log line only, and can be
`0.0` in the deep tail. The `MeanCoverage` column keeps the legacy quantity: the flag
moves the decision, not the rest of the table.
"""
import logging

from advntr.exact_tail import exact_indel_tail, tail_below_cutoff
from advntr.frameshift_background import BackgroundModelError, load_background_model
from advntr import settings


#: Path -> validated model, for the life of the process. `configured_background` runs
#: once per VNTR, not once per run: `advntr/genome_analyzer.py:215-216` loops over
#: `target_vntr_ids` and each iteration reaches
#: `find_frameshift_from_selected_reads`. Re-reading there would let an operator's edit
#: land between two VNTRs of one run and score them against different bases, which is
#: the hazard the no-cache version was reaching for and produced instead. One read per
#: path per process gives every VNTR in a run the same basis; a changed artifact needs a
#: new run, which is what "frozen" is supposed to mean.
_LOADED = {}


def configured_background():
    """The frozen background for this run, or `None` when the exact caller is off.

    Raises `BackgroundModelError` when the flag is on without a valid artifact.
    """
    if not settings.EXACT_FRAMESHIFT_CALLER:
        return None
    path = settings.FRAMESHIFT_BACKGROUND_FILE
    if not path:
        raise BackgroundModelError(
            'the exact frameshift caller is enabled but no background model is '
            'configured: pass --frameshift-background <file>. There is deliberately no '
            'built-in default (SPEC Q-RATE), so this run cannot proceed.')
    if path not in _LOADED:
        _LOADED[path] = load_background_model(path)
        logging.info(_LOADED[path].describe())
    return _LOADED[path]


def aggregate_evidence(records, state):
    """`(k, N)` for one emitted `State`, or `None` when no row mentions it.

    Both halves union rather than sum or max; see the module docstring for the
    measurement that settles it. `spans` is keyed on the span id, so a shape two
    siblings both match contributes its count once.
    """
    siblings = [row for candidate, row in records.items()
                if candidate == state or state in row.get('legacy_states', ())]
    if not siblings:
        return None
    identities = set()
    spans = {}
    for row in siblings:
        identities.update(row['support_identities'])
        spans.update(row['opportunity_spans'])
    return len(identities), sum(spans.values())


def decide(records, state, background, cutoff):
    """`(called, pvalue)` for one candidate. `pvalue` is `None` when nothing was scored."""
    evidence = aggregate_evidence(records, state)
    if evidence is None:
        logging.warning('exact caller: no opportunity row for %s; not called', state)
        return False, None
    support, opportunities = evidence
    if support == 0:
        logging.warning('exact caller: %s has no occurrence-scoped support although the '
                        'legacy count cleared the support floor, so every supporting '
                        'occurrence was ineligible; tail is 1.0 and it is not called',
                        state)
    if support > opportunities:
        logging.warning('exact caller: aggregated support %d exceeds opportunities %d '
                        'for %s, so the sibling denominator is a strict under-count; '
                        'not called', support, opportunities, state)
        return False, None
    probability = background.probability_for(state)
    logging.info('Exact tail inputs: k=%d N=%d p0=%s' % (support, opportunities,
                                                         probability))
    return (tail_below_cutoff(support, opportunities, probability, cutoff),
            exact_indel_tail(support, opportunities, probability))
