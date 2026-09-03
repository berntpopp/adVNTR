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
cases the legacy-named row sits at `support == 0` while its siblings carry the support.

**The two halves are not the same kind of quantity.**

- **`k` is a union over identities, attributed per read.** SPEC line 131: "Any future
  merge must union read/occurrence identities; it must never sum overlapping counts." An
  identity belongs to the `State` that ITS OWN read's whole-read fusion produced, which
  is what a row's `state_identities` map records
  (`advntr/frameshift_opportunities.py:observe_read`). The `legacy_states` list beside it
  is that map's key set -- a name index, unioned across every read behind the row, and
  not an attribution.
- **`N` is the scored `State`'s own row's `opportunities`.** A compound's components must
  be satisfied by ONE occurrence -- `advntr/frameshift_opportunities.py`'s module
  docstring, "an intersection within one occurrence and never a union" -- so a span that
  offers a sibling's shorter component set is not a trial for the longer `State`. The own
  row is always there at a decision site: every `State` that reaches one is a key of
  `mutations` or `prefix_suffix_mutations`, and `finalise` iterates
  `sorted(set(legacy_support) | set(self._support))`.

**Task 8a's version of this paragraph was wrong, and the correction is recorded rather
than quietly dropped.** It unioned BOTH halves, crediting a `State` with every sibling's
entire support and with the union of the siblings' `opportunity_spans`, and argued that
the span union was what made `k <= N` structural. Measured on the public corpus, that
`k` gave the six-deletion state at positions 17-22 of pattern 2 on
`example_dfc3_hg19_subset.bam` `k = 300` where 17 occurrences produced it -- the entire
245-occurrence support of its five-deletion sibling came with it, of which 3 belong --
and gave `I10_6_A_LEN2` on `example_6c28_hg19_subset.bam` `k = 347` where 2 occurrences
produced it. Inflating `k` shrinks the tail and inflating `N` widens it, so the two
errors point opposite ways; the `k` error is hundreds against tens, so the net is
anti-conservative.

**So `k <= N` is no longer structural, and must not be clamped back.** An identity
attributed to a fused `State` can come from an occurrence that never reached every one of
that state's reference positions -- the occurrence supports one component, and the read's
other occurrence supplies the other, but only the intersection is a trial. The
`support > opportunities` guard below is therefore a live path, not a defensive check: it
logs and refuses the call, which is the safe direction, and it is deliberately the only
handler. Clamping either number, or keeping the span union to make the arithmetic tidy,
would put back the over-count this replaced.

The earlier `sum(k)` / `max(N)` pair is recorded here so it is not reinvented: `max` is a
*lower bound* on the siblings' trials, and `sum` double-counts an occurrence two siblings
share. The "146x, flips the decision" figure that used to sit here came from hand-built
records no traversal produces (it needs two siblings of one fused `State` with disjoint
span sets, so that `max(N)` is 20 where the union is 40). Measured instead, across all
eight public `example_*` BAMs: `sum`/`max` differs from the union Task 8a shipped on two
states, in `N` only, by 0.60% and 0.39%, with no decision change demonstrated. The
argument against summing `k` stands on the identity union, not on that figure --
`tests/test_exact_caller_aggregation.py` keeps a hand-built fixture that flips a decision
at 7.8x and says on its face that it is hand-built.

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
    """`(k, N)` for one emitted `State`, or `None` when it has no row of its own.

    `k` unions only the identities each row ATTRIBUTED to this `State`, which is why the
    walk is over `state_identities` and not over `support_identities`; `N` is this
    `State`'s own row's `opportunities`. See the module docstring for both rulings and
    for what the union of the two whole fields cost when it shipped.

    Every row is visited rather than only the rows whose `legacy_states` name the state,
    because those two sets are the same one: `legacy_states` is `state_identities`' key
    set (`advntr/frameshift_opportunities.py:_record`).
    """
    own = records.get(state)
    if own is None:
        return None
    identities = set()
    for row in records.values():
        identities.update(row['state_identities'].get(state, ()))
    return len(identities), own['opportunities']


def decide(records, state, background, cutoff):
    """`(called, pvalue)` for one candidate. `pvalue` is `None` when nothing was scored.

    `k > N` is reachable here and is refused, never clamped: see the module docstring.
    """
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
                        'for %s, so at least one attributed occurrence never offered '
                        'every component of it; not called',
                        support, opportunities, state)
        return False, None
    probability = background.probability_for(state)
    logging.info('Exact tail inputs: k=%d N=%d p0=%s' % (support, opportunities,
                                                         probability))
    return (tail_below_cutoff(support, opportunities, probability, cutoff),
            exact_indel_tail(support, opportunities, probability))
