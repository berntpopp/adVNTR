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

The policy, and it is a policy rather than a derivation:

- **`k` sums** over every row that names this `State` (itself included). The rows count
  distinct `(read, occurrence)` identities, and a fused candidate's support really is
  spread across sibling rows.
- **`N` takes the maximum**, never the sum, because siblings draw on overlapping spans:
  `D11_2` and `D12_2` are offered by many of the same occurrences, so adding their
  denominators would count those occurrences twice and inflate `N` -- which suppresses
  real calls.
- **`k > N` after aggregation declines the candidate**, loudly, and calls nothing. This
  is not the clamp SPEC 3.3 forbids: `max` is a *lower bound* on the union of the
  siblings' spans, and the finalised records carry counts, not the identity sets that
  union would need. When the summed `k` passes that bound the denominator is known to be
  wrong, and the honest response is to refuse rather than to report a p-value computed
  from a denominator too small. Task 7's own per-row invariant still raises, by
  construction, where it applies (`frameshift_opportunities.finalise`).

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


def configured_background():
    """The frozen background for this run, or `None` when the exact caller is off.

    Raises `BackgroundModelError` when the flag is on without a valid artifact. Loaded
    once per `find_frameshift_from_selected_reads` call rather than cached: `-fs` runs
    one VNTR, and a cache would silently outlive an operator's edit to the file.
    """
    if not settings.EXACT_FRAMESHIFT_CALLER:
        return None
    path = settings.FRAMESHIFT_BACKGROUND_FILE
    if not path:
        raise BackgroundModelError(
            'the exact frameshift caller is enabled but no background model is '
            'configured: pass --frameshift-background <file>. There is deliberately no '
            'built-in default (SPEC Q-RATE), so this run cannot proceed.')
    background = load_background_model(path)
    logging.info(background.describe())
    return background


def aggregate_evidence(records, state):
    """`(k, N)` for one emitted `State`, or `None` when no row mentions it.

    See the module docstring for why `k` sums and `N` does not.
    """
    siblings = [row for candidate, row in records.items()
                if candidate == state or state in row.get('legacy_states', ())]
    if not siblings:
        return None
    return (sum(row['support'] for row in siblings),
            max(row['opportunities'] for row in siblings))


def decide(records, state, background, cutoff):
    """`(called, pvalue)` for one candidate. `pvalue` is `None` when nothing was scored."""
    evidence = aggregate_evidence(records, state)
    if evidence is None:
        logging.warning('exact caller: no opportunity row for %s; not called', state)
        return False, None
    support, opportunities = evidence
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
