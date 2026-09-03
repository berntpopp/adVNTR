"""Occurrence-aware candidate opportunity counters for `genotype -fs`, in shadow mode.

PLAN Task 7 needs an integer Bernoulli pair `(k, N)` per frameshift candidate so Task 8
can run an exact binomial test. SPEC 3.1 rejects `round(ru_bp_coverage / ru_length)` as
`N`: `ru_bp_coverage` counts emitted `M` and `I` states and no `D` at all (Q-COV,
`advntr/hmm_utils.py:326` via `is_matching_state`, `:135-139`), while `k` is a per-read
count that has already lost the repeat occurrence (Q-OCC). Nothing here changes a call:
the counters reach the outside world only through
`VNTRFinder.last_frameshift_opportunities` and one INFO log line, never through the six
column result table `advntr/genome_analyzer.py:215-223` prints.

The observation identity is `(selected_read_index, query_name, repeat_occurrence)` --
the first three fields of Task 5's `FrameshiftEvidence`
(`advntr/mutation_keys.py:13-16`), reused rather than re-declared. Both `k` and `N`
deduplicate on exactly that triple. `query_name` is `None` on the unmapped-recruited
path (`advntr/vntr_finder.py:182` constructs a `SelectedRead` without one, so the
default at `:41` applies), so `selected_read_index` is the primary key and `query_name`
is descriptive; nothing here groups by `query_name` alone.

Slot semantics, read off `get_repeat_matcher_enhanced_hmm`
(`advntr/hmm_utils.py:622-701`) and the two flank matchers (`:360-423`, `:427-490`):

- deletion site `D i p`: the occurrence visited `M{i}_{p}` or `D{i}_{p}`, the only two
  states at reference position `i` (`advntr/hmm_utils.py:644-649`).
- insertion slot `I i p`: the occurrence visited `I{i}_{p}` itself, or it reached the
  reference position before the slot AND the one after it. `I_i` is entered only from
  `M_i`/`D_i`/itself (`advntr/hmm_utils.py:683-685`) and left only to
  `M_{i+1}`/`D_{i+1}`/itself (`:687-688`), with `I_0` entered from `unit_start`
  (`:661-663`) and `I_L` left to `unit_end` (`:671-680`), so the slot was genuinely
  available exactly when the path crossed it. The one-sided form ("reached position i")
  would credit a slot to a read whose last emitted base sits at `i`, where no insertion
  could have been observed; that inflates `N` at read ends, and does so differently for
  prefix and suffix reads. The reverse temptation -- crediting `D i p` to a read that
  stopped at `i-1` because a deletion emits nothing -- is rejected too: it means
  enumerating DP paths the decoder did not take. The rule is "the reference position was
  actually reached".
- compound `A&B&...`: the SAME occurrence satisfies EVERY component, an intersection
  within one occurrence and never a union. The pattern index comes off field 1 of each
  component exactly as `advntr/vntr_finder.py:474` reads it, and every component must
  match the occurrence's own submodel, so a candidate for pattern `p` can only ever draw
  on occurrences of `p`.

**Eligibility is a counterfactual, not a transcription.** Every legacy filter runs only
at an `I` or `D` state, because the read loop `continue`s otherwise
(`advntr/vntr_finder.py:336-337`); a clean occurrence reaches none of them. The
predicate here therefore asks *"would this occurrence have been eligible had a candidate
indel sat at this slot?"*.

- Applied: `settings.USE_ONLY_FULLY_COVERED_RU` (`advntr/vntr_finder.py:341-342`), the
  partial-occurrence `M >= 5` and `S < 4` tests (`:345-348`), the three read-level
  rejections (`:366`, `:373`, `:379`) evaluated against the occurrence's own
  `pattern_length`, and the 0.9 flank mutation/match ratio (`:414`, `:418`).
- **Not** applied: the `I == D` balance tests (`:349` for partial occurrences, `:359`
  for complete ones). A clean occurrence has `I == D == 0`, so transcribing them
  literally would reject every clean occurrence and delete the entire zero-support
  inventory -- exactly the selection conditioning PLAN Task 7 forbids. The consequence is
  deliberate and worth stating: an occurrence carrying a balanced 1I/1D pair is an
  opportunity that contributes no support.
- **Not** applied: `settings.MIN_SUPPORTING_READ_COUNT` or `INDEL_MUTATION_MIN_PVALUE`.
  `N` is a property of the data, not of the caller's threshold.

Two places where `k` deliberately does not equal the legacy per-read count, both carried
out to the caller rather than left to be inferred:

- **The candidate NAME can differ.** Rebuilding candidates per occurrence both un-fuses
  adjacent deletions and renumbers an insertion's `_LEN` suffix, so the legacy-named row
  can sit at `support == 0` beside the occurrence-scoped rows that carry its support.
  Every row's `legacy_states` field names the shipped `State` strings its support belongs
  to -- see `per_occurrence_candidates`.
- **Flank support can exceed the legacy count.** `advntr/vntr_finder.py:402-403`
  short-circuits the whole prefix/suffix block for a read with no repeat-unit mutation,
  so the legacy never counts that read's flank indel; `observe_read` is called before
  that `continue` and `accepted_raw_mutations` already holds the flank raws. A flank row
  can therefore show `support >= 1` with `legacy_support == 0`. The shadow number is the
  more complete one, but Task 8 must not read the difference as an error.

Every row carries the identity sets behind its two counts, not only the counts:
`support_identities` is the deduplicated `(read, occurrence)` pairs that produced the
candidate, and `opportunity_spans` is `(span id, count)` for every distinct span shape
that offered it. Task 8's caller has to UNION those across sibling rows rather than sum
or max them (SPEC line 131), and cardinalities alone cannot be unioned. Span ids stand
in for the identities behind each denominator because spans partition the identities --
one `(read, occurrence)` is recorded under exactly one signature -- so a union over span
ids costs distinct shapes rather than candidates x reads. Neither field is encoded into
the diagnostics line; see `UNENCODED_FIELDS`.

The read-level rejections are read-scoped in the legacy loop (`is_valid_read = False`
then `break`), but they are evaluated here per occurrence. That is the tightest form
that still preserves the subset property: an occurrence that produced support must have
passed them when the loop reached its indel, so support is always a subset of
opportunity and `k <= N` holds by construction. A violation is therefore an
implementation bug and raises.
"""
from collections import OrderedDict, defaultdict, namedtuple
import json
import logging

from advntr.mutation_keys import legacy_mutation_candidates, occurrence_labels
from advntr import settings


DIAGNOSTICS_VERSION = 1

#: `advntr/advntr_commands.py:96-101` greps the run log for "alignment file for" and
#: "INFO:find_frameshift_from_alignment" to resume in append mode. A diagnostic carrying
#: either substring would corrupt that resume count, so this prefix contains neither.
LOG_PREFIX = 'frameshift opportunity counters (shadow, no call effect): '

#: Carried on every record for Task 8's exact caller, and deliberately kept OUT of the
#: encoded diagnostics. Two reasons, in order: the encoding is already 213 KB on
#: example_66bf's 1014 candidates and these fields are O(observations) and O(distinct
#: span shapes) per candidate on top of that, and they are an in-process interface for
#: `advntr/exact_caller.py`, not a diagnostic anyone reads out of a log. They carry no
#: read name either way -- see `anonymous_identities`.
UNENCODED_FIELDS = ('support_identities', 'opportunity_spans')

FLANK_OCCURRENCES = ('suffix_flank', 'prefix_flank')
PARTIAL_OCCURRENCES = ('partial_start', 'partial_end')

#: `signature` is `(pattern_index, reached_mask, inserted_mask, saw_start, saw_end)` --
#: everything the slot rules need. Spans with an identical signature are pooled, so
#: `finalise` costs candidates x distinct shapes rather than candidates x reads.
OccurrenceSpan = namedtuple('OccurrenceSpan', 'occurrence signature')


def _position_of(state):
    """(kind, position) for an `M`/`I`/`D` state, else `(None, None)`.

    `start_random_matches` and `end_random_matches` (`advntr/hmm_utils.py:824-826`)
    consume a read base but belong to no submodel and no occurrence, so they fall
    through here and contribute to no slot.
    """
    head = state.split('_')[0]
    if head[:1] in ('M', 'I', 'D') and head[1:].isdigit():
        return head[0], int(head[1:])
    return None, None


def anonymous_identities(identities):
    """The `(read, occurrence)` half of the observation identity, deduplicated and sorted.

    Task 8's caller unions these across sibling rows (SPEC line 131: "Any future merge
    must union read/occurrence identities; it must never sum overlapping counts"), so
    the identities have to leave this module rather than only their cardinality.

    `query_name` is dropped rather than carried. It is descriptive, not part of the key
    -- one selected read has exactly one name, so the pair and the triple have identical
    cardinality -- and dropping it means nothing derived from these fields can leak a
    read name, which is the anonymity property `tests/test_frameshift_context.py:199`
    pins for Task 5's Context column and this module's own diagnostics test pins here.
    """
    return tuple(sorted(set((index, occurrence)
                            for index, _query_name, occurrence in identities)))


def occurrence_spans(visited_states):
    """One span per `(read, occurrence)`, whether or not an indel occurred.

    This is the piece Task 5 deliberately did not build: nothing else in the tree
    enumerates an occurrence that produced no mutation, because
    `advntr/vntr_finder.py:402-403` short-circuits on an empty mutation map before any
    evidence is recorded, so a clean read leaves no trace beyond `ru_bp_coverage`.
    """
    labels = occurrence_labels(visited_states)
    patterns = {}
    reached = defaultdict(int)
    inserted = defaultdict(int)
    saw_start = defaultdict(bool)
    saw_end = defaultdict(bool)
    order = []
    for index, state in enumerate(visited_states):
        occurrence = labels[index]
        if occurrence not in patterns:
            order.append(occurrence)
            patterns[occurrence] = set()
        kind, position = _position_of(state)
        if kind is not None:
            patterns[occurrence].add(state.split('_')[-1])
            if kind == 'I':
                inserted[occurrence] |= 1 << position
            else:
                reached[occurrence] |= 1 << position
        elif _is_start_terminator(state):
            patterns[occurrence].add(state.split('_')[-1])
            saw_start[occurrence] = True
        elif _is_end_terminator(state):
            patterns[occurrence].add(state.split('_')[-1])
            saw_end[occurrence] = True
    spans = []
    for occurrence in order:
        found = patterns[occurrence]
        pattern_index = list(found)[0] if len(found) == 1 else None
        if len(found) > 1:
            logging.debug('opportunity: occurrence %s carries %d model types; skipped',
                          occurrence, len(found))
        elif not found:
            # The ordinary leading `start_random_matches` span, which belongs to no
            # submodel (`advntr/hmm_utils.py:824-826`) -- not a model conflict.
            logging.debug('opportunity: occurrence %s has no submodel state; skipped',
                          occurrence)
        spans.append(OccurrenceSpan(occurrence,
                                    (pattern_index, reached[occurrence],
                                     inserted[occurrence], saw_start[occurrence],
                                     saw_end[occurrence])))
    return spans


def _is_start_terminator(state):
    """The flank matchers name the same two silent states their own way:
    `suffix_start_suffix`/`suffix_end_suffix` (`advntr/hmm_utils.py:445-446`) and
    `prefix_start_prefix`/`prefix_end_prefix` (`:378-379`)."""
    return (state.startswith('unit_start') or state.startswith('suffix_start')
            or state.startswith('prefix_start'))


def _is_end_terminator(state):
    return (state.startswith('unit_end') or state.startswith('suffix_end')
            or state.startswith('prefix_end'))


def parse_components(candidate):
    """`(kind, position, pattern_index)` per `A&B&...` component, or `None`.

    `I2_1_T_LEN2` -> `('I', 2, '1')`, `D3_1` -> `('D', 3, '1')`,
    `I0_prefix_LEN1` -> `('I', 0, 'prefix')`. Field 1 is the pattern index, exactly as
    `advntr/vntr_finder.py:474` takes it off a candidate.
    """
    components = []
    for component in candidate.split('&'):
        fields = component.split('_')
        kind, position = _position_of(component)
        if kind is None or kind == 'M' or len(fields) < 2:
            return None
        components.append((kind, position, fields[1]))
    return components


#: Submodels whose end terminator does NOT identify the last reference position: the
#: prefix matcher alone, because `advntr/hmm_utils.py:416` gives EVERY match state a 0.01
#: transition to `prefix_end_prefix`, so a prefix span can carry the end terminator
#: having exited one base in. The set is written negatively because its complement is
#: open: repeat submodels are named by pattern index `'1'`, `'2'`, ... with no fixed
#: upper bound, so no positive literal can enumerate them. Everything absent from here --
#: the repeat submodels (`advntr/hmm_utils.py:670-680`: `D_L`, `M_L`, `I_L` and nothing
#: else) and the suffix matcher (`:465-472`, same shape) -- reaches its end terminator
#: only from the last position, so seeing that terminator identifies the position without
#: the counter knowing the model's length: a path is monotone forward, so the highest
#: position it reached IS where it exited.
_END_DOES_NOT_MARK_LAST_POSITION = ('prefix',)


def _insertion_slot_crossed(reached, inserted, saw_start, saw_end, position, end_marks_last):
    """Did this occurrence cross insertion slot `position`?

    `before` at `position == 0` is the start terminator and needs no further gate: `I0`
    has exactly one non-self predecessor in all three submodels -- `unit_start`
    (`advntr/hmm_utils.py:663`), `suffix_start` (`:457`), `prefix_start` (`:391`) -- and
    exactly one set of successors, `{I0, M1, D1}` (`:666-668`, `:461-463`, `:393-395`).
    So the left flank's any-offset entry (`:459`, `unit_start -> match_states[i]` for
    EVERY i) cannot inflate it: a span that entered at `M5_suffix` never reaches position
    1, so `after` is false and the slot is not credited. That is why `saw_end` must not
    satisfy `after` at position 0 either -- no `I0` anywhere transitions to an end
    terminator, so an end terminator can never be evidence that slot 0 was crossed.

    `after` at `position >= 1` is the end terminator only when it pins the last reference
    position (see `_END_DOES_NOT_MARK_LAST_POSITION`). Without that gate a read exiting
    `M1_prefix -> prefix_end_prefix` -- its last emitted base sitting AT the slot -- would
    be credited for `I1_prefix`, which is precisely the read-end inflation this module
    rejects, and it is asymmetric between prefix and suffix reads.

    The cost of excluding the prefix matcher is exactly one slot: `I{L}_prefix` is not
    credited to a prefix span that ran to `prefix_end_prefix` without visiting `I{L}`.
    That is conservative about what the counter CLAIMS -- it never asserts an opportunity
    the decoder did not demonstrate -- and it is emphatically not conservative about
    calling: in the exact binomial a smaller `N` at the same `k` LOWERS the p-value, so
    under-counting a denominator pushes toward a call, not away from one. The
    load-bearing half of the argument is therefore the other one: no candidate at that
    slot is callable at all, because `advntr/vntr_finder.py:524` gates prefix candidates
    on a small leading-nucleotide-run boundary. (The flank length is also not reachable
    from here without new arguments the LOC ratchet has no room for.)
    """
    if (inserted >> position) & 1:
        return True
    if position == 0:
        return saw_start and bool((reached >> 1) & 1)
    if not (reached >> position) & 1:
        return False
    if (reached >> (position + 1)) & 1:
        return True
    # A monotone forward path exits from its highest reached position.
    return saw_end and end_marks_last and position == reached.bit_length() - 1


def _signature_supports(signature, components):
    """Whether one occurrence offered every component of a candidate."""
    pattern_index, reached, inserted, saw_start, saw_end = signature
    if pattern_index is None:
        return False
    for kind, position, pattern in components:
        if pattern != pattern_index:
            return False
        if kind == 'D':
            if not (reached >> position) & 1:
                return False
        elif not _insertion_slot_crossed(
                reached, inserted, saw_start, saw_end, position,
                pattern_index not in _END_DOES_NOT_MARK_LAST_POSITION):
            return False
    return True


def occurrence_counts(ru_state_count, occurrence):
    """The M/I/D/S counts of one occurrence, or `None` when it has no entry at all.

    `ru_state_count` is a `defaultdict(lambda: defaultdict(int))`
    (`advntr/hmm_utils.py:158`), so a missing occurrence silently reads as all-zero and
    the legacy `M >= 5` test at `advntr/vntr_finder.py:345` then skips with no
    diagnostic. Membership is checked first so the denominator does not inherit that
    silence -- and so reading the counts cannot insert an entry into the caller's dict.
    """
    if occurrence not in ru_state_count:
        return None
    counts = ru_state_count[occurrence]
    return dict((key, counts.get(key, 0)) for key in ('M', 'I', 'D', 'S'))


def flank_ratio_gates(visited_states):
    """Mirror the flank mutation/match ratio filter at `advntr/vntr_finder.py:414`, `:418`.

    Note the direction: a flank candidate is KEPT only when its mutations are at least
    0.9x its matches, and the test is skipped entirely when the flank has no match state.
    Every state ending in `fix` counts, including the silent `prefix_start_prefix` and
    `suffix_end_suffix`, which do not start with `M` and so land on the mutation side --
    that is the legacy behaviour, mirrored rather than corrected.
    """
    tally = {'prefix': [0, 0], 'suffix': [0, 0]}
    for state in visited_states:
        if not state.endswith('fix'):
            continue
        side = 'prefix' if state.endswith('prefix') else 'suffix'
        tally[side][0 if state.startswith('M') else 1] += 1
    gates = {}
    for side, (matches, mutations) in tally.items():
        gates['%s_flank' % side] = matches == 0 or mutations / float(matches) >= 0.9
    return gates


def is_eligible(span, ru_state_count, pattern_clusters, gates):
    """Would this occurrence have been eligible had a candidate indel sat in it?

    See the module docstring for which legacy filters this applies and which two it
    deliberately does not.
    """
    pattern_index = span.signature[0]
    if pattern_index is None:
        return False
    if span.occurrence in FLANK_OCCURRENCES:
        return gates.get(span.occurrence, False)
    counts = occurrence_counts(ru_state_count, span.occurrence)
    if counts is None:
        logging.debug('opportunity: occurrence %s has no state counts; not an opportunity',
                      span.occurrence)
        return False
    if span.occurrence in PARTIAL_OCCURRENCES:
        if settings.USE_ONLY_FULLY_COVERED_RU:
            return False
        return counts['M'] >= 5 and counts['S'] < 4
    if not pattern_index.isdigit():
        return False
    index = int(pattern_index) - 1
    if index < 0 or index >= len(pattern_clusters):
        return False
    pattern_length = len(pattern_clusters[index][0])
    half_pattern = pattern_length // 2  # `/` in the legacy tests; both operands are ints
    if abs(counts['M'] + counts['I'] - pattern_length) > half_pattern:
        return False
    if counts['I'] + counts['D'] > half_pattern:
        return False
    return counts['S'] < 4


def _visit_count(raw_mutation):
    """The legacy `_LEN` suffix counts state visits, not mutation records.

    `extract_raw_mutations` collapses an insertion run to its first index
    (`advntr/mutation_keys.py:138-158`), so the run length is the visit count
    `mutation_count_temp` would have accumulated at `advntr/vntr_finder.py:353`/`:389`.
    """
    if raw_mutation.event.type == 'I':
        return len(raw_mutation.event.inserted_sequence)
    return 1


def _flank_candidates(counts):
    """Mirror the flank candidate names built at `advntr/vntr_finder.py:422`."""
    return [(state + ('_LEN%d' % count if state.startswith('I') else ''), (state,))
            for state, count in counts.items()]


def per_occurrence_candidates(accepted_raw_mutations):
    """Rebuild legacy candidates once per occurrence, not once per read.

    `mutation_count_temp` is keyed on the state string alone and summed over the whole
    read, so `legacy_mutation_candidates` then fuses adjacent deletions ACROSS
    occurrences (`advntr/mutation_keys.py:189`) and derives `_LEN%d` from the whole-read
    count -- Q-OCC exactly. Reconstructing per occurrence undoes that for the shadow
    counters while leaving the legacy `State` untouched.

    The per-occurrence map must be an `OrderedDict`: `legacy_mutation_candidates` walks
    its input sequentially and compares each key with its predecessor
    (`advntr/mutation_keys.py:182-209`), so first-appearance order within an occurrence
    is load bearing. The insertion base tag inside `legacy_key` is left alone -- it is
    the FIRST emitted base for that state name across the WHOLE read
    (`advntr/hmm_utils.py:125-132`, mirrored at `advntr/mutation_keys.py:124-125`), and
    recomputing it per occurrence would mint candidate keys the shipped caller never
    emits.

    Reconstruction changes the candidate NAME in two ways, not one, and both are carried
    out to the caller as `legacy_states` rather than left for Task 8 to infer:

    - fusion: two deletions in different occurrences give `D11_2` and `D12_2` here where
      the whole-read map gives the single fused `D11_2&D12_2`;
    - length renumbering: one read inserting at `I2_1` in two occurrences gives
      `I2_1_T_LEN1` twice here where the whole-read map gives `I2_1_T_LEN2`, because
      `legacy_mutation_candidates` derives `_LEN%d` from its input count
      (`advntr/mutation_keys.py:167-170`).

    In both cases the legacy-named row ends up with `support == 0` while its
    occurrence-scoped siblings carry the support, so a consumer keying `(k, N)` off the
    `State` column would read zero for a genuinely supported candidate. Each row's
    `legacy_states` names the shipped `State` strings its support belongs to; the
    whole-read reconstruction that produces them is run here on the same
    `accepted_raw_mutations`, minus the flank raws that `mutation_count_temp` never sees
    (`advntr/vntr_finder.py:320` diverts them).

    Returns `occurrence -> [(candidate, contributing raw mutations, legacy State names)]`.
    """
    grouped = OrderedDict()
    sources = OrderedDict()
    whole_read = OrderedDict()
    for raw in sorted(accepted_raw_mutations, key=lambda item: item.visited_index):
        occurrence = raw.repeat_occurrence
        counts = grouped.setdefault(occurrence, OrderedDict())
        counts[raw.legacy_key] = counts.get(raw.legacy_key, 0) + _visit_count(raw)
        sources.setdefault(occurrence, []).append(raw)
        if occurrence not in FLANK_OCCURRENCES:
            whole_read[raw.legacy_key] = whole_read.get(raw.legacy_key, 0) + _visit_count(raw)
    legacy_state_of_key = {}
    if whole_read:
        for state, legacy_keys in legacy_mutation_candidates(whole_read):
            for key in legacy_keys:
                legacy_state_of_key[key] = state
    candidates = OrderedDict()
    for occurrence, counts in grouped.items():
        if occurrence in FLANK_OCCURRENCES:
            built = _flank_candidates(counts)
        else:
            built = legacy_mutation_candidates(counts)
        rows = []
        for candidate, legacy_keys in built:
            raws = tuple(raw for raw in sources[occurrence]
                         if raw.legacy_key in legacy_keys)
            if occurrence in FLANK_OCCURRENCES:
                # One flank pseudo-occurrence per read per flank, and no fusion at
                # `advntr/vntr_finder.py:422`, so the two names always agree.
                states = (candidate,)
            else:
                states = tuple(sorted(set(legacy_state_of_key[key] for key in legacy_keys
                                          if key in legacy_state_of_key)))
            rows.append((candidate, raws, states))
        candidates[occurrence] = rows
    return candidates


class OpportunityCounter(object):
    """Accumulate integer `(k, N)` per candidate over one `find_frameshift` invocation."""

    def __init__(self, pattern_clusters, estimated_ru_count, hmm_match_count, is_haploid):
        self._pattern_clusters = pattern_clusters
        self._estimated_ru_count = estimated_ru_count
        self._hmm_match_count = hmm_match_count
        self._is_haploid = is_haploid
        self._support = defaultdict(list)
        self._rollup = defaultdict(set)
        self._spans = OrderedDict()

    def observe_read(self, selected_read_index, query_name, visited_states,
                     accepted_raw_mutations, ru_state_count):
        """Record every occurrence of one valid read, indel or no indel."""
        gates = flank_ratio_gates(visited_states)
        supported = per_occurrence_candidates(accepted_raw_mutations)
        for span in occurrence_spans(visited_states):
            if not is_eligible(span, ru_state_count, self._pattern_clusters, gates):
                continue
            identity = (selected_read_index, query_name, span.occurrence)
            self._spans.setdefault(span.signature, []).append(identity)
            for candidate, raw_mutations, legacy_states in supported.get(span.occurrence, ()):
                for _raw in raw_mutations:
                    self._support[candidate].append(identity)
                self._rollup[candidate].update(legacy_states)

    def finalise(self, mutations, prefix_suffix_mutations, ru_bp_coverage):
        """Integer `(k, N)` per candidate, with the rejected denominator beside them.

        The candidate universe is the union of the legacy candidates and the
        occurrence-scoped ones, so a fused `D11_2&D12_2` keeps a row with `k = 0` next to
        the `D11_2` and `D12_2` rows that actually carry its support.
        """
        legacy_support = dict(mutations)
        legacy_support.update(prefix_suffix_mutations)
        # `self._spans` is an OrderedDict, so enumeration gives each distinct span shape
        # a stable id within one invocation. Spans partition the identities -- each
        # `(read, occurrence)` is recorded under exactly one signature in
        # `observe_read` -- so unioning span ids and adding their counts is exactly
        # unioning the identities behind them, at the cost of the distinct shapes rather
        # than of candidates x reads.
        span_counts = [(span_id, signature, len(set(identities)))
                       for span_id, (signature, identities)
                       in enumerate(self._spans.items())]
        records = OrderedDict()
        for candidate in sorted(set(legacy_support) | set(self._support)):
            components = parse_components(candidate)
            identities = anonymous_identities(self._support.get(candidate, ()))
            support = len(identities)
            spans = ()
            if components is not None:
                spans = tuple((span_id, count) for span_id, signature, count
                              in span_counts
                              if _signature_supports(signature, components))
            opportunities = sum(count for _span_id, count in spans)
            if not 0 <= support <= opportunities:
                raise ValueError(
                    'frameshift opportunity invariant violated for candidate %s: '
                    'support %d, opportunities %d (0 <= k <= N must hold by '
                    'construction; do not clamp)' % (candidate, support, opportunities))
            records[candidate] = self._record(candidate, components, support,
                                              opportunities, identities, spans,
                                              legacy_support.get(candidate, 0),
                                              ru_bp_coverage)
        if logging.getLogger().isEnabledFor(logging.INFO):
            # Encoding is not free -- 213 KB on example_66bf's 1014 candidates -- and the
            # attribute, not the log line, is the interface the harness reads.
            logging.info('%s%s', LOG_PREFIX, encode_opportunity_diagnostics(records))
        return records

    def _record(self, candidate, components, support, opportunities, identities, spans,
                legacy, ru_bp_coverage):
        """One diagnostic row: `(k, N)` beside the denominator SPEC 3.1 rejects.

        `identities` and `spans` are the `(read, occurrence)` evidence behind `support`
        and `opportunities`; `support == len(identities)` and
        `opportunities == sum(count for _, count in spans)` hold by construction, and
        Task 8 needs the sets rather than the counts so siblings can be unioned.

        `round(ru_bp_coverage / ru_length)` is carried to QUANTIFY the unit mismatch that
        PLAN Task 7 Step 3 asks about, never as a definition of `N`. `hmm_match_count` is
        a `defaultdict(int)` (`advntr/vntr_finder.py:214`), so `ru_length` can be `0` and
        `advntr/vntr_finder.py:451`'s division is a live `ZeroDivisionError`; the ratio
        is guarded here and reported as `None` instead. Flank candidates report `None`
        too: the legacy flank denominator borrows a repeat-unit index from
        `reference_repeat_order` (`advntr/vntr_finder.py:515`, `:525`), and rebuilding
        that is Task 8's business, not a shadow counter's.
        """
        pattern_index = components[0][2] if components else None
        ru_length = self._hmm_match_count.get(pattern_index)
        total_bps = ru_bp_coverage.get(pattern_index)
        copies = self._estimated_ru_count.get(pattern_index)
        ratio = None
        average = None
        if ru_length and total_bps is not None:
            ratio = int(round(total_bps / float(ru_length)))
            if copies:
                # Left-to-right exactly as `advntr/vntr_finder.py:451`/`:454` associate
                # it. `x / (a * b * c)` is equal in exact arithmetic but can differ in
                # the last ulp, and this field exists only to sit beside the printed
                # MeanCoverage. Dividing by 1 is exact, so the haploid branch, which has
                # no ploidy divisor at all, is reproduced by the same expression.
                ploidy = 1 if self._is_haploid else 2
                average = float(total_bps) / ru_length / ploidy / copies
        return {'candidate': candidate, 'support': support,
                'opportunities': opportunities, 'support_identities': identities,
                'opportunity_spans': spans, 'legacy_support': legacy,
                'legacy_states': sorted(self._rollup.get(candidate, ())),
                'pattern_index': pattern_index, 'ru_bp_coverage': total_bps,
                'ru_length': ru_length, 'ru_bp_coverage_ratio': ratio,
                'avg_bp_coverage': average}


def encode_opportunity_diagnostics(records, version=DIAGNOSTICS_VERSION):
    """Deterministic, versioned, and free of read names.

    Same idiom as `advntr/mutation_keys.encode_frameshift_context` (`:237-251`). Sorting
    the ENCODED strings is the load-bearing step, not decoration: `mutations` and
    `candidate_evidence` are plain `defaultdict`s (`advntr/vntr_finder.py:213-215`) whose
    Python 2 iteration order is hash order, and a single-threaded unit test would not
    catch the difference. No `query_name` appears in any field, mirroring the anonymity
    property `tests/test_frameshift_context.py:199` pins for Task 5's Context column.
    """
    encoded = [json.dumps(dict((key, value) for key, value in record.items()
                               if key not in UNENCODED_FIELDS),
                          sort_keys=True, separators=(',', ':'))
               for record in records.values()]
    encoded.sort()
    return '{"v":%d,"candidates":[%s]}' % (version, ','.join(encoded))
