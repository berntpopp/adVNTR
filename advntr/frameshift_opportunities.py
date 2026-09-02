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
path (`advntr/vntr_finder.py:181` constructs a `SelectedRead` without one, so the
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
  component exactly as `advntr/vntr_finder.py:432` reads it, and every component must
  match the occurrence's own submodel, so a candidate for pattern `p` can only ever draw
  on occurrences of `p`.

**Eligibility is a counterfactual, not a transcription.** Every legacy filter runs only
at an `I` or `D` state, because the read loop `continue`s otherwise
(`advntr/vntr_finder.py:332-333`); a clean occurrence reaches none of them. The
predicate here therefore asks *"would this occurrence have been eligible had a candidate
indel sat at this slot?"*.

- Applied: `settings.USE_ONLY_FULLY_COVERED_RU` (`advntr/vntr_finder.py:337-338`), the
  partial-occurrence `M >= 5` and `S < 4` tests (`:341-344`), the three read-level
  rejections (`:362`, `:369`, `:375`) evaluated against the occurrence's own
  `pattern_length`, and the 0.9 flank mutation/match ratio (`:410`, `:414`).
- **Not** applied: the `I == D` balance tests (`:345` for partial occurrences, `:355`
  for complete ones). A clean occurrence has `I == D == 0`, so transcribing them
  literally would reject every clean occurrence and delete the entire zero-support
  inventory -- exactly the selection conditioning PLAN Task 7 forbids. The consequence is
  deliberate and worth stating: an occurrence carrying a balanced 1I/1D pair is an
  opportunity that contributes no support.
- **Not** applied: `settings.MIN_SUPPORTING_READ_COUNT` or `INDEL_MUTATION_MIN_PVALUE`.
  `N` is a property of the data, not of the caller's threshold.

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

FLANK_OCCURRENCES = ('suffix_flank', 'prefix_flank')
PARTIAL_OCCURRENCES = ('partial_start', 'partial_end')

#: `pattern_index`, plus the bit sets and terminator flags the slot rules need. Spans
#: with an identical signature are pooled, so `finalise` costs candidates x distinct
#: shapes rather than candidates x reads.
OccurrenceSpan = namedtuple('OccurrenceSpan', 'occurrence signature patterns')


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


def occurrence_spans(visited_states):
    """One span per `(read, occurrence)`, whether or not an indel occurred.

    This is the piece Task 5 deliberately did not build: nothing else in the tree
    enumerates an occurrence that produced no mutation, because
    `advntr/vntr_finder.py:398-399` short-circuits on an empty mutation map before any
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
        if pattern_index is None:
            logging.debug('opportunity: occurrence %s carries %d model types; skipped',
                          occurrence, len(found))
        spans.append(OccurrenceSpan(occurrence,
                                    (pattern_index, reached[occurrence],
                                     inserted[occurrence], saw_start[occurrence],
                                     saw_end[occurrence]),
                                    found))
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
    `advntr/vntr_finder.py:432` takes it off a candidate.
    """
    components = []
    for component in candidate.split('&'):
        fields = component.split('_')
        kind, position = _position_of(component)
        if kind is None or kind == 'M' or len(fields) < 2:
            return None
        components.append((kind, position, fields[1]))
    return components


def _insertion_slot_crossed(reached, inserted, saw_start, saw_end, position):
    if (inserted >> position) & 1:
        return True
    if position == 0:
        before = saw_start
    else:
        before = bool((reached >> position) & 1)
    after = bool((reached >> (position + 1)) & 1) or saw_end
    return before and after


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
        elif not _insertion_slot_crossed(reached, inserted, saw_start, saw_end, position):
            return False
    return True


def occurrence_counts(ru_state_count, occurrence):
    """The M/I/D/S counts of one occurrence, or `None` when it has no entry at all.

    `ru_state_count` is a `defaultdict(lambda: defaultdict(int))`
    (`advntr/hmm_utils.py:158`), so a missing occurrence silently reads as all-zero and
    the legacy `M >= 5` test at `advntr/vntr_finder.py:341` then skips with no
    diagnostic. Membership is checked first so the denominator does not inherit that
    silence -- and so reading the counts cannot insert an entry into the caller's dict.
    """
    if occurrence not in ru_state_count:
        return None
    counts = ru_state_count[occurrence]
    return dict((key, counts.get(key, 0)) for key in ('M', 'I', 'D', 'S'))


def flank_ratio_gates(visited_states):
    """Mirror the flank mutation/match ratio filter at `advntr/vntr_finder.py:410`, `:414`.

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
    `mutation_count_temp` would have accumulated at `advntr/vntr_finder.py:349`/`:385`.
    """
    if raw_mutation.event.type == 'I':
        return len(raw_mutation.event.inserted_sequence)
    return 1


def _flank_candidates(counts):
    """Mirror the flank candidate names built at `advntr/vntr_finder.py:418`."""
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
    """
    grouped = OrderedDict()
    sources = OrderedDict()
    for raw in sorted(accepted_raw_mutations, key=lambda item: item.visited_index):
        occurrence = raw.repeat_occurrence
        counts = grouped.setdefault(occurrence, OrderedDict())
        counts[raw.legacy_key] = counts.get(raw.legacy_key, 0) + _visit_count(raw)
        sources.setdefault(occurrence, []).append(raw)
    candidates = OrderedDict()
    for occurrence, counts in grouped.items():
        if occurrence in FLANK_OCCURRENCES:
            built = _flank_candidates(counts)
        else:
            built = legacy_mutation_candidates(counts)
        candidates[occurrence] = [
            (candidate, tuple(raw for raw in sources[occurrence]
                              if raw.legacy_key in legacy_keys))
            for candidate, legacy_keys in built]
    return candidates


class OpportunityCounter(object):
    """Accumulate integer `(k, N)` per candidate over one `find_frameshift` invocation."""

    def __init__(self, pattern_clusters, estimated_ru_count, hmm_match_count, is_haploid):
        self._pattern_clusters = pattern_clusters
        self._estimated_ru_count = estimated_ru_count
        self._hmm_match_count = hmm_match_count
        self._is_haploid = is_haploid
        self._support = defaultdict(list)
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
            for candidate, raw_mutations in supported.get(span.occurrence, ()):
                for _raw in raw_mutations:
                    self._support[candidate].append(identity)

    def finalise(self, mutations, prefix_suffix_mutations, ru_bp_coverage):
        """Integer `(k, N)` per candidate, with the rejected denominator beside them.

        The candidate universe is the union of the legacy candidates and the
        occurrence-scoped ones, so a fused `D11_2&D12_2` keeps a row with `k = 0` next to
        the `D11_2` and `D12_2` rows that actually carry its support.
        """
        legacy_support = dict(mutations)
        legacy_support.update(prefix_suffix_mutations)
        span_counts = [(signature, len(set(identities)))
                       for signature, identities in self._spans.items()]
        records = OrderedDict()
        for candidate in sorted(set(legacy_support) | set(self._support)):
            components = parse_components(candidate)
            support = len(set(self._support.get(candidate, ())))
            opportunities = 0
            if components is not None:
                for signature, count in span_counts:
                    if _signature_supports(signature, components):
                        opportunities += count
            if not 0 <= support <= opportunities:
                raise ValueError(
                    'frameshift opportunity invariant violated for candidate %s: '
                    'support %d, opportunities %d (0 <= k <= N must hold by '
                    'construction; do not clamp)' % (candidate, support, opportunities))
            records[candidate] = self._record(candidate, components, support,
                                              opportunities,
                                              legacy_support.get(candidate, 0),
                                              ru_bp_coverage)
        logging.info('%s%s', LOG_PREFIX, encode_opportunity_diagnostics(records))
        return records

    def _record(self, candidate, components, support, opportunities, legacy, ru_bp_coverage):
        """One diagnostic row: `(k, N)` beside the denominator SPEC 3.1 rejects.

        `round(ru_bp_coverage / ru_length)` is carried to QUANTIFY the unit mismatch that
        PLAN Task 7 Step 3 asks about, never as a definition of `N`. `hmm_match_count` is
        a `defaultdict(int)` (`advntr/vntr_finder.py:210`), so `ru_length` can be `0` and
        `advntr/vntr_finder.py:444`'s division is a live `ZeroDivisionError`; the ratio
        is guarded here and reported as `None` instead. Flank candidates report `None`
        too: the legacy flank denominator borrows a repeat-unit index from
        `reference_repeat_order` (`advntr/vntr_finder.py:493`, `:524`), and rebuilding
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
                ploidy = 1 if self._is_haploid else 2
                average = total_bps / (float(ru_length) * ploidy * copies)
        return {'candidate': candidate, 'support': support,
                'opportunities': opportunities, 'legacy_support': legacy,
                'pattern_index': pattern_index, 'ru_bp_coverage': total_bps,
                'ru_length': ru_length, 'ru_bp_coverage_ratio': ratio,
                'avg_bp_coverage': average}


def encode_opportunity_diagnostics(records, version=DIAGNOSTICS_VERSION):
    """Deterministic, versioned, and free of read names.

    Same idiom as `advntr/mutation_keys.encode_frameshift_context` (`:237-251`). Sorting
    the ENCODED strings is the load-bearing step, not decoration: `mutations` and
    `candidate_evidence` are plain `defaultdict`s (`advntr/vntr_finder.py:209-211`) whose
    Python 2 iteration order is hash order, and a single-threaded unit test would not
    catch the difference. No `query_name` appears in any field, mirroring the anonymity
    property `tests/test_frameshift_context.py:199` pins for Task 5's Context column.
    """
    encoded = [json.dumps(record, sort_keys=True, separators=(',', ':'))
               for record in records.values()]
    encoded.sort()
    return '{"v":%d,"candidates":[%s]}' % (version, ','.join(encoded))
