"""Sibling aggregation: how Task 7's per-occurrence rows become one emitted `State`.

Split out of `tests/test_exact_caller.py`, which drives the whole
`find_frameshift_from_selected_reads` traversal, because that file reached the 650-line
new-file ratchet (`scripts/loc_ratchet.py:10`) -- the same split Task 7 made between
`tests/test_frameshift_opportunities.py` and `tests/test_frameshift_opportunity_slots.py`.

**The first class drives reads, not record dicts, and that is the point.** The defect
Task 8f repairs lives in how `OpportunityCounter._support` and its per-candidate legacy
rollup accumulate ACROSS reads: one read whose whole-read fusion differs from the
majority's used to attach that whole component's support to the minority `State`
permanently. No hand-built `records` dict can express that, because a hand-built row
already asserts the association the counter lost. This repository has been burned once by
exactly that -- the "146x, flips the decision" figure below came from records no
traversal produces -- so every claim about attribution here is made through
`OpportunityCounter.observe_read`, reached through the real
`VNTRFinder.find_frameshift_from_selected_reads` from synthetic vpaths (no BAM, no HMM,
no decoder), in the idiom of `tests/test_frameshift_opportunities.py:74-118`.

The second class keeps hand-built records, for the two decline paths that are about the
record SHAPE rather than about attribution.

The fixture builders are duplicated from the other modules rather than imported, so
neither file constrains the other's fixtures.
"""
import logging
import unittest

from advntr import exact_caller
from advntr import settings
from advntr.exact_tail import exact_indel_tail
from advntr.reference_vntr import ReferenceVNTR
import advntr.vntr_finder as vntr_finder_module
from advntr.vntr_finder import SelectedRead, VNTRFinder


REFERENCE_UNIT = 'ACGTACGTACGT'
UNIT_LENGTH = len(REFERENCE_UNIT)

#: A power of two, absurd for an indel background and calibrated on nothing -- the rule
#: `tests/test_exact_caller.py` follows, from PLAN Global Constraints. It is chosen so
#: the MISATTRIBUTED pair of each fixture below falls under the shipped 1e-3 cutoff and
#: the correctly attributed pair does not: the defect is anti-conservative, and these
#: fixtures show it deciding.
SYNTHETIC_RATE = 0.03125
CUTOFF = 0.001


class _FakeState(object):
    def __init__(self, name):
        self.name = name


class _FakeHMM(object):
    read_length_used_to_build_model = UNIT_LENGTH


class _ListHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class _StubBackground(object):
    """A background with one rate, for the tests that need no artifact."""

    def __init__(self, probability):
        self.probability = probability

    def probability_for(self, _state):
        return self.probability


def _capture_log(level, action):
    """Run `action` with the root logger collected into a list, then restore it."""
    root = logging.getLogger()
    saved_level, saved_handlers = root.level, root.handlers[:]
    handler = _ListHandler()
    root.handlers = [handler]
    root.setLevel(level)
    try:
        result = action()
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
    return result, handler.messages


def _vpath(state_names):
    return ([(0, _FakeState('start'))] +
            [(i + 1, _FakeState(name)) for i, name in enumerate(state_names)] +
            [(len(state_names) + 1, _FakeState('end'))])


def _unit(events=None, pattern='1', last=UNIT_LENGTH, close=True):
    """One repeat occurrence. `events` maps a reference position to `('I', bases)` or
    `('D', '')`; `close=False` leaves `unit_end` unvisited, which is what makes the last
    occurrence of a read a `partial_end` one."""
    events = events or {}
    states = ['unit_start_%s' % pattern]
    sequence = []
    for position in range(1, last + 1):
        operations = events.get(position, [])
        if any(kind == 'D' for kind, _payload in operations):
            states.append('D%d_%s' % (position, pattern))
        else:
            states.append('M%d_%s' % (position, pattern))
            sequence.append(REFERENCE_UNIT[position - 1])
        for kind, payload in operations:
            if kind == 'I':
                for inserted in payload:
                    states.append('I%d_%s' % (position, pattern))
                    sequence.append(inserted)
    if close:
        states.append('unit_end_%s' % pattern)
    return states, ''.join(sequence)


def _read(units, query_name):
    """Concatenate `(states, sequence)` pieces into one SelectedRead."""
    states = []
    sequence = []
    for unit_states, unit_sequence in units:
        states.extend(unit_states)
        sequence.append(unit_sequence)
    return SelectedRead(''.join(sequence), -1.0, _vpath(states), query_name=query_name)


def _row(support_identities, opportunity_spans, legacy_states=()):
    """One finalised opportunity row, in the shape `finalise` produces.

    `support` and `opportunities` are derived from the identity sets rather than given
    independently, exactly as `advntr/frameshift_opportunities.py:finalise` derives them
    -- a fixture that let the two disagree would not be testing the real record.

    Every identity is attributed to every `legacy_states` entry, which is the shape a
    traversal produces only when EVERY read behind the row fused it the same way. The
    divergence that makes this helper unable to express the real thing is exactly what
    the first class drives through `observe_read` instead.
    """
    identities = tuple(sorted(set(support_identities)))
    return {'support': len(identities),
            'opportunities': sum(count for _span, count in opportunity_spans),
            'support_identities': identities,
            'opportunity_spans': tuple(sorted(opportunity_spans)),
            'legacy_states': sorted(legacy_states),
            'state_identities': dict((state, identities) for state in legacy_states)}


class TestAggregationThroughTheReadLoop(unittest.TestCase):
    """Task 7's rows are per occurrence; the emitted `State` is per read.

    `k` is the union of the `(read, occurrence)` identities THAT READ'S OWN whole-read
    fusion attributed to the scored `State`; `N` is the scored `State`'s own row's
    `opportunities`. See `advntr/exact_caller.py`'s docstring for both rulings.
    """

    def setUp(self):
        self.saved = (settings.USE_REF_ALIGNMENT, settings.USE_ONLY_FULLY_COVERED_RU,
                      vntr_finder_module.get_pattern_clusters)
        settings.USE_REF_ALIGNMENT = False
        settings.USE_ONLY_FULLY_COVERED_RU = False
        # Two clusters, so a read may name pattern `2` and reproduce the `D11_2&D12_2`
        # example `advntr/frameshift_opportunities.py:per_occurrence_candidates` uses.
        vntr_finder_module.get_pattern_clusters = lambda patterns: [[patterns[0]],
                                                                    [patterns[1]]]
        reference = ReferenceVNTR(1, REFERENCE_UNIT, 100, 'chr1', None, None)
        reference.init_from_xml([REFERENCE_UNIT, REFERENCE_UNIT], 'TTTTTTTT', 'GGGGGGGG')
        self.finder = VNTRFinder(reference)
        self.finder.hmm = _FakeHMM()

    def tearDown(self):
        (settings.USE_REF_ALIGNMENT, settings.USE_ONLY_FULLY_COVERED_RU,
         vntr_finder_module.get_pattern_clusters) = self.saved

    def _records(self, reads):
        self.finder.find_frameshift_from_selected_reads(reads)
        return self.finder.last_frameshift_opportunities

    def _fusion_reads(self):
        """Three reads whose whole-read fusions disagree about `D11_2`.

        - `split` deletes 11 in one occurrence and 12 in the next, so its whole-read map
          fuses them (`advntr/mutation_keys.py:189`) and both of its per-occurrence rows
          belong to `D11_2&D12_2`;
        - `eleven` deletes 11 alone, so ITS `D11_2` row belongs to the unfused `D11_2`
          and to nothing else;
        - `partial` carries no indel at all and contributes two spans, the second of
          which stops at position 11 and so offers `D11_2` without offering
          `D11_2&D12_2`.
        """
        return [_read([_unit({11: [('D', '')]}, pattern='2'),
                       _unit({12: [('D', '')]}, pattern='2')], 'split'),
                _read([_unit({11: [('D', '')]}, pattern='2')], 'eleven'),
                _read([_unit(pattern='2'), _unit(pattern='2', last=11, close=False)],
                      'partial')]

    def _renumbered_reads(self):
        """The `_LEN` analogue of the same disagreement, and the shape measured on
        `example_6c28_hg19_subset.bam` as `I10_6_A_LEN1` / `I10_6_A_LEN2`.

        `twice` inserts at the same slot in two occurrences, so its whole-read map counts
        two visits and names the state `_LEN2` while both per-occurrence rows are named
        `_LEN1` (`advntr/mutation_keys.py:167-170`); `once` inserts in a single
        occurrence and its `_LEN1` row belongs to `_LEN1`.
        """
        return [_read([_unit({2: [('I', 'T')]}), _unit({2: [('I', 'T')]})], 'twice'),
                _read([_unit({2: [('I', 'T')]})], 'once')]

    def test_a_fused_state_counts_only_the_reads_that_actually_fused_it(self):
        """`eleven`'s occurrence supports the unfused `D11_2`; crediting the whole
        `D11_2` row to `D11_2&D12_2` because SOME read fused it there is the defect."""
        records = self._records(self._fusion_reads())

        self.assertEqual(records['D11_2&D12_2']['legacy_support'], 1)
        self.assertEqual(records['D11_2']['support'], 2)
        self.assertEqual(exact_caller.aggregate_evidence(records, 'D11_2&D12_2'), (2, 4))

    def test_the_unfused_sibling_counts_only_its_own_reads_in_the_other_direction(self):
        """The same misattribution seen from the other side: `split`'s two occurrences
        belong to the fused state, not to `D11_2`."""
        records = self._records(self._fusion_reads())

        self.assertEqual(exact_caller.aggregate_evidence(records, 'D11_2'), (1, 5))

    def test_the_denominator_is_the_scored_states_own_row_not_the_sibling_union(self):
        """A compound's components must be satisfied by ONE occurrence
        (`advntr/frameshift_opportunities.py`'s module docstring: "an intersection within
        one occurrence and never a union"), so the `partial` read's second span -- which
        stops at position 11 -- is a trial for `D11_2` and not for `D11_2&D12_2`.
        Unioning the siblings' spans would hand the fused state that span anyway."""
        records = self._records(self._fusion_reads())
        union = {}
        for candidate, row in records.items():
            if candidate == 'D11_2&D12_2' or 'D11_2&D12_2' in row['legacy_states']:
                union.update(row['opportunity_spans'])

        self.assertEqual(records['D11_2&D12_2']['opportunities'], 4)
        self.assertEqual(records['D11_2']['opportunities'], 5)
        self.assertEqual(sum(union.values()), 5)
        self.assertEqual(exact_caller.aggregate_evidence(records, 'D11_2&D12_2')[1], 4)

    def test_the_length_renumbered_sibling_splits_its_support_between_two_states(self):
        """One `I2_1_T_LEN1` row, three supporting occurrences, two emitted `State`s: two
        of those occurrences belong to `_LEN2` and one to `_LEN1`."""
        records = self._records(self._renumbered_reads())

        self.assertEqual(records['I2_1_T_LEN1']['support'], 3)
        self.assertEqual(records['I2_1_T_LEN2']['support'], 0)
        self.assertEqual(exact_caller.aggregate_evidence(records, 'I2_1_T_LEN2'), (2, 3))
        self.assertEqual(exact_caller.aggregate_evidence(records, 'I2_1_T_LEN1'), (1, 3))

    def test_a_legacy_named_row_at_zero_support_still_recovers_its_siblings(self):
        """The case the sibling walk exists for, and it must keep working: with only the
        `twice` read, `I2_1_T_LEN2` has no support of its own and both `_LEN1`
        occurrences belong to it."""
        records = self._records(self._renumbered_reads()[:1])

        self.assertEqual(records['I2_1_T_LEN2']['support'], 0)
        self.assertEqual(exact_caller.aggregate_evidence(records, 'I2_1_T_LEN2'), (2, 2))

    def test_the_misattributed_pair_would_be_called_and_the_attributed_one_is_not(self):
        """Both fixtures, at a rate where the difference decides. This is the direction
        that matters: the defect inflates `k`, which shrinks the tail."""
        background = _StubBackground(SYNTHETIC_RATE)
        fused = self._records(self._fusion_reads())
        called_fused, _pvalue = exact_caller.decide(fused, 'D11_2&D12_2', background,
                                                    CUTOFF)
        renumbered = self._records(self._renumbered_reads())
        called_len, _pvalue = exact_caller.decide(renumbered, 'I2_1_T_LEN2', background,
                                                 CUTOFF)

        self.assertFalse(called_fused)
        self.assertFalse(called_len)
        # What the misattribution scored instead: every sibling's whole support.
        self.assertLess(exact_indel_tail(3, 5, SYNTHETIC_RATE), CUTOFF)
        self.assertLess(exact_indel_tail(3, 3, SYNTHETIC_RATE), CUTOFF)

    def test_a_traversal_can_now_produce_support_above_opportunities_and_it_declines(self):
        """`k <= N` is no longer structural, and this is the shape that breaks it.

        The second occurrence stops at position 11, so it produces a `D11_2` row whose
        identity the whole-read fusion attributes to `D11_2&D12_2` -- while its span
        never reached position 12 and is therefore not a trial for the fused state. The
        conservative direction is to refuse the call, not to clamp either number: see
        `advntr/exact_caller.py:decide`.
        """
        read = _read([_unit({11: [('D', '')], 12: [('D', '')]}, pattern='2'),
                      _unit({11: [('D', '')]}, pattern='2', last=11, close=False)],
                     'over-attributed')
        records = self._records([read])

        self.assertEqual(exact_caller.aggregate_evidence(records, 'D11_2&D12_2'), (2, 1))
        (called, pvalue), messages = _capture_log(
            logging.WARNING,
            lambda: exact_caller.decide(records, 'D11_2&D12_2',
                                        _StubBackground(SYNTHETIC_RATE), CUTOFF))

        self.assertFalse(called)
        self.assertIsNone(pvalue)
        self.assertIn('exceeds opportunities', ' '.join(messages))

    def test_every_row_still_satisfies_the_per_row_invariant(self):
        """`finalise` raises on `k > N` per ROW and keeps doing so: the aggregate above
        is the only place the invariant can now break."""
        for records in (self._records(self._fusion_reads()),
                        self._records(self._renumbered_reads())):
            for row in records.values():
                self.assertLessEqual(row['support'], row['opportunities'])


class TestTheDeclineGuards(unittest.TestCase):
    """The paths that are about the record shape rather than about attribution."""

    def test_a_state_with_no_row_at_all_yields_nothing(self):
        self.assertIsNone(exact_caller.aggregate_evidence({}, 'D3_1'))

    def test_a_state_named_only_by_a_sibling_still_needs_its_own_row(self):
        """`N` comes from the scored state's own row, so a `State` that only appears in
        some sibling's `legacy_states` has no denominator. Every `State` that reaches a
        decision site is a key of `mutations` or `prefix_suffix_mutations` and `finalise`
        iterates the union of those with the occurrence-scoped candidates, so this is a
        shape only a hand-built record can have."""
        records = {'D11_2': _row([(0, 0)], [(0, 4)], legacy_states=['D11_2&D12_2'])}

        self.assertIsNone(exact_caller.aggregate_evidence(records, 'D11_2&D12_2'))

    def test_a_row_that_names_only_itself_is_its_own_evidence(self):
        records = {'D3_1': _row([(0, 0), (1, 0), (2, 0)], [(0, 9)],
                                legacy_states=['D3_1'])}

        self.assertEqual(exact_caller.aggregate_evidence(records, 'D3_1'), (3, 9))

    def test_an_identity_two_siblings_share_is_counted_once(self):
        """Union, never sum -- SPEC line 131: "Any future merge must union
        read/occurrence identities; it must never sum overlapping counts"."""
        records = {
            'F': _row([], [(0, 9)]),
            'A': _row([(0, 0), (1, 0)], [(0, 9)], legacy_states=['F']),
            'B': _row([(1, 0), (2, 0)], [(0, 9)], legacy_states=['F']),
        }

        self.assertEqual(exact_caller.aggregate_evidence(records, 'F'), (3, 9))

    def test_summing_the_siblings_flips_this_hand_built_call_and_the_union_does_not(self):
        """The regime the `k > N` guard never sees -- on records no traversal produces.

        Two siblings drawing on one shared span, one shared supporting occurrence
        between them. `sum(k)` is 6 where the union is 5, over the same `N = 20`:

            sum      (k=6, N=20) -> p = 3.2929e-04   called
            union    (k=5, N=20) -> p = 2.5739e-03   not called

        7.8x, in the anti-conservative direction -- the one
        `advntr/frameshift_opportunities.py:126-129` identifies as the wrong one to
        optimise against, since a smaller `N` or a larger `k` lowers the p-value. The
        figure is illustrative and hand-built: measured across all eight public BAMs,
        `sum`/`max` differs from the shipped union on two states, in `N` only, by under
        1%, with no decision change demonstrated. Task 8f's report has the numbers.
        """
        records = {
            'D11_2&D12_2': _row([], [(0, 20)]),
            'D11_2': _row([(0, 0), (1, 0), (2, 0)], [(0, 20)],
                          legacy_states=['D11_2&D12_2']),
            'D12_2': _row([(2, 0), (3, 0), (4, 0)], [(0, 20)],
                          legacy_states=['D11_2&D12_2']),
        }

        self.assertEqual(exact_caller.aggregate_evidence(records, 'D11_2&D12_2'), (5, 20))

        called, pvalue = exact_caller.decide(records, 'D11_2&D12_2',
                                             _StubBackground(0.05), CUTOFF)
        self.assertFalse(called)
        self.assertAlmostEqual(pvalue, exact_indel_tail(5, 20, 0.05),
                               delta=abs(pvalue) * 1e-9)
        self.assertTrue(exact_indel_tail(6, 20, 0.05) < CUTOFF)

    def test_a_state_with_no_row_declines_rather_than_calling(self):
        (called, pvalue), messages = _capture_log(
            logging.WARNING,
            lambda: exact_caller.decide({}, 'D3_1', _StubBackground(0.001), CUTOFF))

        self.assertFalse(called)
        self.assertIsNone(pvalue)
        self.assertIn('no opportunity row', ' '.join(messages))

    def test_zero_aggregated_support_says_so_instead_of_going_quiet(self):
        """`k == 0` has a well-defined tail -- exactly 1.0 -- so unlike the two decline
        paths above it reports a p-value rather than `None`. But the legacy caller only
        reaches a decision site with support at or above
        `settings.MIN_SUPPORTING_READ_COUNT`, so an occurrence-scoped `k` of zero means
        every supporting occurrence was ineligible: a divergence worth a line in the run
        log, not a silent drop."""
        records = {'D3_1': _row([], [(0, 9)], legacy_states=['D3_1'])}

        (called, pvalue), messages = _capture_log(
            logging.WARNING,
            lambda: exact_caller.decide(records, 'D3_1', _StubBackground(0.001), CUTOFF))

        self.assertFalse(called)
        self.assertEqual(pvalue, 1.0)
        self.assertIn('no occurrence-scoped support', ' '.join(messages))

if __name__ == '__main__':
    unittest.main()
