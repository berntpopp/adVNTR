"""Slot and eligibility predicates of `advntr/frameshift_opportunities.py`, unit level.

Split out of `tests/test_frameshift_opportunities.py`, which drives the whole
`find_frameshift_from_selected_reads` traversal, because that file reached the 650-line
new-file ratchet (`scripts/loc_ratchet.py:10`). These tests exercise the module directly:
no finder, no reads, just a vpath and a candidate.
"""
import logging
import unittest

from advntr import frameshift_opportunities
from advntr import settings


REFERENCE_UNIT = 'ACGTACGTACGT'


class _ListHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _capture_debug(action):
    """Run `action` with the root logger silenced into a list, restoring it afterwards."""
    root = logging.getLogger()
    saved_level, saved_handlers = root.level, root.handlers[:]
    handler = _ListHandler()
    root.handlers = [handler]
    root.setLevel(logging.DEBUG)
    try:
        action()
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
    return handler.messages


class TestOpportunityPredicates(unittest.TestCase):
    """Unit-level cover for the slot and eligibility predicates."""

    def setUp(self):
        self.original_full_ru = settings.USE_ONLY_FULLY_COVERED_RU
        settings.USE_ONLY_FULLY_COVERED_RU = False

    def tearDown(self):
        settings.USE_ONLY_FULLY_COVERED_RU = self.original_full_ru

    @staticmethod
    def _spans(states):
        return frameshift_opportunities.occurrence_spans(states)

    def test_a_deletion_slot_needs_only_that_reference_position_reached(self):
        spans = self._spans(['unit_start_1', 'M1_1', 'D2_1', 'M3_1', 'unit_end_1'])
        signature = spans[0].signature

        self.assertTrue(frameshift_opportunities._signature_supports(
            signature, [('D', 2, '1')]))
        self.assertFalse(frameshift_opportunities._signature_supports(
            signature, [('D', 4, '1')]))

    def test_an_insertion_slot_needs_both_sides_reached(self):
        """The one-sided form would credit a slot to a read whose last emitted base sits
        at the slot, inflating `N` at read ends."""
        stops_at_two = self._spans(['unit_start_1', 'M1_1', 'M2_1'])[0].signature
        crosses_two = self._spans(['unit_start_1', 'M1_1', 'M2_1', 'M3_1'])[0].signature

        self.assertFalse(frameshift_opportunities._signature_supports(
            stops_at_two, [('I', 2, '1')]))
        self.assertTrue(frameshift_opportunities._signature_supports(
            crosses_two, [('I', 2, '1')]))

    def test_a_visited_insertion_state_is_an_opportunity_on_its_own(self):
        signature = self._spans(['unit_start_1', 'M1_1', 'I1_1'])[0].signature

        self.assertTrue(frameshift_opportunities._signature_supports(
            signature, [('I', 1, '1')]))

    def test_a_candidate_never_draws_on_another_pattern(self):
        signature = self._spans(['unit_start_1', 'M1_1', 'M2_1', 'M3_1'])[0].signature

        self.assertFalse(frameshift_opportunities._signature_supports(
            signature, [('D', 2, '2')]))

    def test_flank_spans_carry_the_flank_pseudo_occurrence(self):
        spans = self._spans(['M1_suffix', 'unit_start_1', 'M1_1', 'unit_end_1', 'I0_prefix'])

        self.assertEqual([span.occurrence for span in spans],
                         ['suffix_flank', 0, 'prefix_flank'])

    def test_a_balanced_insertion_deletion_pair_is_an_opportunity_without_support(self):
        """The deliberate divergence: the legacy `I == D` tests
        (`advntr/vntr_finder.py:345`, `:355`) are support-side, not eligibility."""
        counts = {0: {'M': 12, 'I': 1, 'D': 1, 'S': 0}}
        span = self._spans(['unit_start_1', 'M1_1', 'unit_end_1'])[0]

        self.assertTrue(frameshift_opportunities.is_eligible(
            span, counts, [[REFERENCE_UNIT]], {'suffix_flank': True, 'prefix_flank': True}))

    def test_a_missing_occurrence_key_is_an_explicit_rejection(self):
        """`ru_state_count` is a defaultdict of defaultdicts
        (`advntr/hmm_utils.py:158`), so a missing key otherwise reads as all-zero."""
        span = self._spans(['unit_start_1', 'M1_1', 'unit_end_1'])[0]

        self.assertFalse(frameshift_opportunities.is_eligible(
            span, {}, [[REFERENCE_UNIT]], {}))

    def test_fully_covered_ru_setting_drops_partial_occurrences(self):
        settings.USE_ONLY_FULLY_COVERED_RU = True
        span = self._spans(['M1_1', 'M2_1', 'unit_end_1'])[0]
        counts = {'partial_start': {'M': 9, 'I': 0, 'D': 0, 'S': 0}}

        self.assertFalse(frameshift_opportunities.is_eligible(
            span, counts, [[REFERENCE_UNIT]], {}))

    def test_a_read_level_rejection_removes_the_occurrence_from_the_denominator(self):
        span = self._spans(['unit_start_1', 'M1_1', 'unit_end_1'])[0]
        counts = {0: {'M': 12, 'I': 0, 'D': 0, 'S': 4}}

        self.assertFalse(frameshift_opportunities.is_eligible(
            span, counts, [[REFERENCE_UNIT]], {}))

    def test_the_end_terminator_pins_the_last_position_only_where_the_model_forces_it(self):
        """The suffix matcher reaches `suffix_end_suffix` only from its last reference
        position (`advntr/hmm_utils.py:465-472`), so the terminator identifies that
        position. The prefix matcher does not (`:416`), so it identifies nothing."""
        suffix = self._spans(['suffix_start_suffix', 'M1_suffix', 'M2_suffix',
                              'suffix_end_suffix'])[0]
        prefix = self._spans(['prefix_start_prefix', 'M1_prefix', 'M2_prefix',
                              'prefix_end_prefix'])[0]

        self.assertTrue(frameshift_opportunities._signature_supports(
            suffix.signature, [('I', 2, 'suffix')]))
        self.assertFalse(frameshift_opportunities._signature_supports(
            prefix.signature, [('I', 2, 'prefix')]))

    def test_the_end_terminator_never_satisfies_the_start_slot(self):
        """No `I0` state anywhere transitions to an end terminator
        (`advntr/hmm_utils.py:666-668`, `:461-463`, `:393-395`), so reaching one can
        never be evidence that slot 0 was crossed."""
        span = self._spans(['unit_start_1', 'M5_1', 'M6_1', 'unit_end_1'])[0]

        self.assertFalse(frameshift_opportunities._signature_supports(
            span.signature, [('I', 0, '1')]))

    def test_the_span_carries_only_what_the_slot_rules_read(self):
        """A per-span set of pattern names was retained and never read again."""
        self.assertEqual(frameshift_opportunities.OccurrenceSpan._fields,
                         ('occurrence', 'signature'))

    def test_a_span_with_no_submodel_state_is_not_reported_as_a_model_conflict(self):
        """`start_random_matches` belongs to no submodel (`advntr/hmm_utils.py:824-826`),
        which is ordinary, not a conflict between two of them."""
        records = _capture_debug(lambda: self._spans(
            ['start_random_matches', 'unit_start_1', 'M1_1', 'unit_end_1']))

        self.assertIn('no submodel state', ' '.join(records))
        self.assertNotIn('model types', ' '.join(records))

    def test_the_encoder_sorts_records_and_prefixes_a_version(self):
        records = {'B': {'candidate': 'B'}, 'A': {'candidate': 'A'}}

        self.assertEqual(frameshift_opportunities.encode_opportunity_diagnostics(records),
                         '{"v":1,"candidates":[{"candidate":"A"},{"candidate":"B"}]}')

    def test_the_diagnostic_line_cannot_corrupt_append_mode_resume(self):
        """`advntr/advntr_commands.py:96-101` greps the run log for both substrings to
        resume in append mode."""
        self.assertNotIn('alignment file for', frameshift_opportunities.LOG_PREFIX)
        self.assertNotIn('INFO:find_frameshift_from_alignment',
                         'INFO:' + frameshift_opportunities.LOG_PREFIX)

if __name__ == '__main__':
    unittest.main()
