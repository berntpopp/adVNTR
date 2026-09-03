"""Sibling aggregation: how Task 7's per-occurrence rows become one emitted `State`.

Split out of `tests/test_exact_caller.py`, which drives the whole
`find_frameshift_from_selected_reads` traversal, because that file reached the 650-line
new-file ratchet (`scripts/loc_ratchet.py:10`) -- the same split Task 7 made between
`tests/test_frameshift_opportunities.py` and `tests/test_frameshift_opportunity_slots.py`.
These tests exercise `advntr/exact_caller.py` directly: no finder, no reads, just
hand-built records.

The two helpers below are duplicated from the other module rather than imported, so
neither file constrains the other's fixtures.
"""
import logging
import unittest

from advntr import exact_caller
from advntr.exact_tail import exact_indel_tail


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


def _row(support_identities, opportunity_spans, legacy_states=()):
    """One finalised opportunity row, in the shape `finalise` now produces.

    `support` and `opportunities` are derived from the identity sets rather than given
    independently, exactly as `advntr/frameshift_opportunities.py:finalise` derives them
    -- a fixture that let the two disagree would not be testing the real record.
    """
    return {'support': len(set(support_identities)),
            'opportunities': sum(count for _span, count in opportunity_spans),
            'support_identities': tuple(sorted(set(support_identities))),
            'opportunity_spans': tuple(sorted(opportunity_spans)),
            'legacy_states': sorted(legacy_states)}


class TestSiblingAggregation(unittest.TestCase):
    """Task 7's rows are per occurrence; the emitted `State` is per read.

    Both halves of the aggregate are a UNION over `(read, occurrence)` identities --
    SPEC line 131: "Any future merge must union read/occurrence identities; it must
    never sum overlapping counts." See `advntr/exact_caller.py`'s docstring for why the
    earlier sum/max pair was wrong in both directions.
    """

    def test_support_unions_the_identities_of_the_siblings_a_row_names(self):
        """One read inserting in two occurrences is named `I2_1_T_LEN2` by the shipped
        caller and `I2_1_T_LEN1` twice by the per-occurrence rebuild
        (`advntr/frameshift_opportunities.py:per_occurrence_candidates`), so the
        legacy-named row carries no support at all."""
        records = {
            'I2_1_T_LEN2': _row([], [(0, 2)]),
            'I2_1_T_LEN1': _row([(0, 0), (0, 1)], [(0, 2)],
                                legacy_states=['I2_1_T_LEN2']),
        }

        self.assertEqual(exact_caller.aggregate_evidence(records, 'I2_1_T_LEN2'), (2, 2))

    def test_opportunities_union_the_span_sets_instead_of_taking_the_largest(self):
        """Two deletions in different occurrences of one read fuse into one `State`
        (`advntr/mutation_keys.py:189`). Slot 11 and slot 12 are not offered by the same
        occurrences, so the trials behind the fused state are the UNION of the two span
        sets -- taking the larger one alone under-counts `N`, and a smaller `N` at the
        same `k` lowers the p-value (`advntr/frameshift_opportunities.py:126-129`)."""
        records = {
            'D11_2&D12_2': _row([], []),
            'D11_2': _row([(0, 0)], [(0, 4)], legacy_states=['D11_2&D12_2']),
            'D12_2': _row([(0, 1)], [(1, 3)], legacy_states=['D11_2&D12_2']),
        }

        self.assertEqual(exact_caller.aggregate_evidence(records, 'D11_2&D12_2'), (2, 7))

    def test_a_span_two_siblings_share_is_counted_once(self):
        """The other direction: siblings really do draw on the same occurrences, so the
        union must not double-count a span both of them match."""
        records = {
            'D11_2&D12_2': _row([], []),
            'D11_2': _row([(0, 0)], [(0, 4), (1, 3)], legacy_states=['D11_2&D12_2']),
            'D12_2': _row([(0, 1)], [(0, 4)], legacy_states=['D11_2&D12_2']),
        }

        self.assertEqual(exact_caller.aggregate_evidence(records, 'D11_2&D12_2'), (2, 7))

    def test_an_identity_two_siblings_share_is_counted_once(self):
        records = {
            'F': _row([], []),
            'A': _row([(0, 0), (1, 0)], [(0, 9)], legacy_states=['F']),
            'B': _row([(1, 0), (2, 0)], [(0, 9)], legacy_states=['F']),
        }

        self.assertEqual(exact_caller.aggregate_evidence(records, 'F'), (3, 9))

    def test_a_row_that_names_only_itself_is_its_own_evidence(self):
        records = {'D3_1': _row([(0, 0), (1, 0), (2, 0)], [(0, 9)],
                                legacy_states=['D3_1'])}

        self.assertEqual(exact_caller.aggregate_evidence(records, 'D3_1'), (3, 9))

    def test_a_state_with_no_row_at_all_yields_nothing(self):
        self.assertIsNone(exact_caller.aggregate_evidence({}, 'D3_1'))

    def test_summing_and_maxing_would_flip_this_call_and_unioning_does_not(self):
        """The sub-invariant regime the `k > N` guard never sees.

        Two siblings, one shared supporting occurrence, disjoint span sets. `sum(k)` is
        6 and `max(N)` is 20, so the guard stays silent and the candidate is called at
        the shipped 1e-3 cutoff. The union is `k = 5` over `N = 40`, which is not called.

            sum/max  (k=6, N=20) -> p = 3.2929e-04   called
            union    (k=5, N=40) -> p = 4.8028e-02   not called

        146x, and it flips the decision -- in the anti-conservative direction, which is
        the one `a0b0207` identified as the wrong one to optimise against.
        """
        records = {
            'D11_2&D12_2': _row([], []),
            'D11_2': _row([(0, 0), (1, 0), (2, 0)], [(0, 20)],
                          legacy_states=['D11_2&D12_2']),
            'D12_2': _row([(2, 0), (3, 0), (4, 0)], [(1, 20)],
                          legacy_states=['D11_2&D12_2']),
        }

        self.assertEqual(exact_caller.aggregate_evidence(records, 'D11_2&D12_2'), (5, 40))

        called, pvalue = exact_caller.decide(records, 'D11_2&D12_2',
                                             _StubBackground(0.05), 0.001)
        self.assertFalse(called)
        self.assertAlmostEqual(pvalue, exact_indel_tail(5, 40, 0.05),
                               delta=abs(pvalue) * 1e-9)
        self.assertTrue(exact_indel_tail(6, 20, 0.05) < 0.001)

    def test_the_union_makes_the_invariant_structural(self):
        """Every sibling's support identities lie inside the spans it matches, so the
        union of supports lies inside the union of spans: `k <= N` cannot fail through
        aggregation any more. The guard below is kept for hand-built records only."""
        records = {
            'F': _row([], []),
            'A': _row([(0, 0), (1, 0), (2, 0)], [(0, 3)], legacy_states=['F']),
            'B': _row([(3, 0), (4, 0), (5, 0)], [(1, 3)], legacy_states=['F']),
        }
        support, opportunities = exact_caller.aggregate_evidence(records, 'F')

        self.assertEqual((support, opportunities), (6, 6))
        self.assertLessEqual(support, opportunities)

    def test_an_aggregate_that_breaks_the_invariant_declines_rather_than_calls(self):
        """Unreachable from `finalise` now that both halves union (see above), so this
        pins the defensive guard against a hand-built or future record shape: refuse
        loudly, never clamp, never call on a denominator known to be too small."""
        records = {
            'F': _row([], []),
            'A': _row([(0, 0), (1, 0), (2, 0)], [(0, 1)], legacy_states=['F']),
        }

        (called, pvalue), messages = _capture_log(
            logging.WARNING,
            lambda: exact_caller.decide(records, 'F', _StubBackground(0.001), 0.001))

        self.assertFalse(called)
        self.assertIsNone(pvalue)
        self.assertIn('exceeds opportunities', ' '.join(messages))

    def test_a_state_with_no_row_declines_rather_than_calling(self):
        (called, pvalue), messages = _capture_log(
            logging.WARNING,
            lambda: exact_caller.decide({}, 'D3_1', _StubBackground(0.001), 0.001))

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
            lambda: exact_caller.decide(records, 'D3_1', _StubBackground(0.001), 0.001))

        self.assertFalse(called)
        self.assertEqual(pvalue, 1.0)
        self.assertIn('no occurrence-scoped support', ' '.join(messages))

if __name__ == '__main__':
    unittest.main()
