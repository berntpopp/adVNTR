"""Invented candidates prove policy-dependent traversal and no-visit semantics."""
import pickle
import unittest
from collections import OrderedDict

from advntr.frameshift_decisions import resolve_policy


class TestFrameshiftTraversal(unittest.TestCase):
    def test_repeat_order_is_canonical_but_flank_order_is_preserved(self):
        from advntr.frameshift_traversal import freeze_traversal, plan_visits
        traversal = freeze_traversal({'D2_1': 4, 'D1_1': 4, 'D3_2': 1},
                                    OrderedDict([('I0_prefix_LEN1', 4), ('I12_suffix_LEN1', 4)]))
        self.assertEqual((('D3_2', 1), ('D1_1', 4), ('D2_1', 4)), traversal.repeat_candidates)
        self.assertEqual((('I0_prefix_LEN1', 4), ('I12_suffix_LEN1', 4)), traversal.flank_candidates)
        visits = plan_visits(traversal, ['L', '1', '2', 'R'], 12, 0, resolve_policy())
        self.assertEqual(['repeat', 'repeat', 'repeat', 'prefix', 'suffix'], [v.site for v in visits])
        self.assertEqual(['2', '1', '1', '2', '1'], [v.repeat_unit_index for v in visits])
        self.assertEqual([0, 1, 2, 0, 1], [v.source_index for v in visits])
        self.assertEqual(range(5), [v.ordinal for v in visits])
        self.assertEqual(['insufficient-read-support'] + ['ready-to-score'] * 4,
                         [v.disposition for v in visits])

    def test_support_change_can_reach_a_later_independent_flank_branch(self):
        from advntr.frameshift_traversal import freeze_traversal, plan_visits
        # The source loop uses two independent substring checks. At high support,
        # suffix's continue skips prefix; with equality at the floor, both execute.
        traversal = freeze_traversal({}, OrderedDict([('I5_prefix_suffix', 2)]))
        high = plan_visits(traversal, ['L', '1', '2', 'R'], 0, 10, resolve_policy(minimum_read_support=3))
        low = plan_visits(traversal, ['L', '1', '2', 'R'], 0, 10, resolve_policy(minimum_read_support=2))
        self.assertEqual([('suffix', 'insufficient-read-support')], [(v.site, v.disposition) for v in high])
        self.assertEqual([('suffix', 'ready-to-score'), ('prefix', 'ready-to-score')],
                         [(v.site, v.disposition) for v in low])
        self.assertEqual(['1', '2'], [v.repeat_unit_index for v in low])
        self.assertEqual([0, 0], [v.source_index for v in low])
        self.assertEqual((('I5_prefix_suffix', 2),), traversal.flank_candidates)

    def test_boundary_suppression_does_not_skip_the_other_branch(self):
        from advntr.frameshift_traversal import freeze_traversal, plan_visits
        traversal = freeze_traversal({}, {'I5_prefix_suffix': 3})
        suffix_suppressed = plan_visits(traversal, ['L', '1', '2', 'R'], 6, 5, resolve_policy())
        prefix_suppressed = plan_visits(traversal, ['L', '1', '2', 'R'], 5, 4, resolve_policy())
        self.assertEqual(['outside-boundary', 'ready-to-score'], [v.disposition for v in suffix_suppressed])
        self.assertEqual([None, '2'], [v.repeat_unit_index for v in suffix_suppressed])
        self.assertEqual(['ready-to-score', 'outside-boundary'], [v.disposition for v in prefix_suppressed])
        self.assertEqual(['1', None], [v.repeat_unit_index for v in prefix_suppressed])

    def test_no_visit_is_distinct_from_boundary_support_and_scoring(self):
        from advntr.frameshift_traversal import freeze_traversal, plan_visits
        empty = freeze_traversal({}, {})
        self.assertEqual((), plan_visits(empty, [], 0, 0, resolve_policy()))
        # No branch label in a flank traversal entry means neither source if runs.
        unmatched = freeze_traversal({}, {'D2_1': 3})
        self.assertEqual((), plan_visits(unmatched, [], 0, 0, resolve_policy()))
        boundary = freeze_traversal({}, {'I1_prefix_LEN1': 3})
        self.assertEqual('outside-boundary', plan_visits(boundary, [], 0, 0, resolve_policy())[0].disposition)
        support = freeze_traversal({'D1_1': 1}, {})
        self.assertEqual('insufficient-read-support', plan_visits(support, [], 0, 0, resolve_policy())[0].disposition)
        self.assertEqual('ready-to-score', plan_visits(support, [], 0, 0,
            resolve_policy(minimum_read_support=1))[0].disposition)

    def test_source_lookup_order_does_not_invent_geometry_for_suppressed_flanks(self):
        from advntr.frameshift_traversal import freeze_traversal, plan_visits
        traversal = freeze_traversal({}, {'I0_prefix_LEN1': 1})
        # At an allowed boundary, the source reads reference_order[-2] BEFORE
        # testing supporting reads. An invalid basis must still fail there.
        with self.assertRaises(IndexError):
            plan_visits(traversal, [], 0, 0, resolve_policy())
        with self.assertRaises(IndexError):
            plan_visits(freeze_traversal({}, {'I0_suffix_LEN1': 1}), [], 0, 0, resolve_policy())

    def test_compound_repeat_candidate_keeps_first_component_index(self):
        from advntr.frameshift_traversal import freeze_traversal, plan_visits
        traversal = freeze_traversal({'D3_2&D4_7': 3}, {})
        visit = plan_visits(traversal, [], -1, 0, resolve_policy())[0]
        self.assertEqual(('repeat', '2', 3), (visit.site, visit.repeat_unit_index, visit.read_support))

    def test_frozen_traversal_is_closed_validated_and_pickle_safe(self):
        from advntr.frameshift_traversal import CandidateTraversal, freeze_traversal, validate_traversal
        repeat = {'D1_1': 3}
        traversal = freeze_traversal(repeat, {})
        repeat['D1_1'] = 9
        self.assertEqual((('D1_1', 3),), traversal.repeat_candidates)
        self.assertIs(traversal, validate_traversal(traversal))
        self.assertEqual(traversal, CandidateTraversal._make(tuple(traversal)))
        self.assertEqual(traversal, traversal._replace(flank_candidates=()))
        with self.assertRaises(AttributeError):
            traversal.repeat_candidates = ()
        for bad in (None, ((), ())):
            with self.assertRaises(ValueError):
                validate_traversal(bad)
        for pairs in ((('D1_1', 0),), (('D1_1', True),), (('D1_1', 1.5),),
                      (('D1_1', -1),), ((None, 1),), (('', 1),),
                      (('D1_1', 1), ('D1_1', 1)), (('D2_1', 1), ('D1_1', 1)),
                      (('D1_1',),)):
            with self.assertRaises(ValueError):
                CandidateTraversal(pairs, ())
        with self.assertRaises(ValueError):
            CandidateTraversal([], ())
        with self.assertRaises(ValueError):
            traversal._replace(unknown=())
        forged = tuple.__new__(CandidateTraversal, ((('D1_1', 0),), ()))
        for protocol in (0, 1, 2):
            self.assertEqual(traversal, pickle.loads(pickle.dumps(traversal, protocol)))
            with self.assertRaises(ValueError):
                pickle.loads(pickle.dumps(forged, protocol))

    def test_invalid_policy_and_boundary_types_fail_before_traversal(self):
        from advntr.frameshift_traversal import freeze_traversal, plan_visits
        traversal = freeze_traversal({}, {})
        for suffix, prefix in ((True, 0), (0.0, 0), (0, False), (0, -1), (0, 1.0)):
            with self.assertRaises(ValueError):
                plan_visits(traversal, [], suffix, prefix, resolve_policy())
        with self.assertRaises(ValueError):
            plan_visits(traversal, [], 0, 0, None)


class TestTraversalAgainstProduction(unittest.TestCase):
    def test_three_site_order_and_support_dispositions_match_current_read_loop(self):
        from advntr import settings
        from advntr.frameshift_traversal import freeze_traversal, plan_visits
        from tests import test_exact_caller as reference
        fixture = reference._ExactCallerTestCase('setUp')
        fixture.setUp()
        old = dict((name, getattr(settings, name)) for name in (
            'FILTER_ADAPTER_READTHROUGH', 'MIN_RELATIVE_RU_COVERAGE', 'FRAMESHIFT_CALIBRATION_OUT'))
        try:
            settings.FILTER_ADAPTER_READTHROUGH = False
            settings.MIN_RELATIVE_RU_COVERAGE = None
            settings.FRAMESHIFT_CALIBRATION_OUT = None
            traversal = freeze_traversal({'D3_2': 4}, OrderedDict([
                ('I12_suffix_LEN1', 4), ('I0_prefix_LEN1', 4)]))
            for support in (1, 3, 4, 5):
                policy = resolve_policy(minimum_read_support=support)
                fixture.finder.frameshift_policy = policy
                results, messages = fixture._run_capturing_info()
                plans = plan_visits(traversal, ['L', '1', '2', '2', '2', '3', 'R'], 12, 0, policy)
                native_states = [line.split('Occurrence ')[1].split(':')[0] for line in messages
                                 if line.startswith('Frameshift Candidate and Occurrence ')]
                self.assertEqual(native_states, [plan.state for plan in plans])
                native_skips = [line for line in messages if line.startswith('Skipped due to too small')]
                self.assertEqual(len(native_skips), sum(plan.disposition == 'insufficient-read-support'
                                                        for plan in plans))
                self.assertEqual([row[0] for row in results or ()],
                                 [plan.state for plan in plans if plan.disposition == 'ready-to-score'])
                self.assertEqual(['2', '1', '3'], [plan.repeat_unit_index for plan in plans])
        finally:
            for name, value in old.items():
                setattr(settings, name, value)
            fixture.tearDown()

    def test_failed_prefix_support_continues_to_next_source_candidate(self):
        from advntr.frameshift_traversal import freeze_traversal, plan_visits
        traversal = freeze_traversal({}, OrderedDict([
            ('I0_prefix_LEN1', 1), ('I8_suffix_LEN1', 3)]))
        plans = plan_visits(traversal, ['L', '1', '2', 'R'], 8, 0, resolve_policy())
        self.assertEqual([('prefix', 'insufficient-read-support'), ('suffix', 'ready-to-score')],
                         [(plan.site, plan.disposition) for plan in plans])
        self.assertEqual([0, 1], [plan.source_index for plan in plans])

    def test_invalid_source_state_keeps_parse_failure_instead_of_manufacturing_a_visit(self):
        from advntr.frameshift_traversal import freeze_traversal, plan_visits
        with self.assertRaises(IndexError):
            plan_visits(freeze_traversal({'D1': 3}, {}), [], 0, 0, resolve_policy())
        with self.assertRaises(ValueError):
            plan_visits(freeze_traversal({}, {'invalid': 3}), [], 0, 0, resolve_policy())
