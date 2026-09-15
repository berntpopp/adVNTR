"""Invented visit facts keep coverage/read/occurrence evidence units separate."""
import math
import unittest

from advntr.frameshift_decisions import resolve_policy
from advntr.frameshift_traversal import freeze_traversal, plan_visits


def plan(support=4, policy=None, state='D3_1'):
    return plan_visits(freeze_traversal({state: support}, {}), [], 0, 0,
                       resolve_policy() if policy is None else policy)[0]


class Background(object):
    def __init__(self, probability):
        self.probability = probability
        self.requests = []

    def probability_for(self, state):
        self.requests.append(state)
        return self.probability


def records(k, n, state='D3_1'):
    return {state: {'opportunities': n,
                    'state_identities': {state: [(index, 0) for index in range(k)]}}}


class TestVisitDecisions(unittest.TestCase):
    def test_coverage_keeps_original_association_and_expected_indel_rate(self):
        from advntr.frameshift_visit_decisions import coverage_basis
        diploid = coverage_basis(44, 12, 3, False, 1.0)
        haploid = coverage_basis(44, 12, 3, True, 1.0)
        self.assertEqual(float(44) / 12 / 2 / 3, diploid.mean_coverage)
        self.assertNotEqual(float(44) / (12 * 2 * 3), diploid.mean_coverage)
        self.assertEqual(float(44) / 12 / 3, haploid.mean_coverage)
        self.assertEqual(0.99 / 6, diploid.expected_indels)
        self.assertEqual(0.99 / 3, haploid.expected_indels)
        for length, copies in ((0, 1), (1, 0)):
            with self.assertRaises(ZeroDivisionError):
                coverage_basis(1, length, copies, False, 1.0)

    def test_prescore_suppression_never_reads_missing_geometry_or_background(self):
        from advntr.frameshift_visit_decisions import assess_visit
        background = Background(0.1)
        suppressed = plan(1)
        result = assess_visit(suppressed, None, resolve_policy(), {}, background)
        self.assertEqual('insufficient-read-support', result.disposition)
        self.assertIsNone(result.statistic)
        self.assertEqual([], background.requests)
        outside = plan_visits(freeze_traversal({}, {'I1_prefix_LEN1': 4}), [], 0, 0, resolve_policy())[0]
        self.assertEqual('outside-boundary', assess_visit(outside, None, resolve_policy(), {}, background).disposition)
        with self.assertRaises(ValueError):
            assess_visit(suppressed, None, resolve_policy(minimum_read_support=1), {}, background)
        with self.assertRaises(ValueError):
            assess_visit(plan(), None, resolve_policy(minimum_read_support=5), {}, background)

    def test_rare_coverage_guard_is_strict_and_precedes_statistics(self):
        from advntr.frameshift_visit_decisions import coverage_basis, assess_visit
        basis = coverage_basis(12, 12, 1, False, 2.0)
        background = Background(0.01)
        rejected = assess_visit(plan(), basis, resolve_policy(), records(1, 4), background,
                                rare_fraction=0.2500001)
        self.assertEqual('rare-unit-coverage', rejected.disposition)
        self.assertEqual(0.5, rejected.mean_coverage)
        self.assertIsNone(rejected.statistic)
        self.assertEqual([], background.requests)
        for limit in (None, 0.0, 0.25):
            result = assess_visit(plan(), basis, resolve_policy(), records(1, 4), background, rare_fraction=limit)
            self.assertIsNotNone(result.statistic)
        no_locus = coverage_basis(12, 12, 1, False, 0.0)
        self.assertIsNotNone(assess_visit(plan(), no_locus, resolve_policy(), records(1, 4),
                                         background, rare_fraction=100.0).statistic)

    def test_legacy_read_count_above_mean_coverage_remains_a_call(self):
        from advntr.frameshift_visit_decisions import coverage_basis, assess_visit
        result = assess_visit(plan(), coverage_basis(44, 12, 3, False, 1.0), resolve_policy(), {}, None)
        self.assertTrue(result.statistic.called)
        self.assertEqual(0, result.statistic.pvalue)
        self.assertEqual((0, 1.0), (result.sequencing_error_probability, result.frameshift_probability))
        self.assertIsNone(result.exact_assessment)

    def test_exact_occurrences_are_not_replaced_by_global_read_support(self):
        from advntr.frameshift_visit_decisions import coverage_basis, assess_visit
        background = Background(0.5)
        result = assess_visit(plan(4), coverage_basis(120, 12, 1, False, 1.0), resolve_policy(),
                              records(1, 1), background)
        self.assertEqual((1, 1), (result.exact_assessment.support, result.exact_assessment.opportunities))
        self.assertEqual(0.5, result.statistic.pvalue)
        self.assertFalse(result.statistic.called)
        self.assertIsNone(result.sequencing_error_probability)
        self.assertEqual(['D3_1'], background.requests)

    def test_exact_missing_row_invalid_counts_and_zero_support_remain_distinct(self):
        from advntr.frameshift_visit_decisions import coverage_basis, assess_visit
        basis = coverage_basis(120, 12, 1, False, 1.0)
        for evidence, disposition, expected_requests in (
                ({}, 'missing-opportunity-row', 0),
                (records(2, 1), 'support-exceeds-opportunities', 0),
                (records(0, 0), 'no-trials', 1),
                (records(0, 4), 'no-occurrence-support', 1)):
            background = Background(0.1)
            result = assess_visit(plan(), basis, resolve_policy(), evidence, background)
            self.assertEqual(disposition, result.disposition)
            self.assertFalse(result.statistic.called)
            self.assertEqual(expected_requests, len(background.requests))

    def test_exact_strict_tie_and_underflow_use_existing_log_tail(self):
        from advntr.frameshift_visit_decisions import coverage_basis, assess_visit
        basis = coverage_basis(120, 12, 1, False, 1.0)
        tied_policy = resolve_policy(0.25)
        result = assess_visit(plan(policy=tied_policy), basis, tied_policy, records(1, 1), Background(0.25))
        self.assertFalse(result.statistic.called)
        result = assess_visit(plan(), basis, resolve_policy(), records(1000, 1000), Background(0.01))
        self.assertTrue(result.statistic.called)
        self.assertEqual(0.0, result.statistic.pvalue)
        self.assertFalse(math.isinf(result.statistic.log_tail))

    def test_legacy_callback_preserves_nan_noncall_and_receives_explicit_rate(self):
        from advntr.frameshift_visit_decisions import coverage_basis, assess_visit
        received = []
        def statistic(coverage, support, expected, error_rate):
            received.append((coverage, support, expected, error_rate))
            return float('nan'), float('nan'), float('nan')
        basis = coverage_basis(120, 12, 1, False, 1.0)
        result = assess_visit(plan(), basis, resolve_policy(), {}, None,
                              legacy_error_rate=0.2, legacy_statistic=statistic)
        self.assertEqual([(5.0, 4, 0.495, 0.2)], received)
        self.assertEqual('legacy-nonfinite', result.disposition)
        self.assertFalse(result.statistic.called)

    def test_pure_exact_assessment_keeps_union_counts_without_emitting_logs(self):
        import logging
        from advntr import exact_caller
        from tests.test_exact_caller import _capture_log
        evidence = records(2, 3)
        evidence['D4_1'] = {'opportunities': 99, 'state_identities': {'D3_1': [(0, 0)]}}
        assessment, messages = _capture_log(logging.DEBUG, lambda: exact_caller.assess_evidence(
            evidence, 'D3_1', Background(0.5), resolve_policy()))
        self.assertEqual([], messages)
        self.assertEqual((2, 3, 0.5), (assessment.support, assessment.opportunities, assessment.probability))
        self.assertEqual(0.5, assessment.statistic.pvalue)
        self.assertFalse(assessment.statistic.called)


class TestProductionVisitReceipts(unittest.TestCase):
    def test_production_stores_the_shared_traversal_and_assessment_receipts(self):
        from tests import test_exact_caller as reference
        from advntr.frameshift_traversal import CandidateTraversal
        fixture = reference._ExactCallerTestCase('setUp')
        fixture.setUp()
        try:
            fixture._run()
            self.assertIsInstance(fixture.finder.last_frameshift_traversal, CandidateTraversal)
            visits = fixture.finder.last_frameshift_visits
            self.assertEqual(['repeat', 'suffix', 'prefix'], [visit.plan.site for visit in visits])
            self.assertEqual(['called'] * 3, [visit.disposition for visit in visits])
            self.assertEqual([0, 1, 2], [visit.plan.ordinal for visit in visits])
            self.assertEqual(['D3_2', 'I12_suffix_LEN1', 'I0_prefix_LEN1'],
                             [visit.plan.state for visit in visits])
            fixture.finder.frameshift_policy = resolve_policy(minimum_read_support=5)
            self.assertIsNone(fixture._run())
            self.assertEqual(['insufficient-read-support'] * 3,
                             [visit.disposition for visit in fixture.finder.last_frameshift_visits])
            self.assertTrue(all(visit.statistic is None for visit in fixture.finder.last_frameshift_visits))
        finally:
            fixture.tearDown()

    def test_flank_boundaries_keep_source_endpoint_and_negative_suffix_behavior(self):
        from advntr.frameshift_visit_decisions import flank_boundaries
        self.assertEqual((-1, 4), flank_boundaries(['A'], 'AAAAA', 'AAAA', 3))
        self.assertEqual((1, 2), flank_boundaries(['A'], 'TTAA', 'AATT', 3))
        self.assertEqual((3, 0), flank_boundaries(['A'], 'TT', 'GG', 3))
        self.assertEqual((3, 0), flank_boundaries(['A'], '', '', 3))
        with self.assertRaises(IndexError):
            flank_boundaries([], '', '', 3)

    def test_boundary_suppression_has_a_receipt_without_candidate_log_or_geometry_lookup(self):
        from collections import OrderedDict
        import logging
        from advntr.frameshift_calling import call_frameshift_candidates
        from tests import test_exact_caller as reference
        fixture = reference._ExactCallerTestCase('setUp')
        fixture.setUp()
        try:
            candidates = OrderedDict([('I1_prefix_LEN1', 4), ('I11_suffix_LEN1', 4), ('I0_unknown', 4)])
            calls, messages = reference._capture_log(logging.INFO, lambda: call_frameshift_candidates(
                fixture.finder, {}, candidates, {}, {}, {}, [], None))
            self.assertEqual([], calls)
            self.assertEqual([], messages)
            self.assertEqual((12, 0), fixture.finder.last_frameshift_boundaries)
            visits = fixture.finder.last_frameshift_visits
            self.assertEqual(['I1_prefix_LEN1', 'I11_suffix_LEN1'], [visit.plan.state for visit in visits])
            self.assertEqual(['outside-boundary'] * 2, [visit.disposition for visit in visits])
            self.assertTrue(all(visit.statistic is None for visit in visits))
        finally:
            fixture.tearDown()
