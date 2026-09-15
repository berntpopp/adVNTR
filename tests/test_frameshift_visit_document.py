"""Finite JSON receipts preserve strict-tail distinctions and original call units."""
import json
import math
import unittest

from advntr.frameshift_decisions import resolve_policy
from advntr.frameshift_traversal import freeze_traversal, plan_visits
from advntr.frameshift_visit_decisions import assess_visit, coverage_basis
from tests.test_frameshift_visit_decisions import Background, records


class TestVisitDocument(unittest.TestCase):
    def test_scored_and_suppressed_visits_have_distinct_closed_receipts(self):
        from advntr.frameshift_visit_document import visit_document
        policy = resolve_policy()
        plan = plan_visits(freeze_traversal({'D3_1': 4}, {}), [], 0, 0, policy)[0]
        result = assess_visit(plan, coverage_basis(44, 12, 3, False, 1.0), policy, {}, None)
        raw = visit_document(result)
        self.assertEqual('called', raw['disposition'])
        self.assertEqual(4, raw['plan']['read_support'])
        self.assertEqual(0.0, raw['statistic']['pvalue'])
        self.assertIsNone(raw['statistic']['log_tail'])
        low = plan_visits(freeze_traversal({'D3_1': 1}, {}), [], 0, 0, policy)[0]
        rejected = visit_document(assess_visit(low, None, policy, {}, None))
        self.assertEqual('insufficient-read-support', rejected['disposition'])
        self.assertIsNone(rejected['statistic'])
        self.assertIsNone(rejected['mean_coverage'])
        json.dumps(raw, allow_nan=False)

    def test_nan_legacy_noncall_is_explicitly_null_not_nonstandard_json(self):
        from advntr.frameshift_visit_document import visit_document
        policy = resolve_policy()
        plan = plan_visits(freeze_traversal({'D3_1': 4}, {}), [], 0, 0, policy)[0]
        def statistic(*args, **kwargs):
            return float('nan'), float('nan'), float('nan')
        result = assess_visit(plan, coverage_basis(120, 12, 1, False, 1.0), policy, {}, None, legacy_statistic=statistic)
        raw = visit_document(result)
        self.assertFalse(raw['statistic']['called'])
        self.assertEqual('legacy-nonfinite', raw['statistic']['disposition'])
        self.assertIsNone(raw['statistic']['pvalue'])
        self.assertIsNone(raw['sequencing_error_probability'])
        self.assertIsNone(raw['frameshift_probability'])
        json.dumps(raw, allow_nan=False)

    def test_exact_underflow_keeps_finite_log_and_genuine_zero_has_explicit_kind(self):
        from advntr.frameshift_visit_document import visit_document
        policy = resolve_policy()
        plan = plan_visits(freeze_traversal({'D3_1': 4}, {}), [], 0, 0, policy)[0]
        result = assess_visit(plan, coverage_basis(120, 12, 1, False, 1.0), policy,
                              records(1000, 1000), Background(0.01))
        raw = visit_document(result)
        self.assertEqual(0.0, raw['statistic']['pvalue'])
        self.assertEqual('finite', raw['statistic']['log_tail']['kind'])
        self.assertAlmostEqual(1000 * math.log(0.01), raw['statistic']['log_tail']['value'])
        changed = result._replace(statistic=result.statistic._replace(log_tail=float('-inf')))
        self.assertEqual({'kind': 'negative-infinity', 'value': None}, visit_document(changed)['statistic']['log_tail'])
        with self.assertRaises(ValueError):
            visit_document(result._replace(statistic=result.statistic._replace(log_tail=float('inf'))))

    def test_warning_receipts_distinguish_native_warnings_from_subset_audit(self):
        from advntr.frameshift_visit_document import decision_warnings
        policy = resolve_policy()
        plan = plan_visits(freeze_traversal({'D3_1': 4}, {}), [], 0, 0, policy)[0]
        result = assess_visit(plan, coverage_basis(120, 12, 1, False, 1.0), policy, {}, Background(0.1))
        expected = [{'origin': 'caller', 'ordinal': 0, 'state': 'D3_1', 'disposition': 'missing-opportunity-row'},
                    {'origin': 'calibration-audit', 'ordinal': None, 'state': 'D4_1',
                     'disposition': 'attribution-outside-trials'}]
        self.assertEqual(expected, decision_warnings([result], ('D4_1',)))
        legacy = assess_visit(plan, coverage_basis(44, 12, 3, False, 1.0), policy, {}, None)
        self.assertEqual([], decision_warnings([legacy], ()))
        scored = assess_visit(plan, coverage_basis(44, 12, 3, False, 1.0), policy, records(1, 2), Background(0.1))
        self.assertEqual([], decision_warnings([scored], ()))
