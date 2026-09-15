"""Invented vectors pin the shared production/replay statistic result ABI."""
import math
import unittest

from scipy.stats import binom, chi2

from advntr import exact_caller
from advntr.frameshift_decisions import resolve_policy
from advntr.frameshift_statistics import exact_result, legacy_result, legacy_statistic
from advntr.vntr_finder import VNTRFinder


class TestStatisticResults(unittest.TestCase):

    def test_exact_strict_tie_and_adjacent_cutoff(self):
        tied = exact_result(1, 1, 0.25, resolve_policy(0.25))
        self.assertEqual((False, 0.25, math.log(0.25), 'cutoff'), tuple(tied))
        self.assertEqual('called', exact_result(1, 1, 0.25,
                                              resolve_policy(0.25000000000000006)).disposition)

    def test_exact_scipy_oracle_and_underflow(self):
        for k, n, p0 in ((3, 20, 0.01), (10, 10, 0.2), (0, 10, 0.1)):
            result = exact_result(k, n, p0, resolve_policy())
            expected = binom.sf(k - 1, n, p0)
            self.assertAlmostEqual(expected, result.pvalue, places=14)
            self.assertEqual(expected < 0.001, result.called)
        deep = exact_result(200, 1000, 0.0001, resolve_policy(1e-300))
        self.assertEqual(0.0, deep.pvalue)
        self.assertFalse(math.isinf(deep.log_tail))
        self.assertFalse(math.isnan(deep.log_tail))
        self.assertTrue(deep.called)
        self.assertEqual(float('-inf'), exact_result(1, 1, 0.0, resolve_policy()).log_tail)

    def test_exact_zero_and_invalid_opportunity_dispositions(self):
        self.assertEqual((False, 1.0, 0.0, 'no-trials'),
                         tuple(exact_result(0, 0, 0.1, resolve_policy())))
        self.assertEqual('no-occurrence-support',
                         exact_result(0, 10, 0.1, resolve_policy()).disposition)
        self.assertEqual((False, None, None, 'support-exceeds-opportunities'),
                         tuple(exact_result(3, 2, None, resolve_policy())))
        for k, n in ((True, 1), (1, False), (-1, 2), (1.5, 2), (1, -2)):
            with self.assertRaises(ValueError):
                exact_result(k, n, 0.1, resolve_policy())
        with self.assertRaises(ValueError):
            exact_result(0, 0, float('nan'), resolve_policy())

    def test_legacy_nan_is_noncall_and_strict_tie_is_suppressed(self):
        result = legacy_result(float('nan'), resolve_policy())
        self.assertFalse(result.called)
        self.assertTrue(math.isnan(result.pvalue))
        self.assertIsNone(result.log_tail)
        self.assertEqual('legacy-nonfinite', result.disposition)
        self.assertEqual('cutoff', legacy_result(0.001, resolve_policy()).disposition)
        self.assertEqual('called', legacy_result(0.0009, resolve_policy()).disposition)
        with self.assertRaises(ValueError):
            legacy_result(float('inf'), resolve_policy())

    def test_legacy_statistic_matches_inherited_formula(self):
        coverage, reads, expected, error = 10.0, 3, 0.495, 0.01
        statistic = -2 * (binom.logpmf(reads, coverage, error)
                          - binom.logpmf(reads, coverage, expected))
        oracle = (binom.pmf(reads, coverage, error),
                  binom.pmf(reads, coverage, expected), chi2.sf(statistic, 1))
        self.assertEqual(oracle, legacy_statistic(coverage, reads, expected, error))
        self.assertEqual(oracle, VNTRFinder.identify_frameshift(coverage, reads, expected, error))

    def test_public_synthetic_read_count_exceeds_mean_base_coverage_reproducer(self):
        # Three supporting reads need not give three mean-covered copies: for
        # example 12 covered bases / 6 bases / 2 diploid copies gives coverage 1.
        # These evidence units differ. Preserve the inherited p=0 call here;
        # deciding whether it is scientifically defensible is a separate fix.
        coverage = 12.0 / 6 / 2
        self.assertEqual((0, 1.0, 0), legacy_statistic(coverage, 3, 0.495, 0.01))
        self.assertTrue(legacy_result(0, resolve_policy()).called)
        self.assertFalse(exact_result(3, 1, None, resolve_policy()).called)

    def test_exact_wrapper_result_preserves_tuple_api_and_missing_row(self):
        class Background(object):
            def probability_for(self, state):
                return 0.25
        records = {'D1_1': {'state_identities': {'D1_1': ((0, 0),)},
                             'opportunities': 1}}
        result = exact_caller.decide_result(records, 'D1_1', Background(), cutoff=0.3)
        self.assertEqual((result.called, result.pvalue),
                         exact_caller.decide(records, 'D1_1', Background(), cutoff=0.3))
        self.assertEqual('missing-opportunity-row',
                         exact_caller.decide_result({}, 'D1_1', None).disposition)
        records['D1_1']['opportunities'] = 0
        self.assertEqual('support-exceeds-opportunities',
                         exact_caller.decide_result(records, 'D1_1', None).disposition)


if __name__ == '__main__':
    unittest.main()
