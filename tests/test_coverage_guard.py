"""Unit tests for Task 9: Rare-unit coverage guard.

A candidate whose repeat-unit model has implausibly low coverage relative to the locus-wide
sequencing depth is guarded against false positive calling caused by coverage collapse,
while genuine calls at normal relative coverage are unaffected.
"""
import unittest

from advntr import coverage_guard
from advntr import settings


class TestCoverageGuard(unittest.TestCase):

    def test_compute_locus_coverage(self):
        hmm_match_count = {'1': 60, '2': 60}
        estimated_ru_count = {'1': 10, '2': 1}
        ru_bp_coverage = {'1': 120000, '2': 12000}
        locus_cov = coverage_guard.compute_locus_coverage(
            ru_bp_coverage, hmm_match_count, estimated_ru_count, is_haploid=False)
        self.assertAlmostEqual(locus_cov, 100.0)

    def test_is_rare_unit_coverage_collapsed_fires_on_collapse(self):
        self.assertTrue(coverage_guard.is_rare_unit_coverage_collapsed(
            ru_coverage=5.0, locus_coverage=100.0, min_relative_coverage=0.15))

    def test_is_rare_unit_coverage_collapsed_passes_on_normal_coverage(self):
        self.assertFalse(coverage_guard.is_rare_unit_coverage_collapsed(
            ru_coverage=95.0, locus_coverage=100.0, min_relative_coverage=0.15))

    def test_guard_scales_with_sequencing_depth(self):
        self.assertTrue(coverage_guard.is_rare_unit_coverage_collapsed(
            ru_coverage=3.0, locus_coverage=30.0, min_relative_coverage=0.15))
        self.assertTrue(coverage_guard.is_rare_unit_coverage_collapsed(
            ru_coverage=30.0, locus_coverage=300.0, min_relative_coverage=0.15))
        self.assertFalse(coverage_guard.is_rare_unit_coverage_collapsed(
            ru_coverage=50.0, locus_coverage=300.0, min_relative_coverage=0.15))

    def test_guard_inactive_when_locus_coverage_zero_or_threshold_none(self):
        self.assertFalse(coverage_guard.is_rare_unit_coverage_collapsed(
            ru_coverage=0.0, locus_coverage=0.0, min_relative_coverage=0.15))
        self.assertFalse(coverage_guard.is_rare_unit_coverage_collapsed(
            ru_coverage=5.0, locus_coverage=100.0, min_relative_coverage=None))


class TestCoverageGuardIntegration(unittest.TestCase):

    def setUp(self):
        self._orig_threshold = settings.MIN_RELATIVE_RU_COVERAGE

    def tearDown(self):
        settings.MIN_RELATIVE_RU_COVERAGE = self._orig_threshold

    def test_guard_filters_candidate_on_collapsed_ru(self):
        # Mock ru_bp_coverage with collapsed coverage on RU2
        ru_bp_coverage = {'1': 10000, '2': 50}
        hmm_match_count = {'1': 8, '2': 8}
        estimated_ru_count = {'1': 9, '2': 1}
        locus_cov = coverage_guard.compute_locus_coverage(
            ru_bp_coverage, hmm_match_count, estimated_ru_count, is_haploid=False)
        ru2_cov = float(ru_bp_coverage['2']) / hmm_match_count['2'] / 2 / estimated_ru_count['2']

        # RU2 coverage ratio is ~3.125 / ~62.8 = ~0.05 (< 0.15)
        settings.MIN_RELATIVE_RU_COVERAGE = 0.15
        self.assertTrue(coverage_guard.is_rare_unit_coverage_collapsed(
            ru2_cov, locus_cov, settings.MIN_RELATIVE_RU_COVERAGE))

        # With guard disabled (None), it does not collapse
        settings.MIN_RELATIVE_RU_COVERAGE = None
        self.assertFalse(coverage_guard.is_rare_unit_coverage_collapsed(
            ru2_cov, locus_cov, settings.MIN_RELATIVE_RU_COVERAGE))


if __name__ == '__main__':
    unittest.main()
