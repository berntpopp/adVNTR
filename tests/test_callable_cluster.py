"""Tests for callable cluster size bounds and validation (Issue #7)."""
import unittest

from advntr.callable_cluster import (
    max_callable_cluster_size,
    practical_callable_cluster_size,
    is_cluster_callable,
    validate_cluster_sizes,
    CallableClusterError
)


class TestCallableCluster(unittest.TestCase):

    def test_max_callable_cluster_size_diploid(self):
        # Diploid: 0.99 / (2 * 0.01) = 49.5 -> max callable is 49
        self.assertEqual(max_callable_cluster_size(is_haploid=False, error_rate=0.01), 49)

    def test_max_callable_cluster_size_haploid(self):
        # Haploid: 0.99 / 0.01 = 99.0 -> max callable is 98 (strictly below crossover)
        self.assertEqual(max_callable_cluster_size(is_haploid=True, error_rate=0.01), 98)

    def test_is_cluster_callable(self):
        self.assertTrue(is_cluster_callable(1, is_haploid=False))
        self.assertTrue(is_cluster_callable(6, is_haploid=False))
        self.assertTrue(is_cluster_callable(49, is_haploid=False))
        self.assertFalse(is_cluster_callable(50, is_haploid=False))
        self.assertFalse(is_cluster_callable(100, is_haploid=False))

    def test_practical_callable_cluster_size(self):
        # With min_ratio 2.0: 0.99 / (2 * 0.02) = 24.75 -> 24
        practical = practical_callable_cluster_size(is_haploid=False, error_rate=0.01, min_ratio=2.0)
        self.assertEqual(practical, 24)

    def test_validate_cluster_sizes_clean(self):
        # Shipped MUC1 clusters: sizes 1 and 6
        clusters = [['ACGT'] * 6, ['ACGT'] * 1]
        issues = validate_cluster_sizes(clusters, is_haploid=False, strict=True)
        self.assertEqual(len(issues), 0)

    def test_validate_cluster_sizes_detects_inversion(self):
        clusters = [['ACGT'] * 60, ['ACGT'] * 5]
        issues = validate_cluster_sizes(clusters, is_haploid=False, strict=False)
        self.assertIn(0, issues)
        self.assertEqual(issues[0], (60, 49))

        with self.assertRaises(CallableClusterError):
            validate_cluster_sizes(clusters, is_haploid=False, strict=True)


if __name__ == '__main__':
    unittest.main()
