"""The installed fitter and development benchmark share one pure evaluator."""
import os
import sys
import unittest

from advntr import background_evaluation


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import accuracy_bench  # noqa: E402


def _records():
    """Hand-written cases covering both truths, a missed call, and a compound."""
    return [
        {'sample_id': 'carrier-positive', 'truth': True,
         'baseline_call': True, 'candidate_call': True,
         'variant_class': 'deletion', 'array_length': 30},
        {'sample_id': 'control-negative', 'truth': False,
         'baseline_call': False, 'candidate_call': False,
         'variant_class': 'negative', 'array_length': 31},
        {'sample_id': 'carrier-missed', 'truth': True,
         'baseline_call': False, 'candidate_call': False,
         'variant_class': 'missing', 'array_length': 32},
        {'sample_id': 'compound-carrier', 'truth': True,
         'baseline_call': True, 'candidate_call': False,
         'variant_class': 'compound', 'array_length': 33},
    ]


class TestSharedEvaluator(unittest.TestCase):

    def test_packaged_report_matches_the_benchmark_report(self):
        expected = accuracy_bench.build_report(_records(), compare=True)
        self.assertEqual(expected,
                         background_evaluation.build_report(_records(), compare=True))

    def test_packaged_statistical_helpers_match_the_benchmark(self):
        self.assertEqual(accuracy_bench.mcnemar_exact(1, 9),
                         background_evaluation.mcnemar_exact(1, 9))
        self.assertEqual(accuracy_bench.wilson_ci(7, 11),
                         background_evaluation.wilson_ci(7, 11))

    def test_missing_candidate_call_is_rejected_identically(self):
        records = _records()
        del records[0]['candidate_call']
        for evaluator in (accuracy_bench, background_evaluation):
            with self.assertRaisesRegexp(
                    ValueError, 'comparison record carrier-positive lacks candidate_call'):
                evaluator.build_report(records, compare=True)


if __name__ == '__main__':
    unittest.main()
