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

    def test_unknown_lengths_remain_null_without_changing_accuracy(self):
        records = _records()
        expected = background_evaluation.build_report(records, compare=True)
        records[0]['array_length'] = None
        records[3]['array_length'] = None
        report = background_evaluation.build_report(records, compare=True)
        self.assertEqual(expected['metrics'], report['metrics'])
        self.assertEqual(expected['comparison']['decision'], report['comparison']['decision'])
        self.assertEqual([31, 32, None],
                         [item['value'] for item in report['strata']['array_length']])
        self.assertEqual(2, report['strata']['array_length'][-1]['n'])
        self.assertIsNone(report['comparison']['discordances'][0]['array_length'])
        for record in records:
            record['array_length'] = None
        report = background_evaluation.build_report(records, compare=True)
        self.assertEqual(1, len(report['strata']['array_length']))
        self.assertIsNone(report['strata']['array_length'][0]['value'])
        self.assertEqual(4, report['strata']['array_length'][0]['n'])

    def test_only_null_or_nonnegative_integer_lengths_are_accepted(self):
        for invalid in (True, False, 1.5, 'unknown', -1):
            records = _records()
            records[0]['array_length'] = invalid
            with self.assertRaises((TypeError, ValueError)):
                background_evaluation.build_report(records, compare=True)

    def test_benchmark_uses_the_packaged_evaluator(self):
        self.assertIs(accuracy_bench.build_report,
                      background_evaluation.build_report)
        self.assertIs(accuracy_bench.mcnemar_exact,
                      background_evaluation.mcnemar_exact)
        self.assertIs(accuracy_bench.wilson_ci,
                      background_evaluation.wilson_ci)

    def test_packaged_report_has_the_expected_counts_and_discordance(self):
        report = background_evaluation.build_report(_records(), compare=True)

        self.assertEqual(1, report['schema_version'])
        self.assertEqual('comparison', report['mode'])
        self.assertEqual(4, report['sample_count'])
        self.assertEqual({'carriers': 3, 'controls': 1},
                         report['partitions'])
        self.assertEqual({
            'numerator': 2,
            'denominator': 3,
            'estimate': 2.0 / 3.0,
            'ci95': [0.2076596008020477, 0.9385080552796037],
        }, report['metrics']['baseline']['sensitivity'])
        self.assertEqual({
            'numerator': 1,
            'denominator': 3,
            'estimate': 1.0 / 3.0,
            'ci95': [0.06149194472039621, 0.7923403991979522],
        }, report['metrics']['candidate']['sensitivity'])
        self.assertEqual({
            'baseline_only': 1,
            'candidate_only': 0,
            'discordant_total': 1,
            'p_value': 1.0,
        }, report['comparison']['mcnemar']['carriers'])
        self.assertEqual([{
            'sample_id': 'compound-carrier',
            'truth': True,
            'baseline_call': True,
            'candidate_call': False,
            'direction': 'baseline_correct_to_candidate_incorrect',
            'cause': 'candidate_false_negative',
            'variant_class': 'compound',
            'array_length': 33,
        }], report['comparison']['discordances'])

    def test_packaged_statistical_helpers_have_known_values(self):
        self.assertEqual({
            'p_value': 0.021484374999999997,
            'baseline_only': 1,
            'discordant_total': 10,
            'candidate_only': 9,
        }, background_evaluation.mcnemar_exact(1, 9))
        lower, upper = background_evaluation.wilson_ci(7, 11)
        self.assertAlmostEqual(0.35380117450784887, lower)
        self.assertAlmostEqual(0.8483352890463243, upper)
        with self.assertRaisesRegexp(ValueError, 'total must be positive'):
            background_evaluation.wilson_ci(0, 0)

    def test_missing_candidate_call_is_rejected_identically(self):
        records = _records()
        del records[0]['candidate_call']
        with self.assertRaisesRegexp(
                ValueError, 'comparison record carrier-positive lacks candidate_call'):
            background_evaluation.build_report(records, compare=True)


if __name__ == '__main__':
    unittest.main()
