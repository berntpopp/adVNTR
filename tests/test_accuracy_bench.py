"""Behavioral tests for the external-only accuracy benchmark harness."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, 'scripts', 'accuracy_bench.py')
sys.path.insert(0, os.path.join(REPO, 'scripts'))

_saved_dont_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    try:
        import accuracy_bench
    except ImportError:
        accuracy_bench = None
finally:
    sys.dont_write_bytecode = _saved_dont_write_bytecode


def _record(sample_id, truth, baseline, candidate=None,
            variant_class='duplication', array_length=30):
    record = {
        'sample_id': sample_id,
        'truth': truth,
        'baseline_call': baseline,
        'variant_class': variant_class,
        'array_length': array_length,
    }
    if candidate is not None:
        record['candidate_call'] = candidate
    return record


class _TemporaryDirectoryTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix='advntr-accuracy-test-')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def require_module(self):
        self.assertIsNotNone(
            accuracy_bench,
            'scripts/accuracy_bench.py must provide the accuracy harness')
        return accuracy_bench


class TestWilsonInterval(_TemporaryDirectoryTest):
    def test_perfect_proportion_has_non_degenerate_wilson_interval(self):
        bench = self.require_module()
        lower, upper = bench.wilson_ci(10, 10)
        self.assertAlmostEqual(lower, 0.7224672001371106, places=14)
        self.assertEqual(upper, 1.0)
        self.assertLess(lower, upper)

    def test_zero_proportion_has_non_degenerate_wilson_interval(self):
        bench = self.require_module()
        lower, upper = bench.wilson_ci(0, 10)
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 0.2775327998628892, places=14)

    def test_wilson_rejects_invalid_counts(self):
        bench = self.require_module()
        invalid = [(-1, 10), (11, 10), (1, -1), (0, 0),
                   (1.0, 10), (True, 10), (1, 10.0)]
        for successes, total in invalid:
            with self.assertRaises((TypeError, ValueError)):
                bench.wilson_ci(successes, total)


class TestMcNemarExact(_TemporaryDirectoryTest):
    def test_uses_only_discordant_pairs(self):
        bench = self.require_module()
        result = bench.mcnemar_exact(1, 9)
        self.assertEqual(result['baseline_only'], 1)
        self.assertEqual(result['candidate_only'], 9)
        self.assertEqual(result['discordant_total'], 10)
        self.assertAlmostEqual(result['p_value'], 0.021484375, places=15)

    def test_no_discordant_pairs_has_p_value_one(self):
        bench = self.require_module()
        self.assertEqual(bench.mcnemar_exact(0, 0)['p_value'], 1.0)

    def test_mcnemar_rejects_non_count_inputs(self):
        bench = self.require_module()
        for baseline_only, candidate_only in [(-1, 0), (0, -1),
                                               (1.0, 2), (True, 2)]:
            with self.assertRaises((TypeError, ValueError)):
                bench.mcnemar_exact(baseline_only, candidate_only)


class TestAccuracyReport(_TemporaryDirectoryTest):
    def test_carriers_and_controls_have_separate_paired_tests(self):
        bench = self.require_module()
        records = [
            _record('carrier-1', True, True, False),
            _record('carrier-2', True, True, False),
            _record('carrier-3', True, True, True),
            _record('control-1', False, True, False),
            _record('control-2', False, True, False),
            _record('control-3', False, True, False),
        ]
        comparison = bench.build_report(records, compare=True)['comparison']
        carriers = comparison['mcnemar']['carriers']
        controls = comparison['mcnemar']['controls']
        self.assertEqual((carriers['baseline_only'], carriers['candidate_only']),
                         (2, 0))
        self.assertEqual(carriers['p_value'], 0.5)
        self.assertEqual((controls['baseline_only'], controls['candidate_only']),
                         (0, 3))
        self.assertEqual(controls['p_value'], 0.25)

    def test_metrics_strata_discordances_and_decision_flags_are_exposed(self):
        bench = self.require_module()
        records = []
        for index in range(10):
            records.append(_record('common-carrier-%02d' % index,
                                   True, True, False))
            records.append(_record('common-control-%02d' % index,
                                   False, False, True))
        records.extend([
            _record('rare-carrier', True, True, True,
                    variant_class='deletion', array_length=41),
            _record('rare-control', False, False, False,
                    variant_class='deletion', array_length=41),
        ])

        report = bench.build_report(records, compare=True)
        baseline = report['metrics']['baseline']
        candidate = report['metrics']['candidate']
        self.assertEqual(baseline['sensitivity']['estimate'], 1.0)
        self.assertEqual(baseline['specificity']['estimate'], 1.0)
        self.assertLess(baseline['sensitivity']['ci95'][0], 1.0)
        self.assertEqual(candidate['sensitivity']['numerator'], 1)
        self.assertEqual(candidate['sensitivity']['denominator'], 11)
        self.assertEqual(candidate['specificity']['numerator'], 1)
        self.assertEqual(candidate['specificity']['denominator'], 11)
        self.assertIn('ppv', candidate)
        self.assertIn('npv', candidate)

        class_strata = report['strata']['variant_class']
        self.assertEqual([(item['value'], item['n'], item['interpretation'])
                          for item in class_strata],
                         [('deletion', 2, 'report_only'),
                          ('duplication', 20, 'interpreted')])
        length_strata = report['strata']['array_length']
        self.assertEqual([(item['value'], item['n'], item['interpretation'])
                          for item in length_strata],
                         [(30, 20, 'interpreted'),
                          (41, 2, 'report_only')])

        discordances = report['comparison']['discordances']
        self.assertEqual(len(discordances), 20)
        self.assertEqual([item['sample_id'] for item in discordances],
                         sorted(item['sample_id'] for item in discordances))
        for item in discordances:
            self.assertIn(item['direction'],
                          ('baseline_correct_to_candidate_incorrect',
                           'baseline_incorrect_to_candidate_correct'))
            self.assertIn(item['cause'],
                          ('candidate_false_negative',
                           'candidate_false_positive',
                           'candidate_fixed_false_negative',
                           'candidate_fixed_false_positive'))

        decision = report['comparison']['decision']
        self.assertTrue(decision['carrier_p_below_0_01'])
        self.assertTrue(decision['control_p_below_0_01'])
        self.assertTrue(decision['specificity_fell'])
        self.assertTrue(
            decision['candidate_sensitivity_lower_ci_below_baseline_point'])

    def test_zero_prediction_denominator_is_reported_as_undefined(self):
        bench = self.require_module()
        report = bench.build_report([
            _record('carrier', True, False),
            _record('control', False, False),
        ], compare=False)
        ppv = report['metrics']['baseline']['ppv']
        self.assertEqual(ppv['denominator'], 0)
        self.assertIsNone(ppv['estimate'])
        self.assertIsNone(ppv['ci95'])

    def test_empty_records_comparison_and_truth_partitions_are_rejected(self):
        bench = self.require_module()
        with self.assertRaises(ValueError):
            bench.build_report([], compare=False)
        with self.assertRaises(ValueError):
            bench.build_report([
                _record('carrier', True, True),
                _record('control', False, False),
            ], compare=True)
        with self.assertRaises(ValueError):
            bench.build_report([
                _record('carrier-1', True, True),
                _record('carrier-2', True, False),
            ], compare=False)


class TestExternalOutputBoundary(_TemporaryDirectoryTest):
    def test_direct_in_repository_output_is_refused(self):
        bench = self.require_module()
        forbidden = os.path.join(REPO, '.ignored-accuracy-output')
        for path in (REPO, forbidden):
            with self.assertRaises(ValueError):
                bench.publish_report({'mode': 'baseline'}, path)
        self.assertFalse(os.path.exists(forbidden))

    def test_symlink_resolving_inside_repository_is_refused(self):
        bench = self.require_module()
        forbidden = os.path.join(REPO, '.ignored-accuracy-output')
        link = os.path.join(self.tempdir, 'output-link')
        os.symlink(forbidden, link)
        with self.assertRaises(ValueError):
            bench.publish_report({'mode': 'baseline'}, link)
        self.assertFalse(os.path.exists(forbidden))

    def test_cli_writes_only_deterministic_json_outside_repository(self):
        self.require_module()
        records_path = os.path.join(self.tempdir, 'records.jsonl')
        output_dir = os.path.join(self.tempdir, 'output')
        records = [
            _record('carrier', True, True, False,
                    variant_class='insertion', array_length=29),
            _record('control', False, False, False,
                    variant_class='insertion', array_length=29),
        ]
        with open(records_path, 'w') as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + '\n')

        before = self._repository_paths()
        subprocess.check_call([
            sys.executable, SCRIPT, '--records', records_path,
            '--mode', 'compare', '--out', output_dir,
        ], cwd=REPO)
        after = self._repository_paths()
        self.assertEqual(after, before)

        output_path = os.path.join(output_dir, 'accuracy-report.json')
        self.assertEqual(os.listdir(output_dir), ['accuracy-report.json'])
        with open(output_path) as handle:
            first_bytes = handle.read()
        parsed = json.loads(first_bytes)
        self.assertEqual(parsed['mode'], 'comparison')

        subprocess.check_call([
            sys.executable, SCRIPT, '--records', records_path,
            '--mode', 'compare', '--out', output_dir,
        ], cwd=REPO)
        with open(output_path) as handle:
            self.assertEqual(handle.read(), first_bytes)

    @staticmethod
    def _repository_paths():
        paths = []
        for root, dirs, files in os.walk(REPO):
            dirs[:] = sorted(name for name in dirs
                              if name not in ('.git', '.superpowers'))
            relative_root = os.path.relpath(root, REPO)
            paths.extend(('d', os.path.join(relative_root, name))
                         for name in dirs)
            paths.extend(('f', os.path.join(relative_root, name))
                         for name in sorted(files))
        return paths


if __name__ == '__main__':
    unittest.main()
