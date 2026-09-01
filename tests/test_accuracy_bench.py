"""Behavioral tests for the external-only accuracy benchmark harness."""
import hashlib
import json
import os
import shutil
import stat
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


def _registered_worktrees():
    output = subprocess.check_output(
        ['git', 'worktree', 'list', '--porcelain', '-z'], cwd=REPO)
    return [os.path.realpath(field[len('worktree '):])
            for field in output.split('\0') if field.startswith('worktree ')]


def _common_git_directory():
    path = subprocess.check_output(
        ['git', 'rev-parse', '--git-common-dir'], cwd=REPO).strip()
    if not os.path.isabs(path):
        path = os.path.join(REPO, path)
    return os.path.realpath(path)


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

    def test_near_one_confidence_uses_a_finite_non_collapsed_interval(self):
        bench = self.require_module()
        lower, upper = bench.wilson_ci(10, 10, 1.0 - 1e-16)
        self.assertAlmostEqual(lower, 0.12696276142998747, places=14)
        self.assertEqual(upper, 1.0)

    def test_wilson_rejects_non_finite_confidence(self):
        bench = self.require_module()
        for confidence in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                bench.wilson_ci(1, 2, confidence)


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
        self.assertEqual((baseline['ppv']['numerator'],
                          baseline['ppv']['denominator']), (11, 11))
        self.assertEqual((baseline['npv']['numerator'],
                          baseline['npv']['denominator']), (11, 11))
        self.assertEqual((candidate['ppv']['numerator'],
                          candidate['ppv']['denominator']), (1, 11))
        self.assertEqual((candidate['npv']['numerator'],
                          candidate['npv']['denominator']), (1, 11))

        class_strata = report['strata']['variant_class']
        self.assertEqual([(item['value'], item['n']) for item in class_strata],
                         [('deletion', 2), ('duplication', 20)])
        length_strata = report['strata']['array_length']
        self.assertEqual([(item['value'], item['n']) for item in length_strata],
                         [(30, 20), (41, 2)])
        for stratum in class_strata + length_strata:
            self.assertNotIn('interpretation', stratum)
            for caller_metrics in stratum['metrics'].values():
                for metric in caller_metrics.values():
                    self.assertIn(metric['interpretation'],
                                  ('interpreted', 'report_only'))

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

    def test_stratum_interpretation_uses_each_callers_metric_denominator(self):
        bench = self.require_module()
        records = []
        for index in range(5):
            records.append(_record('carrier-%02d' % index,
                                   True, True, False))
        for index in range(15):
            records.append(_record('control-%02d' % index,
                                   False, True, False))

        stratum = bench.build_report(
            records, compare=True)['strata']['variant_class'][0]
        self.assertEqual(stratum['n'], 20)
        self.assertNotIn('interpretation', stratum)
        baseline = stratum['metrics']['baseline']
        candidate = stratum['metrics']['candidate']
        self.assertEqual((baseline['sensitivity']['denominator'],
                          baseline['sensitivity']['interpretation']),
                         (5, 'report_only'))
        self.assertEqual((baseline['specificity']['denominator'],
                          baseline['specificity']['interpretation']),
                         (15, 'report_only'))
        self.assertEqual((baseline['ppv']['denominator'],
                          baseline['ppv']['interpretation']),
                         (20, 'interpreted'))
        self.assertEqual((baseline['npv']['denominator'],
                          baseline['npv']['interpretation']),
                         (0, 'report_only'))
        self.assertEqual((candidate['sensitivity']['denominator'],
                          candidate['sensitivity']['interpretation']),
                         (5, 'report_only'))
        self.assertEqual((candidate['specificity']['denominator'],
                          candidate['specificity']['interpretation']),
                         (15, 'report_only'))
        self.assertEqual((candidate['ppv']['denominator'],
                          candidate['ppv']['interpretation']),
                         (0, 'report_only'))
        self.assertEqual((candidate['npv']['denominator'],
                          candidate['npv']['interpretation']),
                         (20, 'interpreted'))

    def test_every_discordance_direction_and_cause_mapping_is_exact(self):
        bench = self.require_module()
        records = [
            _record('a-new-false-negative', True, True, False),
            _record('b-new-false-positive', False, False, True),
            _record('c-fixed-false-negative', True, False, True),
            _record('d-fixed-false-positive', False, True, False),
        ]
        discordances = bench.build_report(
            records, compare=True)['comparison']['discordances']
        self.assertEqual(discordances, [
            {'sample_id': 'a-new-false-negative', 'truth': True,
             'baseline_call': True, 'candidate_call': False,
             'variant_class': 'duplication', 'array_length': 30,
             'direction': 'baseline_correct_to_candidate_incorrect',
             'cause': 'candidate_false_negative'},
            {'sample_id': 'b-new-false-positive', 'truth': False,
             'baseline_call': False, 'candidate_call': True,
             'variant_class': 'duplication', 'array_length': 30,
             'direction': 'baseline_correct_to_candidate_incorrect',
             'cause': 'candidate_false_positive'},
            {'sample_id': 'c-fixed-false-negative', 'truth': True,
             'baseline_call': False, 'candidate_call': True,
             'variant_class': 'duplication', 'array_length': 30,
             'direction': 'baseline_incorrect_to_candidate_correct',
             'cause': 'candidate_fixed_false_negative'},
            {'sample_id': 'd-fixed-false-positive', 'truth': False,
             'baseline_call': True, 'candidate_call': False,
             'variant_class': 'duplication', 'array_length': 30,
             'direction': 'baseline_incorrect_to_candidate_correct',
             'cause': 'candidate_fixed_false_positive'},
        ])

    def test_duplicate_sample_ids_are_rejected(self):
        bench = self.require_module()
        with self.assertRaises(ValueError):
            bench.build_report([
                _record('duplicate', True, True),
                _record('duplicate', False, False),
            ])

    def test_candidate_field_is_forbidden_in_baseline_mode(self):
        bench = self.require_module()
        with self.assertRaises(ValueError):
            bench.build_report([
                _record('carrier', True, True, True),
                _record('control', False, False, False),
            ], compare=False)

    def test_malformed_record_field_types_are_rejected(self):
        bench = self.require_module()
        malformed = []
        for field, value in [('sample_id', 1), ('truth', 1),
                             ('baseline_call', 0), ('variant_class', 1),
                             ('array_length', 30.5), ('array_length', True)]:
            record = _record('malformed-%s-%d' % (field, len(malformed)),
                             True, True)
            record[field] = value
            malformed.append(record)
        for record in malformed:
            with self.assertRaises((TypeError, ValueError)):
                bench.build_report([
                    record,
                    _record('valid-control', False, False),
                ])

        with self.assertRaises(TypeError):
            bench.build_report([
                _record('carrier', True, True, candidate='yes'),
                _record('control', False, False, candidate=False),
            ], compare=True)

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
    def test_nonexistent_nested_output_creates_no_components(self):
        bench = self.require_module()
        first_missing = os.path.join(self.tempdir, 'missing')
        output_dir = os.path.join(first_missing, 'nested', 'output')
        with self.assertRaises(ValueError):
            bench.publish_report({'mode': 'baseline'}, output_dir)
        self.assertFalse(os.path.lexists(first_missing))

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

    def test_main_checkout_and_every_registered_worktree_are_refused(self):
        bench = self.require_module()
        worktrees = _registered_worktrees()
        self.assertIn(os.path.realpath(REPO), worktrees)
        for worktree in worktrees:
            for path in (worktree, os.path.join(worktree, 'ignored-child')):
                with self.assertRaises(ValueError):
                    bench._resolved_external_path(path)

    def test_containment_handles_multiple_roots_without_prefix_confusion(self):
        bench = self.require_module()
        self.assertTrue(hasattr(bench, '_path_is_within'),
                        'containment must be independently testable')
        roots = ('/synthetic/main', '/synthetic/linked')
        for root in roots:
            self.assertTrue(bench._path_is_within(root, root))
            self.assertTrue(bench._path_is_within(
                os.path.join(root, 'nested'), root))
            self.assertFalse(bench._path_is_within(root + '-sibling', root))
        self.assertTrue(bench._path_is_within('/synthetic', os.sep))

    def test_common_git_metadata_directory_is_refused(self):
        bench = self.require_module()
        common_git = _common_git_directory()
        self.assertTrue(os.path.isdir(common_git))
        for path in (common_git, os.path.join(common_git, 'ignored-child')):
            with self.assertRaises(ValueError):
                bench._resolved_external_path(path)

    def test_git_boundary_discovery_failure_refuses_publication(self):
        bench = self.require_module()
        self.assertTrue(hasattr(bench, 'subprocess'),
                        'publication must discover Git boundaries')
        real_check_output = bench.subprocess.check_output

        def unavailable(*_args, **_kwargs):
            raise OSError('synthetic git failure')

        bench.subprocess.check_output = unavailable
        try:
            with self.assertRaises(ValueError):
                bench.publish_report({'mode': 'baseline'}, self.tempdir)
        finally:
            bench.subprocess.check_output = real_check_output

    def test_records_collision_is_refused_for_direct_and_symlink_paths(self):
        self.require_module()
        output_dir = os.path.join(self.tempdir, 'collision-output')
        os.mkdir(output_dir)
        output_path = os.path.join(output_dir, 'accuracy-report.json')
        records = [
            _record('collision-carrier', True, True),
            _record('collision-control', False, False),
        ]
        original = ''.join(json.dumps(record, sort_keys=True) + '\n'
                           for record in records)
        with open(output_path, 'w') as handle:
            handle.write(original)
        records_link = os.path.join(self.tempdir, 'records-link.jsonl')
        os.symlink(output_path, records_link)

        for records_path in (output_path, records_link):
            process = subprocess.Popen([
                sys.executable, SCRIPT, '--records', records_path,
                '--mode', 'baseline', '--out', output_dir,
            ], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _stdout, _stderr = process.communicate()
            self.assertNotEqual(process.returncode, 0)
            with open(output_path) as handle:
                self.assertEqual(handle.read(), original)

    def test_published_report_mode_is_owner_read_write_only(self):
        bench = self.require_module()
        output_dir = os.path.join(self.tempdir, 'private-output')
        os.mkdir(output_dir)
        output_path = bench.publish_report({'mode': 'baseline'}, output_dir)
        self.assertEqual(stat.S_IMODE(os.stat(output_path).st_mode), 0o600)

    def test_cli_help_requires_a_preexisting_output_directory(self):
        output = subprocess.check_output([sys.executable, SCRIPT, '--help'],
                                         cwd=REPO)
        self.assertIn('must already exist', output)

    def test_cli_validation_failure_creates_no_report(self):
        self.require_module()
        records_path = os.path.join(self.tempdir, 'invalid-records.jsonl')
        output_dir = os.path.join(self.tempdir, 'invalid-output')
        with open(records_path, 'w') as handle:
            handle.write(json.dumps(_record('carrier', True, True)) + '\n')
            handle.write(json.dumps(_record('control', False, False)) + '\n')
        process = subprocess.Popen([
            sys.executable, SCRIPT, '--records', records_path,
            '--mode', 'compare', '--out', output_dir,
        ], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _stdout, _stderr = process.communicate()
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(os.path.exists(
            os.path.join(output_dir, 'accuracy-report.json')))

    def test_repository_snapshot_detects_same_name_same_size_content_change(self):
        probe_root = os.path.join(self.tempdir, 'snapshot-probe')
        os.mkdir(probe_root)
        probe_path = os.path.join(probe_root, 'probe.txt')
        with open(probe_path, 'w') as handle:
            handle.write('before')
        before = self._repository_snapshot(probe_root)
        with open(probe_path, 'w') as handle:
            handle.write('after!')
        after = self._repository_snapshot(probe_root)
        self.assertNotEqual(after, before)

    def test_cli_writes_only_deterministic_json_outside_repository(self):
        self.require_module()
        records_path = os.path.join(self.tempdir, 'records.jsonl')
        output_dir = os.path.join(self.tempdir, 'output')
        os.mkdir(output_dir)
        records = [
            _record('carrier', True, True, False,
                    variant_class='insertion', array_length=29),
            _record('control', False, False, False,
                    variant_class='insertion', array_length=29),
        ]
        with open(records_path, 'w') as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + '\n')

        before = self._repository_snapshot()
        subprocess.check_call([
            sys.executable, SCRIPT, '--records', records_path,
            '--mode', 'compare', '--out', output_dir,
        ], cwd=REPO)
        after = self._repository_snapshot()
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
    def _repository_snapshot(snapshot_root=REPO):
        paths = []
        for root, dirs, files in os.walk(snapshot_root):
            dirs[:] = sorted(name for name in dirs
                              if name not in ('.git', '.superpowers'))
            relative_root = os.path.relpath(root, snapshot_root)
            paths.extend(('d', os.path.join(relative_root, name))
                         for name in dirs)
            for name in sorted(files):
                path = os.path.join(root, name)
                relative_path = os.path.join(relative_root, name)
                if os.path.islink(path):
                    digest = 'symlink:%s' % os.readlink(path)
                else:
                    hasher = hashlib.sha256()
                    with open(path, 'rb') as handle:
                        while True:
                            chunk = handle.read(1024 * 1024)
                            if not chunk:
                                break
                            hasher.update(chunk)
                    digest = hasher.hexdigest()
                paths.append(('f', relative_path, digest))
        return paths


if __name__ == '__main__':
    unittest.main()
