"""Build distributions and run fit-background with no checkout on sys.path."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

from tests.test_frameshift_capture_record import capture_document as complete_capture_document
from tests.test_frameshift_replay_policy import replay_policy


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_OUTPUTS = (
    'installed.background.json',
    'installed.build-report.json',
    'installed.build-report.md',
    'installed.cv.json',
    'installed.falsification.json',
    'installed.predictions.json',
    'installed.sidecar.json',
    'installed.states.tsv',
)


def _ignore_build_products(_directory, names):
    ignored = set(name for name in names if name in ('.git', 'build', 'dist'))
    ignored.update(name for name in names
                   if name.endswith(('.pyc', '.pyo', '.so')))
    ignored.update(name for name in names
                   if name in ('base.c', 'hmm.c', 'hmm_instrumented.c'))
    return ignored


def _capture_document(opportunities):
    return {
        'schema': 'advntr.frameshift.calibration',
        'version': 1,
        'vntr_id': 25561,
        'read_length': 1,
        'is_haploid': False,
        'spans': [['1', 2, 0, False, False, opportunities]],
        'candidates': [{
            'candidate': 'D1_1',
            'legacy_states': [],
            'legacy_support': 0,
            'opportunities': opportunities,
            'pattern_index': '1',
            'support': 0,
            'support_identities': [],
            'state_identities': {},
            'avg_bp_coverage': 10.0,
            'ru_bp_coverage': 10,
            'ru_bp_coverage_ratio': 1,
            'ru_length': 1,
        }],
    }


def _write_inputs(root, zero_control_opportunities=False):
    records = []
    specifications = (
        ('control-a', False, 'pair-a'),
        ('carrier-a', True, 'pair-a'),
        ('control-b', False, 'pair-b'),
        ('carrier-b', True, 'pair-b'),
    )
    for sample_id, truth, pair_id in specifications:
        records.append({'sample_id': sample_id, 'truth': truth,
                        'partition': 'calibration', 'pair_id': pair_id,
                        'variant_class': 'negative' if not truth else 'compound',
                        'array_length': 30})
        run = os.path.join(root, 'runs', sample_id)
        output = os.path.join(run, 'output')
        work = os.path.join(run, 'work')
        os.makedirs(output)
        os.makedirs(work)
        opportunities = 0 if zero_control_opportunities and not truth else 10
        with open(os.path.join(output, 'calibration.jsonl'), 'w') as handle:
            handle.write(json.dumps(_capture_document(opportunities),
                                    sort_keys=True, separators=(',', ':')) + '\n')
        with open(os.path.join(run, 'result.json'), 'w') as handle:
            json.dump({'data_rows': 0}, handle)
        with open(os.path.join(work, 'log_%s.bam.log' % sample_id), 'w') as handle:
            handle.write('INFO:Using read length 1\nINFO:RU1 A\n')
    labels = os.path.join(root, 'labels.json')
    with open(labels, 'w') as handle:
        json.dump({'samples': records}, handle)
    return labels


def _write_v2_inputs(root, producer):
    records = []
    specifications = (
        ('control-a', False, 'pair-a'),
        ('carrier-a', True, 'pair-a'),
        ('control-b', False, 'pair-b'),
        ('carrier-b', True, 'pair-b'),
    )
    for sample_id, truth, pair_id in specifications:
        records.append({'sample_id': sample_id, 'truth': truth,
                        'partition': 'calibration', 'pair_id': pair_id,
                        'variant_class': 'negative' if not truth else 'compound',
                        'array_length': None})
        output = os.path.join(root, 'runs', sample_id, 'output')
        os.makedirs(output)
        document = complete_capture_document()
        document['producer'] = dict(producer)
        with open(os.path.join(output, 'calibration.jsonl'), 'w') as handle:
            handle.write(json.dumps(document, sort_keys=True, separators=(',', ':')) + '\n')
    labels = os.path.join(root, 'labels.json')
    with open(labels, 'w') as handle:
        json.dump({'samples': records}, handle)
    return labels


class TestInstalledBackgroundFitter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp(prefix='advntr-installed-fit-')
        source = os.path.join(cls.tempdir, 'source')
        shutil.copytree(REPO, source, ignore=_ignore_build_products)
        subprocess.check_call(['git', 'init', '-q'], cwd=source)
        subprocess.check_call(['git', 'add', '-A'], cwd=source)
        subprocess.check_call(['git', '-c', 'user.name=Fixture',
                               '-c', 'user.email=fixture@example.invalid',
                               'commit', '-qm', 'synthetic source snapshot'], cwd=source)
        cls.source_revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=source).strip()
        dist = os.path.join(cls.tempdir, 'dist')
        os.makedirs(dist)
        subprocess.check_call(
            [sys.executable, 'setup.py', 'sdist', '--dist-dir', dist], cwd=source)
        subprocess.check_call(
            [sys.executable, 'setup.py', 'bdist_wheel', '--dist-dir', dist], cwd=source)
        wheels = sorted(os.path.join(dist, name) for name in os.listdir(dist)
                        if name.endswith('.whl'))
        sdists = sorted(os.path.join(dist, name) for name in os.listdir(dist)
                        if name.endswith('.tar.gz'))
        if len(wheels) != 1:
            raise AssertionError('expected one wheel, found %r' % wheels)
        if len(sdists) != 1:
            raise AssertionError('expected one sdist, found %r' % sdists)
        artifacts = wheels + sdists
        cls.installations = []
        for artifact in artifacts:
            cls._assert_benchmark_script_absent(artifact)
            target = os.path.join(cls.tempdir, 'site-%d' % len(cls.installations))
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                 '--no-deps', '--target', target, artifact])
            cls.installations.append((os.path.basename(artifact), target))
        shutil.rmtree(source)
        shutil.rmtree(dist)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir, ignore_errors=True)

    @classmethod
    def _assert_benchmark_script_absent(cls, artifact):
        if artifact.endswith('.whl'):
            archive = zipfile.ZipFile(artifact)
        else:
            archive = tarfile.open(artifact)
        try:
            names = archive.namelist() if hasattr(archive, 'namelist') else archive.getnames()
        finally:
            archive.close()
        if any(name == 'scripts/accuracy_bench.py'
               or name.endswith('/scripts/accuracy_bench.py') for name in names):
            raise AssertionError('%s shipped the development benchmark' % artifact)
        if any(name == 'hmm/hmm_instrumented.pyx'
               or name.endswith('/hmm/hmm_instrumented.pyx') for name in names):
            raise AssertionError('%s shipped the test-only decoder' % artifact)

    def _environment(self, site):
        environment = dict(os.environ)
        environment['PYTHONPATH'] = site
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        return environment

    def _command(self, capture_root, labels, out_dir):
        return [
            sys.executable, '-m', 'advntr', 'fit-background',
            '--capture-root', capture_root,
            '--labels', labels,
            '--partition', 'calibration',
            '--out-dir', out_dir,
            '--profile', 'installed',
            '--folds', '2',
            '--insert-lengths', '1',
        ]

    def _installed_capabilities(self, site, execution):
        return json.loads(subprocess.check_output(
            [sys.executable, '-m', 'advntr', 'capabilities', '--json'],
            cwd=execution, env=self._environment(site)))

    def test_installed_capabilities_bind_clean_source_and_actual_package_bytes(self):
        from advntr.capabilities import canonical_bytes, payload_build_id, payload_manifest
        for name, site in self.installations:
            execution = tempfile.mkdtemp(prefix='advntr-cap-execution-', dir=self.tempdir)
            output = subprocess.check_output(
                [sys.executable, '-m', 'advntr', 'capabilities', '--json'],
                cwd=execution, env=self._environment(site))
            document = json.loads(output)
            self.assertEqual(self.source_revision, document['source_revision'], name)
            with open(os.path.join(site, 'advntr', '_build_identity.json')) as handle:
                identity = json.load(handle)
            source_digest = hashlib.sha256(canonical_bytes(identity['source'])).hexdigest()
            self.assertEqual(payload_build_id(payload_manifest(site), source_digest), document['build_id'])
            self.assertEqual([1, 2], document['capture_schema_versions'])
            self.assertEqual([
                'advntr-frameshift-policy-v1',
                'advntr-frameshift-replay-policy-v1',
            ], document['policy_schema_versions'])
            self.assertIn('frameshift-calibration-capture-v2', document['capabilities'])
            self.assertIn('frameshift-replay-v1', document['capabilities'])
            self.assertNotIn(self.tempdir, output)
            self.assertTrue(os.path.isfile(os.path.join(site, 'advntr', '_build_identity.json')))

    def test_wheel_and_sdist_run_v2_replay_and_fit_outside_the_checkout(self):
        for name, site in self.installations:
            execution = tempfile.mkdtemp(prefix='advntr-v2-execution-', dir=self.tempdir)
            capabilities = self._installed_capabilities(site, execution)
            producer = dict((key, capabilities[key])
                            for key in ('package_version', 'build_id', 'source_revision'))

            capture_root = os.path.join(execution, 'replay captures')
            os.makedirs(capture_root)
            capture = complete_capture_document()
            capture['producer'] = dict(producer)
            capture_bytes = json.dumps(capture, sort_keys=True, separators=(',', ':')) + '\n'
            capture_name = 'invented capture.jsonl'
            with open(os.path.join(capture_root, capture_name), 'w') as handle:
                handle.write(capture_bytes)
            manifest = os.path.join(execution, 'manifest.json')
            with open(manifest, 'w') as handle:
                json.dump({
                    'schema_version': 'advntr-frameshift-replay-manifest-v1',
                    'captures': [{
                        'key': 'invented', 'filename': capture_name,
                        'sha256': hashlib.sha256(capture_bytes).hexdigest(), 'vntr_ids': [17],
                    }],
                }, handle)
            policy = os.path.join(execution, 'policy.json')
            with open(policy, 'w') as handle:
                json.dump(replay_policy(support=5), handle)
            replay_output = os.path.join(execution, 'replay output')
            subprocess.check_call([
                sys.executable, '-m', 'advntr', 'replay-frameshift',
                '--capture-root', capture_root, '--manifest', manifest,
                '--policy', policy, '--output', replay_output,
            ], cwd=execution, env=self._environment(site))
            replay = json.load(open(os.path.join(replay_output, 'replay.json')))
            result = replay['results'][0]['vntrs'][0]['result']
            self.assertTrue(result['baseline_parity'], name)
            self.assertEqual([], result['calls'], name)
            self.assertEqual(producer, replay['replay_producer'], name)

            fit_root = os.path.join(execution, 'fit captures')
            labels = _write_v2_inputs(fit_root, producer)
            fit_output = os.path.join(execution, 'fit output')
            subprocess.check_call(
                self._command(fit_root, labels, fit_output), cwd=execution,
                env=self._environment(site))
            sidecar = json.load(open(os.path.join(fit_output, 'installed.sidecar.json')))
            self.assertEqual('recipe-v1', sidecar['background_recipe_id'], name)
            self.assertEqual(producer, sidecar['capture_identity']['producer'], name)
            cv = json.load(open(os.path.join(fit_output, 'installed.cv.json')))
            self.assertEqual([None], [row['value'] for row in cv['accuracy_bench_report']['strata']['array_length']], name)

    def test_wheel_and_sdist_run_the_fitter_outside_the_checkout(self):
        for name, site in self.installations:
            execution = tempfile.mkdtemp(prefix='advntr-fit-execution-', dir=self.tempdir)
            evaluator_file = subprocess.check_output(
                [sys.executable, '-c',
                 'import os; import advntr.background_evaluation as e; '
                 'print(os.path.realpath(e.__file__))'],
                cwd=execution, env=self._environment(site)).strip()
            real_site = os.path.realpath(site)
            self.assertTrue(
                evaluator_file.startswith(real_site + os.sep),
                '%s imported evaluator from %s instead of %s' %
                (name, evaluator_file, real_site))
            capture_root = os.path.join(execution, 'captures')
            labels = _write_inputs(capture_root)
            out_dir = os.path.join(execution, 'output')
            output = subprocess.check_output(
                self._command(capture_root, labels, out_dir), cwd=execution,
                env=self._environment(site), stderr=subprocess.STDOUT)
            self.assertEqual(sorted(EXPECTED_OUTPUTS), sorted(
                item for item in os.listdir(out_dir) if item != 'loader-refusal-probe.json'),
                '%s output:\n%s' % (name, output))
            artifact = json.load(open(os.path.join(out_dir, 'installed.background.json')))
            self.assertEqual('advntr.frameshift.background', artifact['schema'])
            sidecar = json.load(open(os.path.join(out_dir, 'installed.sidecar.json')))
            self.assertEqual({}, sidecar['preregistration_overrides'])
            self.assertEqual({
                'apply_floor': True,
                'dispersion_min_events': 20,
                'dispersion_min_samples': 15,
                'dispersion_threshold': 3.0,
                'floor_target': 0.001,
                'kprot': 4,
                'min_events': 10,
                'phi': 2.0,
            }, sidecar['preregistered_hyperparameters'])

    def test_all_control_opportunities_zero_is_refused(self):
        name, site = self.installations[0]
        execution = tempfile.mkdtemp(prefix='advntr-zero-opportunity-', dir=self.tempdir)
        capture_root = os.path.join(execution, 'captures')
        labels = _write_inputs(capture_root, zero_control_opportunities=True)
        out_dir = os.path.join(execution, 'output')
        process = subprocess.Popen(
            self._command(capture_root, labels, out_dir), cwd=execution,
            env=self._environment(site), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = process.communicate()[0]
        self.assertNotEqual(0, process.returncode, '%s output:\n%s' % (name, output))
        self.assertIn('all control opportunities are zero', output)

    def test_explicit_legacy_worktree_is_a_named_deprecation_error(self):
        name, site = self.installations[0]
        execution = tempfile.mkdtemp(prefix='advntr-worktree-option-', dir=self.tempdir)
        capture_root = os.path.join(execution, 'captures')
        labels = _write_inputs(capture_root)
        command = self._command(capture_root, labels, os.path.join(execution, 'output'))
        command.extend(['--worktree', execution])
        process = subprocess.Popen(
            command, cwd=execution, env=self._environment(site),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = process.communicate()[0]
        self.assertNotEqual(0, process.returncode, '%s output:\n%s' % (name, output))
        self.assertIn('--worktree is deprecated', output)


if __name__ == '__main__':
    unittest.main()
