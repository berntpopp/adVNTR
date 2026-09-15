"""Build distributions and run fit-background with no checkout on sys.path."""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


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


class TestInstalledBackgroundFitter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp(prefix='advntr-installed-fit-')
        source = os.path.join(cls.tempdir, 'source')
        shutil.copytree(REPO, source, ignore=_ignore_build_products)
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
