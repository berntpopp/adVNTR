"""Capabilities describe only implemented features and the actual runtime payload."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from advntr import __version__
from advntr.capabilities import (describe_capabilities, payload_manifest, payload_build_id,
                                  canonical_bytes, file_manifest, validate_manifest)


class TestCapabilities(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='advntr-capability-test-')
        for package in ('advntr', 'hmm'):
            os.makedirs(os.path.join(self.root, package))
            self._write(package + '/__init__.py', '# synthetic package\n')
        self._write('advntr/decision.py', 'VALUE = 1\n')
        self._write('hmm/base.so', 'synthetic base bytes')
        self._write('hmm/hmm.so', 'synthetic decoder bytes')

    def tearDown(self):
        shutil.rmtree(self.root)

    def _write(self, name, data):
        with open(os.path.join(self.root, name), 'wb') as handle:
            handle.write(data)

    def test_unidentified_payload_has_exact_honest_capabilities(self):
        result = describe_capabilities(self.root)
        self.assertEqual(set(('schema_version', 'package_version', 'build_id',
                              'source_revision', 'capabilities', 'capture_schema_versions',
                              'policy_schema_versions', 'background_recipe_ids')), set(result))
        self.assertEqual('advntr-capabilities-v1', result['schema_version'])
        self.assertEqual(__version__, result['package_version'])
        self.assertIsNone(result['source_revision'])
        self.assertEqual(['fit-background-installed-v1', 'frameshift-calibration-capture-v1',
                          'frameshift-run-local-policy-v1'], result['capabilities'])
        self.assertEqual([1], result['capture_schema_versions'])
        self.assertEqual([], result['policy_schema_versions'])
        self.assertEqual(['recipe-v1'], result['background_recipe_ids'])
        self.assertNotIn(self.root, json.dumps(result))

    def test_build_id_is_the_canonical_actual_file_manifest_hash(self):
        manifest = payload_manifest(self.root)
        self.assertEqual(sorted(row['path'] for row in manifest), [row['path'] for row in manifest])
        decision = next(row for row in manifest if row['path'] == 'advntr/decision.py')
        self.assertEqual(hashlib.sha256('VALUE = 1\n').hexdigest(), decision['sha256'])
        self.assertEqual(10, decision['size_bytes'])
        expected = hashlib.sha256(json.dumps(
            {'schema_version': 'advntr-runtime-payload-v1', 'files': manifest,
             'source_identity_sha256': None},
            sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()
        self.assertEqual(expected, describe_capabilities(self.root)['build_id'])
        self.assertEqual(expected, payload_build_id(manifest))

    def test_path_move_and_caches_do_not_change_identity_but_code_and_binary_do(self):
        initial = describe_capabilities(self.root)['build_id']
        self._write('advntr/decision.pyc', 'bytecode cache')
        self._write('advntr/decision.pyo', 'optimized cache')
        self.assertEqual(initial, describe_capabilities(self.root)['build_id'])
        moved = self.root + '-moved'
        shutil.copytree(self.root, moved)
        try:
            self.assertEqual(initial, describe_capabilities(moved)['build_id'])
        finally:
            shutil.rmtree(moved)
        self._write('advntr/decision.py', 'VALUE = 2\n')
        changed = describe_capabilities(self.root)['build_id']
        self.assertNotEqual(initial, changed)
        self._write('hmm/hmm.so', 'different decoder bytes')
        self.assertNotEqual(changed, describe_capabilities(self.root)['build_id'])

    def test_extra_compiled_extension_changes_payload_identity(self):
        first = describe_capabilities(self.root)['build_id']
        self._write('hmm/hmm_instrumented.so', 'additional compiled code')
        self.assertNotEqual(first, describe_capabilities(self.root)['build_id'])

    def test_symlinked_payload_refused(self):
        os.symlink('decision.py', os.path.join(self.root, 'advntr', 'alias.py'))
        with self.assertRaisesRegexp(ValueError, 'symlink'):
            describe_capabilities(self.root)

    def test_malformed_file_records_and_paths_are_rejected(self):
        valid = {'path': 'advntr/module.py', 'size_bytes': 0, 'sha256': 'a' * 64}
        for records in ([], (), [None], [dict(valid, extra=True)],
                        [dict(valid, path='../escape.py')], [dict(valid, path='/absolute.py')],
                        [dict(valid, size_bytes=True)], [dict(valid, size_bytes=-1)],
                        [dict(valid, sha256='a' * 64 + '\n')], [valid, valid]):
            with self.assertRaises(ValueError):
                validate_manifest(records)
        for path in ('../escape.py', '/absolute.py', 'advntr/./module.py', 'advntr\\module.py'):
            with self.assertRaises(ValueError):
                file_manifest(self.root, [path])
        with self.assertRaises(ValueError):
            file_manifest(self.root, ['advntr'])
        with self.assertRaises(ValueError):
            payload_build_id([valid], 'invalid')

    def test_missing_package_and_symlinked_subdirectory_are_rejected(self):
        shutil.rmtree(os.path.join(self.root, 'hmm'))
        with self.assertRaises(ValueError):
            describe_capabilities(self.root)
        os.mkdir(os.path.join(self.root, 'hmm'))
        os.symlink('../hmm', os.path.join(self.root, 'advntr', 'linked'))
        with self.assertRaisesRegexp(ValueError, 'symlink'):
            describe_capabilities(self.root)

    def test_duplicate_or_nonfinite_build_metadata_is_rejected(self):
        for content in ('{"value":1,"value":2}', '{"value":NaN}', '{"value":Infinity}'):
            self._write('advntr/_build_identity.json', content)
            with self.assertRaises(ValueError):
                describe_capabilities(self.root)

    def test_wrong_metadata_schema_and_symlink_are_rejected(self):
        self._write('advntr/_build_identity.json', canonical_bytes({'unexpected': True}))
        with self.assertRaises(ValueError):
            describe_capabilities(self.root)
        os.unlink(os.path.join(self.root, 'advntr', '_build_identity.json'))
        os.symlink('decision.py', os.path.join(self.root, 'advntr', '_build_identity.json'))
        with self.assertRaisesRegexp(ValueError, 'symlink'):
            describe_capabilities(self.root)

    def test_corrupt_present_identity_is_not_silently_unidentified(self):
        self._write('advntr/_build_identity.json', '{bad')
        with self.assertRaises(ValueError):
            describe_capabilities(self.root)


class TestCapabilitiesCommand(unittest.TestCase):

    def _run(self, arguments):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        environment = dict(os.environ, PYTHONPATH=root)
        process = subprocess.Popen([sys.executable, '-m', 'advntr', 'capabilities'] + arguments,
                                   cwd=tempfile.gettempdir(), env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return process.returncode, stdout, stderr

    def test_cli_json_uses_installed_location_not_cwd(self):
        status, stdout, stderr = self._run(['--json'])
        self.assertEqual(0, status, stderr)
        self.assertEqual(describe_capabilities(), json.loads(stdout))

    def test_cli_refuses_missing_duplicate_abbreviated_and_unknown_options(self):
        for arguments in ([], ['--json', '--json'], ['--j'], ['--json=true'], ['--json', '--other']):
            status, stdout, stderr = self._run(arguments)
            self.assertEqual(2, status, (arguments, stderr))
            self.assertEqual('', stdout)


if __name__ == '__main__':
    unittest.main()
