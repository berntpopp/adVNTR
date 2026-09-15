"""Source provenance needs both a clean checkout and actual input-byte agreement."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from build_identity import source_identity, write_build_identity
from advntr.capabilities import describe_capabilities, canonical_bytes, validate_source_identity


class TestBuildIdentity(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='advntr-source-identity-')
        for package in ('advntr', 'hmm'):
            os.makedirs(os.path.join(self.root, package))
            self._write(package + '/__init__.py', '# fixture\n')
        for name in ('setup.py', 'build_identity.py', 'build_config.py', 'MANIFEST.in'):
            self._write(name, '# fixture\n')
        for name in ('base.pyx', 'base.pxd', 'hmm.pyx', 'cqueue.pxd', '_viterbi_fill_core.pxi', 'queue.c', 'queue.h'):
            self._write('hmm/' + name, 'fixture\n')
        self._write('.gitignore', '*.so\n*.pyc\n_source_identity.json\n_build_identity.json\n')

    def tearDown(self):
        shutil.rmtree(self.root)

    def _write(self, name, data):
        with open(os.path.join(self.root, name), 'wb') as handle:
            handle.write(data)

    def _git(self, *args):
        return subprocess.check_output(['git', '-C', self.root] + list(args), stderr=subprocess.STDOUT).strip()

    def _commit(self):
        self._git('init', '-q')
        self._git('add', '-A')
        self._git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                  'commit', '-qm', 'synthetic source')
        return self._git('rev-parse', 'HEAD')

    def test_unidentified_and_dirty_sources_are_not_given_a_revision(self):
        self.assertIsNone(source_identity(self.root)['source_revision'])
        revision = self._commit()
        self.assertEqual(revision, source_identity(self.root)['source_revision'])
        self._write('advntr/__init__.py', '# changed\n')
        self.assertIsNone(source_identity(self.root)['source_revision'])

    def test_ignored_binary_cannot_create_clean_source_build_provenance(self):
        self._commit()
        self._write('hmm/base.so', 'old base')
        self._write('hmm/hmm.so', 'old hmm')
        self.assertIsNone(describe_capabilities(self.root)['source_revision'])

    def test_assume_unchanged_index_does_not_hide_modified_build_input(self):
        self._commit()
        self._git('update-index', '--assume-unchanged', 'hmm/queue.h')
        self._write('hmm/queue.h', 'modified hidden header\n')
        self.assertIsNone(source_identity(self.root)['source_revision'])

    def test_source_archive_attestation_retains_revision_only_with_matching_inputs(self):
        revision = self._commit()
        identity = source_identity(self.root)
        self._write('_source_identity.json', canonical_bytes(identity))
        shutil.rmtree(os.path.join(self.root, '.git'))
        self.assertEqual(revision, source_identity(self.root)['source_revision'])
        self._write('hmm/queue.c', 'changed native source\n')
        with self.assertRaisesRegexp(ValueError, 'source'):
            source_identity(self.root)

    def test_source_revision_cannot_carry_a_trailing_newline(self):
        identity = source_identity(self.root)
        identity['source_revision'] = 'a' * 40 + '\n'
        with self.assertRaisesRegexp(ValueError, 'revision'):
            validate_source_identity(identity)

    def test_environment_claimed_revision_cannot_replace_source_evidence(self):
        saved = os.environ.get('GIT_COMMIT')
        os.environ['GIT_COMMIT'] = 'a' * 40
        try:
            self.assertIsNone(source_identity(self.root)['source_revision'])
        finally:
            if saved is None:
                del os.environ['GIT_COMMIT']
            else:
                os.environ['GIT_COMMIT'] = saved

    def test_source_attestation_schema_and_digest_are_checked(self):
        valid = source_identity(self.root)
        for broken in (None, dict(valid, extra=True), dict(valid, source_manifest_sha256='0' * 64)):
            with self.assertRaises(ValueError):
                validate_source_identity(broken)
        self._write('record.json', canonical_bytes(valid))
        os.symlink('record.json', os.path.join(self.root, '_source_identity.json'))
        with self.assertRaisesRegexp(ValueError, 'symlink'):
            source_identity(self.root)

    def test_present_torn_source_metadata_is_an_error(self):
        self._write('_source_identity.json', '{')
        with self.assertRaises(ValueError):
            source_identity(self.root)

    def test_tampered_revision_is_not_consistent_with_the_existing_build_id(self):
        source = source_identity(self.root)
        write_build_identity(self.root, source)
        path = os.path.join(self.root, 'advntr', '_build_identity.json')
        with open(path) as handle:
            identity = json.load(handle)
        identity['source']['source_revision'] = 'a' * 40
        self._write('advntr/_build_identity.json', canonical_bytes(identity))
        with self.assertRaisesRegexp(ValueError, 'runtime payload'):
            describe_capabilities(self.root)

    def test_runtime_file_sizes_cannot_be_booleans_even_when_equal_to_one(self):
        source = source_identity(self.root)
        self._write('hmm/hmm.so', 'x')
        write_build_identity(self.root, source)
        with open(os.path.join(self.root, 'advntr', '_build_identity.json')) as handle:
            identity = json.load(handle)
        next(row for row in identity['files'] if row['path'] == 'hmm/hmm.so')['size_bytes'] = True
        self._write('advntr/_build_identity.json', canonical_bytes(identity))
        with self.assertRaisesRegexp(ValueError, 'integer'):
            describe_capabilities(self.root)

    def test_same_runtime_bytes_with_different_recorded_provenance_get_different_ids(self):
        source = source_identity(self.root)
        write_build_identity(self.root, source)
        unidentified = describe_capabilities(self.root)['build_id']
        source['source_revision'] = 'a' * 40
        write_build_identity(self.root, source)
        self.assertNotEqual(unidentified, describe_capabilities(self.root)['build_id'])

    def test_build_identity_binds_source_attestation_and_actual_compiled_bytes(self):
        revision = self._commit()
        source = source_identity(self.root)
        self._write('hmm/base.so', 'compiled base')
        self._write('hmm/hmm.so', 'compiled decoder')
        write_build_identity(self.root, source)
        first = describe_capabilities(self.root)
        self.assertEqual(revision, first['source_revision'])
        write_build_identity(self.root, source)
        self.assertEqual(first, describe_capabilities(self.root))
        self._write('hmm/hmm.so', 'swapped decoder')
        with self.assertRaisesRegexp(ValueError, 'runtime payload'):
            describe_capabilities(self.root)


if __name__ == '__main__':
    unittest.main()
