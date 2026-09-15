"""Tests for the strict installed frameshift replay command."""
import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from advntr import frameshift_replay_command as command
from advntr.capture_policy import policy_document, resolve_capture_policy
from advntr.frameshift_background import BackgroundModel
from tests.test_frameshift_capture_record import capture_document as complete_capture_document
from tests.test_frameshift_replay_policy import replay_policy as complete_replay_policy


def _bytes(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'


class TestFrameshiftReplayCommand(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='advntr replay command ')
        self.capture_root = os.path.join(self.tmp, 'capture root')
        os.mkdir(self.capture_root)
        self.capture = {
            'schema_version': 'advntr-frameshift-capture-v2',
            'producer': command._installed_producer(),
            'locus': {'vntr_id': 25561},
            'rows': [],
        }
        capture_bytes = self._write_capture(self.capture)
        self.policy = os.path.join(self.tmp, 'policy.json')
        with open(self.policy, 'wb') as handle:
            handle.write(_bytes({
                'schema_version': 'advntr-frameshift-replay-policy-v1',
                'capture_policy': policy_document(resolve_capture_policy(frameshift_mode=True)),
                'caller_policy': {
                    'schema_version': 'advntr-frameshift-policy-v1',
                    'mode': 'legacy', 'cutoff': 0.001, 'minimum_read_support': 3,
                },
            }))
        self.manifest = os.path.join(self.tmp, 'manifest.json')
        with open(self.manifest, 'wb') as handle:
            handle.write(_bytes({
                'schema_version': 'advntr-frameshift-replay-manifest-v1',
                'captures': [{
                    'key': 'synthetic-a', 'filename': 'capture one.json',
                    'sha256': hashlib.sha256(capture_bytes).hexdigest(),
                    'vntr_ids': [25561],
                }],
            }))
        self.output = os.path.join(self.tmp, 'output result')

    def _write_capture(self, document):
        content = _bytes(document)
        with open(os.path.join(self.capture_root, 'capture one.json'), 'wb') as handle:
            handle.write(content)
        if hasattr(self, 'manifest'):
            with open(self.manifest, 'rb') as handle:
                manifest = json.loads(handle.read())
            manifest['captures'][0]['sha256'] = hashlib.sha256(content).hexdigest()
            manifest['captures'][0]['vntr_ids'] = [document['locus']['vntr_id']]
            with open(self.manifest, 'wb') as handle:
                handle.write(_bytes(manifest))
        return content

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _args(self, extra=None):
        values = ['--capture-root', self.capture_root, '--manifest', self.manifest,
                  '--policy', self.policy, '--output', self.output]
        values.extend(extra or [])
        return command.parse_args(values)

    def test_replays_exact_manifest_roster_and_atomically_publishes_private_output(self):
        calls = []

        def evaluator(capture, policy, background=None):
            calls.append((capture, policy, background))
            return {'schema_version': 'synthetic-result-v1', 'vntr_id': 25561, 'calls': [], 'visits': []}

        self.assertEqual(0, command.run(self._args(), evaluator=evaluator))

        self.assertEqual(1, len(calls))
        self.assertEqual(self.capture, calls[0][0])
        self.assertIsNone(calls[0][2])
        with open(os.path.join(self.output, 'replay.json'), 'rb') as handle:
            result = json.loads(handle.read())
        self.assertEqual(['synthetic-a'], [row['key'] for row in result['results']])
        with open(self.manifest, 'rb') as handle:
            self.assertEqual(hashlib.sha256(handle.read()).hexdigest(), result['manifest_file_sha256'])
        with open(self.policy, 'rb') as handle:
            self.assertEqual(hashlib.sha256(handle.read()).hexdigest(), result['policy_file_sha256'])
        self.assertEqual(0o700, os.stat(self.output).st_mode & 0o777)
        self.assertEqual(0o600, os.stat(os.path.join(self.output, 'replay.json')).st_mode & 0o777)

    def test_existing_output_fails_before_any_input_or_evaluator_access(self):
        os.mkdir(self.output)
        opened = []

        def reader(*args, **kwargs):
            opened.append(args)
            raise AssertionError('input opened')

        original = command._read_regular
        command._read_regular = reader
        try:
            with self.assertRaisesRegexp(ValueError, 'already exists'):
                command.run(self._args(), evaluator=lambda *_args, **_kwargs: {})
        finally:
            command._read_regular = original
        self.assertEqual([], opened)

    def test_evaluator_failure_cleans_staging_and_publishes_nothing(self):
        def fail(*_args, **_kwargs):
            raise ValueError('synthetic evaluator failure')

        with self.assertRaisesRegexp(ValueError, 'synthetic evaluator failure'):
            command.run(self._args(), evaluator=fail)
        self.assertFalse(os.path.lexists(self.output))
        self.assertEqual([], [name for name in os.listdir(self.tmp) if name.startswith('.output result.')])

    def test_evaluator_result_must_bind_the_capture_target(self):
        with self.assertRaisesRegexp(ValueError, 'VNTR identifier'):
            command.run(
                self._args(),
                evaluator=lambda *_args, **_kwargs: {
                    'schema_version': 'synthetic-result-v1', 'vntr_id': 25562,
                    'calls': [], 'visits': [],
                })
        self.assertFalse(os.path.lexists(self.output))

    def test_capture_producer_must_equal_the_installed_replay_producer(self):
        self.capture['producer']['build_id'] = '0' * 64
        self._write_capture(self.capture)
        calls = []

        with self.assertRaisesRegexp(ValueError, 'producer'):
            command.run(self._args(), evaluator=lambda *_args, **_kwargs: calls.append(True))
        self.assertEqual([], calls)
        self.assertFalse(os.path.lexists(self.output))

    def test_actual_packaged_evaluator_replays_a_complete_invented_capture(self):
        capture = complete_capture_document()
        capture['producer'] = command._installed_producer()
        self._write_capture(capture)
        with open(self.policy, 'wb') as handle:
            handle.write(_bytes(complete_replay_policy(support=5)))

        self.assertEqual(0, command.run(self._args()))

        with open(os.path.join(self.output, 'replay.json'), 'rb') as handle:
            result = json.loads(handle.read())
        replay = result['results'][0]['vntrs'][0]['result']
        self.assertTrue(replay['baseline_parity'])
        self.assertEqual([], replay['calls'])
        self.assertEqual(result['replay_producer'], capture['producer'])

    def test_manifest_digest_and_safe_flat_filename_are_enforced(self):
        with open(self.manifest, 'rb') as handle:
            manifest = json.loads(handle.read())
        for filename, message in (('../escape.json', 'basename'), ('capture one.json', 'SHA256')):
            changed = dict(manifest)
            changed['captures'] = [dict(manifest['captures'][0])]
            changed['captures'][0]['filename'] = filename
            if filename == 'capture one.json':
                changed['captures'][0]['sha256'] = '0' * 64
            with open(self.manifest, 'wb') as handle:
                handle.write(_bytes(changed))
            with self.assertRaisesRegexp(ValueError, message):
                command.run(self._args(), evaluator=lambda *_args, **_kwargs: {})

    def test_duplicate_nonfinite_and_unknown_json_are_rejected(self):
        bad_documents = (
            b'{"schema_version":"advntr-frameshift-replay-manifest-v1","captures":[],"captures":[]}\n',
            b'{"schema_version":"advntr-frameshift-replay-manifest-v1","captures":NaN}\n',
            b'{"schema_version":"advntr-frameshift-replay-manifest-v1","captures":[],"extra":1}\n',
        )
        for content in bad_documents:
            with open(self.manifest, 'wb') as handle:
                handle.write(content)
            with self.assertRaises(ValueError):
                command.run(self._args(), evaluator=lambda *_args, **_kwargs: {})

    def test_partial_jsonl_and_wrong_or_duplicate_target_roster_are_rejected(self):
        capture_path = os.path.join(self.capture_root, 'capture one.json')
        cases = (
            (_bytes(self.capture).rstrip(b'\n'), 'partial JSONL'),
            (_bytes(dict(self.capture, locus={'vntr_id': 25562})), 'manifest vntr_ids'),
            (_bytes(self.capture) + _bytes(self.capture), 'manifest vntr_ids'),
        )
        for content, message in cases:
            with open(capture_path, 'wb') as handle:
                handle.write(content)
            with open(self.manifest, 'rb') as handle:
                manifest = json.loads(handle.read())
            manifest['captures'][0]['sha256'] = hashlib.sha256(content).hexdigest()
            with open(self.manifest, 'wb') as handle:
                handle.write(_bytes(manifest))
            with self.assertRaisesRegexp(ValueError, message):
                command.run(self._args(), evaluator=lambda *_args, **_kwargs: {})

    def test_capture_root_inventory_is_exact_and_symlinks_are_rejected(self):
        extra = os.path.join(self.capture_root, 'undeclared.json')
        with open(extra, 'wb') as handle:
            handle.write(b'{}\n')
        with self.assertRaisesRegexp(ValueError, 'inventory'):
            command.run(self._args(), evaluator=lambda *_args, **_kwargs: {})
        os.unlink(extra)
        capture = os.path.join(self.capture_root, 'capture one.json')
        target = os.path.join(self.tmp, 'moved.json')
        os.rename(capture, target)
        os.symlink(target, capture)
        with self.assertRaisesRegexp(ValueError, 'symlink'):
            command.run(self._args(), evaluator=lambda *_args, **_kwargs: {})

    def test_unavailable_atomic_primitive_fails_before_input_reads(self):
        opened = []
        original_rename, original_read = command._RENAME_AT2, command._read_regular
        command._RENAME_AT2 = None

        def reader(*args, **kwargs):
            opened.append(args)
            raise AssertionError('input opened')

        command._read_regular = reader
        try:
            with self.assertRaisesRegexp(RuntimeError, 'renameat2'):
                command.run(self._args(), evaluator=lambda *_args, **_kwargs: {})
        finally:
            command._RENAME_AT2, command._read_regular = original_rename, original_read
        self.assertEqual([], opened)

    def test_activation_race_preserves_the_competing_destination(self):
        original = command._RENAME_AT2

        def racing_rename(_old_dir, _old_name, _new_dir, _new_name, _flags):
            os.mkdir(self.output)
            with open(os.path.join(self.output, 'winner'), 'wb') as handle:
                handle.write(b'other producer\n')
            ctypes.set_errno(errno.EEXIST)
            return -1

        command._RENAME_AT2 = racing_rename
        try:
            with self.assertRaisesRegexp(ValueError, 'already exists'):
                command.run(
                    self._args(),
                    evaluator=lambda *_args, **_kwargs: {
                        'schema_version': 'synthetic-result-v1', 'vntr_id': 25561,
                        'calls': [], 'visits': [],
                    })
        finally:
            command._RENAME_AT2 = original
        with open(os.path.join(self.output, 'winner'), 'rb') as handle:
            self.assertEqual(b'other producer\n', handle.read())
        self.assertEqual([], [name for name in os.listdir(self.tmp) if name.startswith('.output result.')])

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'POSIX FIFO required')
    def test_fifo_input_without_writer_is_rejected_promptly_as_nonregular(self):
        fifo = os.path.join(self.tmp, 'manifest fifo')
        os.mkfifo(fifo)
        args = self._args()
        args.manifest = fifo

        with self.assertRaisesRegexp(ValueError, 'regular file'):
            command.run(args, evaluator=lambda *_args, **_kwargs: {})
        self.assertFalse(os.path.lexists(self.output))

    def test_exact_policy_requires_background_before_capture_reads(self):
        with open(self.policy, 'rb') as handle:
            policy = json.loads(handle.read())
        policy['caller_policy']['mode'] = 'exact'
        policy['capture_policy'] = policy_document(
            resolve_capture_policy(frameshift_mode=True, caller_mode='exact'))
        with open(self.policy, 'wb') as handle:
            handle.write(_bytes(policy))
        opened = []
        original = command._load_capture

        def capture_reader(*args, **kwargs):
            opened.append(args)
            raise AssertionError('capture opened')

        command._load_capture = capture_reader
        try:
            with self.assertRaisesRegexp(ValueError, 'requires --background'):
                command.run(self._args(), evaluator=lambda *_args, **_kwargs: {})
        finally:
            command._load_capture = original
        self.assertEqual([], opened)

    def test_exact_policy_loads_one_stable_background_and_binds_its_bytes(self):
        with open(self.policy, 'rb') as handle:
            policy = json.loads(handle.read())
        policy['caller_policy']['mode'] = 'exact'
        policy['capture_policy'] = policy_document(
            resolve_capture_policy(frameshift_mode=True, caller_mode='exact'))
        with open(self.policy, 'wb') as handle:
            handle.write(_bytes(policy))
        background = os.path.join(self.tmp, 'background.json')
        background_bytes = _bytes({
            'schema': 'advntr.frameshift.background', 'version': 1,
            'provenance': 'SYNTHETIC TEST', 'default_probability': 0.5, 'states': {},
        })
        with open(background, 'wb') as handle:
            handle.write(background_bytes)
        loaded = []

        def evaluator(_capture, _policy, model=None):
            self.assertIsInstance(model, BackgroundModel)
            loaded.append(model)
            return {'schema_version': 'synthetic-result-v1', 'vntr_id': 25561, 'calls': [], 'visits': []}

        self.assertEqual(0, command.run(self._args(['--background', background]), evaluator=evaluator))
        self.assertEqual(1, len(loaded))
        with open(os.path.join(self.output, 'replay.json'), 'rb') as handle:
            result = json.loads(handle.read())
        self.assertEqual(hashlib.sha256(background_bytes).hexdigest(), result['background_file_sha256'])

    def test_argument_validation_rejects_prefixes_duplicates_and_missing_values(self):
        parser = argparse.ArgumentParser()
        command.add_replay_arguments(parser)
        for arguments in (
                ['--capture-r', 'x'],
                ['--policy', 'a', '--policy=b'],
                ['--manifest']):
            with self.assertRaises(SystemExit) as caught:
                command.validate_replay_arguments(parser, arguments)
                parser.parse_args(arguments)
            self.assertEqual(2, caught.exception.code)

    def test_cli_help_is_installed_without_changing_existing_commands(self):
        output = subprocess.check_output([sys.executable, '-m', 'advntr', 'replay-frameshift', '--help'])
        self.assertIn('replay-frameshift', output)
        self.assertIn('--capture-root', output)
        genotype = subprocess.check_output([sys.executable, '-m', 'advntr', 'genotype', '--help'])
        self.assertIn('--frameshift-capture-version', genotype)

    def test_installed_fit_command_runs_its_exact_duplicate_option_precheck(self):
        process = subprocess.Popen(
            [sys.executable, '-m', 'advntr', 'fit-background', '--profile', 'one', '--profile', 'two'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _stdout, stderr = process.communicate()
        self.assertEqual(2, process.returncode)
        self.assertIn('duplicate fit-background option: --profile', stderr)


if __name__ == '__main__':
    unittest.main()
