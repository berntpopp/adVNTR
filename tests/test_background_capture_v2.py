"""Invented completed captures retain zero-event trials and native replay gates."""
import copy
import json
import os
import shutil
import tempfile
import unittest

from tests.test_frameshift_capture_record import capture_document


class TestBackgroundCaptureV2(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='advntr-fit-v2-')
        self.path = os.path.join(self.root, 'capture.jsonl')

    def tearDown(self):
        shutil.rmtree(self.root)

    def write(self, documents):
        with open(self.path, 'w') as handle:
            for document in documents:
                handle.write(json.dumps(document) + '\n')

    def test_projection_keeps_occurrence_union_and_unobserved_denominators(self):
        from advntr.background_capture_v2 import load_fit_capture
        self.write([capture_document()])
        capture = load_fit_capture('invented-a', self.path)
        self.assertEqual((1, 3), capture.evidence('D2_1'))
        self.assertEqual((0, 3), capture.evidence('D3_1'))
        self.assertEqual([], capture.round_trip_failures())
        self.assertEqual([], capture.shipped_aggregation_disagreements())
        self.assertEqual(['D2_1'], capture.completed_decisions['called'])
        self.assertEqual({'D2_1': 4}, capture.completed_decisions['tested'])

    def test_completed_record_validation_precedes_projection(self):
        from advntr.background_capture_v2 import load_fit_capture
        from advntr.background_estimator import FitterError
        raw = capture_document()
        raw['decision_visits'] = []
        self.write([raw])
        with self.assertRaises(FitterError):
            load_fit_capture('invented-a', self.path)

    def test_special_files_and_ambiguous_documents_are_refused(self):
        from advntr.background_capture_v2 import load_fit_capture
        from advntr.background_estimator import FitterError
        for contents in ('', '\n', '[]\n', '{"a":1,"a":2}\n', '{"a":NaN}\n',
                         '{"schema_version":"unsupported"}\n'):
            with open(self.path, 'wb') as handle:
                handle.write(contents)
            with self.assertRaises(FitterError):
                load_fit_capture('invented-a', self.path)
        os.unlink(self.path)
        os.mkfifo(self.path)
        with self.assertRaises(FitterError):
            load_fit_capture('invented-a', self.path)
        os.unlink(self.path)
        os.symlink('missing', self.path)
        with self.assertRaises(OSError):
            load_fit_capture('invented-a', self.path)
        os.unlink(self.path)
        self.write([capture_document(), capture_document()])
        with self.assertRaises(FitterError):
            load_fit_capture('invented-a', self.path)
        self.write([capture_document()])
        with open(self.path, 'rb+') as handle:
            handle.seek(-1, 2)
            handle.truncate()
        with self.assertRaises(FitterError):
            load_fit_capture('invented-a', self.path)

    def test_fit_identity_changes_with_capture_policy_model_and_build(self):
        from advntr.background_capture_v2 import load_fit_capture, require_fit_identity
        from advntr.background_estimator import FitterError
        raw = capture_document()
        self.write([raw])
        first = load_fit_capture('invented-a', self.path)
        for key in ('model_sha256',):
            changed = copy.deepcopy(raw)
            changed['assets'][key] = 'c' * 64
            self.write([changed])
            with self.assertRaises(FitterError):
                require_fit_identity(first, load_fit_capture('invented-b', self.path))
        changed = copy.deepcopy(raw)
        changed['producer']['build_id'] = 'c' * 64
        self.write([changed])
        with self.assertRaises(FitterError):
            require_fit_identity(first, load_fit_capture('invented-b', self.path))
        self.write([raw])
        require_fit_identity(first, load_fit_capture('invented-b', self.path))

    def test_legacy_projection_uses_read_bytes_even_if_path_is_replaced(self):
        from advntr import background_capture_v2 as adapter
        from tests.test_background_installed import _capture_document
        self.write([_capture_document(10)])
        original = adapter._pairs
        owner = self
        def swapped(pairs):
            owner.write([_capture_document(20)])
            return original(pairs)
        adapter._pairs = swapped
        try:
            capture = adapter.load_fit_capture('invented-a', self.path)
        finally:
            adapter._pairs = original
        self.assertEqual((0, 10), capture.evidence('D1_1'))

    def test_v2_replay_uses_requested_support_and_cutoff(self):
        from advntr.background_capture_v2 import load_fit_capture, replay_fit_capture
        from advntr.background_validation import _StaticModel
        self.write([capture_document()])
        capture = load_fit_capture('invented-a', self.path)
        model = _StaticModel({}, 0.00001)
        loose = {'schema_version': 'advntr-frameshift-policy-v1', 'mode': 'exact',
                 'cutoff': 0.001, 'minimum_read_support': 4}
        self.assertTrue(replay_fit_capture(capture, model, loose)['called'])
        strict = dict(loose, minimum_read_support=5)
        result = replay_fit_capture(capture, model, strict)
        self.assertFalse(result['called'])
        self.assertEqual('insufficient-read-support', result['details'][0]['disposition'])
        strict = dict(loose, cutoff=0.000001)
        self.assertFalse(replay_fit_capture(capture, model, strict)['called'])

    def test_full_fitter_uses_completed_receipts_and_declared_policy_without_logs(self):
        from advntr.background_fit_command import parse_args, run
        from advntr.capabilities import describe_capabilities
        described = describe_capabilities()
        producer = dict((key, described[key]) for key in ('package_version', 'build_id', 'source_revision'))
        records = []
        for name, truth, group in (('control-a', False, 'group-a'), ('carrier-a', True, 'group-a'),
                                   ('control-b', False, 'group-b'), ('carrier-b', True, 'group-b')):
            output = os.path.join(self.root, 'runs', name, 'output')
            os.makedirs(output)
            with open(os.path.join(output, 'calibration.jsonl'), 'w') as handle:
                handle.write(json.dumps(dict(capture_document(), producer=producer)) + '\n')
            records.append({'sample_id': name, 'truth': truth, 'pair_id': group, 'partition': 'training',
                            'variant_class': 'invented' if truth else 'negative', 'array_length': 30})
        labels = os.path.join(self.root, 'labels.json')
        with open(labels, 'w') as handle:
            json.dump({'samples': records}, handle)
        policy_path = os.path.join(self.root, 'policy.json')
        policy = {'schema_version': 'advntr-frameshift-policy-v1', 'mode': 'exact',
                  'cutoff': 0.002, 'minimum_read_support': 5}
        with open(policy_path, 'w') as handle:
            json.dump(policy, handle)
        output = os.path.join(self.root, 'fit')
        args = parse_args(['--capture-root', self.root, '--labels', labels, '--partition', 'training',
                           '--out-dir', output, '--profile', 'invented', '--folds', '2',
                           '--diagnostic-policy', policy_path, '--insert-lengths', '1'])
        self.assertEqual(0, run(args))
        with open(os.path.join(output, 'invented.sidecar.json')) as handle:
            sidecar = json.load(handle)
        self.assertEqual('recipe-v1', sidecar['background_recipe_id'])
        self.assertEqual(policy, sidecar['diagnostic_policy'])
        self.assertEqual(0.001, sidecar['preregistered_hyperparameters']['floor_target'])
        self.assertEqual(producer, sidecar['capture_identity']['producer'])
        self.assertEqual(2, sidecar['independent_control_groups'])
        with open(os.path.join(output, 'invented.cv.json')) as handle:
            cv = json.load(handle)
        self.assertEqual(2, cv['fold_count'])
        self.assertTrue(all(fold['diagnostic_policy'] == policy for fold in cv['folds']))
        self.assertTrue(all(not call for fold in cv['folds'] for call in fold['calls'].values()))
        from advntr.background_fit_command import ingest
        from advntr.background_estimator import FitterError
        duplicated = copy.deepcopy(records)
        duplicated[2]['pair_id'] = duplicated[0]['pair_id']
        with self.assertRaises(FitterError) as caught:
            ingest(args, duplicated)
        self.assertIn('one primary negative', str(caught.exception))
        path = os.path.join(self.root, 'runs', 'control-a', 'output', 'calibration.jsonl')
        with open(path, 'w') as handle:
            handle.write(json.dumps(capture_document()) + '\n')
        with self.assertRaises(FitterError) as caught:
            ingest(args, records)
        self.assertIn('installed fitting producer', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
