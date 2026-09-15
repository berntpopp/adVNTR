"""Invented assets bind the bytes actually queried, never a mutable source path."""
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import unittest


def _module():
    from advntr import capture_assets
    return capture_assets


class TestCaptureAssets(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='advntr-assets-test-')
        self.model = os.path.join(self.root, 'model.db')
        connection = sqlite3.connect(self.model)
        connection.execute('CREATE TABLE invented (value INTEGER)')
        connection.execute('INSERT INTO invented VALUES (17)')
        connection.commit()
        connection.close()
        self.background = os.path.join(self.root, 'background.json')
        with open(self.background, 'w') as handle:
            json.dump({'schema': 'advntr.frameshift.background', 'version': 1,
                       'provenance': 'SYNTHETIC ONLY', 'default_probability': 0.1,
                       'states': {'D3_1': 0.2}}, handle)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_model_snapshot_is_private_bound_and_independent_of_source_edits(self):
        module = _module()
        with open(self.model, 'rb') as handle:
            expected = hashlib.sha256(handle.read()).hexdigest()
        with module.prepare_capture_assets(self.model) as assets:
            snapshot = assets.model_path
            self.assertNotEqual(self.model, snapshot)
            self.assertEqual(0700, os.stat(os.path.dirname(snapshot)).st_mode & 0777)
            self.assertEqual(0400, os.stat(snapshot).st_mode & 0777)
            document = assets.document(None)
            self.assertEqual(expected, document['assets']['model_sha256'])
            self.assertIsNone(document['assets']['background_sha256'])
            self.assertIsNone(document['assets']['loaded_background_sha256'])
            self.assertEqual(set(('producer', 'assets')), set(document))
            self.assertNotIn(self.root, json.dumps(document))
            connection = sqlite3.connect(snapshot)
            self.assertEqual([(17,)], connection.execute('SELECT value FROM invented').fetchall())
            connection.close()
            with open(self.model, 'wb') as handle:
                handle.write('source replaced after snapshot')
            self.assertEqual(document, assets.document(None))
        self.assertFalse(os.path.exists(snapshot))
        with self.assertRaises(ValueError):
            assets.document(None)
        assets.close()

    def test_loaded_background_is_snapshot_bound_and_rechecked_before_completion(self):
        module = _module()
        with module.prepare_capture_assets(self.model, self.background) as assets:
            document = assets.document(assets.background)
            self.assertEqual(0.2, assets.background.probability_for('D3_1'))
            self.assertEqual(64, len(document['assets']['loaded_background_sha256']))
            semantic = assets.loaded_background_document()
            self.assertNotIn('provenance', semantic)
            self.assertNotIn('path', semantic)
            self.assertEqual(document['assets']['loaded_background_sha256'],
                             hashlib.sha256(module.canonical_bytes(semantic)).hexdigest())
            semantic['states']['D3_1'] = 0.9
            self.assertEqual(0.2, assets.loaded_background_document()['states']['D3_1'])
            with open(self.background, 'wb') as handle:
                handle.write('changed original')
            self.assertEqual(document, assets.document(assets.background))
            assets.background.states['D3_1'] = 0.3
            with self.assertRaises(ValueError):
                assets.document(assets.background)

    def test_foreign_background_and_mutated_snapshot_are_refused(self):
        module = _module()
        with module.prepare_capture_assets(self.model, self.background) as assets:
            with self.assertRaises(ValueError):
                assets.document(None)
            os.chmod(assets.model_path, 0600)
            with open(assets.model_path, 'ab') as handle:
                handle.write('changed')
            with self.assertRaises(ValueError):
                assets.document(assets.background)

    def test_runtime_identity_drift_is_refused(self):
        module = _module()
        with module.prepare_capture_assets(self.model) as assets:
            original = module.describe_capabilities
            changed = original()
            changed['build_id'] = '0' * 64
            module.describe_capabilities = lambda: changed
            try:
                with self.assertRaises(ValueError):
                    assets.document(None)
            finally:
                module.describe_capabilities = original

    def test_active_sqlite_sidecars_and_wal_format_are_refused(self):
        module = _module()
        for suffix in ('-wal', '-shm', '-journal'):
            sidecar = self.model + suffix
            with open(sidecar, 'wb') as handle:
                handle.write('active')
            try:
                with self.assertRaises(ValueError):
                    module.prepare_capture_assets(self.model)
            finally:
                os.unlink(sidecar)
        with open(self.model, 'r+b') as handle:
            handle.seek(18)
            handle.write('\x02\x02')
        with self.assertRaises(ValueError):
            module.prepare_capture_assets(self.model)

    def test_symlinks_non_sqlite_and_duplicate_background_keys_refused(self):
        module = _module()
        link = os.path.join(self.root, 'linked.db')
        os.symlink(self.model, link)
        with self.assertRaises(ValueError):
            module.prepare_capture_assets(link)
        with self.assertRaises(ValueError):
            module.prepare_capture_assets(self.background)
        with open(self.background, 'wb') as handle:
            handle.write('{"schema":"advntr.frameshift.background","version":1,"version":1}')
        with self.assertRaises(ValueError):
            module.prepare_capture_assets(self.model, self.background)



    def test_source_change_during_copy_is_not_accepted(self):
        module = _module()
        original = module.os.read
        changed = []
        def read_and_change(descriptor, count):
            data = original(descriptor, count)
            if data and not changed:
                changed.append(True)
                with open(self.model, 'r+b') as handle:
                    handle.seek(100)
                    handle.write('CHANGED')
            return data
        module.os.read = read_and_change
        try:
            with self.assertRaises(ValueError):
                module.prepare_capture_assets(self.model)
        finally:
            module.os.read = original

    def test_native_background_method_cannot_be_replaced(self):
        module = _module()
        with module.prepare_capture_assets(self.model, self.background) as assets:
            assets.background.probability_for = lambda state: 0.9
            with self.assertRaises(ValueError):
                assets.document(assets.background)

    def test_invalid_paths_and_nonregular_files_fail_cleanly(self):
        module = _module()
        for path in (None, '', [], self.root, os.path.join(self.root, 'absent.db')):
            with self.assertRaises(ValueError):
                module.prepare_capture_assets(path)

    def test_legacy_snapshot_has_no_semantic_background(self):
        with _module().prepare_capture_assets(self.model) as assets:
            self.assertIsNone(assets.loaded_background_document())


if __name__ == '__main__':
    unittest.main()
