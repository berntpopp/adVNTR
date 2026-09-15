"""V2 command assets are preflighted before sinks and closed on every outcome."""
import __builtin__
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

from advntr import advntr_commands
from tests.test_run_context import parse_args


class TestCaptureAssetCommand(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='advntr-v2-command-')
        self.model = os.path.join(self.root, 'model.db')
        connection = sqlite3.connect(self.model)
        connection.execute('CREATE TABLE synthetic (value INTEGER)')
        connection.commit()
        connection.close()
        self.sink = os.path.join(self.root, 'capture.jsonl')
        self.contexts, self.model_paths = [], []
        self.originals = (advntr_commands.load_unique_vntrs_data, advntr_commands.GenomeAnalyzer,
                          advntr_commands.logging.basicConfig)
        owner = self
        class Reference(object):
            id = 25561
        def load(**kwargs):
            owner.model_paths.append(kwargs['db_file'])
            owner.assertTrue(os.path.isfile(kwargs['db_file']))
            owner.assertFalse(os.path.exists(owner.sink))
            return [Reference()]
        class Analyzer(object):
            def __init__(self, *args, **kwargs):
                owner.contexts.append(kwargs['run_context'])
            def find_frameshift_from_alignment_file(self, path):
                owner.assertTrue(os.path.exists(owner.sink))
        advntr_commands.load_unique_vntrs_data = load
        advntr_commands.GenomeAnalyzer = Analyzer
        advntr_commands.logging.basicConfig = lambda **kwargs: None

    def tearDown(self):
        (advntr_commands.load_unique_vntrs_data, advntr_commands.GenomeAnalyzer,
         advntr_commands.logging.basicConfig) = self.originals
        shutil.rmtree(self.root)

    def run_command(self, extra=None):
        args, parser = parse_args(['-a', 'synthetic.bam', '-fs', '-vid', '25561', '-m', self.model,
                                   '--working_directory', self.root, '--frameshift-capture-version', '2',
                                   '--frameshift-calibration-out', self.sink] + (extra or []))
        advntr_commands.genotype(args, parser)

    def test_query_uses_verified_snapshot_and_closes_after_success(self):
        self.run_command()
        context = self.contexts[0]
        self.assertEqual(2, context.capture_version)
        self.assertIsNotNone(context.capture_assets)
        self.assertNotEqual(self.model, context.model_path)
        self.assertEqual([context.model_path], self.model_paths)
        self.assertFalse(os.path.exists(context.model_path))
        self.assertTrue(os.path.exists(self.model))

    def test_bad_model_or_missing_exact_background_creates_no_sink(self):
        with self.assertRaises(SystemExit):
            self.run_command(['--exact-frameshift-caller'])
        self.assertFalse(os.path.exists(self.sink))
        with open(self.model, 'wb') as handle:
            handle.write('not a database')
        with self.assertRaises(SystemExit):
            self.run_command()
        self.assertFalse(os.path.exists(self.sink))
        self.assertEqual([], self.contexts)

    def test_analyzer_failure_still_closes_snapshots(self):
        owner = self
        class Failing(object):
            def __init__(self, *args, **kwargs):
                owner.contexts.append(kwargs['run_context'])
            def find_frameshift_from_alignment_file(self, path):
                raise RuntimeError('synthetic analysis failure')
        advntr_commands.GenomeAnalyzer = Failing
        with self.assertRaises(RuntimeError):
            self.run_command()
        self.assertFalse(os.path.exists(self.contexts[0].model_path))

    def test_v2_output_redirect_is_restored_and_closed_on_success_and_failure(self):
        original_stdout = sys.stdout
        output = os.path.join(self.root, 'genotype output.txt')
        second_output = os.path.join(self.root, 'failed genotype output.txt')
        redirected = []

        def tracked_open(path, *args, **kwargs):
            handle = __builtin__.open(path, *args, **kwargs)
            if path in (output, second_output):
                redirected.append(handle)
            return handle

        advntr_commands.open = tracked_open
        try:
            self.run_command(['--outfile', output])
            self.assertIs(original_stdout, sys.stdout)

            owner = self
            class Failing(object):
                def __init__(self, *args, **kwargs):
                    owner.contexts.append(kwargs['run_context'])
                def find_frameshift_from_alignment_file(self, path):
                    raise RuntimeError('synthetic analysis failure')
            advntr_commands.GenomeAnalyzer = Failing
            self.sink = os.path.join(self.root, 'failed capture.jsonl')
            with self.assertRaises(RuntimeError):
                self.run_command(['--outfile', second_output])
            self.assertIs(original_stdout, sys.stdout)
        finally:
            del advntr_commands.open

        self.assertEqual(2, len(redirected))
        self.assertTrue(all(handle.closed for handle in redirected))

    def test_missing_target_and_source_sink_collision_fail_before_writes(self):
        advntr_commands.load_unique_vntrs_data = lambda **kwargs: []
        with self.assertRaises(SystemExit):
            self.run_command()
        self.assertFalse(os.path.exists(self.sink))
        self.sink = self.model
        with open(self.model, 'rb') as handle:
            original = handle.read()
        with self.assertRaises(SystemExit):
            self.run_command()
        with open(self.model, 'rb') as handle:
            self.assertEqual(original, handle.read())

    def test_existing_v2_sink_refuses_before_log_or_model_access(self):
        with open(self.sink, 'wb') as handle:
            handle.write('prior evidence')
        log = os.path.join(self.root, 'log_synthetic.bam.log')
        with open(log, 'wb') as handle:
            handle.write('prior log')
        with self.assertRaises(SystemExit):
            self.run_command()
        self.assertEqual([], self.model_paths)
        with open(log, 'rb') as handle:
            self.assertEqual('prior log', handle.read())
        with open(self.sink, 'rb') as handle:
            self.assertEqual('prior evidence', handle.read())


if __name__ == '__main__':
    unittest.main()
