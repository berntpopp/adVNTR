"""Invented command inputs prove policies and assets cannot leak between runs."""
import importlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest

from advntr import advntr_commands, settings
from advntr.capture_policy import resolve_capture_policy
from advntr.frameshift_decisions import resolve_policy
from advntr.frameshift_background import BackgroundModel


def parse_args(extra):
    cli = importlib.import_module('advntr.__main__')
    old_argv, old_genotype = sys.argv, cli.genotype
    captured = []
    try:
        sys.argv = ['advntr', 'genotype'] + extra
        cli.genotype = lambda args, parser: captured.append((args, parser))
        cli.main()
    finally:
        sys.argv, cli.genotype = old_argv, old_genotype
    return captured[0]


class TestRunContext(unittest.TestCase):
    def test_closed_validated_context_and_pickle(self):
        from advntr.run_context import RunContext, validate_context
        policy = resolve_capture_policy(frameshift_mode=True)
        context = RunContext(policy, resolve_policy(), None, None, 'invented.db')
        self.assertIs(context, validate_context(context))
        self.assertEqual(0.60, context.effective_match_ratio)
        for protocol in (0, 1, 2):
            self.assertEqual(context, pickle.loads(pickle.dumps(context, protocol)))
        with self.assertRaises(AttributeError):
            context.model_path = 'other.db'
        for values in ((None, resolve_policy(), None, None, 'x'),
                       (policy, None, None, None, 'x'),
                       (policy, resolve_policy(), object(), None, 'x'),
                       (policy, resolve_policy(), None, 3, 'x'),
                       (policy, resolve_policy(), None, None, '')):
            with self.assertRaises(ValueError):
                RunContext(*values)
        exact = policy._replace(caller_mode='exact')
        with self.assertRaises(ValueError):
            RunContext(exact, resolve_policy(), None, None, 'x')
        background = BackgroundModel(1, 'invented', 0.1, {}, 'invented.json')
        self.assertIs(background, RunContext(exact, resolve_policy(), background, None, 'x').background)
        with self.assertRaises(ValueError):
            RunContext(policy, resolve_policy(), background, None, 'x')
        forged = tuple.__new__(RunContext, (None, resolve_policy(), None, None, 'x'))
        for protocol in (0, 1, 2):
            with self.assertRaises(ValueError):
                pickle.loads(pickle.dumps(forged, protocol))
        with self.assertRaises(ValueError):
            context._replace(model_path='')

    def test_explicit_context_ignores_ambient_values_at_every_access(self):
        from advntr.run_context import RunContext, runtime_value
        class Owner(object):
            pass
        owner = Owner()
        owner.run_context = RunContext(resolve_capture_policy(), resolve_policy(), None, None, 'x')
        original = settings.MAX_ERROR_RATE
        try:
            settings.MAX_ERROR_RATE = 0.9
            self.assertEqual(0.05, runtime_value(owner, 'maximum_error_rate'))
            owner.run_context = None
            self.assertEqual(0.9, runtime_value(owner, 'maximum_error_rate'))
        finally:
            settings.MAX_ERROR_RATE = original


class TestConsecutiveCommands(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='advntr-context-test-')
        self.old_loader = advntr_commands.load_unique_vntrs_data
        self.old_analyzer = advntr_commands.GenomeAnalyzer
        self.old_logging = advntr_commands.logging.basicConfig
        self.contexts, self.databases, self.dispatches = [], [], []
        contexts, databases, dispatches = self.contexts, self.databases, self.dispatches
        def loader(**kwargs):
            databases.append(kwargs)
            return []
        class Analyzer(object):
            def __init__(self, *args, **kwargs):
                contexts.append(kwargs.get('run_context'))
            def find_frameshift_from_alignment_file(self, path):
                dispatches.append(('frameshift', path))
            def find_repeat_counts_from_pacbio_alignment_file(self, path):
                dispatches.append(('pacbio', path))
            def find_repeat_counts_from_alignment_file(self, *args):
                dispatches.append(('repeat', args[0]))
        advntr_commands.load_unique_vntrs_data = loader
        advntr_commands.GenomeAnalyzer = Analyzer
        advntr_commands.logging.basicConfig = lambda **kwargs: None
        self.before = dict((name, getattr(settings, name)) for name in (
            'MAX_ERROR_RATE', 'CORES', 'MIN_READ_LENGTH', 'PRUNE_REVERSE_DECODE',
            'EXACT_FRAMESHIFT_CALLER', 'FRAMESHIFT_BACKGROUND_FILE',
            'MIN_RELATIVE_RU_COVERAGE', 'FILTER_ADAPTER_READTHROUGH',
            'MIN_READ_MATCH_RATIO', 'FRAMESHIFT_CALIBRATION_OUT',
            'TRAINED_MODELS_DB', 'TRAINED_HMMS_DIR', 'USE_REF_ALIGNMENT',
            'USE_ONLY_FULLY_COVERED_RU'))

    def tearDown(self):
        advntr_commands.load_unique_vntrs_data = self.old_loader
        advntr_commands.GenomeAnalyzer = self.old_analyzer
        advntr_commands.logging.basicConfig = self.old_logging
        for name, value in self.before.items():
            setattr(settings, name, value)
        shutil.rmtree(self.directory)

    def run_command(self, extra):
        args, parser = parse_args(['-a', 'invented.bam', '--working_directory', self.directory] + extra)
        advntr_commands.genotype(args, parser)

    def test_conflicting_commands_resolve_fresh_without_global_writes(self):
        self.run_command(['-n', '-m', 'first.db', '--noref_aln', '--fullru',
                          '--min_read_length', '80', '--rare-unit-coverage-guard', '0.2',
                          '--filter-adapter-readthrough', '--min-read-match-ratio', '0.8',
                          '--prune-reverse', '-t', '2', '--frameshift-pvalue-cutoff', '0.02',
                          '--min-frameshift-read-support', '5'])
        self.run_command(['-fs', '-m', 'second.db', '--filter-adapter-readthrough'])
        self.assertIsNotNone(self.contexts[0])
        first, second = self.contexts
        self.assertEqual((0.3, 0.05), (first.capture.maximum_error_rate, second.capture.maximum_error_rate))
        self.assertEqual((False, True), (first.capture.use_reference_alignment, second.capture.use_reference_alignment))
        self.assertEqual((True, False), (first.capture.fully_covered_ru_only, second.capture.fully_covered_ru_only))
        self.assertEqual((80, None), (first.capture.minimum_read_length, second.capture.minimum_read_length))
        self.assertEqual((0.2, None), (first.capture.minimum_relative_ru_coverage,
                                    second.capture.minimum_relative_ru_coverage))
        self.assertEqual((0.8, 0.60), (first.effective_match_ratio, second.effective_match_ratio))
        self.assertEqual((2, 1), (first.capture.threads, second.capture.threads))
        self.assertEqual((True, False), (first.capture.prune_reverse, second.capture.prune_reverse))
        self.assertEqual((0.02, 0.001), (first.frameshift.cutoff, second.frameshift.cutoff))
        self.assertEqual((5, 3), (first.frameshift.minimum_read_support, second.frameshift.minimum_read_support))
        self.assertEqual(['first.db', 'second.db'], [row['db_file'] for row in self.databases])
        self.assertEqual(['first.db', 'second.db'], [item.model_path for item in self.contexts])
        self.assertEqual(self.before, dict((name, getattr(settings, name)) for name in self.before))

    def test_background_loaded_once_per_command_and_sink_is_run_local(self):
        from tests.test_exact_caller import SYNTHETIC_BACKGROUND
        path = os.path.join(self.directory, 'background.json')
        sink = os.path.join(self.directory, 'capture.jsonl')
        loader = advntr_commands.frameshift_background.load_background_model
        loaded = []
        def recording_loader(filename):
            result = loader(filename)
            loaded.append(result)
            return result
        advntr_commands.frameshift_background.load_background_model = recording_loader
        try:
            for probability in (0.1, 0.2):
                with open(path, 'w') as handle:
                    json.dump(dict(SYNTHETIC_BACKGROUND, default_probability=probability), handle)
                self.run_command(['-fs', '-m', 'invented.db', '--exact-frameshift-caller',
                                  '--frameshift-background', path, '--frameshift-calibration-out', sink])
            self.run_command(['-fs', '-m', 'invented.db'])
        finally:
            advntr_commands.frameshift_background.load_background_model = loader
        self.assertEqual(2, len(loaded))
        self.assertIs(loaded[0], self.contexts[0].background)
        self.assertIs(loaded[1], self.contexts[1].background)
        self.assertEqual((0.1, 0.2), tuple(item.default_probability for item in loaded))
        self.assertIsNone(self.contexts[2].background)
        self.assertEqual([sink, sink, None], [item.capture_path for item in self.contexts])
        self.assertEqual(self.before, dict((name, getattr(settings, name)) for name in self.before))

    def test_invalid_mode_fails_before_model_or_sink_io(self):
        sink = os.path.join(self.directory, 'not-created.jsonl')
        with self.assertRaises(SystemExit):
            self.run_command(['--exact-frameshift-caller', '--frameshift-calibration-out', sink])
        self.assertEqual([], self.databases)
        self.assertFalse(os.path.exists(sink))
        for flag in ('USE_TRAINED_HMMS', 'USE_ENHANCED_HMM'):
            old = getattr(settings, flag)
            try:
                setattr(settings, flag, flag == 'USE_TRAINED_HMMS')
                with self.assertRaises(SystemExit):
                    self.run_command(['-fs', '--frameshift-calibration-out', sink])
                self.assertFalse(os.path.exists(sink))
            finally:
                setattr(settings, flag, old)
        with self.assertRaises(SystemExit):
            self.run_command(['-fs', '-n', '-p'])
        for platform in ('-p', '-n'):
            with self.assertRaises(SystemExit):
                self.run_command(['-fs', platform, '--frameshift-calibration-out', sink])
            self.assertFalse(os.path.exists(sink))
        self.assertEqual([], self.databases)
        self.assertEqual([], self.dispatches)
        self.run_command(['-p', '-m', 'invented.db'])
        self.run_command(['-n', '-m', 'invented.db'])
        self.assertEqual(['pacbio', 'repeat'], [kind for kind, path in self.dispatches])


class TestStrictCalibrationArguments(unittest.TestCase):
    def test_abbreviations_and_duplicates_cannot_redefine_policy(self):
        for extra in (['--frameshift-pvalue-c', '0.1'],
                      ['--frameshift-pvalue-cutoff', '0.1', '--frameshift-pvalue-cutoff=0.2'],
                      ['--fullru', '--fullru'], ['-t', '1', '--threads', '2'],
                      ['--filter-adapter-readthrough', '--filter-adapter-readthrough']):
            with self.assertRaises(SystemExit):
                parse_args(extra)
        args, _parser = parse_args(['--min-read-match-ratio=0.6', '--rare-unit-coverage-guard'])
        self.assertEqual(0.6, args.min_read_match_ratio)
        self.assertEqual(0.15, args.rare_unit_coverage_guard)


class TestCaptureVersion(unittest.TestCase):
    def test_default_v1_and_exact_version_option(self):
        args, _ = parse_args([])
        self.assertEqual(1, args.frameshift_capture_version)
        args, _ = parse_args(['--frameshift-capture-version', '2'])
        self.assertEqual(2, args.frameshift_capture_version)
        for extra in (['--frameshift-capture-v', '2'], ['--frameshift-capture-version', '3'],
                      ['--frameshift-capture-version', '1', '--frameshift-capture-version=2']):
            with self.assertRaises(SystemExit):
                parse_args(extra)

    def test_v2_requires_explicit_frameshift_sink_and_run_assets(self):
        from advntr.run_context import RunContext, command_policies
        for extra in (['--frameshift-capture-version', '2'],
                      ['-fs', '--frameshift-capture-version', '2'],
                      ['--frameshift-calibration-out', 'x', '--frameshift-capture-version', '2'],
                      ['-fs', '--frameshift-calibration-out', 'x', '--frameshift-capture-version', '2', '--append']):
            args, _ = parse_args(extra)
            with self.assertRaises(ValueError):
                command_policies(args)
        policy = resolve_capture_policy(frameshift_mode=True)
        for version in (True, 0, 3, '2'):
            with self.assertRaises(ValueError):
                RunContext(policy, resolve_policy(), None, 'capture', 'model', capture_version=version)
        with self.assertRaises(ValueError):
            RunContext(policy, resolve_policy(), None, 'capture', 'model', capture_version=2)
