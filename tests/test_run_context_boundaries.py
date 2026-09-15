"""Legacy adapters and explicit opportunity/quality boundary regressions."""
import os
import shutil
import tempfile
import unittest
from collections import defaultdict

from advntr import finder_hmm, settings, utils
from advntr.calibration_arguments import validate_genotype_arguments
from advntr.frameshift_opportunities import OpportunityCounter
from advntr.run_context import RunContext, validate_context
from tests.test_run_context import parse_args
from tests.test_run_context_production import context, _make_finder


class TestContextBoundaries(unittest.TestCase):
    def test_context_factory_and_invalid_reconstruction_boundaries(self):
        resolved = context()
        self.assertEqual(resolved, RunContext._make(tuple(resolved)))
        with self.assertRaises(ValueError):
            validate_context(None)
        with self.assertRaises(ValueError):
            resolved._replace(unknown=True)
        with self.assertRaises(ValueError):
            RunContext._make(tuple(resolved)[:-1] + ('',))

    def test_unknown_options_and_end_of_options_still_belong_to_argparse(self):
        args, parser = parse_args([])
        self.assertIsNone(validate_genotype_arguments(parser, ['--', '--fullru', '--fullru']))
        with self.assertRaises(SystemExit):
            parse_args(['--unknown-calibration-option'])
        with self.assertRaises(SystemExit):
            parse_args(['--', '--fullru'])

    def test_partial_opportunities_follow_run_policy_in_both_directions(self):
        finder = _make_finder()
        old = settings.USE_ONLY_FULLY_COVERED_RU
        try:
            for full in (False, True, False):
                settings.USE_ONLY_FULLY_COVERED_RU = not full
                finder.run_context = context(fully_covered_ru_only=full)
                counter = OpportunityCounter([['ACGTACGT']], {'1': 1}, {'1': 8}, False, finder)
                counter.observe_read(0, 'invented',
                    ['M3_1', 'M4_1', 'M5_1', 'M6_1', 'M7_1', 'M8_1', 'unit_end_1'],
                    [], {'partial_start': {'M': 6}})
                row = counter.finalise({'D5_1': 0}, {}, defaultdict(int))['D5_1']
                self.assertEqual((0, 0 if full else 1), (row['support'], row['opportunities']))
        finally:
            settings.USE_ONLY_FULLY_COVERED_RU = old

    def test_short_low_quality_runs_preserve_strict_fraction_and_run_rules(self):
        class Read(object):
            mapq = 60
            query_qualities = [30] * 100
        read = Read()
        resolved = context().capture
        for positions, rejected in (([0], False), ([0, 2, 4], False), ([0, 1], True)):
            read.query_qualities = [19 if index in positions else 30 for index in range(100)]
            self.assertEqual(rejected, utils.is_low_quality_read(read, resolved))
            self.assertEqual(rejected, utils.is_low_quality_read(read))
        read.query_qualities = []
        self.assertTrue(utils.is_low_quality_read(read, resolved))


class TestLegacyFinderHMMAdapter(unittest.TestCase):
    def test_unsupported_legacy_backend_and_cache_calls_keep_their_behavior(self):
        finder = _make_finder()
        old = dict((key, getattr(settings, key)) for key in ('USE_ENHANCED_HMM', 'USE_TRAINED_HMMS', 'TRAINED_HMMS_DIR'))
        directory = tempfile.mkdtemp(prefix='advntr-legacy-hmm-test-')
        original_model = finder_hmm.Model
        original_builder = finder_hmm.get_read_matcher_model
        seen = []
        class FakeModel(object):
            def from_json(self, filename):
                seen.append(filename)
                return self
            def to_json(self):
                return 'invented-model'
        try:
            settings.USE_ENHANCED_HMM = False
            finder_hmm.get_read_matcher_model = lambda *args: seen.append(args) or 'legacy-result'
            self.assertEqual('legacy-result', finder.build_vntr_matcher_hmm(2, 4))
            self.assertEqual(2, seen[-1][-1])
            settings.USE_TRAINED_HMMS = True
            settings.TRAINED_HMMS_DIR = directory + '/'
            finder_hmm.Model = FakeModel
            finder.build_vntr_matcher_hmm = lambda *args: FakeModel()
            first = finder.get_vntr_matcher_hmm(8)
            filename = os.path.join(directory, '1_8.json')
            with open(filename) as handle:
                self.assertEqual('invented-model', handle.read())
            second = finder.get_vntr_matcher_hmm(8)
            self.assertIsInstance(first, FakeModel)
            self.assertIsInstance(second, FakeModel)
            self.assertEqual(filename, seen[-1])
            settings.USE_TRAINED_HMMS = False
            self.assertIsInstance(finder.get_vntr_matcher_hmm(8), FakeModel)
            self.assertEqual(filename, seen[-1])
        finally:
            finder_hmm.Model, finder_hmm.get_read_matcher_model = original_model, original_builder
            for key, value in old.items():
                setattr(settings, key, value)
            shutil.rmtree(directory)
