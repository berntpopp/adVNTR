"""Explicit run values reach synthetic HMMs, evidence and quality decisions."""
import unittest
from collections import defaultdict

from advntr import settings, hmm_utils, utils
from advntr import vntr_finder as finder_module
from tests.test_frameshift_context import _vpath
from advntr.capture_policy import resolve_capture_policy
from advntr.frameshift_decisions import resolve_policy
from advntr.frameshift_background import BackgroundModel
from advntr.genome_analyzer import GenomeAnalyzer
from advntr.reference_vntr import ReferenceVNTR
from advntr.run_context import RunContext
from advntr.vntr_finder import VNTRFinder
from tests.test_exact_caller import _all_three_sites, _FakeHMM, UNITS, SEGMENTS
from tests.test_hmm_construction_policy import _fingerprint


def _make_finder():
    reference = ReferenceVNTR(1, UNITS['1'], 100, 'chr1', None, None)
    reference.init_from_xml(SEGMENTS, 'TTTTTTTT', 'GGGGGGGG')
    finder = VNTRFinder(reference, is_frameshift_mode=True)
    finder.hmm = _FakeHMM()
    return finder


def context(**fields):
    fields.setdefault('frameshift_mode', True)
    return RunContext(resolve_capture_policy(**fields), resolve_policy(), None, None, 'invented.db')


class TestRunContextProduction(unittest.TestCase):
    def test_analyzer_and_finder_share_the_same_validated_context(self):
        reference = _make_finder().reference_vntr
        resolved = context(use_reference_alignment=False)
        analyzer = GenomeAnalyzer([reference], [reference.id], is_frameshift_mode=True, run_context=resolved)
        self.assertIs(resolved, analyzer.run_context)
        self.assertIs(resolved, analyzer.vntr_finder[reference.id].run_context)
        self.assertIs(resolved.frameshift, analyzer.frameshift_policy)
        with self.assertRaises(ValueError):
            GenomeAnalyzer([], [], run_context=resolved)
        with self.assertRaises(ValueError):
            VNTRFinder(reference, is_frameshift_mode=True, run_context=resolved,
                       frameshift_policy=resolve_policy(0.2))

    def test_finder_hmm_uses_context_rate_even_after_global_changes(self):
        reference = ReferenceVNTR(1, 'ACG', 0, 'chr1', None, None)
        reference.init_from_xml(['ACG'], 'TGC', 'GCA')
        old_rate, old_trained = settings.MAX_ERROR_RATE, settings.USE_TRAINED_HMMS
        try:
            expected = {}
            for platform in ('illumina', 'nanopore', 'illumina'):
                finder = VNTRFinder(reference, is_frameshift_mode=True, run_context=context(platform=platform))
                settings.MAX_ERROR_RATE = 0.9
                settings.USE_TRAINED_HMMS = True
                actual = _fingerprint(finder.get_vntr_matcher_hmm(3))
                baseline = _fingerprint(hmm_utils.get_read_matcher_model_enhanced(
                    'TGC', 'GCA', ['ACG'], finder.get_copies_for_hmm(3), None, True,
                    maximum_error_rate=0.05 if platform == 'illumina' else 0.3))
                self.assertEqual(baseline, actual)
                if platform in expected:
                    self.assertEqual(expected[platform], actual)
                expected[platform] = actual
            self.assertNotEqual(expected['illumina'], expected['nanopore'])
        finally:
            settings.MAX_ERROR_RATE, settings.USE_TRAINED_HMMS = old_rate, old_trained

    def test_quality_cutoffs_are_explicit_and_keep_original_inequalities(self):
        class Read(object):
            mapq = 10
            query_qualities = [30] * 100
        read = Read()
        self.assertFalse(utils.is_low_quality_read(read, resolve_capture_policy()))
        self.assertTrue(utils.is_low_quality_read(read, resolve_capture_policy(mapq_cutoff=10)))
        self.assertTrue(utils.is_low_quality_read(read, resolve_capture_policy(base_quality_cutoff=31)))
        read.query_qualities = [30] * 90 + [19] * 10
        self.assertTrue(utils.is_low_quality_read(read, resolve_capture_policy()))
        self.assertFalse(utils.is_low_quality_read(read, resolve_capture_policy(base_quality_cutoff=19)))

    def test_full_ru_coverage_helper_uses_explicit_false_and_true(self):
        old = settings.USE_ONLY_FULLY_COVERED_RU
        states = ['M1_1', 'M2_1', 'M1_2', 'M2_2']
        try:
            settings.USE_ONLY_FULLY_COVERED_RU = True
            counts = defaultdict(int)
            hmm_utils.update_number_of_repeat_bp_matches_in_vpath_for_each_hmm(states, counts, 2, 4, False)
            self.assertEqual({'1': 2, '2': 2}, counts)
            settings.USE_ONLY_FULLY_COVERED_RU = False
            counts = defaultdict(int)
            hmm_utils.update_number_of_repeat_bp_matches_in_vpath_for_each_hmm(states, counts, 2, 4, True)
            self.assertEqual({'2': 2}, counts)
        finally:
            settings.USE_ONLY_FULLY_COVERED_RU = old

    def test_read_loop_uses_context_for_background_guards_and_sink(self):
        keys = ('USE_REF_ALIGNMENT', 'USE_ONLY_FULLY_COVERED_RU', 'MIN_RELATIVE_RU_COVERAGE',
                'FILTER_ADAPTER_READTHROUGH', 'EXACT_FRAMESHIFT_CALLER',
                'FRAMESHIFT_BACKGROUND_FILE', 'FRAMESHIFT_CALIBRATION_OUT')
        old = dict((key, getattr(settings, key)) for key in keys)
        try:
            settings.USE_REF_ALIGNMENT = False
            settings.USE_ONLY_FULLY_COVERED_RU = False
            settings.MIN_RELATIVE_RU_COVERAGE = None
            settings.FILTER_ADAPTER_READTHROUGH = False
            settings.EXACT_FRAMESHIFT_CALLER = False
            settings.FRAMESHIFT_CALIBRATION_OUT = None
            baseline = _make_finder().find_frameshift_from_selected_reads(_all_three_sites())
            resolved = context(use_reference_alignment=False)
            finder = _make_finder()
            finder.run_context = resolved
            settings.USE_REF_ALIGNMENT = True
            settings.USE_ONLY_FULLY_COVERED_RU = True
            settings.MIN_RELATIVE_RU_COVERAGE = 100.0
            settings.FILTER_ADAPTER_READTHROUGH = True
            settings.EXACT_FRAMESHIFT_CALLER = True
            settings.FRAMESHIFT_BACKGROUND_FILE = 'missing-background.json'
            settings.FRAMESHIFT_CALIBRATION_OUT = 'missing-directory/capture.jsonl'
            self.assertEqual(baseline, finder.find_frameshift_from_selected_reads(_all_three_sites()))
            finder.run_context = resolved._replace(capture=resolved.capture._replace(
                minimum_relative_ru_coverage=100.0))
            self.assertIsNone(finder.find_frameshift_from_selected_reads(_all_three_sites()))
            exact = resolved._replace(capture=resolved.capture._replace(caller_mode='exact'),
                background=BackgroundModel(1, 'invented', 0.99, {}, 'invented.json'))
            finder.run_context = exact
            self.assertIsNone(finder.find_frameshift_from_selected_reads(_all_three_sites()))
        finally:
            for key, value in old.items():
                setattr(settings, key, value)


class TestRunContextReadSelection(unittest.TestCase):
    def test_consecutive_read_selection_uses_explicit_length_quality_adapter_and_threads(self):
        class Read(object):
            seq = 'A' * 100
            is_unmapped = is_duplicate = False
            reference_start, reference_end = 101, 201
            mapq = 10
            query_qualities = [30] * 100
            query_name = 'invented'
        class Samfile(object):
            def head(self, count):
                return [Read()] * count
            def fetch(self, *args):
                return [Read()]
        class Model(object):
            dp_score_threshold = -1000.0
            def viterbi(self, sequence, **kwargs):
                return -200.0, _vpath(['M1_1'] * 60 + ['I1_1'] * 40)
        old_alignment = finder_module.pysam.AlignmentFile
        old_reference = finder_module.get_reference_genome_of_alignment_file
        old_decode = finder_module.read_selection.decode_pending
        recorded = []
        def decode(model, pending, threads, prune):
            recorded.append((threads, prune))
            old_decode(model, pending, threads, prune)
        finder_module.pysam.AlignmentFile = lambda *args, **kwargs: Samfile()
        finder_module.get_reference_genome_of_alignment_file = lambda samfile: 'HG19'
        finder_module.read_selection.decode_pending = decode
        old = dict((key, getattr(settings, key)) for key in (
            'MIN_READ_LENGTH', 'CORES', 'PRUNE_REVERSE_DECODE',
            'FILTER_ADAPTER_READTHROUGH', 'MIN_READ_MATCH_RATIO'))
        try:
            settings.MIN_READ_LENGTH, settings.CORES = 200, 7
            settings.PRUNE_REVERSE_DECODE = True
            settings.FILTER_ADAPTER_READTHROUGH, settings.MIN_READ_MATCH_RATIO = True, 0.99
            cases = (
                (dict(filter_adapter_readthrough=True), 1),  # None resolves to 0.60, exact boundary.
                (dict(filter_adapter_readthrough=True, minimum_read_match_ratio=0.8), 0),
                (dict(minimum_read_length=101), 0),
                (dict(mapq_cutoff=10), 0),
                (dict(base_quality_cutoff=31), 0),
                (dict(threads=2, prune_reverse=True), 1),
                (dict(), 1),
            )
            for fields, count in cases:
                finder = _make_finder()
                finder.run_context = context(**fields)
                finder.get_vntr_matcher_hmm = lambda read_length: Model()
                selected = finder.select_illumina_reads('invented.bam', [])
                self.assertEqual(count, len(selected), fields)
                self.assertEqual((fields.get('threads', 1), fields.get('prune_reverse', False)), recorded[-1])
                if selected:
                    self.assertEqual(('invented', -200.0), (selected[0].query_name, selected[0].logp))
        finally:
            finder_module.pysam.AlignmentFile = old_alignment
            finder_module.get_reference_genome_of_alignment_file = old_reference
            finder_module.read_selection.decode_pending = old_decode
            for key, value in old.items():
                setattr(settings, key, value)
